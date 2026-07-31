# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This is an eager-attention compatibility implementation of the Apache-2.0
# GestaltCog/zeus and BasicTS modules. Module names and tensor shapes are kept
# compatible with the official Zeus checkpoint. It intentionally omits the
# forecasting/generation code that is not used by the ChatTS feature adapter.

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


_ACT2FN = {
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "prelu": nn.PReLU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
}


class ZeusConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: list[int] | None = None,
        n_heads: list[int] | None = None,
        intermediate_size: list[int] | None = None,
        dropout: float = 0.1,
        hidden_act: str = "silu",
        num_reg_tokens: int = 4,
        num_layers: list[int] | None = None,
        scales: list[int] | None = None,
        quantiles: list[float] | None = None,
        initializer_range: float = 0.02,
        num_latent_tokens: int = 2,
        attn_implementation: str = "eager",
        use_latent_tokens: bool = True,
        **kwargs,
    ) -> None:
        # Zeus' published checkpoint also contains `num_heads`, but the
        # official code reads `n_heads`; preserve that artifact behavior.
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size or [256, 256, 512, 512, 512, 256, 256]
        self.n_heads = n_heads or [4, 4, 8, 8, 8, 4, 4]
        self.intermediate_size = intermediate_size or [1024, 1024, 2048, 2048, 2048, 1024, 1024]
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.num_reg_tokens = num_reg_tokens
        self.num_layers = num_layers or [1, 1, 1, 1, 1, 1, 1]
        self.scales = scales or [1, 4, 16, 64, 16, 4, 1]
        self.quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.initializer_range = initializer_range
        self.num_latent_tokens = num_latent_tokens
        self.attn_implementation = attn_implementation
        self.use_latent_tokens = use_latent_tokens


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.eps)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self._set_cos_sin_cache(max_position_embeddings, torch.device("cpu"))

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device) -> None:
        self.max_seq_len_cached = seq_len
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device))
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("_cos", embeddings.cos(), persistent=False)
        self.register_buffer("_sin", embeddings.sin(), persistent=False)

    @staticmethod
    def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_position = int(position_ids.max()) + 1
        if max_position > self.max_seq_len_cached or self._cos.device != query.device:
            self._set_cos_sin_cache(max(max_position, self.max_seq_len_cached), query.device)
        cosine = self._cos[position_ids].unsqueeze(1).to(query.dtype)
        sine = self._sin[position_ids].unsqueeze(1).to(query.dtype)
        query = query * cosine + self._rotate_half(query) * sine
        key = key * cosine + self._rotate_half(key) * sine
        return query, key


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.0,
        rope: RotaryPositionEmbedding | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden size {hidden_size} is not divisible by {n_heads} heads")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        # The published checkpoint was trained with ZeusFlashAttention, whose
        # output projection has no bias. Keeping this exact shape prevents a
        # silently initialized `out_proj.bias` in eager mode.
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def _shape(self, tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        return tensor.view(tensor.size(0), seq_len, self.n_heads, self.head_size).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        batch_size, seq_len, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), seq_len)
        key = self._shape(self.k_proj(hidden_states), seq_len)
        value = self._shape(self.v_proj(hidden_states), seq_len)
        if self.rope is not None:
            if position_ids is None:
                raise ValueError("position_ids is required by Zeus RoPE")
            query, key = self.rope(query, key, position_ids)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(context), None, None


class ZeusMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = _ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class ZeusInputEmbedding(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, hidden_act: str = "gelu") -> None:
        super().__init__()
        intermediate_size = 4 * hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.res_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(input_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(input_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = _ACT2FN[hidden_act]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.res_proj(inputs) + self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))


class ZeusEncoderLayer(nn.Module):
    def __init__(self, config: ZeusConfig, stage: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size[stage]
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=config.n_heads[stage],
            dropout=config.dropout,
            rope=RotaryPositionEmbedding(hidden_size // config.n_heads[stage]),
        )
        self.ffn_layer = ZeusMLP(hidden_size, config.intermediate_size[stage], config.hidden_act)
        self.pre_attn_norm = RMSNorm(hidden_size)
        self.pre_ffn_norm = RMSNorm(hidden_size)
        self.post_attn_norm = None
        self.post_ffn_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.pre_attn_norm(hidden_states)
        attention_output, _, _ = self.self_attn(
            normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.pre_ffn_norm(hidden_states)
        return residual + self.ffn_layer(hidden_states)


class ZeusEncoder(nn.Module):
    def __init__(self, config: ZeusConfig, stage: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ZeusEncoderLayer(config, stage) for _ in range(config.num_layers[stage])])
        self.layer_norm = RMSNorm(config.hidden_size[stage])
        self.num_reg_tokens = config.num_reg_tokens
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(
                torch.randn(1, self.num_reg_tokens, config.hidden_size[stage]) * config.initializer_range
            )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        if self.num_reg_tokens > 0:
            reg_tokens = self.reg_tokens.expand(batch_size, -1, -1)
            hidden_states = torch.cat((reg_tokens, hidden_states), dim=1)
            reg_positions = torch.zeros(1, self.num_reg_tokens, dtype=torch.long, device=hidden_states.device)
            position_ids = torch.cat((reg_positions, position_ids), dim=1)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids.expand(batch_size, -1),
            )
        hidden_states = self.layer_norm(hidden_states)
        reg_tokens = hidden_states[:, : self.num_reg_tokens]
        return hidden_states[:, self.num_reg_tokens :], reg_tokens


class ZeusPoolingLayer(nn.Module):
    def __init__(self, config: ZeusConfig, stage: int) -> None:
        super().__init__()
        self.stage = stage
        self.config = config
        self.factor = config.scales[stage] // config.scales[stage - 1]
        self.proj = nn.Linear(self.factor * config.hidden_size[stage - 1], config.hidden_size[stage], bias=False)

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, hidden_size = hidden_states.size()
        hidden_states = hidden_states.reshape(batch_size, -1, self.factor * hidden_size)
        hidden_states = self.proj(hidden_states)
        padding_mask = padding_mask.reshape(batch_size, -1, self.factor, 1).any(dim=2)
        return hidden_states, padding_mask


class ZeusUnpoolingLayer(nn.Module):
    def __init__(self, config: ZeusConfig, stage: int) -> None:
        super().__init__()
        self.stage = stage
        self.config = config
        self.factor = config.scales[stage - 1] // config.scales[stage]
        self.proj = nn.Linear(config.hidden_size[stage - 1], self.factor * config.hidden_size[stage], bias=False)

    def forward(self, hidden_states: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = skip_connection.size()
        hidden_states = self.proj(hidden_states).reshape(batch_size, -1, hidden_size)
        return hidden_states + skip_connection


class ZeusForPrediction(PreTrainedModel):
    r"""Official-checkpoint-compatible Zeus forward pass using eager attention."""

    config_class = ZeusConfig
    base_model_prefix = ""
    main_input_name = "inputs"

    def __init__(self, config: ZeusConfig) -> None:
        super().__init__(config)
        self.scales = config.scales
        self.num_reg_tokens = config.num_reg_tokens
        self.num_scales = len(self.scales)
        self.input_mlp = ZeusInputEmbedding(config.input_dim, config.hidden_size[0], config.hidden_act)
        self.special_tokens = nn.Embedding(2, config.hidden_size[0])
        self.pad_token_id = 0
        self.mask_token_id = 1
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.encoders.append(ZeusEncoder(config, 0))
        for stage in range(1, self.num_scales // 2 + 1):
            self.encoders.append(ZeusEncoder(config, stage))
            self.downsamplers.append(ZeusPoolingLayer(config, stage))
        for stage in range(self.num_scales // 2 + 1, self.num_scales):
            self.encoders.append(ZeusEncoder(config, stage))
            self.upsamplers.append(ZeusUnpoolingLayer(config, stage))
        self.head = nn.Linear(config.hidden_size[-1], len(config.quantiles))
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _prepare_embedding(
        self, inputs: torch.Tensor, targets_mask: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = inputs.shape
        input_embeds = self.input_mlp(inputs)
        is_target = targets_mask == 1
        target_tokens = self.special_tokens(torch.full_like(targets_mask.squeeze(-1), self.mask_token_id))
        input_embeds = torch.where(is_target, target_tokens, input_embeds)
        if padding_mask is None:
            padding_mask = torch.ones(batch_size, seq_len, 1, device=input_embeds.device, dtype=torch.long)
        padding_tokens = self.special_tokens(torch.full_like(padding_mask.squeeze(-1), self.pad_token_id))
        input_embeds = torch.where(padding_mask == 0, padding_tokens, input_embeds)

        max_scale = max(self.scales)
        pad_len = math.ceil(seq_len / max_scale) * max_scale - seq_len
        if pad_len:
            pad_tokens = self.special_tokens(
                torch.full((batch_size, pad_len), self.pad_token_id, device=input_embeds.device)
            )
            input_embeds = torch.cat((input_embeds, pad_tokens), dim=1)
            padding_mask = torch.cat(
                (
                    padding_mask,
                    torch.zeros(batch_size, pad_len, 1, device=input_embeds.device, dtype=padding_mask.dtype),
                ),
                dim=1,
            )
        return input_embeds, padding_mask

    def _prepare_attn_mask(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        if self.num_reg_tokens > 0:
            reg_mask = torch.ones(
                batch_size,
                self.num_reg_tokens,
                1,
                device=hidden_states.device,
                dtype=padding_mask.dtype,
            )
            padding_mask = torch.cat((reg_mask, padding_mask), dim=1)
        attention_mask = padding_mask.view(batch_size, 1, 1, -1)
        return (1 - attention_mask.float()) * torch.finfo(hidden_states.dtype).min

    def forward(
        self,
        inputs: torch.Tensor,
        targets_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | float | None]:
        original_length = inputs.shape[1]
        original_padding_mask = padding_mask
        hidden_states, padding_mask = self._prepare_embedding(inputs, targets_mask, padding_mask)
        scale_outputs: list[torch.Tensor] = []
        scale_padding_masks: list[torch.Tensor] = []
        all_hidden_states: list[torch.Tensor] = []
        reg_token_emb = None

        for stage in range(self.num_scales):
            if stage > 0 and stage <= self.num_scales // 2:
                scale_padding_masks.append(padding_mask)
                hidden_states, padding_mask = self.downsamplers[stage - 1](hidden_states, padding_mask)
            elif stage > self.num_scales // 2:
                index = stage - self.num_scales // 2 - 1
                hidden_states = self.upsamplers[index](hidden_states, scale_outputs[self.num_scales - stage - 1])
                padding_mask = scale_padding_masks[self.num_scales - stage - 1]

            attention_mask = self._prepare_attn_mask(hidden_states, padding_mask)
            hidden_states, reg_tokens = self.encoders[stage](hidden_states, attention_mask)
            if stage == self.num_scales - 2:
                reg_token_emb = reg_tokens.mean(dim=1)
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)
            scale_outputs.append(hidden_states)

        quantile_preds = self.head(hidden_states)[:, :original_length]
        loss: torch.Tensor | float = 0.0
        if targets is not None:
            if original_padding_mask is None:
                original_padding_mask = torch.ones_like(targets_mask)
            loss_mask = (targets_mask * original_padding_mask).float()
            quantiles = torch.tensor(
                self.config.quantiles, device=quantile_preds.device, dtype=quantile_preds.dtype
            ).view(1, 1, -1)
            loss = 2 * torch.abs((targets - quantile_preds) * ((targets <= quantile_preds).float() - quantiles))
            loss = (loss * loss_mask).sum() / (loss_mask.sum() * len(self.config.quantiles))

        return {
            "prediction": quantile_preds,
            "loss": loss,
            "all_hidden_states": all_hidden_states,
            "reg_token_emb": reg_token_emb,
        }
