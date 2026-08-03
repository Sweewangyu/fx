# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from ...extras import logging
from .timesfm2_5 import (
    CHATTS_MIN_INPUT_FEATURES,
    CHATTS_VALID_MASK_THRESHOLD,
    ExternalTimeSeriesProjector,
    load_external_projector_from_checkpoint,
)
from .zeus_modeling import ZeusConfig, ZeusForPrediction


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)

ZEUS_ENCODER = "zeus"
ZEUS_HIDDEN_SIZE = 768
ZEUS_OUTPUT_SCALE = 32
ZEUS_CENTER_STAGE = 2
ZEUS_CONTEXT_LIMIT = 4096
ZEUS_NORM_EPSILON = 1e-5


class _ZeusHandle:
    r"""Non-Module holder, so frozen Zeus weights are not checkpointed."""

    def __init__(self, model: ZeusForPrediction) -> None:
        self.model = model


class ZeusTimeSeriesEncoder(nn.Module):
    r"""Frozen Zeus U-shaped encoder plus a trainable TS-to-text projector."""

    llamafactory_lora_target_prefixes: ClassVar[tuple[str, ...]] = ()
    llamafactory_modules_to_save: ClassVar[tuple[str, ...]] = ("ts_encoder.projector",)

    def __init__(
        self,
        llm_hidden_size: int,
        model_name_or_path: str,
        num_features: int = 2,
        context_limit: int = ZEUS_CONTEXT_LIMIT,
    ) -> None:
        super().__init__()
        if num_features < CHATTS_MIN_INPUT_FEATURES:
            raise ValueError("Zeus requires ChatTS inputs with value and valid-mask features.")

        self.patch_size = ZEUS_OUTPUT_SCALE
        self.hidden_size = llm_hidden_size
        self.num_features = num_features
        self.context_limit = context_limit
        self.model_name_or_path = model_name_or_path
        self.projector = ExternalTimeSeriesProjector(ZEUS_HIDDEN_SIZE, llm_hidden_size)
        self._zeus: _ZeusHandle | None = None

    def _load_zeus(self) -> _ZeusHandle:
        logger.info_rank0("Loading frozen Zeus backbone from `%s`.", self.model_name_or_path)
        config = ZeusConfig.from_pretrained(self.model_name_or_path)
        config.attn_implementation = "eager"
        model = ZeusForPrediction.from_pretrained(
            self.model_name_or_path,
            config=config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.requires_grad_(False)
        return _ZeusHandle(model)

    def _get_backbone(self, device: torch.device) -> ZeusForPrediction:
        if self._zeus is None:
            self._zeus = self._load_zeus()

        backbone = self._zeus.model
        backbone.to(device=device, dtype=torch.float32)
        backbone.eval()
        backbone.requires_grad_(False)
        return backbone

    def _normalize(self, values: torch.Tensor, valid_mask: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.to(torch.float32)
        denominator = valid_lengths.to(torch.float32).view(-1, 1).clamp_min(1.0)
        mean = (values.to(torch.float32) * mask).sum(dim=1, keepdim=True) / denominator
        centered = (values.to(torch.float32) - mean) * mask
        variance = centered.square().sum(dim=1, keepdim=True) / denominator
        normalized = torch.arcsinh(centered / torch.sqrt(variance + ZEUS_NORM_EPSILON))
        return normalized.masked_fill(~valid_mask, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)
        values = x[:, :, 0]
        valid_mask = x[:, :, -1] > CHATTS_VALID_MASK_THRESHOLD
        valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)

        if torch.any(valid_lengths == 0):
            raise ValueError("Zeus received an empty time series; every `<ts>` input must contain data.")
        if torch.any(valid_lengths > self.context_limit):
            longest = int(valid_lengths.max().item())
            raise ValueError(f"Zeus supports at most {self.context_limit} input points, but received {longest}.")

        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size
        packed_length = int(valid_lengths.max().item())
        packed_values = torch.zeros(batch_size, packed_length, device=x.device, dtype=torch.float32)
        packed_mask = torch.zeros(batch_size, packed_length, device=x.device, dtype=torch.bool)
        for index, length in enumerate(valid_lengths.tolist()):
            packed_values[index, :length] = values[index][valid_mask[index]].to(torch.float32)
            packed_mask[index, :length] = True

        normalized = self._normalize(packed_values, packed_mask, valid_lengths).unsqueeze(-1)
        padding_mask = packed_mask.to(torch.int32).unsqueeze(-1)
        targets_mask = torch.zeros_like(padding_mask)
        backbone = self._get_backbone(x.device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False) if x.device.type == "cuda" else nullcontext()
        )

        with torch.no_grad(), autocast_context:
            outputs = backbone(
                normalized,
                targets_mask=targets_mask,
                padding_mask=padding_mask,
                return_all_hidden_states=True,
            )
            center_features = outputs["all_hidden_states"][ZEUS_CENTER_STAGE]

        unpadded_features = [center_features[index, :count] for index, count in enumerate(patch_cnt.tolist())]
        zeus_features = torch.cat(unpadded_features, dim=0)
        projector_param = next(self.projector.parameters())
        zeus_features = zeus_features.to(device=projector_param.device, dtype=projector_param.dtype)
        return self.projector(zeus_features), patch_cnt


def replace_with_zeus_encoder(
    model: PreTrainedModel,
    config: Any,
    model_args: ModelArguments,
    checkpoint_encoder_type: str,
) -> None:
    if not hasattr(model, "ts_encoder"):
        raise ValueError("The selected model does not expose a `ts_encoder` module.")
    if model_args.use_unsloth:
        raise ValueError("Zeus encoder replacement is not supported with Unsloth.")

    ts_config = getattr(config, "ts", {}) or {}
    model_name_or_path = model_args.zeus_model_name_or_path
    if model_args.ts_encoder_type == "auto":
        model_name_or_path = getattr(config, "zeus_model_name_or_path", model_name_or_path)

    encoder = ZeusTimeSeriesEncoder(
        llm_hidden_size=config.hidden_size,
        model_name_or_path=model_name_or_path,
        num_features=int(ts_config.get("num_features", 2)),
    )
    reference_weight = model.get_input_embeddings().weight
    encoder.projector.to(device=reference_weight.device, dtype=reference_weight.dtype)
    model.ts_encoder = encoder

    config.ts_encoder_type = ZEUS_ENCODER
    config.zeus_model_name_or_path = model_name_or_path
    config.zeus_hidden_size = ZEUS_HIDDEN_SIZE
    config.zeus_output_scale = ZEUS_OUTPUT_SCALE
    if not hasattr(config, "ts") or config.ts is None:
        config.ts = {}
    config.ts["patch_size"] = ZEUS_OUTPUT_SCALE
    model.config = config

    restored = False
    if checkpoint_encoder_type == ZEUS_ENCODER:
        restored = load_external_projector_from_checkpoint(encoder, model_args.model_name_or_path, "Zeus")
    if not restored:
        logger.info_rank0("Initialized a new Zeus TS-to-text projector.")

    logger.info_rank0(
        "Replaced ChatTS native encoder with frozen Zeus (%s); center scale 32, trainable projector: 768 -> %d.",
        model_name_or_path,
        config.hidden_size,
    )
