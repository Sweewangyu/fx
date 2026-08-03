# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)

TIMESFM2_5_ENCODER = "timesfm2_5"
TIMESFM2_5_HIDDEN_SIZE = 1280
TIMESFM2_5_PATCH_SIZE = 32
TIMESFM2_5_CONTEXT_LIMIT = 16384
TIMESFM2_5_NORM_EPSILON = 1e-6
CHATTS_MIN_INPUT_FEATURES = 2
CHATTS_VALID_MASK_THRESHOLD = 0.5
_PROJECTOR_PREFIX = "ts_encoder.projector."


class ExternalTimeSeriesProjector(nn.Module):
    r"""Map frozen backbone embeddings into the language-model hidden space."""

    def __init__(self, input_hidden_size: int, llm_hidden_size: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_hidden_size)
        self.linear_in = nn.Linear(input_hidden_size, llm_hidden_size)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.output_norm = nn.LayerNorm(llm_hidden_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.input_norm(features)
        features = self.linear_in(features)
        features = self.activation(features)
        features = self.linear_out(features)
        return self.output_norm(features)


class TimesFMProjector(ExternalTimeSeriesProjector):
    r"""Backward-compatible TimesFM 2.5 projector."""

    def __init__(self, llm_hidden_size: int) -> None:
        super().__init__(TIMESFM2_5_HIDDEN_SIZE, llm_hidden_size)


class TimesFM2_5TimeSeriesEncoder(nn.Module):
    r"""Frozen TimesFM 2.5 backbone plus a trainable TS-to-text projector.

    The input contract is identical to ChatTS' native ``TimeSeriesEmbedding``:
    ``x`` contains interleaved ``[value, valid_mask]`` features and the return
    value is ``(flattened_patch_features, patch_count_per_series)``.

    TimesFM itself is deliberately held by a non-Module wrapper. Its frozen
    200M parameters are downloaded from ``timesfm_model_name_or_path`` at first
    use and are not duplicated in ChatTS checkpoints. Only ``projector`` is a
    registered trainable submodule.
    """

    llamafactory_lora_target_prefixes: ClassVar[tuple[str, ...]] = ()
    llamafactory_modules_to_save: ClassVar[tuple[str, ...]] = ("ts_encoder.projector",)

    def __init__(
        self,
        llm_hidden_size: int,
        model_name_or_path: str,
        num_features: int = 2,
        context_limit: int = TIMESFM2_5_CONTEXT_LIMIT,
    ) -> None:
        super().__init__()
        if num_features < CHATTS_MIN_INPUT_FEATURES:
            raise ValueError("TimesFM 2.5 requires ChatTS inputs with value and valid-mask features.")

        self.patch_size = TIMESFM2_5_PATCH_SIZE
        self.hidden_size = llm_hidden_size
        self.num_features = num_features
        self.context_limit = context_limit
        self.model_name_or_path = model_name_or_path
        self.projector = TimesFMProjector(llm_hidden_size)

        # ``timesfm.TimesFM_2p5_200M_torch`` is a plain Python wrapper rather
        # than an nn.Module. Keeping it here prevents its frozen weights from
        # being serialized into every ChatTS checkpoint.
        self._timesfm: Any | None = None

    def _load_timesfm(self) -> Any:
        if sys.version_info < (3, 10):
            raise ImportError("TimesFM 2.5 requires Python 3.10 or newer.")

        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                'TimesFM 2.5 is not installed. Run `pip install -e ".[timesfm]"` from the ChatTS-Training repository.'
            ) from exc

        model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if model_class is None:
            raise ImportError(
                "The installed `timesfm` package does not expose `TimesFM_2p5_200M_torch`. Install timesfm>=2.0.2."
            )

        logger.info_rank0("Loading frozen TimesFM 2.5 backbone from `%s`.", self.model_name_or_path)
        timesfm_model = model_class.from_pretrained(self.model_name_or_path, torch_compile=False)
        backbone = timesfm_model.model
        backbone.eval()
        backbone.requires_grad_(False)
        return timesfm_model

    def _get_backbone(self, device: torch.device) -> nn.Module:
        if self._timesfm is None:
            self._timesfm = self._load_timesfm()

        backbone: nn.Module = self._timesfm.model
        backbone.to(device=device, dtype=torch.float32)
        backbone.eval()
        backbone.requires_grad_(False)
        return backbone

    @staticmethod
    def _update_running_stats(
        n: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        values: torch.Tensor,
        padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Match TimesFM 2.5's cumulative per-patch normalization."""
        valid = ~padding
        increment_n = valid.to(values.dtype).sum(dim=-1)
        safe_increment_n = increment_n.clamp_min(1.0)
        increment_mean = (values * valid).sum(dim=-1) / safe_increment_n
        increment_mean = torch.where(increment_n == 0, 0.0, increment_mean)
        increment_var = (((values - increment_mean.unsqueeze(-1)) ** 2) * valid).sum(dim=-1)
        increment_var = increment_var / safe_increment_n
        increment_var = torch.where(increment_n == 0, 0.0, increment_var)
        increment_std = torch.sqrt(increment_var)

        new_n = n + increment_n
        safe_new_n = new_n.clamp_min(1.0)
        new_mean = (n * mean + increment_n * increment_mean) / safe_new_n
        new_mean = torch.where(new_n == 0, 0.0, new_mean)

        new_var = (
            n * std.square()
            + increment_n * increment_std.square()
            + n * (mean - new_mean).square()
            + increment_n * (increment_mean - new_mean).square()
        ) / safe_new_n
        new_var = torch.where(new_n == 0, 0.0, new_var)
        new_std = torch.sqrt(new_var.clamp_min(0.0))
        return new_n, new_mean, new_std

    def _prepare_timesfm_batch(
        self, values: torch.Tensor, valid_mask: torch.Tensor, patch_cnt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Left-pad, patch, and normalize a variable-length ChatTS batch."""
        batch_size = values.size(0)
        max_patch_cnt = int(patch_cnt.max().item())
        padded_length = max_patch_cnt * self.patch_size
        padded_values = torch.zeros(batch_size, padded_length, device=values.device, dtype=torch.float32)
        padded_mask = torch.ones(batch_size, padded_length, device=values.device, dtype=torch.bool)

        for index in range(batch_size):
            series_values = values[index][valid_mask[index]].to(torch.float32)
            series_length = series_values.numel()
            if series_length == 0:
                continue

            left_padding = padded_length - series_length
            padded_values[index, left_padding:] = series_values
            padded_mask[index, left_padding:] = False

        patched_values = padded_values.reshape(batch_size, max_patch_cnt, self.patch_size)
        patched_mask = padded_mask.reshape(batch_size, max_patch_cnt, self.patch_size)

        n = torch.zeros(batch_size, device=values.device, dtype=torch.float32)
        mean = torch.zeros_like(n)
        std = torch.zeros_like(n)
        running_means = []
        running_stds = []
        for patch_index in range(max_patch_cnt):
            n, mean, std = self._update_running_stats(
                n,
                mean,
                std,
                patched_values[:, patch_index],
                patched_mask[:, patch_index],
            )
            running_means.append(mean)
            running_stds.append(std)

        cumulative_mean = torch.stack(running_means, dim=1).unsqueeze(-1)
        cumulative_std = torch.stack(running_stds, dim=1).unsqueeze(-1)
        safe_std = torch.where(cumulative_std < TIMESFM2_5_NORM_EPSILON, 1.0, cumulative_std)
        normalized_values = (patched_values - cumulative_mean) / safe_std
        normalized_values = normalized_values.masked_fill(patched_mask, 0.0)
        return normalized_values, patched_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)
        values = x[:, :, 0]
        valid_mask = x[:, :, -1] > CHATTS_VALID_MASK_THRESHOLD
        valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)

        if torch.any(valid_lengths == 0):
            raise ValueError("TimesFM 2.5 received an empty time series; every `<ts>` input must contain data.")

        if torch.any(valid_lengths > self.context_limit):
            longest = int(valid_lengths.max().item())
            raise ValueError(
                f"TimesFM 2.5 supports at most {self.context_limit} input points, but received {longest}."
            )

        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size
        normalized_values, padding_mask = self._prepare_timesfm_batch(values, valid_mask, patch_cnt)

        backbone = self._get_backbone(x.device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False) if x.device.type == "cuda" else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            (model_outputs, _) = backbone(normalized_values, padding_mask)
            output_embeddings = model_outputs[1]

        unpadded_embeddings = []
        for index, count in enumerate(patch_cnt.tolist()):
            unpadded_embeddings.append(output_embeddings[index, -count:])

        timesfm_features = torch.cat(unpadded_embeddings, dim=0)
        projector_param = next(self.projector.parameters())
        timesfm_features = timesfm_features.to(device=projector_param.device, dtype=projector_param.dtype)
        return self.projector(timesfm_features), patch_cnt


def _load_weight_file(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=True)


def _find_projector_weight_files(model_path: str) -> list[str]:
    if not os.path.isdir(model_path):
        return []

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = os.path.join(model_path, index_name)
        if not os.path.isfile(index_path):
            continue

        with open(index_path, encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})

        shard_names = {shard_name for key, shard_name in weight_map.items() if key.startswith(_PROJECTOR_PREFIX)}
        return [os.path.join(model_path, shard_name) for shard_name in sorted(shard_names)]

    for weight_name in ("model.safetensors", "pytorch_model.bin"):
        weight_path = os.path.join(model_path, weight_name)
        if os.path.isfile(weight_path):
            return [weight_path]

    return []


_PROJECTOR_STATE_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _read_external_projector_state(model_path: str) -> dict[str, torch.Tensor]:
    normalized_path = os.path.abspath(model_path)
    if normalized_path in _PROJECTOR_STATE_CACHE:
        return _PROJECTOR_STATE_CACHE[normalized_path]

    projector_state: dict[str, torch.Tensor] = {}
    for weight_path in _find_projector_weight_files(model_path):
        if weight_path.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(weight_path, framework="pt", device="cpu") as state:
                for key in state.keys():
                    if key.startswith(_PROJECTOR_PREFIX):
                        projector_state[key.removeprefix(_PROJECTOR_PREFIX)] = state.get_tensor(key).clone()
            continue

        state = _load_weight_file(weight_path)
        for key, value in state.items():
            if key.startswith(_PROJECTOR_PREFIX):
                projector_state[key.removeprefix(_PROJECTOR_PREFIX)] = value.detach().cpu().clone()
        del state

    _PROJECTOR_STATE_CACHE[normalized_path] = projector_state
    return projector_state


def infer_checkpoint_ts_encoder_type(config: Any, model_path: str) -> str:
    r"""Infer the saved encoder from projector weights when metadata is absent."""
    configured = getattr(config, "ts_encoder_type", None)
    if isinstance(configured, str):
        configured = configured.strip().lower()
    if configured in ("", "auto"):
        configured = None

    projector_state = _read_external_projector_state(model_path)
    input_dims: set[int] = set()
    input_norm = projector_state.get("input_norm.weight")
    if input_norm is not None and input_norm.ndim:
        input_dims.add(int(input_norm.shape[0]))
    linear_in = projector_state.get("linear_in.weight")
    if linear_in is not None and linear_in.ndim >= 2:
        input_dims.add(int(linear_in.shape[-1]))

    if input_dims == {TIMESFM2_5_HIDDEN_SIZE}:
        if configured not in (None, "native", TIMESFM2_5_ENCODER):
            logger.warning_rank0(
                "Checkpoint config says ts_encoder_type=%s, but projector weights are 1280-d TimesFM 2.5; "
                "using the weight evidence.",
                configured,
            )
        return TIMESFM2_5_ENCODER

    if input_dims == {768}:
        if configured in ("chronos2", "zeus"):
            return configured
        ts_config = getattr(config, "ts", {}) or {}
        patch_size = ts_config.get("patch_size")
        try:
            patch_size = int(patch_size)
        except (TypeError, ValueError):
            patch_size = None
        if patch_size == 16:
            return "chronos2"
        if patch_size == 32:
            return "zeus"
        return "external_768_ambiguous"

    if input_dims:
        raise ValueError(f"Unsupported external projector input dimensions in `{model_path}`: {sorted(input_dims)}.")
    return configured or "native"


def _resolve_encoder_type(config: Any, model_args: ModelArguments) -> str:
    requested = model_args.ts_encoder_type
    if requested != "auto":
        return requested

    encoder_type = infer_checkpoint_ts_encoder_type(config, model_args.model_name_or_path)
    if encoder_type == "external_768_ambiguous":
        raise ValueError(
            "The checkpoint contains a 768-d external projector but lacks metadata to distinguish Chronos-2 "
            "from Zeus. Set `--ts_encoder_type chronos2` or `--ts_encoder_type zeus`."
        )
    return encoder_type


def load_external_projector_from_checkpoint(
    encoder: nn.Module, model_path: str, encoder_name: str = "external time-series"
) -> bool:
    projector_state = _read_external_projector_state(model_path)
    if not projector_state:
        return False

    incompatible = encoder.projector.load_state_dict(projector_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"Incomplete {encoder_name} projector checkpoint: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
        )

    logger.info_rank0("Loaded %s projector weights from `%s`.", encoder_name, model_path)
    return True


def _load_projector_from_checkpoint(encoder: nn.Module, model_path: str) -> bool:
    r"""Compatibility alias used by older integrations and tests."""
    return load_external_projector_from_checkpoint(encoder, model_path, "TimesFM 2.5")


def maybe_replace_with_timesfm2_5_encoder(model: PreTrainedModel, config: Any, model_args: ModelArguments) -> None:
    checkpoint_encoder_type = infer_checkpoint_ts_encoder_type(config, model_args.model_name_or_path)
    encoder_type = _resolve_encoder_type(config, model_args)
    if encoder_type == "native":
        return
    if encoder_type != TIMESFM2_5_ENCODER:
        raise ValueError(f"Unsupported time-series encoder type: {encoder_type}.")
    if not hasattr(model, "ts_encoder"):
        raise ValueError("The selected model does not expose a `ts_encoder` module.")
    if model_args.use_unsloth:
        raise ValueError("TimesFM 2.5 encoder replacement is not supported with Unsloth.")

    ts_config = getattr(config, "ts", {}) or {}
    timesfm_model_name_or_path = model_args.timesfm_model_name_or_path
    if model_args.ts_encoder_type == "auto":
        timesfm_model_name_or_path = getattr(config, "timesfm_model_name_or_path", timesfm_model_name_or_path)

    encoder = TimesFM2_5TimeSeriesEncoder(
        llm_hidden_size=config.hidden_size,
        model_name_or_path=timesfm_model_name_or_path,
        num_features=int(ts_config.get("num_features", 2)),
    )

    reference_weight = model.get_input_embeddings().weight
    encoder.projector.to(device=reference_weight.device, dtype=reference_weight.dtype)
    model.ts_encoder = encoder

    config.ts_encoder_type = TIMESFM2_5_ENCODER
    config.timesfm_model_name_or_path = timesfm_model_name_or_path
    config.timesfm_hidden_size = TIMESFM2_5_HIDDEN_SIZE
    if not hasattr(config, "ts") or config.ts is None:
        config.ts = {}
    config.ts["patch_size"] = TIMESFM2_5_PATCH_SIZE
    model.config = config

    restored = False
    if checkpoint_encoder_type == TIMESFM2_5_ENCODER:
        restored = _load_projector_from_checkpoint(encoder, model_args.model_name_or_path)
    if getattr(config, "ts_encoder_type", None) == TIMESFM2_5_ENCODER and not restored:
        logger.info_rank0("Initialized a new TimesFM 2.5 TS-to-text projector.")

    logger.info_rank0(
        "Replaced ChatTS native encoder with frozen TimesFM 2.5 (%s); trainable projector: 1280 -> %d.",
        timesfm_model_name_or_path,
        config.hidden_size,
    )
