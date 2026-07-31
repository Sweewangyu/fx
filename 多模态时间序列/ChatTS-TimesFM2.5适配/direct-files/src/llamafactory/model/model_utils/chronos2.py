# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import sys
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


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)

CHRONOS2_ENCODER = "chronos2"
CHRONOS2_HIDDEN_SIZE = 768
CHRONOS2_PATCH_SIZE = 16
CHRONOS2_CONTEXT_LIMIT = 8192


class _Chronos2Handle:
    r"""Non-Module holder, so the frozen model is omitted from ChatTS checkpoints."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline


class Chronos2TimeSeriesEncoder(nn.Module):
    r"""Frozen Chronos-2 encoder plus a trainable TS-to-text projector.

    Chronos-2 adds a register token and a masked-future token around its
    context representation. This adapter keeps context patch tokens only.
    """

    llamafactory_lora_target_prefixes: ClassVar[tuple[str, ...]] = ()
    llamafactory_modules_to_save: ClassVar[tuple[str, ...]] = ("ts_encoder.projector",)

    def __init__(
        self,
        llm_hidden_size: int,
        model_name_or_path: str,
        num_features: int = 2,
        context_limit: int = CHRONOS2_CONTEXT_LIMIT,
    ) -> None:
        super().__init__()
        if num_features < CHATTS_MIN_INPUT_FEATURES:
            raise ValueError("Chronos-2 requires ChatTS inputs with value and valid-mask features.")

        self.patch_size = CHRONOS2_PATCH_SIZE
        self.hidden_size = llm_hidden_size
        self.num_features = num_features
        self.context_limit = context_limit
        self.model_name_or_path = model_name_or_path
        self.projector = ExternalTimeSeriesProjector(CHRONOS2_HIDDEN_SIZE, llm_hidden_size)
        self._chronos2: _Chronos2Handle | None = None

    def _load_chronos2(self) -> _Chronos2Handle:
        if sys.version_info < (3, 10):
            raise ImportError("Chronos-2 requires Python 3.10 or newer.")

        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                'Chronos-2 is not installed. Run `pip install -e ".[chronos2]"` from the ChatTS-Training repository.'
            ) from exc

        logger.info_rank0("Loading frozen Chronos-2 backbone from `%s`.", self.model_name_or_path)
        pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name_or_path,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        pipeline.model.eval()
        pipeline.model.requires_grad_(False)
        return _Chronos2Handle(pipeline)

    def _get_backbone(self, device: torch.device) -> nn.Module:
        if self._chronos2 is None:
            self._chronos2 = self._load_chronos2()

        backbone: nn.Module = self._chronos2.pipeline.model
        backbone.to(device=device, dtype=torch.float32)
        backbone.eval()
        backbone.requires_grad_(False)
        return backbone

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)
        values = x[:, :, 0]
        valid_mask = x[:, :, -1] > CHATTS_VALID_MASK_THRESHOLD
        valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)

        if torch.any(valid_lengths == 0):
            raise ValueError("Chronos-2 received an empty time series; every `<ts>` input must contain data.")
        if torch.any(valid_lengths > self.context_limit):
            longest = int(valid_lengths.max().item())
            raise ValueError(f"Chronos-2 supports at most {self.context_limit} input points, but received {longest}.")

        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size
        backbone = self._get_backbone(x.device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False) if x.device.type == "cuda" else nullcontext()
        )

        # Calling encode per series preserves the official left-padding behavior
        # for variable lengths and avoids leaking right-padding into patch tokens.
        context_features = []
        with torch.no_grad(), autocast_context:
            for index, count in enumerate(patch_cnt.tolist()):
                context = values[index][valid_mask[index]].to(device=x.device, dtype=torch.float32).unsqueeze(0)
                group_ids = torch.zeros(1, device=x.device, dtype=torch.long)
                encoder_outputs, _, _, num_context_patches = backbone.encode(
                    context=context,
                    group_ids=group_ids,
                )
                returned_count = int(num_context_patches)
                if returned_count != count:
                    raise RuntimeError(
                        f"Chronos-2 returned {returned_count} context patches, expected {count} for this series."
                    )
                context_features.append(encoder_outputs.last_hidden_state[:, :count].squeeze(0))

        chronos_features = torch.cat(context_features, dim=0)
        projector_param = next(self.projector.parameters())
        chronos_features = chronos_features.to(device=projector_param.device, dtype=projector_param.dtype)
        return self.projector(chronos_features), patch_cnt


def replace_with_chronos2_encoder(
    model: PreTrainedModel,
    config: Any,
    model_args: ModelArguments,
    checkpoint_encoder_type: str,
) -> None:
    if not hasattr(model, "ts_encoder"):
        raise ValueError("The selected model does not expose a `ts_encoder` module.")
    if model_args.use_unsloth:
        raise ValueError("Chronos-2 encoder replacement is not supported with Unsloth.")

    ts_config = getattr(config, "ts", {}) or {}
    model_name_or_path = model_args.chronos2_model_name_or_path
    if model_args.ts_encoder_type == "auto":
        model_name_or_path = getattr(config, "chronos2_model_name_or_path", model_name_or_path)

    encoder = Chronos2TimeSeriesEncoder(
        llm_hidden_size=config.hidden_size,
        model_name_or_path=model_name_or_path,
        num_features=int(ts_config.get("num_features", 2)),
    )
    reference_weight = model.get_input_embeddings().weight
    encoder.projector.to(device=reference_weight.device, dtype=reference_weight.dtype)
    model.ts_encoder = encoder

    config.ts_encoder_type = CHRONOS2_ENCODER
    config.chronos2_model_name_or_path = model_name_or_path
    config.chronos2_hidden_size = CHRONOS2_HIDDEN_SIZE
    if not hasattr(config, "ts") or config.ts is None:
        config.ts = {}
    config.ts["patch_size"] = CHRONOS2_PATCH_SIZE

    restored = False
    if checkpoint_encoder_type == CHRONOS2_ENCODER:
        restored = load_external_projector_from_checkpoint(encoder, model_args.model_name_or_path, "Chronos-2")
    if not restored:
        logger.info_rank0("Initialized a new Chronos-2 TS-to-text projector.")

    logger.info_rank0(
        "Replaced ChatTS native encoder with frozen Chronos-2 (%s); trainable projector: 768 -> %d.",
        model_name_or_path,
        config.hidden_size,
    )
