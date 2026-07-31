# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import TYPE_CHECKING, Any

from .chronos2 import CHRONOS2_ENCODER, replace_with_chronos2_encoder
from .timesfm2_5 import TIMESFM2_5_ENCODER, _resolve_encoder_type, maybe_replace_with_timesfm2_5_encoder
from .zeus import ZEUS_ENCODER, replace_with_zeus_encoder


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


SUPPORTED_EXTERNAL_TS_ENCODERS = (TIMESFM2_5_ENCODER, CHRONOS2_ENCODER, ZEUS_ENCODER)


def maybe_replace_timeseries_encoder(model: "PreTrainedModel", config: Any, model_args: "ModelArguments") -> None:
    r"""Replace ChatTS' native MLP with the selected frozen foundation model."""
    checkpoint_encoder_type = getattr(config, "ts_encoder_type", "native")
    encoder_type = _resolve_encoder_type(config, model_args)
    if encoder_type == "native":
        return
    if encoder_type == TIMESFM2_5_ENCODER:
        maybe_replace_with_timesfm2_5_encoder(model, config, model_args)
    elif encoder_type == CHRONOS2_ENCODER:
        replace_with_chronos2_encoder(model, config, model_args, checkpoint_encoder_type)
    elif encoder_type == ZEUS_ENCODER:
        replace_with_zeus_encoder(model, config, model_args, checkpoint_encoder_type)
    else:
        supported = ", ".join(("native",) + SUPPORTED_EXTERNAL_TS_ENCODERS)
        raise ValueError(f"Unsupported time-series encoder type: {encoder_type}. Choose one of: {supported}.")
