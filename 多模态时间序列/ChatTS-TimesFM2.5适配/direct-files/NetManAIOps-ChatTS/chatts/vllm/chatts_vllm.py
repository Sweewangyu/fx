# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Reference: vLLM (https://github.com/vllm-project/vllm)
# Credit: Alexander Chemeris

"""Inference-only Qwen2/Qwen3-ChatTS models compatible with HuggingFace weights."""
from contextlib import nullcontext
from collections.abc import Iterable, Mapping, Sequence
import os
import sys
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig, ProcessorMixin

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (SupportsLoRA,
                                                   SupportsMultiModal,
                                                   SupportsPP)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              init_vllm_registered_model,
                                              maybe_prefix,
                                              merge_multimodal_embeddings)
try:
    from vllm.model_executor.sampling_metadata import SamplingMetadata
except ImportError:
    SamplingMetadata = None

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.parse import MultiModalDataParser, ProcessorBatchItems, EmbeddingItems
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, MultiModalDataDict,
                                        MultiModalDataItems,
                                        MultiModalFieldConfig, PromptReplacement,
                                        PromptUpdateDetails)
try:
    from vllm.multimodal.processing import MultiModalKwargs
except ImportError:
    MultiModalKwargs = None

from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of
from vllm import ModelRegistry
import math


########################MLP TS Embedding (20251015 Version)#####################
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, config):
        super(TimeSeriesEmbedding, self).__init__()
        self.patch_size = config['patch_size']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.num_features = config['num_features']
        self.max_sequence_length = config['max_sequence_length']  # Maximum time series length
        self.use_position_embedding = config.get('use_position_embedding', False)
        self.use_position_idx = config.get('use_position_idx', False)
        self.embedding_dim = config.get('embedding_dim', 16)  # Embedding dimension

        if self.use_position_embedding:
            # Extended vocabulary: [0, max_sequence_length) for real positions, max_sequence_length for padding
            self.position_embedding = nn.Embedding(self.max_sequence_length + 1, self.embedding_dim)
            self.padding_idx = self.max_sequence_length  # Special index for padding
            input_size = 1 * self.patch_size + self.embedding_dim * self.patch_size
        elif self.use_position_idx:
            input_size = 2 * self.patch_size
        else:
            input_size = 1 * self.patch_size

        # Build MLP layers
        layers = []
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(input_size, self.hidden_size))
            layers.append(nn.GELU())
            input_size = self.hidden_size
        layers.append(nn.Linear(input_size, self.hidden_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)

        # Extract mask and calculate valid lengths
        mask = x[:, :, -1].long()
        valid_lengths = mask.sum(dim=1).long()
        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size

        patches_list = []
        # Collect position indices for batch embedding lookup
        all_position_indices = []
        patch_info_list = []  # Store metadata for each patch group

        for i in range(batch_size):
            vl = valid_lengths[i].item()
            pc = patch_cnt[i].item()
            if pc == 0:
                continue

            # Extract time series data (excluding mask)
            xi = x[i, :vl, :1]  # Time-series data
            total_padded_length = pc * self.patch_size
            padding_length = total_padded_length - vl

            # Create position indices: real positions for actual data, special index for padding
            position_indices = torch.arange(vl, device=x.device)

            if padding_length > 0:
                # Pad with last value
                last_value = xi[-1:, :]
                padding = last_value.repeat(padding_length, 1)
                xi = torch.cat([xi, padding], dim=0)

                # Use special padding index for padding positions
                padding_positions = torch.full((padding_length,), self.padding_idx, device=x.device)
                position_indices = torch.cat([position_indices, padding_positions], dim=0)

            # Reshape to patches
            xi = xi.reshape(pc, self.patch_size)  # (num_patches, patch_size)
            position_indices = position_indices.reshape(pc, self.patch_size)  # (num_patches, patch_size)

            if self.use_position_embedding:
                # Collect position indices instead of calling embedding immediately
                all_position_indices.append(position_indices)
                patch_info_list.append({
                    'xi': xi,
                    'pc': pc,
                    'sample_idx': i
                })
            elif self.use_position_idx:
                # Normalize position indices
                pos_indices = torch.arange(vl, device=x.device).unsqueeze(1)
                pos_indices = pos_indices / max(1, valid_lengths.max().item() - 1)
                if padding_length > 0:
                    # Use -1 for padding positions
                    padding_indices = torch.full((padding_length, 1), -1, device=x.device)
                    pos_indices = torch.cat([pos_indices, padding_indices], dim=0)
                # Combine time series data with position indices
                xi_combined = torch.cat([xi.reshape(-1, 1), pos_indices], dim=1)
                patch_input = xi_combined.reshape(pc, self.patch_size * 2)
                patches_list.append(patch_input)
            else:
                # No position embedding, use raw patches
                patch_input = xi
                patches_list.append(patch_input)

        # Batch process position embeddings if needed
        if self.use_position_embedding and all_position_indices:
            # Concatenate all position indices for batch embedding lookup
            batch_position_indices = torch.cat(all_position_indices, dim=0)
            # print(f"{x.shape=}, {x.device=}, {len(all_position_indices)=}, {batch_position_indices=}")
            batch_pos_emb = self.position_embedding(batch_position_indices)  # Single embedding call

            # Split embeddings back and create patch inputs
            emb_start_idx = 0
            for patch_info in patch_info_list:
                xi = patch_info['xi']
                pc = patch_info['pc']

                # Extract corresponding embeddings
                pos_emb = batch_pos_emb[emb_start_idx:emb_start_idx + pc]
                emb_start_idx += pc

                # Flatten and concatenate
                xi = xi.unsqueeze(-1)  # (num_patches, patch_size, 1)
                patch_input = torch.cat([
                    xi.flatten(1),  # (num_patches, patch_size)
                    pos_emb.flatten(1)  # (num_patches, patch_size * embedding_dim)
                ], dim=1)
                patches_list.append(patch_input)

        # Process all patches through MLP
        if patches_list:
            x_patches = torch.cat(patches_list, dim=0)
            x = self.mlp(x_patches)
        else:
            # Handle empty case
            x = torch.empty(0, self.hidden_size, device=x.device)

        return x, patch_cnt


TIMESFM2_5_ENCODER = "timesfm2_5"
TIMESFM2_5_HIDDEN_SIZE = 1280
TIMESFM2_5_PATCH_SIZE = 32
TIMESFM2_5_CONTEXT_LIMIT = 16384
TIMESFM2_5_NORM_EPSILON = 1e-6
CHRONOS2_ENCODER = "chronos2"
CHRONOS2_HIDDEN_SIZE = 768
CHRONOS2_PATCH_SIZE = 16
CHRONOS2_CONTEXT_LIMIT = 8192
ZEUS_ENCODER = "zeus"
ZEUS_HIDDEN_SIZE = 768
ZEUS_OUTPUT_SCALE = 32
ZEUS_CENTER_STAGE = 2
ZEUS_CONTEXT_LIMIT = 4096
ZEUS_NORM_EPSILON = 1e-5
CHATTS_MIN_INPUT_FEATURES = 2
CHATTS_VALID_MASK_THRESHOLD = 0.5
NATIVE_ENCODER_ALIASES = (
    None,
    "native",
    "mlp",
    "mlp_patch",
    "mlp-patch",
    "chatts_mlp",
)


class ExternalTimeSeriesProjector(nn.Module):
    """Map frozen time-series backbone embeddings to the LLM hidden size."""

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


class TimesFM2_5TimeSeriesEncoder(nn.Module):
    """Frozen TimesFM 2.5 backbone plus the trained ChatTS projector.

    TimesFM is kept in a plain-Python wrapper, so only ``projector`` is a
    registered submodule. This matches the checkpoint layout produced by the
    ChatTS-Training adapter: ``ts_encoder.projector.*``.
    """

    def __init__(
        self,
        llm_hidden_size: int,
        model_name_or_path: str,
        num_features: int = 2,
        context_limit: int = TIMESFM2_5_CONTEXT_LIMIT,
    ) -> None:
        super().__init__()
        if num_features < CHATTS_MIN_INPUT_FEATURES:
            raise ValueError(
                "TimesFM 2.5 requires ChatTS inputs with value and valid-mask features."
            )

        self.patch_size = TIMESFM2_5_PATCH_SIZE
        self.hidden_size = llm_hidden_size
        self.num_features = num_features
        self.context_limit = context_limit
        self.model_name_or_path = model_name_or_path
        self.projector = ExternalTimeSeriesProjector(
            TIMESFM2_5_HIDDEN_SIZE, llm_hidden_size
        )
        self._timesfm: Any | None = None

    def _load_timesfm(self, device: torch.device) -> Any:
        if sys.version_info < (3, 10):
            raise ImportError("TimesFM 2.5 requires Python 3.10 or newer.")

        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "TimesFM 2.5 is required by this checkpoint. Install it with "
                "`pip install 'timesfm[torch]>=2.0.2'`."
            ) from exc

        model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if model_class is None:
            raise ImportError(
                "The installed `timesfm` package does not expose "
                "`TimesFM_2p5_200M_torch`; install timesfm>=2.0.2."
            )

        print(
            "[ChatTS vLLM] Loading frozen TimesFM 2.5 backbone from "
            f"`{self.model_name_or_path}`."
        )
        # TimesFM's public ``from_pretrained`` currently stages weights on
        # ``cuda:0`` whenever CUDA is visible. With vLLM tensor parallelism,
        # that makes every worker allocate on rank 0 first. Resolve the weight
        # file ourselves, set the worker-local device, and then use the public
        # checkpoint loader to avoid that transient cross-rank allocation.
        if os.path.isdir(self.model_name_or_path) or os.path.isfile(
            self.model_name_or_path
        ):
            checkpoint_path = self.model_name_or_path
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "`huggingface_hub` is required to download the TimesFM "
                    "checkpoint."
                ) from exc

            checkpoint_path = hf_hub_download(
                repo_id=self.model_name_or_path,
                filename=getattr(
                    model_class, "WEIGHTS_FILENAME", "model.safetensors"
                ),
            )

        timesfm_model = model_class(torch_compile=False)
        timesfm_model.model.device = device
        timesfm_model.load_checkpoint(checkpoint_path, torch_compile=False)
        backbone = timesfm_model.model
        backbone.eval()
        backbone.requires_grad_(False)
        return timesfm_model

    def _get_backbone(self, device: torch.device) -> nn.Module:
        if self._timesfm is None:
            self._timesfm = self._load_timesfm(device)

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
        """Match TimesFM 2.5 cumulative per-patch normalization."""
        valid = ~padding
        increment_n = valid.to(values.dtype).sum(dim=-1)
        safe_increment_n = increment_n.clamp_min(1.0)
        increment_mean = (values * valid).sum(dim=-1) / safe_increment_n
        increment_mean = torch.where(
            increment_n == 0, 0.0, increment_mean
        )
        increment_var = (
            ((values - increment_mean.unsqueeze(-1)) ** 2) * valid
        ).sum(dim=-1)
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
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        patch_cnt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = values.size(0)
        max_patch_cnt = int(patch_cnt.max().item())
        padded_length = max_patch_cnt * self.patch_size
        padded_values = torch.zeros(
            batch_size,
            padded_length,
            device=values.device,
            dtype=torch.float32,
        )
        padded_mask = torch.ones(
            batch_size,
            padded_length,
            device=values.device,
            dtype=torch.bool,
        )

        for index in range(batch_size):
            series_values = values[index][valid_mask[index]].to(torch.float32)
            series_length = series_values.numel()
            if series_length == 0:
                continue

            left_padding = padded_length - series_length
            padded_values[index, left_padding:] = series_values
            padded_mask[index, left_padding:] = False

        patched_values = padded_values.reshape(
            batch_size, max_patch_cnt, self.patch_size
        )
        patched_mask = padded_mask.reshape(
            batch_size, max_patch_cnt, self.patch_size
        )

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
        safe_std = torch.where(
            cumulative_std < TIMESFM2_5_NORM_EPSILON,
            1.0,
            cumulative_std,
        )
        normalized_values = (patched_values - cumulative_mean) / safe_std
        normalized_values = normalized_values.masked_fill(patched_mask, 0.0)
        return normalized_values, patched_mask

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)
        values = x[:, :, 0]
        valid_mask = x[:, :, -1] > CHATTS_VALID_MASK_THRESHOLD
        valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)

        if torch.any(valid_lengths == 0):
            raise ValueError(
                "TimesFM 2.5 received an empty time series; every `<ts>` input "
                "must contain data."
            )

        if torch.any(valid_lengths > self.context_limit):
            longest = int(valid_lengths.max().item())
            raise ValueError(
                f"TimesFM 2.5 supports at most {self.context_limit} input points, "
                f"but received {longest}."
            )

        patch_cnt = (
            valid_lengths + self.patch_size - 1
        ) // self.patch_size
        normalized_values, padding_mask = self._prepare_timesfm_batch(
            values, valid_mask, patch_cnt
        )

        backbone = self._get_backbone(x.device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            model_outputs, _ = backbone(normalized_values, padding_mask)
            output_embeddings = model_outputs[1]

        unpadded_embeddings = []
        for index, count in enumerate(patch_cnt.tolist()):
            unpadded_embeddings.append(output_embeddings[index, -count:])

        timesfm_features = torch.cat(unpadded_embeddings, dim=0)
        projector_param = next(self.projector.parameters())
        timesfm_features = timesfm_features.to(
            device=projector_param.device, dtype=projector_param.dtype
        )
        return self.projector(timesfm_features), patch_cnt


class _Chronos2Handle:
    """Keep the frozen Chronos model outside the registered module tree."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline


class Chronos2TimeSeriesEncoder(nn.Module):
    """Frozen Chronos-2 encoder plus the trained ChatTS projector."""

    def __init__(
        self,
        llm_hidden_size: int,
        model_name_or_path: str,
        num_features: int = 2,
        context_limit: int = CHRONOS2_CONTEXT_LIMIT,
    ) -> None:
        super().__init__()
        if num_features < CHATTS_MIN_INPUT_FEATURES:
            raise ValueError(
                "Chronos-2 requires ChatTS inputs with value and valid-mask features."
            )

        self.patch_size = CHRONOS2_PATCH_SIZE
        self.hidden_size = llm_hidden_size
        self.num_features = num_features
        self.context_limit = context_limit
        self.model_name_or_path = model_name_or_path
        self.projector = ExternalTimeSeriesProjector(
            CHRONOS2_HIDDEN_SIZE, llm_hidden_size
        )
        self._chronos2: _Chronos2Handle | None = None

    def _load_chronos2(self) -> _Chronos2Handle:
        if sys.version_info < (3, 10):
            raise ImportError("Chronos-2 requires Python 3.10 or newer.")

        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 is required by this checkpoint. Install it with "
                "`pip install chronos-forecasting==2.3.1`."
            ) from exc

        print(
            "[ChatTS vLLM] Loading frozen Chronos-2 backbone from "
            f"`{self.model_name_or_path}`."
        )
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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)
        values = x[:, :, 0]
        valid_mask = x[:, :, -1] > CHATTS_VALID_MASK_THRESHOLD
        valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)

        if torch.any(valid_lengths == 0):
            raise ValueError(
                "Chronos-2 received an empty time series; every `<ts>` input "
                "must contain data."
            )
        if torch.any(valid_lengths > self.context_limit):
            longest = int(valid_lengths.max().item())
            raise ValueError(
                f"Chronos-2 supports at most {self.context_limit} input points, "
                f"but received {longest}."
            )

        patch_cnt = (
            valid_lengths + self.patch_size - 1
        ) // self.patch_size
        backbone = self._get_backbone(x.device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else nullcontext()
        )

        # Per-series encoding preserves Chronos-2's official left-padding
        # behavior and keeps batch right-padding out of context tokens.
        context_features = []
        with torch.no_grad(), autocast_context:
            for index, count in enumerate(patch_cnt.tolist()):
                context = (
                    values[index][valid_mask[index]]
                    .to(device=x.device, dtype=torch.float32)
                    .unsqueeze(0)
                )
                group_ids = torch.zeros(1, device=x.device, dtype=torch.long)
                encoder_outputs, _, _, num_context_patches = backbone.encode(
                    context=context,
                    group_ids=group_ids,
                )
                returned_count = int(num_context_patches)
                if returned_count != count:
                    raise RuntimeError(
                        f"Chronos-2 returned {returned_count} context patches, "
                        f"expected {count} for this series."
                    )
                # The final two outputs are Chronos-2's register and future
                # tokens; only the leading context patch tokens belong in LLM.
                context_features.append(
                    encoder_outputs.last_hidden_state[:, :count].squeeze(0)
                )

        chronos_features = torch.cat(context_features, dim=0)
        projector_param = next(self.projector.parameters())
        chronos_features = chronos_features.to(
            device=projector_param.device, dtype=projector_param.dtype
        )
        return self.projector(chronos_features), patch_cnt


class _ZeusHandle:
    """Keep frozen Zeus weights outside the registered module tree."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model


class ZeusTimeSeriesEncoder(nn.Module):
    """Frozen Zeus U-shaped encoder plus the trained ChatTS projector."""

    def __init__(
        self,
        llm_hidden_size: int,
        model_name_or_path: str,
        num_features: int = 2,
        context_limit: int = ZEUS_CONTEXT_LIMIT,
    ) -> None:
        super().__init__()
        if num_features < CHATTS_MIN_INPUT_FEATURES:
            raise ValueError(
                "Zeus requires ChatTS inputs with value and valid-mask features."
            )

        self.patch_size = ZEUS_OUTPUT_SCALE
        self.hidden_size = llm_hidden_size
        self.num_features = num_features
        self.context_limit = context_limit
        self.model_name_or_path = model_name_or_path
        self.projector = ExternalTimeSeriesProjector(
            ZEUS_HIDDEN_SIZE, llm_hidden_size
        )
        self._zeus: _ZeusHandle | None = None

    def _load_zeus(self) -> _ZeusHandle:
        try:
            from chatts.vllm.zeus_modeling import ZeusConfig, ZeusForPrediction
        except ImportError as exc:
            raise ImportError(
                "Zeus checkpoints require `chatts/vllm/zeus_modeling.py`; "
                "copy that file together with `chatts_vllm.py`."
            ) from exc

        print(
            "[ChatTS vLLM] Loading frozen Zeus backbone from "
            f"`{self.model_name_or_path}`."
        )
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

    def _get_backbone(self, device: torch.device) -> nn.Module:
        if self._zeus is None:
            self._zeus = self._load_zeus()

        backbone = self._zeus.model
        backbone.to(device=device, dtype=torch.float32)
        backbone.eval()
        backbone.requires_grad_(False)
        return backbone

    @staticmethod
    def _normalize(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.to(torch.float32)
        denominator = valid_lengths.to(torch.float32).view(-1, 1).clamp_min(1.0)
        mean = (values.to(torch.float32) * mask).sum(
            dim=1, keepdim=True
        ) / denominator
        centered = (values.to(torch.float32) - mean) * mask
        variance = centered.square().sum(
            dim=1, keepdim=True
        ) / denominator
        normalized = torch.arcsinh(
            centered / torch.sqrt(variance + ZEUS_NORM_EPSILON)
        )
        return normalized.masked_fill(~valid_mask, 0.0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)
        values = x[:, :, 0]
        valid_mask = x[:, :, -1] > CHATTS_VALID_MASK_THRESHOLD
        valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)

        if torch.any(valid_lengths == 0):
            raise ValueError(
                "Zeus received an empty time series; every `<ts>` input must "
                "contain data."
            )
        if torch.any(valid_lengths > self.context_limit):
            longest = int(valid_lengths.max().item())
            raise ValueError(
                f"Zeus supports at most {self.context_limit} input points, "
                f"but received {longest}."
            )

        patch_cnt = (
            valid_lengths + self.patch_size - 1
        ) // self.patch_size
        packed_length = int(valid_lengths.max().item())
        packed_values = torch.zeros(
            batch_size,
            packed_length,
            device=x.device,
            dtype=torch.float32,
        )
        packed_mask = torch.zeros(
            batch_size,
            packed_length,
            device=x.device,
            dtype=torch.bool,
        )
        for index, length in enumerate(valid_lengths.tolist()):
            packed_values[index, :length] = values[index][valid_mask[index]].to(
                torch.float32
            )
            packed_mask[index, :length] = True

        normalized = self._normalize(
            packed_values, packed_mask, valid_lengths
        ).unsqueeze(-1)
        padding_mask = packed_mask.to(torch.int32).unsqueeze(-1)
        targets_mask = torch.zeros_like(padding_mask)
        backbone = self._get_backbone(x.device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad(), autocast_context:
            outputs = backbone(
                normalized,
                targets_mask=targets_mask,
                padding_mask=padding_mask,
                return_all_hidden_states=True,
            )
            center_features = outputs["all_hidden_states"][ZEUS_CENTER_STAGE]

        unpadded_features = [
            center_features[index, :count]
            for index, count in enumerate(patch_cnt.tolist())
        ]
        zeus_features = torch.cat(unpadded_features, dim=0)
        projector_param = next(self.projector.parameters())
        zeus_features = zeus_features.to(
            device=projector_param.device, dtype=projector_param.dtype
        )
        return self.projector(zeus_features), patch_cnt


def build_time_series_encoder(config: PretrainedConfig) -> nn.Module:
    """Build the encoder architecture declared by the saved checkpoint."""
    encoder_type = getattr(config, "ts_encoder_type", "native")
    if encoder_type in NATIVE_ENCODER_ALIASES:
        print("[ChatTS vLLM] Using the native ChatTS MLP-Patch encoder.")
        return TimeSeriesEmbedding(config.ts)

    if encoder_type == TIMESFM2_5_ENCODER:
        configured_patch_size = int(
            config.ts.get("patch_size", TIMESFM2_5_PATCH_SIZE)
        )
        if configured_patch_size != TIMESFM2_5_PATCH_SIZE:
            raise ValueError(
                "TimesFM 2.5 checkpoints require `ts.patch_size=32` in "
                f"config.json, but found {configured_patch_size}."
            )

        model_name_or_path = getattr(
            config,
            "timesfm_model_name_or_path",
            "google/timesfm-2.5-200m-pytorch",
        )
        print(
            "[ChatTS vLLM] Using TimesFM 2.5 time-series encoder; "
            "checkpoint projector weights will be loaded from "
            "`ts_encoder.projector.*`."
        )
        return TimesFM2_5TimeSeriesEncoder(
            llm_hidden_size=config.hidden_size,
            model_name_or_path=model_name_or_path,
            num_features=int(config.ts.get("num_features", 2)),
        )

    if encoder_type == CHRONOS2_ENCODER:
        configured_patch_size = int(
            config.ts.get("patch_size", CHRONOS2_PATCH_SIZE)
        )
        if configured_patch_size != CHRONOS2_PATCH_SIZE:
            raise ValueError(
                "Chronos-2 checkpoints require `ts.patch_size=16` in "
                f"config.json, but found {configured_patch_size}."
            )

        model_name_or_path = getattr(
            config,
            "chronos2_model_name_or_path",
            "amazon/chronos-2",
        )
        print(
            "[ChatTS vLLM] Using Chronos-2 time-series encoder; checkpoint "
            "projector weights will be loaded from `ts_encoder.projector.*`."
        )
        return Chronos2TimeSeriesEncoder(
            llm_hidden_size=config.hidden_size,
            model_name_or_path=model_name_or_path,
            num_features=int(config.ts.get("num_features", 2)),
        )

    if encoder_type == ZEUS_ENCODER:
        configured_patch_size = int(
            config.ts.get("patch_size", ZEUS_OUTPUT_SCALE)
        )
        if configured_patch_size != ZEUS_OUTPUT_SCALE:
            raise ValueError(
                "Zeus checkpoints require `ts.patch_size=32` in config.json, "
                f"but found {configured_patch_size}."
            )

        model_name_or_path = getattr(
            config,
            "zeus_model_name_or_path",
            "GestaltCog/zeus",
        )
        print(
            "[ChatTS vLLM] Using Zeus time-series encoder; checkpoint "
            "projector weights will be loaded from `ts_encoder.projector.*`."
        )
        return ZeusTimeSeriesEncoder(
            llm_hidden_size=config.hidden_size,
            model_name_or_path=model_name_or_path,
            num_features=int(config.ts.get("num_features", 2)),
        )

    raise ValueError(
        f"Unsupported `ts_encoder_type={encoder_type}` in ChatTS vLLM. "
        "Supported values are `native`/`mlp`, `timesfm2_5`, `chronos2`, "
        "and `zeus`."
    )


# === TS Encoder === #
# get_patch_cnt: From Time Series Embedding
def get_patch_cnt(x: torch.Tensor, ts_config: PretrainedConfig):
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1, ts_config['num_features'])

    mask = x[:, :, -1]
    valid_lengths = mask.sum(1).long()  # Shape: (batch_size)

    patch_cnt = (valid_lengths + int(ts_config['patch_size']) - 1) // int(
        ts_config['patch_size'])
    return patch_cnt



class Qwen2TSProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(PretrainedConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(ProcessorMixin, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"timeseries": 50}  # Allow up to 50 time series per prompt


class Qwen2TSDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2TSProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<ts><ts/>" * mm_counts.get("timeseries", 0)

    def _get_dummy_timeseries(
        self,
        *,
        length: int,
        num_timeseries: int,
    ):
        if num_timeseries == 0:
            return []
        timeseries = np.zeros(length)
        return [timeseries] * num_timeseries

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        hf_config = self.info.get_hf_config()
        max_ts_length = hf_config.ts.get('max_sequence_length', hf_config.ts['max_length'])
        ts_count = mm_counts.get("timeseries", 0)
        return {
            "timeseries":
            self._get_dummy_timeseries(length=max_ts_length,
                                       num_timeseries=ts_count)
        }


HfTimeSeriesItem = Union[list[float], list[list[float]], np.ndarray,
                                    "torch.Tensor"]

class TimeSeriesProcessorItems(ProcessorBatchItems[HfTimeSeriesItem]):
    """Class for handling time series data items."""

    def __init__(self, data: Sequence[HfTimeSeriesItem]) -> None:
        super().__init__(data, "timeseries")

    def get_processor_data(self) -> Mapping[str, object]:
        # "timeseries" is a special case because it already ends in "s"
        if self.modality == "timeseries":
            return {f"{self.modality}": self.data}
        else:
            return {f"{self.modality}s": self.data}

    def get_series_length(self, item_idx: int) -> int:
        """Get the length of the time series."""
        series = self.get(item_idx)
        if isinstance(series, (np.ndarray, torch.Tensor)):
            return series.shape[0]
        elif isinstance(series, list):
            return len(series)
        else:
            raise TypeError(
                f"Unsupported type for time series: {type(series)}")


class TimeSeriesEmbeddingItems(EmbeddingItems):
    """Class for handling time series embedding items."""

    def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
        super().__init__(data, "timeseries")


class Qwen2TSDataParser(MultiModalDataParser):
    def _parse_timeseries_data(
        self,
        data
    ):
        """Parse time series data."""
        if self._is_empty(data):
            return None

        if self._is_embeddings(data):
            return TimeSeriesEmbeddingItems(data)

        return TimeSeriesProcessorItems(
            data if is_list_of(data, (np.ndarray, torch.Tensor,
                                      list)) else [data])

    def _get_subparsers(self):
        return {
            "audio": self._parse_audio_data,
            "image": self._parse_image_data,
            "video": self._parse_video_data,
            "timeseries": self._parse_timeseries_data,
        }


class Qwen2TSMultiModalProcessor(BaseMultiModalProcessor[Qwen2TSProcessingInfo]
                                 ):
    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2TSDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object]={},
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        ts = mm_data.pop("timeseries", [])

        if ts:
            mm_data["timeseries"] = ts

        mm_kwargs = dict(mm_kwargs)
        mm_kwargs['vllm_flag'] = True
        try:
            result = super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
        except TypeError:
            result = super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
            )

        return result

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Define the field name and configuration for time series data
        return {
            "timeseries": MultiModalFieldConfig.batched("timeseries"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object]={},
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> list[PromptReplacement]:
        hf_config = self.info.get_hf_config()
        placeholder = hf_config.ts_token_start_index

        if 'timeseries' not in mm_items:
            return []

        if 'timeseries' not in out_mm_kwargs:
            return []

        patch_size = hf_config.ts['patch_size']

        # Check if we are using the new structure (list of items with .data) or old structure (list of tuples)
        ts_data = out_mm_kwargs["timeseries"]
        is_new_structure = False
        if len(ts_data) > 0:
            first_item = ts_data[0]
            # In new structure, items are usually dict-like or objects where we access ["timeseries"]
            # In old structure, items are tuples (ts_tokens, encoded_ts_arrays)
            if not isinstance(first_item, (list, tuple)):
                 is_new_structure = True

        if is_new_structure:
            def get_replacement_qwen2_ts(item_idx: int):
                # Get out item of the current modality
                out_item = out_mm_kwargs["timeseries"][item_idx]
                # print(f"{out_item['timeseries'].data=}")
                ts_tokens, encoded_ts_arrays = out_item["timeseries"].data
                patch_cnt = (encoded_ts_arrays.shape[1] // 2 + patch_size - 1) // patch_size

                # Use the pre-tokenized replacements
                tokens = ts_tokens.copy()
                # Extend the tokens with placeholders to match the patch_cnt
                num_placeholders = sum(1 for t in tokens if t == placeholder)
                if num_placeholders < patch_cnt:
                    tokens.extend([placeholder] *
                                  (patch_cnt - num_placeholders))
                # return tokens
                return PromptUpdateDetails.select_token_id(
                    tokens,
                    embed_token_id=placeholder,
                )
        else:
            # Old structure
            ts_tokens, encoded_ts_arrays = zip(*out_mm_kwargs["timeseries"])
            patch_cnt = [
                (encoded_ts_arrays[i].shape[1] // 2 + patch_size - 1) // patch_size
                for i in range(len(encoded_ts_arrays))
            ]

            def get_replacement_qwen2_ts(item_idx: int):
                # Use the pre-tokenized replacements
                tokens = ts_tokens[item_idx].copy()
                # Extend the tokens with placeholders to match the patch_cnt
                num_placeholders = sum(1 for t in tokens if t == placeholder)
                if num_placeholders < patch_cnt[item_idx]:
                    tokens.extend([placeholder] *
                                  (patch_cnt[item_idx] - num_placeholders))
                # return tokens
                return PromptUpdateDetails.select_token_id(
                    tokens,
                    embed_token_id=placeholder,
                )

        return [
            PromptReplacement(
                modality="timeseries",
                target=[placeholder, placeholder + 1],
                replacement=get_replacement_qwen2_ts,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2TSMultiModalProcessor,
    info=Qwen2TSProcessingInfo,
    dummy_inputs=Qwen2TSDummyInputsBuilder,
)
class Qwen2TSForCausalLM(nn.Module, SupportsMultiModal, SupportsPP,
                         SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.ts_encoder = build_time_series_encoder(config)
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_ts_input(self, **kwargs: object) -> torch.Tensor:
        timeseries = kwargs.pop('timeseries', None)
        if timeseries is None:
            return None

        # ChatTS processor returns a list of tuples

        # timeseries (batch x (ts_tokens, num_ts x encoded_ts) or batch x num_ts x (ts_tokens, encoded_ts))
        encoded_ts_arrays = []
        for batch in timeseries:
            if not isinstance(batch[0], list):
                encoded_ts_arrays.append(batch[1])
            else:
                # flatten the ts first
                for ts in batch:
                    encoded_ts_arrays.append(ts[1])

        device = encoded_ts_arrays[0].device

        max_length = max(ts.shape[1] for ts in encoded_ts_arrays)
        total_rows = sum(ts.shape[0] for ts in encoded_ts_arrays)
        feature_dim = encoded_ts_arrays[0].shape[2] if encoded_ts_arrays else 0

        # Pre-allocate the tensor with the right size
        concatenated_ts = torch.zeros((total_rows, max_length, feature_dim),
                                      dtype=torch.float16,
                                      device=device)

        # Copy each array to the right position
        row_offset = 0
        for ts in encoded_ts_arrays:
            ts_tensor = torch.tensor(ts, dtype=torch.float16,
                                     device=device) if isinstance(
                                         ts, np.ndarray) else ts
            concatenated_ts[row_offset:row_offset +
                            ts.shape[0], :ts.shape[1], :] = ts_tensor
            row_offset += ts.shape[0]

        input_features = concatenated_ts

        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of ts input features. "
                             f"Got type: {type(input_features)}")
        return input_features

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        ts_input = self._parse_and_validate_ts_input(**kwargs)
        if ts_input is None:
            return None
        ts_features, patch_cnt = self.ts_encoder(ts_input)

        # Reshape ts_features into a list of 2D tensors
        if ts_features.size(0) > 0:
            features_list = []
            start_idx = 0
            for count in patch_cnt:
                if count > 0:
                    end_idx = start_idx + count
                    features_list.append(ts_features[start_idx:end_idx])
                    start_idx = end_idx
                else:
                    # Add empty tensor for consistency when count is 0
                    # This ensures consistent behavior with prefix caching
                    features_list.append(
                        torch.zeros((0, ts_features.size(1)),
                                    device=ts_features.device,
                                    dtype=ts_features.dtype))
            ts_features = features_list

        return ts_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.ts_token_start_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            ts_features = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, ts_features)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[object] = None,
    ) -> Optional[torch.Tensor]:
        try:
            return self.language_model.compute_logits(hidden_states,
                                                      sampling_metadata)
        except TypeError:
            return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)

        autoloaded_weights = loader.load_weights(weights,
                                                 mapper=self.hf_to_vllm_mapper)

        # The HF config doesn't specify whether these are tied,
        # so we detect it this way
        if "embed_tokens.weight" not in autoloaded_weights:
            self.embed_tokens = self.language_model.model.embed_tokens
            autoloaded_weights.add("embed_tokens.weight")

        return autoloaded_weights


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2TSMultiModalProcessor,
    info=Qwen2TSProcessingInfo,
    dummy_inputs=Qwen2TSDummyInputsBuilder,
)
class Qwen3TSForCausalLM(nn.Module, SupportsMultiModal, SupportsPP,
                         SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.ts_encoder = build_time_series_encoder(config)
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_ts_input(self, **kwargs: object) -> torch.Tensor:
        timeseries = kwargs.pop('timeseries', None)
        if timeseries is None:
            return None

        # ChatTS processor returns a list of tuples

        # timeseries (batch x (ts_tokens, num_ts x encoded_ts) or batch x num_ts x (ts_tokens, encoded_ts))
        encoded_ts_arrays = []
        for batch in timeseries:
            if not isinstance(batch[0], list):
                encoded_ts_arrays.append(batch[1])
            else:
                # flatten the ts first
                for ts in batch:
                    encoded_ts_arrays.append(ts[1])

        device = encoded_ts_arrays[0].device

        max_length = max(ts.shape[1] for ts in encoded_ts_arrays)
        total_rows = sum(ts.shape[0] for ts in encoded_ts_arrays)
        feature_dim = encoded_ts_arrays[0].shape[2] if encoded_ts_arrays else 0

        # Pre-allocate the tensor with the right size
        concatenated_ts = torch.zeros((total_rows, max_length, feature_dim),
                                      dtype=torch.float16,
                                      device=device)

        # Copy each array to the right position
        row_offset = 0
        for ts in encoded_ts_arrays:
            ts_tensor = torch.tensor(ts, dtype=torch.float16,
                                     device=device) if isinstance(
                                         ts, np.ndarray) else ts
            concatenated_ts[row_offset:row_offset +
                            ts.shape[0], :ts.shape[1], :] = ts_tensor
            row_offset += ts.shape[0]

        input_features = concatenated_ts

        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of ts input features. "
                             f"Got type: {type(input_features)}")
        return input_features

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        ts_input = self._parse_and_validate_ts_input(**kwargs)
        if ts_input is None:
            return None
        ts_features, patch_cnt = self.ts_encoder(ts_input)

        # Reshape ts_features into a list of 2D tensors
        if ts_features.size(0) > 0:
            features_list = []
            start_idx = 0
            for count in patch_cnt:
                if count > 0:
                    end_idx = start_idx + count
                    features_list.append(ts_features[start_idx:end_idx])
                    start_idx = end_idx
                else:
                    # Add empty tensor for consistency when count is 0
                    # This ensures consistent behavior with prefix caching
                    features_list.append(
                        torch.zeros((0, ts_features.size(1)),
                                    device=ts_features.device,
                                    dtype=ts_features.dtype))
            ts_features = features_list

        return ts_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.ts_token_start_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            ts_features = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, ts_features)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[object] = None,
    ) -> Optional[torch.Tensor]:
        try:
            return self.language_model.compute_logits(hidden_states,
                                                      sampling_metadata)
        except TypeError:
            return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)

        autoloaded_weights = loader.load_weights(weights,
                                                 mapper=self.hf_to_vllm_mapper)

        # The HF config doesn't specify whether these are tied,
        # so we detect it this way
        if "embed_tokens.weight" not in autoloaded_weights:
            self.embed_tokens = self.language_model.model.embed_tokens
            autoloaded_weights.add("embed_tokens.weight")

        return autoloaded_weights


# Register VLLM
ModelRegistry.register_model("Qwen2TSForCausalLM", Qwen2TSForCausalLM)
print(f"[ChatTS VLLM] Qwen2TSForCausalLM registered in vLLM!")

ModelRegistry.register_model("Qwen3TSForCausalLM", Qwen3TSForCausalLM)
print(f"[ChatTS VLLM] Qwen3TSForCausalLM registered in vLLM!")
