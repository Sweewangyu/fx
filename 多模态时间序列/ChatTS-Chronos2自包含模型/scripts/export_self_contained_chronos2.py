#!/usr/bin/env python3
"""Merge a ChatTS Qwen3+projector checkpoint and local Chronos-2 weights.

This is a tensor-level conversion: it does not instantiate the 1.7B Qwen
model, and it never downloads anything. Each input shard becomes one output
safetensors shard, which keeps peak CPU memory close to the largest source
shard.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
SINGLE_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
CODE_FILES = (
    "configuration_qwen3_ts.py",
    "modeling_qwen3_ts.py",
    "processing_qwen3_ts.py",
)
PROJECTOR_PREFIX = "ts_encoder.projector."
BACKBONE_PREFIX = "ts_encoder.backbone."
REQUIRED_PROJECTOR_SUFFIXES = {
    "input_norm.weight",
    "input_norm.bias",
    "linear_in.weight",
    "linear_in.bias",
    "linear_out.weight",
    "linear_out.bias",
    "output_norm.weight",
    "output_norm.bias",
}


def expected_chronos2_keys(num_layers: int) -> set[str]:
    keys = {
        "shared.weight",
        "encoder.final_layer_norm.weight",
    }
    for block_name in ("input_patch_embedding", "output_patch_embedding"):
        for layer_name in ("hidden_layer", "output_layer", "residual_layer"):
            keys.add(f"{block_name}.{layer_name}.weight")
            keys.add(f"{block_name}.{layer_name}.bias")
    for layer_index in range(num_layers):
        for attention_index in (0, 1):
            prefix = f"encoder.block.{layer_index}.layer.{attention_index}"
            keys.add(f"{prefix}.layer_norm.weight")
            for projection in ("q", "k", "v", "o"):
                keys.add(f"{prefix}.self_attention.{projection}.weight")
        feed_forward_prefix = f"encoder.block.{layer_index}.layer.2"
        keys.add(f"{feed_forward_prefix}.layer_norm.weight")
        keys.add(f"{feed_forward_prefix}.mlp.wi.weight")
        keys.add(f"{feed_forward_prefix}.mlp.wo.weight")
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one offline Qwen3 + Chronos-2 + projector checkpoint directory."
    )
    parser.add_argument(
        "--chatts-checkpoint",
        type=Path,
        required=True,
        help="Existing full ChatTS checkpoint containing Qwen3 and ts_encoder.projector.*.",
    )
    parser.add_argument(
        "--chronos2-checkpoint",
        type=Path,
        required=True,
        help="Local amazon/chronos-2 checkpoint directory, for example /workspace/chronos2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New empty output directory. Existing non-empty directories are rejected.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def discover_weight_files(model_dir: Path) -> list[Path]:
    for index_name in INDEX_NAMES:
        index_path = model_dir / index_name
        if index_path.is_file():
            weight_map = read_json(index_path).get("weight_map", {})
            if not weight_map:
                raise ValueError(f"Weight index is empty: {index_path}")
            files = [model_dir / name for name in sorted(set(weight_map.values()))]
            missing = [str(path) for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Weight index references missing shards: {missing}")
            return files

    for weight_name in SINGLE_WEIGHT_NAMES:
        weight_path = model_dir / weight_name
        if weight_path.is_file():
            return [weight_path]

    candidates = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    candidates = [path for path in candidates if not path.name.startswith("optimizer")]
    if candidates:
        return candidates
    raise FileNotFoundError(f"No model weight files found under {model_dir}")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        state = load_safetensors(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

    if not isinstance(state, dict) or not state:
        raise ValueError(f"Not a non-empty state dict: {path}")
    non_tensors = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise TypeError(f"Non-tensor entries found in {path}: {non_tensors[:5]}")
    return state


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def normalize_chronos_key(key: str, all_keys: set[str]) -> str:
    # Official Chronos2Model checkpoints use shared.weight, encoder.*, etc.
    # Some wrappers save the same model under a leading "model." prefix.
    if "shared.weight" not in all_keys and "model.shared.weight" in all_keys and key.startswith("model."):
        return key.removeprefix("model.")
    return key


def iter_output_shards(
    chatts_files: list[Path],
    chronos_files: list[Path],
) -> Iterator[tuple[str, Path, dict[str, torch.Tensor]]]:
    for path in chatts_files:
        yield "chatts", path, load_state_dict(path)
    for path in chronos_files:
        state = load_state_dict(path)
        all_keys = set(state)
        prefixed = {
            BACKBONE_PREFIX + normalize_chronos_key(key, all_keys): value
            for key, value in state.items()
        }
        yield "chronos2", path, prefixed


def validate_source_configs(chatts_config: dict, chronos_config: dict) -> None:
    if chronos_config.get("architectures", [None])[0] != "Chronos2Model":
        raise ValueError(
            "--chronos2-checkpoint is not an official Chronos-2 model directory: "
            f"architectures={chronos_config.get('architectures')!r}."
        )
    if "chronos_config" not in chronos_config:
        raise ValueError("Chronos-2 config.json has no chronos_config block.")
    if int(chronos_config.get("d_model", -1)) != 768:
        raise ValueError(
            f"Expected Chronos-2 d_model=768, got {chronos_config.get('d_model')!r}."
        )
    if int(chatts_config.get("hidden_size", -1)) <= 0:
        raise ValueError("ChatTS config.json has no valid hidden_size.")
    chatts_architectures = chatts_config.get("architectures")
    if chatts_architectures and "Qwen3" not in " ".join(chatts_architectures):
        raise ValueError(
            "The source ChatTS checkpoint does not identify a Qwen3 architecture: "
            f"{chatts_architectures!r}."
        )


def make_output_config(chatts_config: dict, chronos_config: dict, chronos_dir: Path) -> dict:
    output = dict(chatts_config)
    # Keep the official bytedance-research/ChatTS-8B class and model type.
    # Chronos-2 selection is explicit in ts_encoder_type and the embedded
    # config, so existing ChatTS tooling does not need a renamed base class.
    output["architectures"] = ["Qwen3TSForCausalLM"]
    output["model_type"] = "qwen3ts"
    output["ts_encoder_type"] = "chronos2"
    output["chronos2_embedded"] = True
    output["chronos2_config"] = chronos_config
    output["chronos2_backbone_name"] = str(
        chronos_config.get("_name_or_path") or chronos_dir.name
    )
    output.pop("chronos2_model_name_or_path", None)
    output["projector_config"] = {
        "input_hidden_size": int(chronos_config["d_model"]),
        "activation": "gelu",
        "num_linear_layers": 2,
    }
    output["auto_map"] = {
        "AutoConfig": "configuration_qwen3_ts.Qwen3TSConfig",
        "AutoModel": "modeling_qwen3_ts.Qwen3TSForCausalLM",
        "AutoModelForCausalLM": "modeling_qwen3_ts.Qwen3TSForCausalLM",
        "AutoProcessor": "processing_qwen3_ts.Qwen3TSProcessor",
    }
    ts_config = dict(output.get("ts") or {})
    ts_config["num_features"] = int(ts_config.get("num_features", 2))
    ts_config["patch_size"] = int(chronos_config["chronos_config"]["input_patch_size"])
    ts_config["max_sequence_length"] = int(chronos_config["chronos_config"]["context_length"])
    output["ts"] = ts_config
    return output


def is_weight_or_generated_file(path: Path) -> bool:
    if path.name in INDEX_NAMES or path.name in SINGLE_WEIGHT_NAMES or path.name in CODE_FILES:
        return True
    if path.name in {"config.json", "processor_config.json", "README.md", "weight_audit.json"}:
        return True
    return path.suffix in {".safetensors", ".bin"}


def copy_auxiliary_files(source_dir: Path, output_dir: Path) -> None:
    for source in source_dir.iterdir():
        if source.is_file() and not is_weight_or_generated_file(source):
            shutil.copy2(source, output_dir / source.name)


def copy_remote_code(output_dir: Path) -> None:
    template_dir = Path(__file__).resolve().parents[1] / "hf_files"
    for filename in CODE_FILES:
        source = template_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing bundled model-code template: {source}")
        shutil.copy2(source, output_dir / filename)


def prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {output_dir}. "
                "Choose a new path."
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    # If validation raises, remove only the staging directory created above.
    # A successful atomic rename makes this callback a harmless no-op.
    atexit.register(shutil.rmtree, staging_dir, True)
    return staging_dir


def main() -> None:
    args = parse_args()
    chatts_dir = args.chatts_checkpoint.resolve()
    chronos_dir = args.chronos2_checkpoint.resolve()
    final_output_dir = args.output_dir.resolve()

    chatts_config = read_json(chatts_dir / "config.json")
    chronos_config = read_json(chronos_dir / "config.json")
    validate_source_configs(chatts_config, chronos_config)
    chatts_files = discover_weight_files(chatts_dir)
    chronos_files = discover_weight_files(chronos_dir)
    output_dir = prepare_output_dir(final_output_dir)

    copy_auxiliary_files(chatts_dir, output_dir)
    copy_remote_code(output_dir)
    output_config = make_output_config(chatts_config, chronos_config, chronos_dir)
    write_json(output_dir / "config.json", output_config)
    write_json(
        output_dir / "processor_config.json",
        {
            "auto_map": {
                "AutoProcessor": "processing_qwen3_ts.Qwen3TSProcessor"
            },
            "processor_class": "Qwen3TSProcessor",
        },
    )

    total_shards = len(chatts_files) + len(chronos_files)
    weight_map: dict[str, str] = {}
    seen_keys: set[str] = set()
    total_size = 0
    component_sizes = {"chatts": 0, "chronos2": 0}
    projector_suffixes: set[str] = set()
    native_keys: list[str] = []
    backbone_key_count = 0
    backbone_suffixes: set[str] = set()
    qwen_key_count = 0

    for shard_index, (component, source_path, state) in enumerate(
        iter_output_shards(chatts_files, chronos_files), start=1
    ):
        shard_name = f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        output_state: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key in seen_keys:
                raise ValueError(f"Duplicate state-dict key while merging: {key}")
            if component == "chatts" and key.startswith(BACKBONE_PREFIX):
                raise ValueError(
                    "The ChatTS source already contains ts_encoder.backbone.*. "
                    "It is already self-contained; do not merge Chronos-2 a second time."
                )
            if key.startswith(("ts_encoder.mlp.", "ts_encoder.position_embedding.")):
                native_keys.append(key)
            if key.startswith(PROJECTOR_PREFIX):
                projector_suffixes.add(key.removeprefix(PROJECTOR_PREFIX))
            if key.startswith(BACKBONE_PREFIX):
                backbone_key_count += 1
                backbone_suffixes.add(key.removeprefix(BACKBONE_PREFIX))
            if key.startswith(("model.", "lm_head.")):
                qwen_key_count += 1

            seen_keys.add(key)
            size = tensor_nbytes(tensor)
            total_size += size
            component_sizes[component] += size
            weight_map[key] = shard_name
            output_state[key] = tensor.detach().cpu().contiguous()

        save_safetensors(
            output_state,
            str(output_dir / shard_name),
            metadata={
                "format": "pt",
                "source_component": component,
                "source_shard": source_path.name,
            },
        )
        del output_state, state
        gc.collect()
        print(f"[{shard_index}/{total_shards}] wrote {shard_name} from {source_path}")

    if native_keys:
        raise ValueError(
            "The source contains native ChatTS MLP keys as well as an external checkpoint. "
            f"First examples: {native_keys[:5]}"
        )
    missing_projector = REQUIRED_PROJECTOR_SUFFIXES - projector_suffixes
    unexpected_projector = projector_suffixes - REQUIRED_PROJECTOR_SUFFIXES
    if missing_projector or unexpected_projector:
        raise ValueError(
            "The source projector is not the expected two-linear-layer Chronos-2 projector: "
            f"missing={sorted(missing_projector)}, unexpected={sorted(unexpected_projector)}"
        )
    required_backbone = expected_chronos2_keys(int(chronos_config["num_layers"]))
    missing_backbone = required_backbone - backbone_suffixes
    if missing_backbone:
        raise ValueError(
            "The Chronos-2 source checkpoint is incomplete. Missing tensor keys: "
            f"{sorted(missing_backbone)[:20]}"
        )
    if backbone_key_count == 0 or qwen_key_count == 0:
        raise ValueError(
            f"Incomplete merged checkpoint: qwen_keys={qwen_key_count}, "
            f"chronos2_backbone_keys={backbone_key_count}."
        )

    write_json(
        output_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    audit = {
        "status": "OK",
        "architecture": "Qwen3TSForCausalLM",
        "chatts_source": str(chatts_dir),
        "chronos2_source": str(chronos_dir),
        "total_tensor_count": len(weight_map),
        "qwen_tensor_count": qwen_key_count,
        "projector_tensor_count": len(projector_suffixes),
        "chronos2_backbone_tensor_count": backbone_key_count,
        "total_size_bytes": total_size,
        "component_size_bytes": component_sizes,
        "required_runtime_weight_paths": [str(final_output_dir)],
        "external_chronos2_weight_path_required": False,
    }
    write_json(output_dir / "weight_audit.json", audit)
    (output_dir / "README.md").write_text(
        "# Self-contained ChatTS: Qwen3 + Chronos-2\n\n"
        "This directory contains Qwen3, the trained two-layer time-series projector, "
        "and Chronos-2 weights in one state dict. It never downloads Chronos-2 at runtime.\n\n"
        "Install `torch`, `transformers`, `safetensors`, and "
        "`chronos-forecasting>=2.3.1`, then load with `trust_remote_code=True`.\n",
        encoding="utf-8",
    )

    if final_output_dir.exists():
        # prepare_output_dir proved this directory was empty. rmdir also
        # protects against a concurrent writer appearing during conversion.
        final_output_dir.rmdir()
    output_dir.rename(final_output_dir)

    print("\nConversion complete")
    print(f"output: {final_output_dir}")
    print(f"tensors: {len(weight_map)}")
    print(f"size: {total_size / 1024**3:.2f} GiB")
    print("external Chronos-2 path required at runtime: NO")


if __name__ == "__main__":
    main()
