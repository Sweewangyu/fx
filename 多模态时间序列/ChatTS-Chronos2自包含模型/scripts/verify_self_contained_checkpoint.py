#!/usr/bin/env python3
"""Verify a merged checkpoint without importing torch or loading tensor data."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

PROJECTOR_PREFIX = "ts_encoder.projector."
BACKBONE_PREFIX = "ts_encoder.backbone."
REQUIRED_PROJECTOR_KEYS = {
    PROJECTOR_PREFIX + suffix
    for suffix in (
        "input_norm.weight",
        "input_norm.bias",
        "linear_in.weight",
        "linear_in.bias",
        "linear_out.weight",
        "linear_out.bias",
        "output_norm.weight",
        "output_norm.bias",
    )
}


def expected_chronos2_keys(num_layers: int) -> set[str]:
    keys = {
        BACKBONE_PREFIX + "shared.weight",
        BACKBONE_PREFIX + "encoder.final_layer_norm.weight",
    }
    for block_name in ("input_patch_embedding", "output_patch_embedding"):
        for layer_name in ("hidden_layer", "output_layer", "residual_layer"):
            keys.add(BACKBONE_PREFIX + f"{block_name}.{layer_name}.weight")
            keys.add(BACKBONE_PREFIX + f"{block_name}.{layer_name}.bias")
    for layer_index in range(num_layers):
        for attention_index in (0, 1):
            prefix = BACKBONE_PREFIX + f"encoder.block.{layer_index}.layer.{attention_index}"
            keys.add(f"{prefix}.layer_norm.weight")
            for projection in ("q", "k", "v", "o"):
                keys.add(f"{prefix}.self_attention.{projection}.weight")
        feed_forward_prefix = BACKBONE_PREFIX + f"encoder.block.{layer_index}.layer.2"
        keys.add(f"{feed_forward_prefix}.layer_norm.weight")
        keys.add(f"{feed_forward_prefix}.mlp.wi.weight")
        keys.add(f"{feed_forward_prefix}.mlp.wo.weight")
    return keys


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_safetensors_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))
    header.pop("__metadata__", None)
    return header


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()

    config = read_json(checkpoint / "config.json")
    index = read_json(checkpoint / "model.safetensors.index.json")
    weight_map: dict[str, str] = index.get("weight_map", {})
    failures: list[str] = []

    require(config.get("architectures") == ["Qwen3TSForCausalLM"], "wrong architectures", failures)
    require(config.get("model_type") == "qwen3ts", "wrong model_type", failures)
    require(config.get("ts_encoder_type") == "chronos2", "ts_encoder_type is not chronos2", failures)
    require(config.get("chronos2_embedded") is True, "chronos2_embedded is not true", failures)
    require(isinstance(config.get("chronos2_config"), dict), "chronos2_config is missing", failures)
    require(isinstance(config.get("projector_config"), dict), "projector_config is missing", failures)
    require("chronos2_model_name_or_path" not in config, "external Chronos path is still present", failures)
    require(int(config.get("ts", {}).get("patch_size", -1)) == 16, "ts.patch_size is not 16", failures)

    all_headers: dict[str, dict] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = checkpoint / shard_name
        require(shard_path.is_file(), f"missing shard: {shard_name}", failures)
        if shard_path.is_file():
            for key, metadata in read_safetensors_header(shard_path).items():
                if key in all_headers:
                    failures.append(f"duplicate tensor across shards: {key}")
                all_headers[key] = metadata

    require(set(weight_map) == set(all_headers), "index keys and shard-header keys differ", failures)
    projector_keys = {key for key in weight_map if key.startswith(PROJECTOR_PREFIX)}
    backbone_keys = {key for key in weight_map if key.startswith(BACKBONE_PREFIX)}
    qwen_keys = {key for key in weight_map if key.startswith(("model.", "lm_head."))}
    native_keys = {
        key
        for key in weight_map
        if key.startswith(("ts_encoder.mlp.", "ts_encoder.position_embedding."))
    }

    require(projector_keys == REQUIRED_PROJECTOR_KEYS, "projector keys are incomplete or unexpected", failures)
    require(len(backbone_keys) > 0, "no ts_encoder.backbone.* tensors", failures)
    require(len(qwen_keys) > 0, "no Qwen model.* / lm_head.* tensors", failures)
    require(not native_keys, "native MLP tensors are mixed into the Chronos checkpoint", failures)
    num_chronos_layers = int(config.get("chronos2_config", {}).get("num_layers", 0))
    expected_backbone_keys = expected_chronos2_keys(num_chronos_layers)
    missing_backbone_keys = expected_backbone_keys - backbone_keys
    require(
        not missing_backbone_keys,
        f"Chronos-2 backbone is incomplete; first missing keys: {sorted(missing_backbone_keys)[:20]}",
        failures,
    )

    hidden_size = int(config.get("hidden_size", -1))
    expected_shapes = {
        PROJECTOR_PREFIX + "input_norm.weight": [768],
        PROJECTOR_PREFIX + "input_norm.bias": [768],
        PROJECTOR_PREFIX + "linear_in.weight": [hidden_size, 768],
        PROJECTOR_PREFIX + "linear_in.bias": [hidden_size],
        PROJECTOR_PREFIX + "linear_out.weight": [hidden_size, hidden_size],
        PROJECTOR_PREFIX + "linear_out.bias": [hidden_size],
        PROJECTOR_PREFIX + "output_norm.weight": [hidden_size],
        PROJECTOR_PREFIX + "output_norm.bias": [hidden_size],
    }
    for key, expected_shape in expected_shapes.items():
        actual_shape = all_headers.get(key, {}).get("shape")
        require(actual_shape == expected_shape, f"{key}: shape {actual_shape}, expected {expected_shape}", failures)

    print(f"checkpoint                : {checkpoint}")
    print(f"architecture              : {config.get('architectures')}")
    print(f"ts_encoder_type           : {config.get('ts_encoder_type')}")
    print(f"Qwen tensors              : {len(qwen_keys)}")
    print(f"projector tensors         : {len(projector_keys)}")
    print(f"Chronos-2 backbone tensors: {len(backbone_keys)}")
    print(f"external Chronos path     : {'YES' if 'chronos2_model_name_or_path' in config else 'NO'}")

    if failures:
        print("\nSTATUS: FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nSTATUS: OK — Qwen3 + two-layer projector + Chronos-2 are in one checkpoint.")


if __name__ == "__main__":
    main()
