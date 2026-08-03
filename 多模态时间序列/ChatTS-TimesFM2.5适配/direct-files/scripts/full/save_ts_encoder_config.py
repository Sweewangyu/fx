#!/usr/bin/env python3
"""Persist time-series encoder metadata into saved ChatTS configs.

This helper is intentionally independent of PyTorch and Transformers. Training
shell scripts call it only after DeepSpeed exits successfully, so the final
checkpoint and every ``checkpoint-*`` directory can be loaded without a
manual config.json edit.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ENCODER_METADATA: dict[str, dict[str, Any]] = {
    "native": {},
    "timesfm2_5": {
        "path_field": "timesfm_model_name_or_path",
        "hidden_field": "timesfm_hidden_size",
        "hidden_size": 1280,
        "patch_size": 32,
    },
    "chronos2": {
        "path_field": "chronos2_model_name_or_path",
        "hidden_field": "chronos2_hidden_size",
        "hidden_size": 768,
        "patch_size": 16,
    },
    "zeus": {
        "path_field": "zeus_model_name_or_path",
        "hidden_field": "zeus_hidden_size",
        "hidden_size": 768,
        "patch_size": 32,
        "extra": {"zeus_output_scale": 32},
    },
}

EXTERNAL_METADATA_FIELDS = {
    "timesfm_model_name_or_path",
    "timesfm_hidden_size",
    "chronos2_model_name_or_path",
    "chronos2_hidden_size",
    "zeus_model_name_or_path",
    "zeus_hidden_size",
    "zeus_output_scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="Final training output directory.")
    parser.add_argument(
        "--encoder-type",
        required=True,
        choices=tuple(ENCODER_METADATA),
        help="Encoder architecture actually used to produce the checkpoint.",
    )
    parser.add_argument(
        "--backbone-path",
        help="Frozen external-backbone path or Hugging Face model ID. Required for non-native encoders.",
    )
    return parser.parse_args()


def find_config_paths(checkpoint_dir: Path) -> list[Path]:
    paths = []
    root_config = checkpoint_dir / "config.json"
    if root_config.is_file():
        paths.append(root_config)
    paths.extend(sorted(checkpoint_dir.glob("checkpoint-*/config.json")))
    return list(dict.fromkeys(paths))


def update_config(config: dict[str, Any], encoder_type: str, backbone_path: str | None) -> None:
    for field in EXTERNAL_METADATA_FIELDS:
        config.pop(field, None)

    config["ts_encoder_type"] = encoder_type
    metadata = ENCODER_METADATA[encoder_type]
    if encoder_type == "native":
        return

    if not backbone_path:
        raise ValueError(f"--backbone-path is required for encoder type {encoder_type}.")

    config[metadata["path_field"]] = backbone_path
    config[metadata["hidden_field"]] = metadata["hidden_size"]
    config.update(metadata.get("extra", {}))
    ts_config = config.setdefault("ts", {})
    if not isinstance(ts_config, dict):
        raise TypeError("config.json field `ts` must be an object.")
    ts_config["patch_size"] = metadata["patch_size"]


def atomic_write_json(path: Path, config: dict[str, Any]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint_dir}")

    config_paths = find_config_paths(checkpoint_dir)
    if not config_paths:
        raise SystemExit(
            f"No config.json found in {checkpoint_dir} or its checkpoint-* directories; "
            "refusing to report successful metadata persistence."
        )

    for path in config_paths:
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
        update_config(config, args.encoder_type, args.backbone_path)
        atomic_write_json(path, config)
        print(f"[TS config] saved {args.encoder_type} metadata: {path}")


if __name__ == "__main__":
    main()
