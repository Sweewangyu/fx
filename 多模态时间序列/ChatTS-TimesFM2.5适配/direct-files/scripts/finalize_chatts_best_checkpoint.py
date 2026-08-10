#!/usr/bin/env python3
"""Validate and finalize a LLaMAFactory ChatTS best-model export.

LLaMAFactory reloads ``best_model_checkpoint`` before ``trainer.save_model()``.
Consequently, the files stored directly in ``output_dir`` are the selected
best model.  This utility records that provenance, stamps the Chronos-2
metadata needed by inference, and optionally removes the now-redundant
``checkpoint-N`` directories.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"checkpoint-[0-9]+")
WEIGHT_PATTERNS = (
    "pytorch_model*.bin",
    "model*.safetensors",
    "adapter_model*.bin",
    "adapter_model*.safetensors",
)


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def discover_weight_files(checkpoint_dir: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for pattern in WEIGHT_PATTERNS:
        for path in checkpoint_dir.glob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                files[path.name] = path
    return [files[name] for name in sorted(files)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument("--chronos2-model-path", required=True)
    parser.add_argument("--input-model-dir")
    parser.add_argument("--input-best-model-manifest")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--chronos2-hidden-size", type=int, default=768)
    parser.add_argument("--cleanup-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint_dir}")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.patch_size < 1 or args.chronos2_hidden_size < 1:
        raise SystemExit("Patch and hidden sizes must be positive")

    input_model_dir = None
    if args.input_model_dir:
        input_model_dir = Path(args.input_model_dir).expanduser().resolve()
        if not (input_model_dir / "config.json").is_file():
            raise SystemExit(f"Input model config not found: {input_model_dir / 'config.json'}")

    input_best_model = None
    if args.input_best_model_manifest:
        input_manifest_path = Path(args.input_best_model_manifest).expanduser().resolve()
        if not input_manifest_path.is_file():
            raise SystemExit(f"Input best-model manifest not found: {input_manifest_path}")
        with input_manifest_path.open(encoding="utf-8") as stream:
            parent_manifest = json.load(stream)
        if parent_manifest.get("stage") != "stage1":
            raise SystemExit("Stage2 input manifest must describe a Stage1 best model")
        parent_export = Path(parent_manifest.get("exported_model_dir", "")).expanduser().resolve()
        if input_model_dir is not None and parent_export != input_model_dir:
            raise SystemExit(
                "Input model directory does not match Stage1 best-model manifest: "
                f"{input_model_dir} != {parent_export}"
            )
        input_best_model = {
            "manifest_path": str(input_manifest_path),
            "stage": parent_manifest["stage"],
            "exported_model_dir": str(parent_export),
            "selected_checkpoint": parent_manifest.get("selected_checkpoint"),
            "best_metric": parent_manifest.get("best_metric"),
        }

    config_path = checkpoint_dir / "config.json"
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not config_path.is_file():
        raise SystemExit(f"config.json not found: {config_path}")
    if not trainer_state_path.is_file():
        raise SystemExit(f"trainer_state.json not found: {trainer_state_path}")

    weight_files = discover_weight_files(checkpoint_dir)
    if not weight_files:
        raise SystemExit(
            f"No non-empty model weight file was exported directly under {checkpoint_dir}"
        )

    with trainer_state_path.open(encoding="utf-8") as stream:
        trainer_state = json.load(stream)
    best_checkpoint_value = trainer_state.get("best_model_checkpoint")
    best_metric = trainer_state.get("best_metric")
    if not best_checkpoint_value:
        raise SystemExit(
            "trainer_state.json has no best_model_checkpoint; cannot prove that the "
            "root export contains the validation-selected model"
        )
    if not isinstance(best_metric, (int, float)) or not math.isfinite(float(best_metric)):
        raise SystemExit(f"Invalid best_metric in trainer_state.json: {best_metric!r}")

    best_checkpoint = Path(str(best_checkpoint_value)).expanduser()
    if not best_checkpoint.is_absolute():
        best_checkpoint = checkpoint_dir / best_checkpoint
    best_checkpoint = best_checkpoint.resolve()
    try:
        best_checkpoint.relative_to(checkpoint_dir)
    except ValueError as exc:
        raise SystemExit(
            f"best_model_checkpoint is outside output_dir: {best_checkpoint}"
        ) from exc
    if not CHECKPOINT_RE.fullmatch(best_checkpoint.name):
        raise SystemExit(
            f"Unexpected best checkpoint directory name: {best_checkpoint.name}"
        )
    if not best_checkpoint.is_dir():
        raise SystemExit(f"Selected best checkpoint directory does not exist: {best_checkpoint}")

    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["ts_encoder_type"] = "chronos2"
    config["chronos2_model_name_or_path"] = args.chronos2_model_path
    config["chronos2_hidden_size"] = args.chronos2_hidden_size
    ts_config = config.setdefault("ts", {})
    if not isinstance(ts_config, dict):
        raise SystemExit("config.json field 'ts' exists but is not an object")
    ts_config["patch_size"] = args.patch_size
    atomic_json_dump(config_path, config)

    checkpoint_dirs = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name)
    )
    manifest = {
        "stage": args.stage,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "best_metric": float(best_metric),
        "selected_checkpoint": best_checkpoint.name,
        "exported_model_dir": str(checkpoint_dir),
        "input_model_dir": str(input_model_dir) if input_model_dir is not None else None,
        "input_best_model": input_best_model,
        "ts_encoder_type": "chronos2",
        "chronos2_model_name_or_path": args.chronos2_model_path,
        "chronos2_hidden_size": args.chronos2_hidden_size,
        "patch_size": args.patch_size,
        "model_files": [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in weight_files
        ],
        "checkpoint_directories_before_cleanup": [path.name for path in checkpoint_dirs],
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = checkpoint_dir / "best_model_manifest.json"
    atomic_json_dump(manifest_path, manifest)

    if args.cleanup_checkpoints:
        for path in checkpoint_dirs:
            # The regex and parent check above deliberately constrain deletion
            # to output_dir/checkpoint-N.
            if path.parent != checkpoint_dir or not CHECKPOINT_RE.fullmatch(path.name):
                raise SystemExit(f"Refusing to remove unexpected path: {path}")
            shutil.rmtree(path)
            print(f"Removed redundant checkpoint directory: {path}")

    if any(
        path.is_dir() and CHECKPOINT_RE.fullmatch(path.name)
        for path in checkpoint_dir.iterdir()
    ):
        raise SystemExit("checkpoint-N directories remain after cleanup")
    if not discover_weight_files(checkpoint_dir):
        raise SystemExit("Root model weights disappeared during finalization")

    print(f"Finalized {args.stage} best model: {checkpoint_dir}")
    print(f"Selected checkpoint: {best_checkpoint.name}; eval_loss={float(best_metric):.8g}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
