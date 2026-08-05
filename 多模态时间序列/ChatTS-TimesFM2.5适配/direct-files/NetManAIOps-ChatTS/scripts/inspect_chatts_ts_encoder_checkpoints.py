#!/usr/bin/env python3
"""Inspect ChatTS checkpoint weights and identify the saved TS encoder.

This script is read-only. It inspects tensor names and shapes instead of
trusting ``config.json`` alone, which makes it useful for finding mixed,
mis-saved, or incorrectly targeted checkpoints before starting vLLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


NATIVE_ENCODER = "native"
TIMESFM_ENCODER = "timesfm2_5"
CHRONOS2_ENCODER = "chronos2"
ZEUS_ENCODER = "zeus"

ENCODER_ALIASES = {
    "mlp": NATIVE_ENCODER,
    "mlp_patch": NATIVE_ENCODER,
    "mlp-patch": NATIVE_ENCODER,
    "chatts_mlp": NATIVE_ENCODER,
    "timesfm2.5": TIMESFM_ENCODER,
    "timesfm-2.5": TIMESFM_ENCODER,
    "chronos-2": CHRONOS2_ENCODER,
}

WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "model*.bin",
    "model*.pt",
    "pytorch_model*.pt",
    "model*.pth",
    "pytorch_model*.pth",
)
EXCLUDED_WEIGHT_PARTS = (
    "optimizer",
    "scheduler",
    "trainer_state",
    "rng_state",
    "training_args",
)


@dataclass
class TensorEvidence:
    native_keys: list[str] = field(default_factory=list)
    projector_keys: list[str] = field(default_factory=list)
    projector_dims: set[int] = field(default_factory=set)
    relevant_tensors: list[str] = field(default_factory=list)
    scanned_tensor_count: int = 0
    load_errors: list[str] = field(default_factory=list)


@dataclass
class CheckpointReport:
    checkpoint: str
    detected_encoder: str
    status: str
    reason: str
    config_encoder: str
    patch_size: str
    projector_dims: str
    native_key_count: int
    projector_key_count: int
    weight_files: str
    relevant_tensors: str
    scanned_tensor_count: int


def normalize_encoder_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in ("", "auto"):
        return None
    return ENCODER_ALIASES.get(normalized, normalized)


def shape_text(shape: Iterable[int]) -> str:
    return "[" + ",".join(str(int(size)) for size in shape) + "]"


def record_tensor(evidence: TensorEvidence, name: str, shape: tuple[int, ...]) -> None:
    evidence.scanned_tensor_count += 1
    is_native = (
        "ts_encoder.mlp." in name
        or "ts_encoder.position_embedding." in name
    )
    is_projector = "ts_encoder.projector." in name
    if not is_native and not is_projector:
        return

    evidence.relevant_tensors.append(f"{name}:{shape_text(shape)}")
    if is_native:
        evidence.native_keys.append(name)
    if is_projector:
        evidence.projector_keys.append(name)
        if "ts_encoder.projector.input_norm.weight" in name and shape:
            evidence.projector_dims.add(int(shape[0]))
        elif "ts_encoder.projector.linear_in.weight" in name and len(shape) >= 2:
            evidence.projector_dims.add(int(shape[-1]))


def inspect_safetensors(path: Path, evidence: TensorEvidence) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Install safetensors to inspect .safetensors checkpoints."
        ) from exc

    with safe_open(str(path), framework="pt", device="cpu") as stream:
        for name in stream.keys():
            shape = tuple(stream.get_slice(name).get_shape())
            record_tensor(evidence, name, shape)


def visit_torch_state(
    value: Any,
    torch_module: Any,
    evidence: TensorEvidence,
    prefix: str = "",
    depth: int = 0,
) -> None:
    if isinstance(value, torch_module.Tensor):
        record_tensor(evidence, prefix, tuple(value.shape))
        return
    if isinstance(value, Mapping) and depth < 4:
        for key, child in value.items():
            child_name = f"{prefix}.{key}" if prefix else str(key)
            visit_torch_state(
                child,
                torch_module,
                evidence,
                prefix=child_name,
                depth=depth + 1,
            )


def inspect_torch_checkpoint(
    path: Path,
    evidence: TensorEvidence,
    allow_unsafe_torch_load: bool,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect .pt/.bin weights.") from exc

    load_kwargs = {"map_location": "cpu"}
    state = None
    safe_errors: list[Exception] = []
    for extra_kwargs in (
        {"weights_only": True, "mmap": True},
        {"weights_only": True},
    ):
        try:
            state = torch.load(str(path), **load_kwargs, **extra_kwargs)
            break
        except Exception as exc:
            safe_errors.append(exc)

    if state is None:
        if not allow_unsafe_torch_load:
            details = "; ".join(str(error) for error in safe_errors[-2:])
            raise RuntimeError(
                "Safe torch.load(weights_only=True) failed. Upgrade PyTorch or "
                "rerun with --allow-unsafe-torch-load only if you trust this "
                f"checkpoint. Details: {details}"
            )
        state = torch.load(str(path), **load_kwargs)

    visit_torch_state(state, torch, evidence)
    del state


def list_weight_files(checkpoint: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        paths.extend(checkpoint.glob(pattern))
    unique_paths = []
    seen = set()
    for path in sorted(paths):
        lowered_name = path.name.lower()
        if any(part in lowered_name for part in EXCLUDED_WEIGHT_PARTS):
            continue
        resolved = str(path.resolve())
        if resolved not in seen and path.is_file():
            unique_paths.append(path)
            seen.add(resolved)
    return unique_paths


def load_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def config_patch_size(config: dict[str, Any]) -> int | None:
    ts_config = config.get("ts") or {}
    if not isinstance(ts_config, Mapping):
        return None
    try:
        return int(ts_config.get("patch_size"))
    except (TypeError, ValueError):
        return None


def classify_encoder(
    evidence: TensorEvidence,
    config_encoder: str | None,
    patch_size: int | None,
) -> tuple[str, str, str]:
    has_native = bool(evidence.native_keys)
    has_projector = bool(evidence.projector_keys)
    dims = evidence.projector_dims

    if has_native and has_projector:
        return (
            "mixed_native_external",
            "ERROR",
            "Both native position_embedding/MLP and external projector weights exist.",
        )
    if has_native:
        detected = NATIVE_ENCODER
        reason = "Found native ts_encoder.position_embedding.* or ts_encoder.mlp.* weights."
    elif dims == {1280}:
        detected = TIMESFM_ENCODER
        reason = "External projector input dimension is 1280."
    elif dims == {768}:
        if config_encoder in (CHRONOS2_ENCODER, ZEUS_ENCODER):
            detected = config_encoder
            reason = f"768-d projector disambiguated by config: {config_encoder}."
        elif patch_size == 16:
            detected = CHRONOS2_ENCODER
            reason = "768-d projector disambiguated by patch_size=16."
        elif patch_size == 32:
            detected = ZEUS_ENCODER
            reason = "768-d projector disambiguated by patch_size=32."
        else:
            return (
                "ambiguous_chronos2_or_zeus",
                "AMBIGUOUS",
                "The projector is 768-d and has no reliable Chronos-2/Zeus discriminator.",
            )
    elif dims:
        return (
            "unsupported_external",
            "ERROR",
            f"Unexpected projector input dimensions: {sorted(dims)}.",
        )
    elif has_projector:
        return (
            "external_unknown_dimension",
            "ERROR",
            "Projector weights exist, but input_norm/linear_in did not reveal the input dimension.",
        )
    else:
        return (
            "unknown",
            "ERROR",
            "No native MLP/position-embedding or external projector tensors were found.",
        )

    expected_patch = {
        TIMESFM_ENCODER: 32,
        CHRONOS2_ENCODER: 16,
        ZEUS_ENCODER: 32,
    }.get(detected)
    mismatches = []
    if config_encoder and config_encoder != detected:
        mismatches.append(
            f"config ts_encoder_type={config_encoder} conflicts with weights={detected}"
        )
    if expected_patch is not None and patch_size not in (None, expected_patch):
        mismatches.append(
            f"config patch_size={patch_size} conflicts with expected {expected_patch}"
        )
    if mismatches:
        return detected, "MISMATCH", reason + " " + "; ".join(mismatches) + "."
    return detected, "OK", reason


def inspect_checkpoint(
    checkpoint: Path,
    allow_unsafe_torch_load: bool,
    max_relevant_keys: int,
) -> CheckpointReport:
    config = load_config(checkpoint)
    config_encoder = normalize_encoder_type(config.get("ts_encoder_type"))
    patch_size = config_patch_size(config)
    weight_files = list_weight_files(checkpoint)
    evidence = TensorEvidence()

    for weight_path in weight_files:
        try:
            if weight_path.suffix == ".safetensors":
                inspect_safetensors(weight_path, evidence)
            else:
                inspect_torch_checkpoint(
                    weight_path,
                    evidence,
                    allow_unsafe_torch_load=allow_unsafe_torch_load,
                )
        except Exception as exc:  # Keep scanning/reporting other checkpoints.
            evidence.load_errors.append(f"{weight_path.name}: {exc}")

    detected, status, reason = classify_encoder(
        evidence,
        config_encoder=config_encoder,
        patch_size=patch_size,
    )
    if not weight_files:
        status = "ERROR"
        reason = "No supported model weight files were found in this directory."
    if evidence.load_errors:
        status = "ERROR"
        reason = reason + " Load errors: " + " | ".join(evidence.load_errors)

    relevant = evidence.relevant_tensors[:max_relevant_keys]
    remaining = len(evidence.relevant_tensors) - len(relevant)
    if remaining > 0:
        relevant.append(f"... and {remaining} more")

    return CheckpointReport(
        checkpoint=str(checkpoint),
        detected_encoder=detected,
        status=status,
        reason=reason,
        config_encoder=config_encoder or "<missing>",
        patch_size=str(patch_size) if patch_size is not None else "<missing>",
        projector_dims=",".join(str(value) for value in sorted(evidence.projector_dims)) or "<none>",
        native_key_count=len(evidence.native_keys),
        projector_key_count=len(evidence.projector_keys),
        weight_files=";".join(path.name for path in weight_files) or "<none>",
        relevant_tensors="; ".join(relevant) or "<none>",
        scanned_tensor_count=evidence.scanned_tensor_count,
    )


def discover_checkpoints(root: Path, checkpoint_suffix: str) -> list[Path]:
    suffix = checkpoint_suffix.strip().strip("/")
    if list_weight_files(root):
        return [root]

    checkpoints = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "logs":
            continue
        candidate = child / suffix if suffix else child
        checkpoints.append(candidate)
    return checkpoints


def report_as_dict(report: CheckpointReport) -> dict[str, Any]:
    return {
        field_name: getattr(report, field_name)
        for field_name in CheckpointReport.__dataclass_fields__
    }


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_reports(reports: list[CheckpointReport], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "chatts_ts_encoder_inventory.csv"
    markdown_path = output_dir / "chatts_ts_encoder_inventory.md"
    rows = [report_as_dict(report) for report in reports]
    fieldnames = list(CheckpointReport.__dataclass_fields__)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Checkpoint",
        "Detected",
        "Status",
        "Config",
        "Patch",
        "Projector dim",
        "Native keys",
        "Projector keys",
        "Reason",
    ]
    lines = [
        "# ChatTS TS Encoder Checkpoint Inventory",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for report in reports:
        values = [
            Path(report.checkpoint).name,
            report.detected_encoder,
            report.status,
            report.config_encoder,
            report.patch_size,
            report.projector_dims,
            report.native_key_count,
            report.projector_key_count,
            report.reason,
        ]
        lines.append("| " + " | ".join(markdown_escape(value) for value in values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def print_terminal_table(reports: list[CheckpointReport]) -> None:
    headers = ("CHECKPOINT", "DETECTED", "STATUS", "PROJ_DIM", "NATIVE", "PROJECTOR")
    rows = [
        (
            Path(report.checkpoint).name,
            report.detected_encoder,
            report.status,
            report.projector_dims,
            str(report.native_key_count),
            str(report.projector_key_count),
        )
        for report in reports
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = min(48, max(widths[index], len(value)))

    def format_row(row: Iterable[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            clipped = value if len(value) <= widths[index] else value[: widths[index] - 1] + "…"
            cells.append(clipped.ljust(widths[index]))
        return "  ".join(cells)

    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in rows:
        print(format_row(row))


def run_self_test() -> None:
    cases = [
        (TensorEvidence(native_keys=["ts_encoder.mlp.0.weight"]), None, None, "native"),
        (TensorEvidence(projector_keys=["p"], projector_dims={1280}), None, None, "timesfm2_5"),
        (TensorEvidence(projector_keys=["p"], projector_dims={768}), None, 16, "chronos2"),
        (TensorEvidence(projector_keys=["p"], projector_dims={768}), None, 32, "zeus"),
        (
            TensorEvidence(native_keys=["n"], projector_keys=["p"], projector_dims={1280}),
            None,
            None,
            "mixed_native_external",
        ),
    ]
    for evidence, config_encoder, patch_size, expected in cases:
        detected, _, _ = classify_encoder(evidence, config_encoder, patch_size)
        assert detected == expected, (detected, expected)
    print("Self-test passed: native, TimesFM, Chronos-2, Zeus, and mixed checkpoints.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "search_dir",
        nargs="?",
        help="Checkpoint directory or parent directory containing checkpoint subdirectories.",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        default="",
        help="Optional relative path from each child directory to its actual checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to SEARCH_DIR/logs.",
    )
    parser.add_argument(
        "--max-relevant-keys",
        type=int,
        default=20,
        help="Maximum relevant tensor entries stored per CSV row.",
    )
    parser.add_argument(
        "--allow-unsafe-torch-load",
        action="store_true",
        help="Allow legacy pickle loading only for checkpoints you trust.",
    )
    parser.add_argument(
        "--print-detected-only",
        action="store_true",
        help="Inspect exactly one checkpoint and print only its encoder type.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not args.search_dir:
        raise SystemExit("search_dir is required unless --self-test is used.")

    root = Path(args.search_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Directory does not exist: {root}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "logs"
    )
    checkpoints = discover_checkpoints(root, args.checkpoint_suffix)
    if not checkpoints:
        raise SystemExit(f"No checkpoint directories found under {root}")

    if args.print_detected_only:
        if len(checkpoints) != 1:
            raise SystemExit(
                "--print-detected-only requires a checkpoint directory, not a parent "
                f"containing {len(checkpoints)} checkpoints."
            )
        report = inspect_checkpoint(
            checkpoints[0],
            allow_unsafe_torch_load=args.allow_unsafe_torch_load,
            max_relevant_keys=max(1, args.max_relevant_keys),
        )
        if report.status not in ("OK",):
            raise SystemExit(
                f"Cannot select encoder automatically: {report.detected_encoder}: "
                f"{report.reason}"
            )
        print(report.detected_encoder)
        return 0

    reports = [
        inspect_checkpoint(
            checkpoint,
            allow_unsafe_torch_load=args.allow_unsafe_torch_load,
            max_relevant_keys=max(1, args.max_relevant_keys),
        )
        for checkpoint in checkpoints
    ]
    print_terminal_table(reports)
    csv_path, markdown_path = write_reports(reports, output_dir)

    counts: dict[str, int] = {}
    for report in reports:
        counts[report.detected_encoder] = counts.get(report.detected_encoder, 0) + 1
    print("\nDetected counts:")
    for encoder_type, count in sorted(counts.items()):
        print(f"  {encoder_type}: {count}")
    print(f"\nCSV report:      {csv_path}")
    print(f"Markdown report: {markdown_path}")

    suspicious = [report for report in reports if report.status != "OK"]
    if suspicious:
        print(
            f"\nFound {len(suspicious)} suspicious checkpoint(s). "
            "See the reason and relevant_tensors columns.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(reports)} checkpoint(s) are internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
