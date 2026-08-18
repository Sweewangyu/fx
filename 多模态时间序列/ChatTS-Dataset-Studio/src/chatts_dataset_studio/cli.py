from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .catalog import CatalogCache
from .exporter import export_selection, parse_rules, preview_selection
from .models import StudioError
from .registry_builder import build_registry
from .server import serve


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise StudioError(f"Configuration file does not exist: {resolved}")
    try:
        value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StudioError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudioError("Configuration root must be one mapping")
    paths = value.get("paths", {})
    selection = value.get("selection", {})
    server = value.get("server", {})
    versions = value.get("versions", {})
    integration = value.get("integration", {})
    registry = value.get("registry", {})
    if not all(
        isinstance(item, dict)
        for item in (paths, selection, server, versions, integration, registry)
    ):
        raise StudioError(
            "paths, selection, server, versions, integration, and registry must be mappings"
        )

    def resolve(raw: Any) -> Any:
        if not isinstance(raw, str) or not raw:
            return raw
        candidate = Path(raw).expanduser()
        return str(candidate if candidate.is_absolute() else (resolved.parent / candidate).resolve())

    integration_result = dict(integration)
    for name in (
        "training_root",
        "evaluation_root",
        "pipeline_script",
        "evaluation_pipeline_script",
        "slurm_root",
        "slurm_sbatch",
        "slurm_evaluation_sbatch",
        "slurm_evaluation_root",
        "slurm_evaluation_sif_image",
        "slurm_chronos2_host_root",
        "slurm_tsrbench_host_root",
        "slurm_tinybench_host_root",
        "slurm_ts_haystack_host_root",
        "slurm_timeseriesexam_host_root",
    ):
        if name in integration_result:
            # A simple sbatch filename is resolved relative to slurm_root by the
            # trusted launcher validator, not relative to this YAML file.
            if name in {"slurm_sbatch", "slurm_evaluation_sbatch"} and isinstance(
                integration_result[name], str
            ) and "/" not in integration_result[name]:
                continue
            integration_result[name] = resolve(integration_result[name])

    output_root = resolve(paths.get("output_root"))
    state_root = resolve(paths.get("state_root"))
    if state_root is None and output_root:
        state_root = str(Path(output_root) / ".studio-state")
    version_start = versions.get("start", 3)
    if isinstance(version_start, bool) or not isinstance(version_start, int):
        raise StudioError("versions.start must be an integer")
    registry_auto_build = registry.get("auto_build", True)
    if not isinstance(registry_auto_build, bool):
        raise StudioError("registry.auto_build must be true or false")

    result = {
        "registry_path": resolve(paths.get("registry_path")),
        "annotations_root": resolve(paths.get("annotations_root")),
        "data_root": resolve(paths.get("data_root")),
        "output_root": output_root,
        "state_root": state_root,
        "version_start": version_start,
        "registry_auto_build": registry_auto_build,
        "metadata_registry": resolve(registry.get("metadata_registry")),
        "integration": integration_result,
        "run_name": selection.get("run_name"),
        "stage1": selection.get("stage1"),
        "stage2": selection.get("stage2"),
        "host": server.get("host"),
        "port": server.get("port"),
    }
    return {key: value for key, value in result.items() if value is not None}


def _apply_path_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(config)
    for name in ("registry_path", "annotations_root", "data_root", "output_root"):
        value = getattr(args, name, None)
        if value is not None:
            result[name] = str(value.expanduser().resolve())
    return result


def _path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--annotations-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatts-dataset-studio",
        description="Select annotated ChatTS QA datasets and export Stage1/Stage2 training snapshots",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="Start the local visual frontend")
    serve_parser.add_argument("-c", "--config", type=Path)
    _path_arguments(serve_parser)
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    serve_parser.add_argument("--open", action="store_true", dest="open_browser")
    serve_parser.add_argument("--verbose", action="store_true")

    for name in ("catalog", "preview", "export"):
        command = subparsers.add_parser(name)
        command.add_argument("-c", "--config", required=True, type=Path)
        if name == "catalog":
            _path_arguments(command)
    registry_parser = subparsers.add_parser(
        "build-registry",
        help="Create sources.json from every merged_labels/annotated/*.jsonl file",
    )
    registry_parser.add_argument("--merged-labels-root", required=True, type=Path)
    registry_parser.add_argument("--output", required=True, type=Path)
    registry_parser.add_argument("--data-root", type=Path)
    registry_parser.add_argument("--metadata-registry", type=Path)
    registry_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-registry":
            result = build_registry(
                args.merged_labels_root,
                args.output,
                data_root=args.data_root,
                metadata_registry=args.metadata_registry,
                force=args.force,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        config = _apply_path_args(_load_config(args.config), args)
        if args.command == "serve":
            host = args.host or config.pop("host", "127.0.0.1")
            port = args.port if args.port is not None else int(config.pop("port", 7865))
            if not 0 <= port <= 65535:
                raise StudioError("port must be between 0 and 65535")
            serve(
                config,
                host=host,
                port=port,
                open_browser=args.open_browser,
                verbose=args.verbose,
            )
            return 0
        cache = CatalogCache()
        sources, catalog = cache.get(
            config.get("registry_path"),
            config.get("annotations_root"),
            config.get("data_root"),
        )
        if args.command == "catalog":
            result = catalog
        else:
            stage1, stage2 = parse_rules(config, sources, catalog)
            if args.command == "preview":
                result = preview_selection(catalog, stage1, stage2)
            else:
                result = export_selection(config, sources, catalog)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, StudioError, ValueError) as exc:
        print(f"chatts-dataset-studio: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
