from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .deepseek import DeepSeekError
from .orchestrator import Autoresearch, OrchestrationError

COMMANDS = (
    "preflight",
    "label",
    "prepare-data",
    "baseline",
    "search",
    "resume",
    "freeze",
    "final-eval",
    "report",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatts-autoresearch",
        description="Independent black-box autoresearch controller for ChatTS",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = subparsers.add_parser(name)
        command.add_argument("-c", "--config", required=True, type=Path)
    return parser


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller: Autoresearch | None = None
    try:
        controller = Autoresearch(load_config(args.config))
        methods = {
            "preflight": controller.preflight,
            "label": controller.label,
            "prepare-data": controller.prepare_data,
            "baseline": controller.baseline,
            "search": controller.search,
            "resume": controller.resume,
            "freeze": controller.freeze,
            "final-eval": controller.final_eval,
            "report": controller.report,
        }
        result = methods[args.command]()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_serializable))
        return 0
    except (ConfigError, OrchestrationError, DeepSeekError, OSError, RuntimeError, ValueError) as exc:
        print(f"chatts-autoresearch: {exc}", file=sys.stderr)
        return 2
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
