#!/usr/bin/env python3
"""Create a tiny six-source dataset for local UI and contract checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SOURCES = (
    "chatts_align_256",
    "chatts_align_random",
    "chatts_ift",
    "chatts_sft",
    "time_mqa",
    "tsaqa",
)
QUALITIES = ("unusable", "weak", "acceptable", "good", "excellent")
DIFFICULTIES = ("very_easy", "easy", "moderate", "hard", "very_hard")
ABILITIES = ("PR", "NR", "TSF", "QuantDM", "UNMAPPED")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(output: Path) -> None:
    registry = []
    for source_offset, source in enumerate(SOURCES):
        qa_rows = []
        annotation_rows = []
        for index, (quality, difficulty, ability) in enumerate(
            zip(QUALITIES, DIFFICULTIES, ABILITIES, strict=True)
        ):
            qa_rows.append(
                {
                    "input": f"{source} 示例 {index + 1}：请分析 <ts><ts/>。",
                    "timeseries": [
                        [round(source_offset + index * 0.1 + step * 0.01, 4) for step in range(8)]
                    ],
                    "output": f"这是 {difficulty} / {quality} 的演示答案。",
                }
            )
            annotation_rows.append(
                {
                    "annotation_id": f"{source}:{index + 1}",
                    "annotation_source": source,
                    "source_index": index,
                    "line_number": index + 1,
                    "ability_label": None if ability == "UNMAPPED" else ability,
                    "ability_bucket": ability,
                    "ability_name": ability,
                    "ability_major": "demo",
                    "quality": quality,
                    "difficulty": difficulty,
                    "quality_reason": "本地界面演示标签，不代表真实模型标注结果。",
                }
            )
        write_jsonl(output / "files" / f"{source}.jsonl", qa_rows)
        write_jsonl(
            output / "merged_labels" / "annotations" / f"{source}.jsonl",
            annotation_rows,
        )
        registry.append(
            {
                "name": source,
                "path": f"files/{source}.jsonl",
                "family": "demo",
                "split": "train",
                "training_role": "alignment" if "align" in source else "sft",
            }
        )
    write_json(output / "sources.json", {"schema_version": "demo-v1", "sources": registry})
    write_json(
        output / "DEMO_ONLY.json",
        {
            "warning": "Synthetic records for local UI/format checks only; never use for training.",
            "sources": list(SOURCES),
            "rows_per_source": len(QUALITIES),
        },
    )
    print(f"Demo data created at: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
