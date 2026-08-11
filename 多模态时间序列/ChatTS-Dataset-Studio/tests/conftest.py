from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

TARGET_SOURCES = (
    "chatts_align_256",
    "chatts_align_random",
    "chatts_ift",
    "chatts_sft",
    "time_mqa",
    "tsaqa",
)

LABEL_ROWS = (
    ("unusable", "very_easy", "pattern_recognition"),
    ("weak", "easy", "numerical_reasoning"),
    ("acceptable", "moderate", "trend_analysis"),
    ("good", "hard", "anomaly_detection"),
    ("excellent", "very_hard", "forecasting"),
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def labeled_corpus(tmp_path: Path) -> dict[str, Any]:
    data_root = tmp_path / "datataste"
    annotations_root = tmp_path / "merged_labels"
    registry_path = data_root / "data" / "versions" / "datav2" / "sources.json"
    sources = []

    for source_index, source_name in enumerate(TARGET_SOURCES):
        relative_path = Path("data") / "versions" / "datav2" / "files" / f"{source_name}.jsonl"
        raw_rows = []
        annotation_rows = []
        for row_index, (quality, difficulty, ability) in enumerate(LABEL_ROWS):
            raw_rows.append(
                {
                    "input": f"{source_name} question {row_index}",
                    "timeseries": [[source_index, row_index, row_index + 0.5]],
                    "output": f"answer {row_index}",
                    "raw_metadata_that_must_not_leak": {"source": source_name},
                }
            )
            annotation_rows.append(
                {
                    "annotation_id": f"{source_name}:{row_index + 1}",
                    "annotation_source": source_name,
                    "source_index": row_index,
                    "line_number": row_index + 1,
                    "ability_label": ability,
                    "ability_bucket": ability,
                    "quality": quality,
                    "difficulty": difficulty,
                    "quality_reason": f"fixture-{quality}",
                }
            )
        write_jsonl(data_root / relative_path, raw_rows)
        write_jsonl(
            annotations_root / "annotations" / f"{source_name}.jsonl",
            annotation_rows,
        )
        sources.append(
            {
                "name": source_name,
                "path": relative_path.as_posix(),
                "family": "chatts" if source_name.startswith("chatts_") else "finiverse",
                "split": "train",
                "training_role": "alignment" if "align" in source_name else "sft",
            }
        )

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps({"schema_version": "fixture-v1", "sources": sources}, indent=2),
        encoding="utf-8",
    )
    return {
        "tmp_path": tmp_path,
        "registry_path": registry_path,
        "annotations_root": annotations_root,
        "data_root": data_root,
        "output_root": tmp_path / "exports",
        "sources": TARGET_SOURCES,
    }


@pytest.fixture
def default_selection(labeled_corpus: dict[str, Any]) -> dict[str, Any]:
    selected = list(labeled_corpus["sources"])
    return {
        "stage1": {
            "sources": selected,
            "qualities": ["weak", "acceptable", "good", "excellent"],
            "difficulties": ["very_easy", "easy", "moderate"],
            "abilities": [],
        },
        "stage2": {
            "sources": selected,
            "qualities": ["weak", "acceptable", "good", "excellent"],
            "difficulties": ["moderate", "hard", "very_hard"],
            "abilities": [],
        },
    }
