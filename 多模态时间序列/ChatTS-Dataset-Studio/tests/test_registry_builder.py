from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatts_dataset_studio.models import StudioError
from chatts_dataset_studio.registry_builder import build_registry


def write_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_build_registry_scans_every_annotated_dataset(tmp_path: Path) -> None:
    merged = tmp_path / "merged_labels"
    for name in ("alpha", "beta", "gamma"):
        write_jsonl(
            merged / "annotated" / f"{name}.jsonl",
            {"input": name, "timeseries": [[1.0]], "output": "answer"},
        )
    write_jsonl(merged / "annotations" / "alpha.jsonl", {"quality": "good"})
    metadata_path = tmp_path / "datav2-sources.json"
    metadata_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "beta",
                        "path": "old/beta.jsonl",
                        "family": "datav2",
                        "split": "train",
                        "training_role": "reasoning",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sources.json"

    result = build_registry(
        merged,
        output,
        data_root=tmp_path,
        metadata_registry=metadata_path,
    )

    registry = json.loads(output.read_text(encoding="utf-8"))
    assert result["source_count"] == 3
    assert result["with_sidecar_count"] == 1
    assert result["without_sidecar"] == ["beta", "gamma"]
    assert result["metadata_reused_count"] == 1
    assert [row["name"] for row in registry["sources"]] == ["alpha", "beta", "gamma"]
    assert registry["sources"][0]["path"] == "merged_labels/annotated/alpha.jsonl"
    assert registry["sources"][1]["family"] == "datav2"
    assert registry["sources"][1]["training_role"] == "reasoning"


def test_build_registry_refuses_overwrite_without_force(tmp_path: Path) -> None:
    merged = tmp_path / "merged_labels"
    write_jsonl(
        merged / "annotated" / "alpha.jsonl",
        {"input": "question", "timeseries": [], "output": "answer"},
    )
    output = tmp_path / "sources.json"
    build_registry(merged, output, data_root=tmp_path)

    with pytest.raises(StudioError, match="already exists"):
        build_registry(merged, output, data_root=tmp_path)

    result = build_registry(merged, output, data_root=tmp_path, force=True)
    assert result["source_count"] == 1


def test_build_registry_rejects_non_qa_jsonl(tmp_path: Path) -> None:
    merged = tmp_path / "merged_labels"
    write_jsonl(merged / "annotated" / "broken.jsonl", {"quality": "good"})

    with pytest.raises(StudioError, match="lacks fields"):
        build_registry(merged, tmp_path / "sources.json", data_root=tmp_path)
