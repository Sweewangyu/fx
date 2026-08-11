from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import TARGET_SOURCES, read_jsonl, write_jsonl

from chatts_dataset_studio.catalog import CatalogCache
from chatts_dataset_studio.exporter import export_selection
from chatts_dataset_studio.models import DEFAULT_ALIASES, StudioError


def _export(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any], run_name: str = "fixture-run"
) -> tuple[dict[str, Any], Path]:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    payload = {
        **default_selection,
        "run_name": run_name,
        "output_root": str(labeled_corpus["output_root"]),
    }
    result = export_selection(payload, sources, catalog)
    return result, Path(result["output_dir"])


def test_export_writes_clean_training_data_and_audit_sidecars(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    result, output_dir = _export(labeled_corpus, default_selection)

    assert result["status"] == "completed"
    assert result["counts"] == {"stage1": 12, "stage2": 18, "overlap": 6}
    for source in TARGET_SOURCES:
        stage1_rows = read_jsonl(output_dir / "stage1" / f"{source}.jsonl")
        stage2_rows = read_jsonl(output_dir / "stage2" / f"{source}.jsonl")
        stage1_labels = read_jsonl(output_dir / "stage1_annotations" / f"{source}.jsonl")
        stage2_labels = read_jsonl(output_dir / "stage2_annotations" / f"{source}.jsonl")

        assert len(stage1_rows) == len(stage1_labels) == 2
        assert len(stage2_rows) == len(stage2_labels) == 3
        assert all(set(row) == {"input", "timeseries", "output"} for row in stage1_rows)
        assert all(set(row) == {"input", "timeseries", "output"} for row in stage2_rows)
        assert [row["difficulty"] for row in stage1_labels] == ["easy", "moderate"]
        assert [row["difficulty"] for row in stage2_labels] == [
            "moderate",
            "hard",
            "very_hard",
        ]


def test_export_writes_training_registry_env_and_reproducibility_manifests(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    result, output_dir = _export(labeled_corpus, default_selection)

    dataset_info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    expected_stage1 = [f"stage1_{DEFAULT_ALIASES[name]}" for name in TARGET_SOURCES]
    expected_stage2 = [f"stage2_{DEFAULT_ALIASES[name]}" for name in TARGET_SOURCES]
    assert set(dataset_info) == set(expected_stage1 + expected_stage2)
    for source in TARGET_SOURCES:
        assert dataset_info[f"stage1_{DEFAULT_ALIASES[source]}"]["file_name"] == (
            f"stage1/{source}.jsonl"
        )
        assert dataset_info[f"stage2_{DEFAULT_ALIASES[source]}"]["columns"] == {
            "prompt": "input",
            "response": "output",
            "timeseries": "timeseries",
        }

    env = (output_dir / "training.env").read_text(encoding="utf-8")
    assert f"DATASET_DIR={output_dir}" in env
    assert f"STAGE1_DATASETS={','.join(expected_stage1)}" in env
    assert f"STAGE2_DATASETS={','.join(expected_stage2)}" in env

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "chatts-dataset-studio-export-v1"
    assert manifest["selection_hash"]
    assert manifest["manifest_hash"]
    assert manifest["dataset_names"] == {
        "stage1": expected_stage1,
        "stage2": expected_stage2,
    }
    assert set(manifest["input_identities"]) == set(TARGET_SOURCES)
    assert all(
        identity["raw_sha256"] and identity["annotation_sha256"]
        for identity in manifest["input_identities"].values()
    )
    for relative, expected_digest in manifest["files"].items():
        assert hashlib.sha256((output_dir / relative).read_bytes()).hexdigest() == expected_digest

    for stage, expected_rows in (("stage1", 12), ("stage2", 18)):
        stage_manifest = json.loads(
            (output_dir / stage / "manifest.json").read_text(encoding="utf-8")
        )
        assert stage_manifest["stage"] == stage
        assert stage_manifest["total_rows"] == expected_rows
        assert stage_manifest["selection_hash"] == manifest["selection_hash"]
    assert Path(result["manifest"]) == output_dir / "manifest.json"


def test_raw_and_sidecar_length_mismatch_is_rejected_and_temp_is_cleaned(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    broken = labeled_corpus["annotations_root"] / "annotations" / "chatts_sft.jsonl"
    rows = read_jsonl(broken)
    write_jsonl(broken, rows[:-1])
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    payload = {
        **default_selection,
        "run_name": "broken-alignment",
        "output_root": str(labeled_corpus["output_root"]),
    }

    with pytest.raises(StudioError, match="line counts differ"):
        export_selection(payload, sources, catalog)

    assert not (labeled_corpus["output_root"] / "broken-alignment").exists()
    assert not list(labeled_corpus["output_root"].glob(".broken-alignment.tmp-*"))


def test_declared_line_mismatch_makes_source_unavailable(
    labeled_corpus: dict[str, Any],
) -> None:
    broken = labeled_corpus["annotations_root"] / "annotations" / "time_mqa.jsonl"
    rows = read_jsonl(broken)
    rows[1]["line_number"] = 99
    write_jsonl(broken, rows)

    _, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )

    summary = next(row for row in catalog["sources"] if row["name"] == "time_mqa")
    assert summary["available"] is False
    assert "line mismatch" in summary["errors"][0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_index", 77, "source_index mismatch"),
        ("annotation_id", "wrong:id", "Annotation id mismatch"),
        ("annotation_source", "wrong-source", "Annotation source mismatch"),
    ],
)
def test_canonical_sidecar_identity_mismatch_makes_source_unavailable(
    labeled_corpus: dict[str, Any], field: str, value: Any, message: str
) -> None:
    broken = labeled_corpus["annotations_root"] / "annotations" / "chatts_sft.jsonl"
    rows = read_jsonl(broken)
    rows[2][field] = value
    write_jsonl(broken, rows)

    _, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )

    summary = next(row for row in catalog["sources"] if row["name"] == "chatts_sft")
    assert summary["available"] is False
    assert message in summary["errors"][0]


def test_annotated_file_fallback_exports_without_reading_labels_from_sidecar(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    source = "chatts_ift"
    sidecar_path = labeled_corpus["annotations_root"] / "annotations" / f"{source}.jsonl"
    raw_path = (
        labeled_corpus["data_root"]
        / "data"
        / "versions"
        / "datav2"
        / "files"
        / f"{source}.jsonl"
    )
    raw_rows = read_jsonl(raw_path)
    annotations = read_jsonl(sidecar_path)
    annotated_rows = [
        {**raw, **annotation} for raw, annotation in zip(raw_rows, annotations, strict=True)
    ]
    write_jsonl(
        labeled_corpus["annotations_root"] / "annotated" / f"{source}.jsonl",
        annotated_rows,
    )
    sidecar_path.unlink()

    result, output_dir = _export(labeled_corpus, default_selection, "annotated-fallback")

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    identity = manifest["input_identities"][source]
    assert identity["annotation_mode"] == "annotated"
    assert identity["raw_sha256"] == "embedded-in-annotated-file"
    rows = read_jsonl(output_dir / "stage1" / f"{source}.jsonl")
    assert len(rows) == 2
    assert all(set(row) == {"input", "timeseries", "output"} for row in rows)
