from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from conftest import write_jsonl

from chatts_dataset_studio.catalog import CatalogCache
from chatts_dataset_studio.exporter import export_selection
from chatts_dataset_studio.models import StudioError
from chatts_dataset_studio.versioning import (
    VersionLedger,
    activate_version,
    list_versions,
    next_version,
    normalize_data_version,
    record_version,
    verify_snapshot,
    verify_version,
)


def _export(
    labeled_corpus: dict[str, Any], selection: dict[str, Any], run_name: str
) -> Path:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    result = export_selection(
        {
            **selection,
            "run_name": run_name,
            "output_root": str(labeled_corpus["output_root"]),
        },
        sources,
        catalog,
    )
    return Path(result["output_dir"])


def test_normalize_version_accepts_alias_and_rejects_ambiguous_values() -> None:
    assert normalize_data_version("datav3") == "datav3"
    assert normalize_data_version("data-v004") == "datav4"
    assert normalize_data_version(9) == "datav9"
    for invalid in ("v3", "data3", "datav-3", "datav3-extra", True, None):
        with pytest.raises(StudioError, match="datavN"):
            normalize_data_version(invalid)  # type: ignore[arg-type]


def test_ledger_starts_at_datav3_records_changes_and_is_content_idempotent(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    first = _export(labeled_corpus, default_selection, "first")
    changed_selection = json.loads(json.dumps(default_selection))
    changed_selection["stage1"]["qualities"] = ["acceptable", "good", "excellent"]
    second = _export(labeled_corpus, changed_selection, "second")
    ledger_root = labeled_corpus["tmp_path"] / "dataset-versions"

    assert next_version(ledger_root) == "datav3"
    datav3 = record_version(ledger_root, first, activate=True)
    assert datav3["version"] == "datav3"
    assert datav3["idempotent"] is False
    assert next_version(ledger_root) == "datav4"

    duplicate = VersionLedger(ledger_root).record(first)
    assert duplicate["version"] == "datav3"
    assert duplicate["idempotent"] is True
    assert len(list_versions(ledger_root)) == 1
    with pytest.raises(StudioError, match="already recorded as datav3"):
        VersionLedger(ledger_root).record(first, version="data-v4")

    datav4 = VersionLedger(ledger_root).record(second, version="data-v4")
    assert datav4["version"] == "datav4"
    assert [entry["version"] for entry in list_versions(ledger_root)] == ["datav3", "datav4"]
    activate_version(ledger_root, "data-v4")
    assert VersionLedger(ledger_root).state()["active_version"] == "datav4"
    activate_version(ledger_root, "datav3")
    assert VersionLedger(ledger_root).state()["active_version"] == "datav3"
    assert verify_version(ledger_root, "datav3")["status"] == "ready"


def test_concurrent_record_of_same_content_creates_one_version(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    snapshot = _export(labeled_corpus, default_selection, "concurrent")
    ledger_root = labeled_corpus["tmp_path"] / "dataset-versions"

    def record() -> dict[str, Any]:
        return VersionLedger(ledger_root).record(snapshot)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: record(), range(16)))

    assert {result["version"] for result in results} == {"datav3"}
    assert sum(not result["idempotent"] for result in results) == 1
    assert len(list_versions(ledger_root)) == 1
    assert json.loads((ledger_root / "ledger.json").read_text(encoding="utf-8"))[
        "next_number"
    ] == 4


def test_verify_rejects_snapshot_file_tampering(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    snapshot = _export(labeled_corpus, default_selection, "tamper")
    ledger_root = labeled_corpus["tmp_path"] / "dataset-versions"
    VersionLedger(ledger_root).record(snapshot)
    target = snapshot / "stage1" / "chatts_sft.jsonl"
    target.write_bytes(target.read_bytes() + b"{}\n")

    with pytest.raises(StudioError, match="SHA256 mismatch"):
        verify_snapshot(snapshot)
    with pytest.raises(StudioError, match="SHA256 mismatch"):
        VersionLedger(ledger_root).verify("datav3")


def test_composition_includes_arbitrary_annotated_sources(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    source_name = "sensor_reasoning_42"
    raw_relative = Path("data/versions/datav2/files") / f"{source_name}.jsonl"
    write_jsonl(
        labeled_corpus["data_root"] / raw_relative,
        [
            {"input": "custom question", "timeseries": [[1.0, 2.0]], "output": "custom answer"}
        ],
    )
    write_jsonl(
        labeled_corpus["annotations_root"] / "annotations" / f"{source_name}.jsonl",
        [
            {
                "annotation_id": f"{source_name}:1",
                "annotation_source": source_name,
                "source_index": 0,
                "line_number": 1,
                "quality": "good",
                "difficulty": "hard",
                "ability_bucket": "custom_reasoning",
            }
        ],
    )
    registry_path = labeled_corpus["registry_path"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["sources"].append(
        {
            "name": source_name,
            "path": raw_relative.as_posix(),
            "family": "custom",
            "split": "train",
            "training_role": "reasoning",
        }
    )
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    selection = json.loads(json.dumps(default_selection))
    selection["stage1"]["sources"].append(source_name)
    selection["stage1"]["difficulties"].append("hard")
    selection["stage2"]["sources"].append(source_name)
    snapshot = _export(labeled_corpus, selection, "all-sources")

    entry = VersionLedger(labeled_corpus["tmp_path"] / "dataset-versions").record(snapshot)
    stage1_sources = {row["source"] for row in entry["composition"]["stage1"]["sources"]}
    stage2_sources = {row["source"] for row in entry["composition"]["stage2"]["sources"]}
    assert source_name in stage1_sources
    assert source_name in stage2_sources
