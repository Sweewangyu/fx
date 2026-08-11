from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import chatts_dataset_studio.versioning as versioning
from chatts_dataset_studio.catalog import CatalogCache
from chatts_dataset_studio.exporter import export_selection
from chatts_dataset_studio.models import StudioError
from chatts_dataset_studio.training_registry import (
    activate_training_version,
    register_training_version,
    verify_training_registration,
    versioned_output_root,
)
from chatts_dataset_studio.versioning import VersionLedger


def _export_and_record(
    labeled_corpus: dict[str, Any], selection: dict[str, Any], name: str
) -> tuple[Path, VersionLedger, dict[str, Any]]:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    result = export_selection(
        {
            **selection,
            "run_name": name,
            "output_root": str(labeled_corpus["output_root"]),
        },
        sources,
        catalog,
    )
    ledger = VersionLedger(labeled_corpus["tmp_path"] / "dataset-versions")
    entry = ledger.record(result["output_dir"])
    return Path(result["output_dir"]), ledger, entry


def _training_root(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "ChatTS-Training"
    data = root / "data"
    data.mkdir(parents=True)
    sentinel = b'{"existing":{"file_name":"do-not-touch.jsonl"}}\n'
    (data / "dataset_info.json").write_bytes(sentinel)
    return root, sentinel


def test_register_writes_only_version_profile_env_and_active_pointer(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    snapshot, ledger, entry = _export_and_record(labeled_corpus, default_selection, "register")
    training_root, original_dataset_info = _training_root(labeled_corpus["tmp_path"])
    output_base = labeled_corpus["tmp_path"] / "models" / "ChatTS-msxf-8B-datav1"

    result = register_training_version(
        training_root,
        ledger.root,
        "data-v3",
        model_output_base=output_base,
        activate=True,
    )

    registry = training_root / "data" / "studio_versions"
    assert Path(result["profile_path"]) == registry / "datav3.json"
    assert Path(result["env_path"]) == registry / "datav3.env"
    assert (training_root / "data" / "dataset_info.json").read_bytes() == original_dataset_info
    profile = json.loads((registry / "datav3.json").read_text(encoding="utf-8"))
    active = json.loads((registry / "active.json").read_text(encoding="utf-8"))
    assert profile["version"] == active["version"] == "datav3"
    assert profile["snapshot_path"] == str(snapshot)
    assert profile["dataset_snapshot_hash"] == entry["dataset_snapshot_hash"]
    assert profile["environment"]["DATASET_DIR"] == str(snapshot)
    assert profile["environment"]["OUTPUT_ROOT"].endswith("ChatTS-msxf-8B-datav3")
    assert ledger.state()["active_version"] == "datav3"
    assert verify_training_registration(training_root, "datav3")["status"] == "verified"

    repeated = register_training_version(
        training_root,
        ledger.root,
        "datav3",
        model_output_base=output_base,
    )
    assert repeated["created"] is False
    with pytest.raises(StudioError, match="conflicts"):
        register_training_version(
            training_root,
            ledger.root,
            "datav3",
            model_output_base=labeled_corpus["tmp_path"] / "different-model-root",
        )


def test_training_activation_can_roll_back_without_mutating_profiles(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    _, ledger, _ = _export_and_record(labeled_corpus, default_selection, "v3")
    changed = json.loads(json.dumps(default_selection))
    changed["stage2"]["qualities"] = ["good", "excellent"]
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    result = export_selection(
        {
            **changed,
            "run_name": "v4",
            "output_root": str(labeled_corpus["output_root"]),
        },
        sources,
        catalog,
    )
    ledger.record(result["output_dir"])
    training_root, _ = _training_root(labeled_corpus["tmp_path"])
    register_training_version(training_root, ledger.root, "datav3")
    register_training_version(training_root, ledger.root, "datav4", activate=True)
    registry = training_root / "data" / "studio_versions"
    v4_before = (registry / "datav4.json").read_bytes()

    activate_training_version(training_root, "data-v3")

    active = json.loads((registry / "active.json").read_text(encoding="utf-8"))
    assert active["version"] == "datav3"
    assert (registry / "datav4.json").read_bytes() == v4_before


def test_training_registration_verification_rejects_env_tampering(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    _, ledger, _ = _export_and_record(labeled_corpus, default_selection, "env-tamper")
    training_root, _ = _training_root(labeled_corpus["tmp_path"])
    register_training_version(training_root, ledger.root, "datav3")
    env_path = training_root / "data" / "studio_versions" / "datav3.env"
    env_path.write_text(env_path.read_text(encoding="utf-8") + "EVIL=1\n", encoding="utf-8")

    with pytest.raises(StudioError, match="environment has changed"):
        verify_training_registration(training_root, "datav3")


def test_training_registry_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "ChatTS-Training"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StudioError, match="escapes"):
        activate_training_version(root, "datav3")


def test_versioned_output_root_replaces_existing_data_suffix(tmp_path: Path) -> None:
    assert versioned_output_root(tmp_path / "model-datav1", "data-v3") == (
        tmp_path / "model-datav3"
    ).resolve()
    assert versioned_output_root(tmp_path / "model", "datav4") == (
        tmp_path / "model-datav4"
    ).resolve()
    assert versioned_output_root(tmp_path / "model_datav9", "datav4") == (
        tmp_path / "model-datav4"
    ).resolve()


def test_register_activate_reuses_one_snapshot_verification_and_orders_pointers(
    labeled_corpus: dict[str, Any],
    default_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, ledger, _ = _export_and_record(labeled_corpus, default_selection, "verify-once")
    training_root, _ = _training_root(labeled_corpus["tmp_path"])
    registry = training_root / "data" / "studio_versions"

    verification_calls = 0
    original_verify_snapshot = versioning.verify_snapshot

    def counted_verify_snapshot(snapshot_dir: str | Path) -> dict[str, Any]:
        nonlocal verification_calls
        verification_calls += 1
        return original_verify_snapshot(snapshot_dir)

    ledger_activation_observed_training_pointer: list[str] = []
    original_ledger_activate = VersionLedger.activate

    def observed_ledger_activate(
        self: VersionLedger,
        version: str | int,
        *,
        verified_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = json.loads((registry / "active.json").read_text(encoding="utf-8"))
        ledger_activation_observed_training_pointer.append(active["version"])
        return original_ledger_activate(
            self, version, verified_snapshot=verified_snapshot
        )

    monkeypatch.setattr(versioning, "verify_snapshot", counted_verify_snapshot)
    monkeypatch.setattr(VersionLedger, "activate", observed_ledger_activate)

    register_training_version(training_root, ledger.root, "datav3", activate=True)

    assert verification_calls == 1
    assert ledger_activation_observed_training_pointer == ["datav3"]
    assert ledger.state()["active_version"] == "datav3"
