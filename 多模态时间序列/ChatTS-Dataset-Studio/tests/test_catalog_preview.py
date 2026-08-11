from __future__ import annotations

from typing import Any

from chatts_dataset_studio.catalog import CatalogCache
from chatts_dataset_studio.exporter import parse_rules, preview_selection


def test_catalog_scans_all_six_sources_and_label_dimensions(
    labeled_corpus: dict[str, Any],
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )

    assert [source.name for source in sources] == list(labeled_corpus["sources"])
    assert catalog["total_sources"] == 6
    assert catalog["available_sources"] == 6
    assert catalog["total_rows"] == 30
    assert catalog["abilities"] == [
        "anomaly_detection",
        "forecasting",
        "numerical_reasoning",
        "pattern_recognition",
        "trend_analysis",
    ]
    for summary in catalog["sources"]:
        assert summary["available"] is True
        assert summary["annotation_mode"] == "sidecar"
        assert summary["rows"] == 5
        assert summary["quality"] == {
            "unusable": 1,
            "weak": 1,
            "acceptable": 1,
            "good": 1,
            "excellent": 1,
        }


def test_default_preview_matches_requested_stage_boundaries(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    stage1, stage2 = parse_rules(default_selection, sources)

    preview = preview_selection(catalog, stage1, stage2)

    assert preview["counts"] == {"stage1": 12, "stage2": 18, "overlap": 6}
    assert len(preview["by_source"]) == 6
    assert all(
        row["stage1"] == 2 and row["stage2"] == 3 and row["overlap"] == 1
        for row in preview["by_source"]
    )
    assert preview["distributions"]["stage1"]["difficulty"] == {
        "easy": 6,
        "moderate": 6,
    }
    assert preview["distributions"]["stage2"]["difficulty"] == {
        "moderate": 6,
        "hard": 6,
        "very_hard": 6,
    }


def test_ability_filter_is_optional_and_exact(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    default_selection["stage1"]["abilities"] = ["trend_analysis"]
    default_selection["stage2"]["abilities"] = ["anomaly_detection"]
    stage1, stage2 = parse_rules(default_selection, sources)

    preview = preview_selection(catalog, stage1, stage2)

    assert preview["counts"] == {"stage1": 6, "stage2": 6, "overlap": 0}
