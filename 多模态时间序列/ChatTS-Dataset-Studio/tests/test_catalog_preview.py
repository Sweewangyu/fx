from __future__ import annotations

from typing import Any

import pytest
from conftest import read_jsonl, write_jsonl

from chatts_dataset_studio.catalog import ABILITY_CODES, ABILITY_NAMES, CatalogCache
from chatts_dataset_studio.exporter import parse_rules, preview_selection
from chatts_dataset_studio.models import StudioError


def test_catalog_scans_every_registered_source_and_label_dimension(
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
    assert catalog["ability_levels"] == list(ABILITY_NAMES)
    assert catalog["ability_level_mode"] == "names"
    assert catalog["ability_extras"] == ["forecasting", "trend_analysis"]
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


def test_catalog_detects_code_labels_and_still_exposes_all_fifteen_levels(
    labeled_corpus: dict[str, Any],
) -> None:
    for source in labeled_corpus["sources"]:
        path = labeled_corpus["annotations_root"] / "annotations" / f"{source}.jsonl"
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            row["ability_label"] = ABILITY_CODES[index]
            row["ability_bucket"] = ABILITY_CODES[index]
        write_jsonl(path, rows)

    _, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )

    assert catalog["ability_level_mode"] == "codes"
    assert catalog["ability_levels"] == list(ABILITY_CODES)
    assert catalog["ability_extras"] == []
    assert catalog["abilities"] == sorted(ABILITY_CODES[:5])


def test_catalog_keeps_observed_extra_abilities_separate_from_authoritative_levels(
    labeled_corpus: dict[str, Any],
) -> None:
    source = labeled_corpus["sources"][0]
    path = labeled_corpus["annotations_root"] / "annotations" / f"{source}.jsonl"
    rows = read_jsonl(path)
    rows[0]["ability_label"] = "custom_reasoning"
    rows[0]["ability_bucket"] = "custom_reasoning"
    write_jsonl(path, rows)

    _, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )

    assert catalog["ability_level_mode"] == "names"
    assert catalog["ability_levels"] == list(ABILITY_NAMES)
    assert catalog["ability_extras"] == [
        "custom_reasoning",
        "forecasting",
        "trend_analysis",
    ]
    assert "custom_reasoning" in catalog["abilities"]


def test_catalog_defaults_to_name_levels_when_observations_match_neither_taxonomy(
    labeled_corpus: dict[str, Any],
) -> None:
    for source in labeled_corpus["sources"]:
        path = labeled_corpus["annotations_root"] / "annotations" / f"{source}.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            row["ability_label"] = "unknown_taxonomy_value"
            row["ability_bucket"] = "unknown_taxonomy_value"
        write_jsonl(path, rows)

    _, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )

    assert catalog["ability_level_mode"] == "names"
    assert catalog["ability_levels"] == list(ABILITY_NAMES)
    assert catalog["ability_extras"] == ["unknown_taxonomy_value"]


def test_default_preview_matches_requested_stage_boundaries(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    stage1, stage2 = parse_rules(default_selection, sources, catalog)

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


def test_star_selector_expands_to_every_available_source(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    all_selection = {
        stage: {**default_selection[stage], "sources": ["*"]}
        for stage in ("stage1", "stage2")
    }
    stage1, stage2 = parse_rules(all_selection, sources, catalog)

    assert stage1.sources == stage2.sources == frozenset(labeled_corpus["sources"])
    assert preview_selection(catalog, stage1, stage2)["counts"] == {
        "stage1": 12,
        "stage2": 18,
        "overlap": 6,
    }


def test_star_selector_excludes_sources_that_failed_catalog_validation(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    unavailable_name = "time_mqa"
    annotation_path = (
        labeled_corpus["annotations_root"] / "annotations" / f"{unavailable_name}.jsonl"
    )
    rows = read_jsonl(annotation_path)
    rows[0]["annotation_id"] = "wrong:id"
    write_jsonl(annotation_path, rows)
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    all_selection = {
        stage: {**default_selection[stage], "sources": ["*"]}
        for stage in ("stage1", "stage2")
    }

    stage1, stage2 = parse_rules(all_selection, sources, catalog)

    expected = frozenset(set(labeled_corpus["sources"]) - {unavailable_name})
    assert stage1.sources == stage2.sources == expected
    assert preview_selection(catalog, stage1, stage2)["counts"] == {
        "stage1": 10,
        "stage2": 15,
        "overlap": 5,
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
    stage1, stage2 = parse_rules(default_selection, sources, catalog)

    preview = preview_selection(catalog, stage1, stage2)

    assert preview["counts"] == {"stage1": 6, "stage2": 6, "overlap": 0}


def test_source_rules_apply_different_filters_to_each_selected_dataset(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    source_a, source_b = labeled_corpus["sources"][:2]
    default_selection["stage1"]["sources"] = [source_a, source_b]
    default_selection["stage1"]["source_rules"] = {
        source_a: {
            "qualities": ["good"],
            "difficulties": ["hard"],
            "abilities": ["anomaly_detection"],
        },
        source_b: {
            "qualities": ["excellent"],
            "difficulties": ["very_hard"],
            "abilities": ["forecasting"],
        },
    }

    stage1, stage2 = parse_rules(default_selection, sources, catalog)
    preview = preview_selection(catalog, stage1, stage2)

    assert preview["counts"] == {"stage1": 2, "stage2": 18, "overlap": 2}
    rows = {row["source"]: row for row in preview["by_source"]}
    assert rows[source_a]["stage1"] == 1
    assert rows[source_b]["stage1"] == 1
    assert stage1.to_mapping()["source_rules"] == default_selection["stage1"][
        "source_rules"
    ]


def test_source_rule_rejects_dimensions_not_available_in_that_dataset(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    source = labeled_corpus["sources"][0]
    default_selection["stage1"]["sources"] = [source]
    default_selection["stage1"]["source_rules"] = {
        source: {
            "qualities": ["good"],
            "difficulties": ["hard"],
            "abilities": ["not_in_this_dataset"],
        }
    }

    with pytest.raises(StudioError, match=f"Unavailable abilities for {source}"):
        parse_rules(default_selection, sources, catalog)


def test_source_rule_cannot_override_an_unselected_dataset(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    selected, unselected = labeled_corpus["sources"][:2]
    default_selection["stage1"]["sources"] = [selected]
    default_selection["stage1"]["source_rules"] = {
        unselected: {
            "qualities": ["good"],
            "difficulties": ["hard"],
            "abilities": [],
        }
    }

    with pytest.raises(StudioError, match="source_rules contains unselected datasets"):
        parse_rules(default_selection, sources, catalog)


def test_preview_rejects_a_selected_dataset_whose_joint_filter_has_zero_rows(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    source = labeled_corpus["sources"][0]
    default_selection["stage1"]["sources"] = [source]
    # Each dimension exists in the source, but this exact joint combination does not.
    default_selection["stage1"]["source_rules"] = {
        source: {
            "qualities": ["good"],
            "difficulties": ["moderate"],
            "abilities": ["anomaly_detection"],
        }
    }
    stage1, stage2 = parse_rules(default_selection, sources, catalog)

    with pytest.raises(
        StudioError,
        match=rf"Stage1 filters select zero rows for selected datasets: \['{source}'\]",
    ):
        preview_selection(catalog, stage1, stage2)


@pytest.mark.parametrize("value", [0, 101, True, "50", float("nan"), 33.3333333])
def test_sample_percent_rejects_invalid_values(
    labeled_corpus: dict[str, Any],
    default_selection: dict[str, Any],
    value: Any,
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    default_selection["stage1"]["sample_percent"] = value

    with pytest.raises(StudioError, match="sample_percent"):
        parse_rules(default_selection, sources, catalog)


def test_decimal_sample_percent_is_canonicalized_and_recorded(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    source = labeled_corpus["sources"][0]
    default_selection["stage1"]["sources"] = [source]
    default_selection["stage1"]["source_rules"] = {
        source: {
            "qualities": default_selection["stage1"]["qualities"],
            "difficulties": default_selection["stage1"]["difficulties"],
            "abilities": [],
            "sample_percent": 50.500000,
        }
    }

    stage1, _ = parse_rules(default_selection, sources, catalog)

    assert stage1.effective_rule(source).sample_percent == 50.5
    assert stage1.to_mapping()["source_rules"][source]["sample_percent"] == 50.5
