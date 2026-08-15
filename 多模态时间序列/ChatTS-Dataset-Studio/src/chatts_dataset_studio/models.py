from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUALITY_LEVELS = ("unusable", "weak", "acceptable", "good", "excellent")
DIFFICULTY_LEVELS = ("very_easy", "easy", "moderate", "hard", "very_hard")

QUALITY_LABELS_ZH = {
    "unusable": "不可用",
    "weak": "较弱",
    "acceptable": "可接受",
    "good": "良好",
    "excellent": "优秀",
}
DIFFICULTY_LABELS_ZH = {
    "very_easy": "非常简单",
    "easy": "简单",
    "moderate": "中等",
    "hard": "困难",
    "very_hard": "非常困难",
}

ANNOTATION_FIELDS = {
    "annotation_id",
    "annotation_source",
    "source_index",
    "line_number",
    "taxonomy_sample_id",
    "taxonomy_cluster_id",
    "ability_label",
    "ability_bucket",
    "ability_name",
    "ability_major",
    "ability_secondary_labels",
    "ability_label_source",
    "ability_status",
    "ability_confidence",
    "taxonomy_fit",
    "reasoning_subtype",
    "training_role",
    "taxonomy_verifier_status",
    "ability_join_method",
    "quality",
    "difficulty",
    "quality_reason",
    "quality_template_id",
    "quality_model",
    "quality_prompt_version",
}

DEFAULT_ALIASES = {
    "chatts_align_256": "align_256",
    "chatts_align_random": "align_random",
    "chatts_ift": "ift",
    "chatts_sft": "sft",
    "time_mqa": "finiverse_time_mqa",
    "tsaqa": "finiverse_tsaqa",
}

DEFAULT_TARGET_SOURCES = tuple(DEFAULT_ALIASES)
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class StudioError(RuntimeError):
    """Raised when an input, selection, or export violates the data contract."""


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    family: str
    split: str
    training_role: str
    annotation_path: Path
    annotation_mode: str


@dataclass(frozen=True)
class FilterRule:
    qualities: frozenset[str]
    difficulties: frozenset[str]
    abilities: frozenset[str]

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        field: str,
        required: bool,
    ) -> FilterRule:
        if not isinstance(value, dict):
            raise StudioError(f"{field} must be one JSON/YAML object")
        qualities = _string_set(value.get("qualities"), f"{field}.qualities")
        difficulties = _string_set(value.get("difficulties"), f"{field}.difficulties")
        abilities = _string_set(value.get("abilities", []), f"{field}.abilities")
        invalid_quality = qualities - set(QUALITY_LEVELS)
        invalid_difficulty = difficulties - set(DIFFICULTY_LEVELS)
        if invalid_quality:
            raise StudioError(f"Unknown quality levels in {field}: {sorted(invalid_quality)}")
        if invalid_difficulty:
            raise StudioError(f"Unknown difficulty levels in {field}: {sorted(invalid_difficulty)}")
        if required and not qualities:
            raise StudioError(f"{field} must select at least one quality")
        if required and not difficulties:
            raise StudioError(f"{field} must select at least one difficulty")
        return cls(
            qualities=frozenset(qualities),
            difficulties=frozenset(difficulties),
            abilities=frozenset(abilities),
        )

    def matches(self, annotation: dict[str, Any]) -> bool:
        quality = annotation.get("quality")
        difficulty = annotation.get("difficulty")
        ability = annotation.get("ability_bucket") or annotation.get("ability_label") or "UNMAPPED"
        return (
            quality in self.qualities
            and difficulty in self.difficulties
            and (not self.abilities or ability in self.abilities)
        )

    def to_mapping(self) -> dict[str, list[str]]:
        return {
            "qualities": sorted(self.qualities),
            "difficulties": sorted(self.difficulties),
            "abilities": sorted(self.abilities),
        }


@dataclass(frozen=True)
class StageRule:
    sources: frozenset[str]
    qualities: frozenset[str]
    difficulties: frozenset[str]
    abilities: frozenset[str]
    source_rules: dict[str, FilterRule]

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        available_sources: set[str],
        source_dimensions: dict[str, dict[str, set[str]]] | None = None,
    ) -> StageRule:
        if not isinstance(value, dict):
            raise StudioError("Each stage rule must be one JSON/YAML object")
        sources = _string_set(value.get("sources"), "sources")
        unknown_sources = sources - available_sources
        if unknown_sources:
            raise StudioError(f"Unknown selected datasets: {sorted(unknown_sources)}")
        fallback = FilterRule.from_mapping(value, field="stage rule", required=bool(sources))

        raw_source_rules = value.get("source_rules", {})
        if not isinstance(raw_source_rules, dict) or any(
            not isinstance(name, str) for name in raw_source_rules
        ):
            raise StudioError("source_rules must be an object keyed by dataset name")
        override_names = set(raw_source_rules)
        unknown_overrides = override_names - available_sources
        if unknown_overrides:
            raise StudioError(f"Unknown source_rules datasets: {sorted(unknown_overrides)}")
        unselected_overrides = override_names - sources
        if unselected_overrides:
            raise StudioError(
                f"source_rules contains unselected datasets: {sorted(unselected_overrides)}"
            )

        source_rules: dict[str, FilterRule] = {}
        for source_name in sorted(override_names):
            source_rule = FilterRule.from_mapping(
                raw_source_rules[source_name],
                field=f"source_rules[{source_name!r}]",
                required=True,
            )
            if source_dimensions is not None:
                dimensions = source_dimensions.get(source_name)
                if dimensions is None:
                    raise StudioError(f"No catalog dimensions are available for {source_name}")
                unavailable_quality = source_rule.qualities - dimensions["qualities"]
                unavailable_difficulty = source_rule.difficulties - dimensions["difficulties"]
                unavailable_ability = source_rule.abilities - dimensions["abilities"]
                if unavailable_quality:
                    raise StudioError(
                        f"Unavailable quality levels for {source_name}: "
                        f"{sorted(unavailable_quality)}"
                    )
                if unavailable_difficulty:
                    raise StudioError(
                        f"Unavailable difficulty levels for {source_name}: "
                        f"{sorted(unavailable_difficulty)}"
                    )
                if unavailable_ability:
                    raise StudioError(
                        f"Unavailable abilities for {source_name}: {sorted(unavailable_ability)}"
                    )
            # An override identical to the legacy fallback has no semantic
            # effect. Dropping it preserves the exact legacy selection/hash.
            if source_rule != fallback:
                source_rules[source_name] = source_rule

        return cls(
            sources=frozenset(sources),
            qualities=fallback.qualities,
            difficulties=fallback.difficulties,
            abilities=fallback.abilities,
            source_rules=source_rules,
        )

    def fallback_rule(self) -> FilterRule:
        return FilterRule(self.qualities, self.difficulties, self.abilities)

    def effective_rule(self, source: str) -> FilterRule:
        source_rule = self.source_rules.get(source)
        return source_rule if source_rule is not None else self.fallback_rule()

    def matches(self, source: str, annotation: dict[str, Any]) -> bool:
        if source not in self.sources:
            return False
        source_rule = self.source_rules.get(source)
        if source_rule is not None:
            return source_rule.matches(annotation)
        quality = annotation.get("quality")
        difficulty = annotation.get("difficulty")
        ability = annotation.get("ability_bucket") or annotation.get("ability_label") or "UNMAPPED"
        return (
            quality in self.qualities
            and difficulty in self.difficulties
            and (not self.abilities or ability in self.abilities)
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sources": sorted(self.sources),
            **self.fallback_rule().to_mapping(),
        }
        if self.source_rules:
            result["source_rules"] = {
                name: self.source_rules[name].to_mapping() for name in sorted(self.source_rules)
            }
        return result


def _string_set(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise StudioError(f"{field} must be a list of strings")
    return set(value)


def safe_name(value: str, field: str = "name") -> str:
    if not isinstance(value, str) or not SAFE_NAME_RE.fullmatch(value):
        raise StudioError(f"{field} may contain only letters, digits, dot, underscore, and dash")
    return value
