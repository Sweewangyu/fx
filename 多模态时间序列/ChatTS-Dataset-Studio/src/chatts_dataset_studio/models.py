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
class StageRule:
    sources: frozenset[str]
    qualities: frozenset[str]
    difficulties: frozenset[str]
    abilities: frozenset[str]

    @classmethod
    def from_mapping(cls, value: dict[str, Any], available_sources: set[str]) -> StageRule:
        if not isinstance(value, dict):
            raise StudioError("Each stage rule must be one JSON/YAML object")
        sources = _string_set(value.get("sources"), "sources")
        unknown_sources = sources - available_sources
        if unknown_sources:
            raise StudioError(f"Unknown selected datasets: {sorted(unknown_sources)}")
        qualities = _string_set(value.get("qualities"), "qualities")
        difficulties = _string_set(value.get("difficulties"), "difficulties")
        abilities = _string_set(value.get("abilities", []), "abilities")
        invalid_quality = qualities - set(QUALITY_LEVELS)
        invalid_difficulty = difficulties - set(DIFFICULTY_LEVELS)
        if invalid_quality:
            raise StudioError(f"Unknown quality levels: {sorted(invalid_quality)}")
        if invalid_difficulty:
            raise StudioError(f"Unknown difficulty levels: {sorted(invalid_difficulty)}")
        if sources and not qualities:
            raise StudioError("A stage with selected datasets must select at least one quality")
        if sources and not difficulties:
            raise StudioError("A stage with selected datasets must select at least one difficulty")
        return cls(
            sources=frozenset(sources),
            qualities=frozenset(qualities),
            difficulties=frozenset(difficulties),
            abilities=frozenset(abilities),
        )

    def matches(self, source: str, annotation: dict[str, Any]) -> bool:
        if source not in self.sources:
            return False
        quality = annotation.get("quality")
        difficulty = annotation.get("difficulty")
        ability = annotation.get("ability_bucket") or annotation.get("ability_label") or "UNMAPPED"
        return (
            quality in self.qualities
            and difficulty in self.difficulties
            and (not self.abilities or ability in self.abilities)
        )


def _string_set(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise StudioError(f"{field} must be a list of strings")
    return set(value)


def safe_name(value: str, field: str = "name") -> str:
    if not isinstance(value, str) or not SAFE_NAME_RE.fullmatch(value):
        raise StudioError(f"{field} may contain only letters, digits, dot, underscore, and dash")
    return value
