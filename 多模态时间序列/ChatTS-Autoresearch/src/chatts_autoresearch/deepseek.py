from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from .hashing import hash_object
from .state import StateStore


class DeepSeekError(RuntimeError):
    pass


def _strict_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DeepSeekError(f"DeepSeek returned non-JSON content: {exc}") from exc
    if not isinstance(value, dict):
        raise DeepSeekError("DeepSeek response must be one JSON object")
    return value


def validate_label(value: dict[str, Any]) -> dict[str, Any]:
    required = {"quality_score", "difficulty", "taxonomy", "rationale"}
    if set(value) != required:
        raise DeepSeekError(f"Label keys must be exactly {sorted(required)}")
    if isinstance(value["quality_score"], bool) or not isinstance(
        value["quality_score"], (int, float)
    ):
        raise DeepSeekError("quality_score must be numeric")
    quality = float(value["quality_score"])
    if not 0.0 <= quality <= 1.0:
        raise DeepSeekError("quality_score must be in [0, 1]")
    if value["difficulty"] not in {"easy", "medium", "hard"}:
        raise DeepSeekError("difficulty must be easy, medium, or hard")
    for key in ("taxonomy", "rationale"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise DeepSeekError(f"{key} must be a non-empty string")
    if len(value["taxonomy"]) > 120 or len(value["rationale"]) > 1000:
        raise DeepSeekError("taxonomy or rationale is too long")
    return {
        "quality_score": quality,
        "difficulty": value["difficulty"],
        "taxonomy": value["taxonomy"].strip(),
        "rationale": value["rationale"].strip(),
    }


ALLOWED_PATCH_KEYS = {
    "learning_rate",
    "projector_lr_ratio",
    "warmup_ratio",
    "scheduler",
    "epochs",
    "source_weights",
    "minimum_quality",
    "difficulty_weights",
}


LABEL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["quality_score", "difficulty", "taxonomy", "rationale"],
    "properties": {
        "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "taxonomy": {"type": "string", "minLength": 1, "maxLength": 120},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 1000},
    },
}


def round_analysis_response_schema(
    search: dict[str, Any],
    allowed_sources: set[str],
    allowed_badcase_ids: set[str],
    *,
    require_proposal: bool,
) -> dict[str, Any]:
    low, high = [float(value) for value in search["source_weight_range"]]
    weight_object = {
        "type": "object",
        "minProperties": 1,
        "propertyNames": {"enum": sorted(allowed_sources)},
        "additionalProperties": {"type": "number", "minimum": low, "maximum": high},
    }
    difficulty_object = {
        "type": "object",
        "minProperties": 1,
        "propertyNames": {"enum": ["easy", "medium", "hard"]},
        "additionalProperties": {"type": "number", "minimum": low, "maximum": high},
    }
    patch_schema = {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 1,
        "properties": {
            "learning_rate": {
                "type": "number",
                "enum": [float(value) for value in search["learning_rates"]],
            },
            "projector_lr_ratio": {
                "type": "number",
                "enum": [float(value) for value in search["projector_lr_ratios"]],
            },
            "warmup_ratio": {
                "type": "number",
                "enum": [float(value) for value in search["warmup_ratios"]],
            },
            "scheduler": {"type": "string", "enum": list(search["schedulers"])},
            "epochs": {"type": "integer", "enum": list(search["epochs"])},
            "source_weights": weight_object,
            "minimum_quality": {
                "type": "number",
                "enum": [float(value) for value in search["minimum_qualities"]],
            },
            "difficulty_weights": difficulty_object,
        },
    }
    proposal_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["family", "patch", "rationale"],
        "properties": {
            "family": {"type": "string", "enum": sorted(ALLOWED_PATCH_KEYS)},
            "patch": patch_schema,
            "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
    }
    id_items: dict[str, Any] = {"type": "string"}
    if allowed_badcase_ids:
        id_items["enum"] = sorted(allowed_badcase_ids)
    group_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["error_type", "likely_data_cause", "badcase_ids"],
        "properties": {
            "error_type": {"type": "string", "minLength": 1, "maxLength": 160},
            "likely_data_cause": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1000,
            },
            "badcase_ids": {
                "type": "array",
                "items": id_items,
                "uniqueItems": True,
                "minItems": 1 if allowed_badcase_ids else 0,
                "maxItems": 32,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["error_groups", "recommended_family", "proposal"],
        "properties": {
            "error_groups": {
                "type": "array",
                "items": group_schema,
                "minItems": 1 if allowed_badcase_ids else 0,
                "maxItems": 12,
            },
            "recommended_family": {
                "type": "string",
                "enum": sorted(ALLOWED_PATCH_KEYS),
            },
            "proposal": proposal_schema if require_proposal else {"type": "null"},
        },
    }


def proposal_validator(
    search: dict[str, Any],
    allowed_sources: set[str] | None = None,
    *,
    disallowed_patch_hashes: set[str] | None = None,
    reject_patch: Callable[[dict[str, Any]], bool] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def validate(value: dict[str, Any]) -> dict[str, Any]:
        required = {"family", "patch", "rationale"}
        if set(value) != required:
            raise DeepSeekError(f"Proposal keys must be exactly {sorted(required)}")
        if (
            not isinstance(value["family"], str)
            or not isinstance(value["rationale"], str)
            or not value["rationale"].strip()
        ):
            raise DeepSeekError("family and rationale must be strings")
        patch = value["patch"]
        if not isinstance(patch, dict) or len(patch) != 1:
            raise DeepSeekError("A proposal must change exactly one parameter family")
        key, raw = next(iter(patch.items()))
        if key not in ALLOWED_PATCH_KEYS or value["family"] != key:
            raise DeepSeekError("Proposal family must match its single allowed patch key")
        allowed_lists = {
            "learning_rate": [float(x) for x in search["learning_rates"]],
            "projector_lr_ratio": [float(x) for x in search["projector_lr_ratios"]],
            "warmup_ratio": [float(x) for x in search["warmup_ratios"]],
            "epochs": [int(x) for x in search["epochs"]],
            "minimum_quality": [float(x) for x in search["minimum_qualities"]],
        }
        if key in allowed_lists:
            if key == "epochs":
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise DeepSeekError("epochs must be an integer")
                comparable = raw
            else:
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise DeepSeekError(f"{key} must be numeric")
                comparable = float(raw)
            if comparable not in allowed_lists[key]:
                raise DeepSeekError(f"{key} is outside its whitelist")
            raw = comparable
        elif key == "scheduler":
            if not isinstance(raw, str) or raw not in search["schedulers"]:
                raise DeepSeekError("scheduler is outside its whitelist")
        elif key in {"source_weights", "difficulty_weights"}:
            if not isinstance(raw, dict) or not raw:
                raise DeepSeekError(f"{key} must be a non-empty object")
            low, high = [float(x) for x in search["source_weight_range"]]
            clean = {}
            for name, weight in raw.items():
                if key == "source_weights" and allowed_sources is not None and name not in allowed_sources:
                    raise DeepSeekError(f"Unknown source_weights key: {name}")
                if key == "difficulty_weights" and name not in {"easy", "medium", "hard"}:
                    raise DeepSeekError(f"Unknown difficulty_weights key: {name}")
                if (
                    not isinstance(name, str)
                    or isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not low <= float(weight) <= high
                ):
                    raise DeepSeekError(f"{key} value is outside [{low}, {high}]")
                clean[name] = float(weight)
            raw = clean
        clean = {"family": key, "patch": {key: raw}, "rationale": value["rationale"][:2000]}
        if hash_object(clean["patch"]) in (disallowed_patch_hashes or set()):
            raise DeepSeekError("Proposal duplicates an already used patch")
        if reject_patch is not None and reject_patch(clean["patch"]):
            raise DeepSeekError("Proposal is baseline-equivalent")
        return clean

    return validate


def round_analysis_validator(
    search: dict[str, Any],
    allowed_sources: set[str],
    allowed_badcase_ids: set[str],
    *,
    disallowed_patch_hashes: set[str],
    reject_patch: Callable[[dict[str, Any]], bool],
    require_proposal: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    validate_proposal = proposal_validator(
        search,
        allowed_sources,
        disallowed_patch_hashes=disallowed_patch_hashes,
        reject_patch=reject_patch,
    )

    def validate(value: dict[str, Any]) -> dict[str, Any]:
        required = {"error_groups", "recommended_family", "proposal"}
        if set(value) != required:
            raise DeepSeekError(f"Round analysis keys must be exactly {sorted(required)}")
        groups = value["error_groups"]
        if (
            not isinstance(groups, list)
            or len(groups) > 12
            or (allowed_badcase_ids and not groups)
        ):
            raise DeepSeekError(
                "error_groups must contain 1-12 groups when badcases are supplied, else 0-12"
            )
        clean_groups = []
        grouped_ids: set[str] = set()
        for group in groups:
            group_keys = {"error_type", "likely_data_cause", "badcase_ids"}
            if not isinstance(group, dict) or set(group) != group_keys:
                raise DeepSeekError(
                    f"Each error group must have exactly {sorted(group_keys)}"
                )
            error_type = group["error_type"]
            likely_data_cause = group["likely_data_cause"]
            badcase_ids = group["badcase_ids"]
            if (
                not isinstance(error_type, str)
                or not error_type.strip()
                or not isinstance(likely_data_cause, str)
                or not likely_data_cause.strip()
            ):
                raise DeepSeekError("error_type and likely_data_cause must be non-empty strings")
            if (
                not isinstance(badcase_ids, list)
                or (allowed_badcase_ids and not badcase_ids)
                or len(badcase_ids) > 32
                or len(set(badcase_ids)) != len(badcase_ids)
                or any(not isinstance(item, str) or item not in allowed_badcase_ids for item in badcase_ids)
            ):
                raise DeepSeekError("badcase_ids must be unique IDs from the supplied sample")
            if grouped_ids.intersection(badcase_ids):
                raise DeepSeekError("A badcase_id may appear in only one error group")
            grouped_ids.update(badcase_ids)
            clean_groups.append(
                {
                    "error_type": error_type.strip()[:160],
                    "likely_data_cause": likely_data_cause.strip()[:1000],
                    "badcase_ids": badcase_ids,
                }
            )
        if require_proposal:
            proposal = validate_proposal(value["proposal"])
            if value["recommended_family"] != proposal["family"]:
                raise DeepSeekError("recommended_family must equal proposal.family")
        else:
            if value["proposal"] is not None:
                raise DeepSeekError("The final round proposal must be null")
            if (
                not isinstance(value["recommended_family"], str)
                or value["recommended_family"] not in ALLOWED_PATCH_KEYS
            ):
                raise DeepSeekError("recommended_family is not an allowed parameter family")
            proposal = None
        return {
            "error_groups": clean_groups,
            "recommended_family": (
                proposal["family"] if proposal is not None else value["recommended_family"]
            ),
            "proposal": proposal,
        }

    return validate


class DeepSeekClient:
    def __init__(
        self,
        config: dict[str, Any],
        state: StateStore,
        transport: httpx.BaseTransport | None = None,
    ):
        self.config = config
        self.state = state
        api_key = os.environ.get(config["api_key_env"], "")
        headers = {"Authorization": f"Bearer {api_key or 'EMPTY'}"}
        self.client = httpx.Client(
            base_url=config["base_url"].rstrip("/") + "/",
            headers=headers,
            timeout=float(config["timeout_seconds"]),
            transport=transport,
            trust_env=False,
        )

    def close(self) -> None:
        self.client.close()

    def complete_json(
        self,
        purpose: str,
        system: str,
        user: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
        prompt_version: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response_format: dict[str, Any] = {"type": "json_object"}
        if (
            response_schema is not None
            and self.config.get("response_format", "json_schema") == "json_schema"
        ):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": purpose.replace("-", "_")[:64],
                    "strict": True,
                    "schema": response_schema,
                },
            }
        request = {
            "model": self.config["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": float(self.config.get("temperature", 0.0)),
            "response_format": response_format,
        }
        request_hash = hash_object(
            {"purpose": purpose, "prompt_version": prompt_version, "request": request}
        )
        cache_key = hash_object(
            {
                "model": self.config["model"],
                "prompt_version": prompt_version,
                "request_hash": request_hash,
            }
        )
        cached = self.state.cache_get(cache_key)
        if cached is not None:
            try:
                return validator(cached)
            except DeepSeekError:
                # Policy may have tightened since this response was cached.
                # Re-query and cache only a response valid under the new policy.
                pass
        last_error: Exception | None = None
        retries = int(self.config.get("max_retries", 2))
        for attempt in range(retries + 1):
            try:
                attempt_request = dict(request)
                if isinstance(last_error, DeepSeekError):
                    attempt_request["messages"] = [
                        *request["messages"],
                        {
                            "role": "user",
                            "content": (
                                "The previous JSON violated the schema/policy: "
                                f"{last_error}. Return a different valid JSON object."
                            ),
                        },
                    ]
                response = self.client.post("chat/completions", json=attempt_request)
                response.raise_for_status()
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise DeepSeekError("DeepSeek message content is not text")
                validated = validator(_strict_object(content))
                self.state.cache_put(cache_key, purpose, request_hash, validated)
                return validated
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, DeepSeekError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
        raise DeepSeekError(f"DeepSeek request failed after {retries + 1} attempts: {last_error}")
