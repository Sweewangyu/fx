#!/usr/bin/env python3
"""Hierarchical annotation pipeline for the TSRBench 4x15 taxonomy.

The pipeline never rewrites source training rows during annotation.  It first
creates a compact label index and template-level review queue, optionally asks
one or more OpenAI-compatible classifiers to label only ambiguous templates,
resolves rule/model/human decisions, and finally materializes 15 ChatTS JSONL
buckets containing the original three-field samples.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


TAXONOMY_VERSION = "tsrbench-4x15-v1"
PLACEHOLDER = "<ts><ts/>"

TAXONOMY: Dict[str, Dict[str, str]] = {
    "PR": {
        "major": "perception",
        "name": "pattern_recognition",
        "definition": "Recognize observed trend, cyclicity, stationarity, structural characteristics, or core statistics.",
    },
    "NU": {
        "major": "perception",
        "name": "noise_understanding",
        "definition": "Quantify or characterize stochastic noise scale, magnitude, or profile.",
    },
    "AD": {
        "major": "perception",
        "name": "anomaly_detection",
        "definition": "Identify, localize, or classify out-of-distribution observations or anomalous segments.",
    },
    "CA": {
        "major": "perception",
        "name": "comparative_analysis",
        "definition": "Compare two or more series for shared patterns, distributions, statistics, noise, or trends.",
    },
    "ER": {
        "major": "reasoning",
        "name": "etiological_reasoning",
        "definition": "Infer the generative source or underlying causal factors responsible for an observed series.",
    },
    "CD": {
        "major": "reasoning",
        "name": "causal_discovery",
        "definition": "Determine existence and direction of causal relationships between multiple time series.",
    },
    "AR": {
        "major": "reasoning",
        "name": "abductive_reasoning",
        "definition": "Infer the most plausible latent event explaining an observed change using before/after evidence.",
    },
    "TR": {
        "major": "reasoning",
        "name": "temporal_relation_reasoning",
        "definition": "Localize events and establish their chronological order or temporal relationship.",
    },
    "NR": {
        "major": "reasoning",
        "name": "numerical_reasoning",
        "definition": "Perform contextual quantitative calculations over time-series values.",
    },
    "DR": {
        "major": "reasoning",
        "name": "deductive_reasoning",
        "definition": "Apply predefined rules, equations, or constraints to derive a logically necessary conclusion.",
    },
    "IR": {
        "major": "reasoning",
        "name": "inductive_reasoning",
        "definition": "Infer a latent principle or rule from observations, then apply it to a new or future case.",
    },
    "TSF": {
        "major": "prediction",
        "name": "time_series_forecasting",
        "definition": "Predict future numerical time-series values from history and optional context.",
    },
    "EP": {
        "major": "prediction",
        "name": "event_prediction",
        "definition": "Predict a future discrete event from historical series and contextual/domain knowledge.",
    },
    "QualDM": {
        "major": "decision_making",
        "name": "qualitative_decision_making",
        "definition": "Choose an action using time-series patterns and contextual knowledge without outcome simulation.",
    },
    "QuantDM": {
        "major": "decision_making",
        "name": "quantitative_decision_making",
        "definition": "Choose an optimal action by quantitatively simulating/comparing outcomes under rules and constraints.",
    },
}

VALID_LABELS = set(TAXONOMY)
VALID_FITS = {"exact", "compatible", "closest", "mixed", "out_of_scope"}


@dataclass
class SourceSpec:
    name: str
    path: str
    split: str = "train"
    audit: Optional[str] = None


@dataclass
class Decision:
    primary_label: Optional[str]
    secondary_labels: List[str]
    closest_label: Optional[str]
    taxonomy_fit: str
    confidence: float
    status: str
    method: str
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.primary_label:
            result["major_label"] = TAXONOMY[self.primary_label]["major"]
            result["minor_name"] = TAXONOMY[self.primary_label]["name"]
        else:
            result["major_label"] = None
            result["minor_name"] = None
        return result


def _atomic_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    return Path(temporary), os.fdopen(descriptor, "w", encoding="utf-8")


def _finish_atomic(temp_path: Path, stream: Any, final_path: Path) -> None:
    stream.flush()
    stream.close()
    os.replace(temp_path, final_path)
    os.chmod(final_path, 0o644)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def load_registry(path: Path) -> List[SourceSpec]:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw_sources = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("registry must contain a non-empty `sources` list")
    sources = [SourceSpec(**item) for item in raw_sources]
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("source names in registry must be unique")
    for source in sources:
        if not Path(source.path).exists():
            raise FileNotFoundError(f"source does not exist: {source.path}")
        if source.audit and not Path(source.audit).exists():
            raise FileNotFoundError(f"audit does not exist: {source.audit}")
    return sources


def iter_source(
    source: SourceSpec,
    invalid_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Iterator[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]], str]]:
    audit_stream = Path(source.audit).open("r", encoding="utf-8") if source.audit else None
    try:
        with Path(source.path).open("r", encoding="utf-8") as data_stream:
            for index, line in enumerate(data_stream):
                if not line.strip():
                    continue
                audit = None
                if audit_stream:
                    audit_line = audit_stream.readline()
                    if not audit_line:
                        raise ValueError(f"audit ended before source data: {source.audit}")
                    audit = json.loads(audit_line)
                    if int(audit.get("sample_index", -1)) != index:
                        raise ValueError(f"audit/source index mismatch at {source.name}:{index}")
                digest = hashlib.sha256(line.rstrip("\n").encode("utf-8")).hexdigest()
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row is not a JSON object")
                    if set(row) != {"input", "timeseries", "output"}:
                        raise ValueError("row is not exact ChatTS three-field schema")
                except (json.JSONDecodeError, ValueError) as exc:
                    invalid = {
                        "source": source.name,
                        "source_index": index,
                        "line_number": index + 1,
                        "source_sha256": digest,
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                        "line_prefix": line[:200],
                    }
                    if invalid_callback is None:
                        raise ValueError(
                            f"invalid source row at {source.path}:{index + 1}: {exc}"
                        ) from exc
                    invalid_callback(invalid)
                    continue
                yield index, row, audit, digest
        if audit_stream and audit_stream.readline().strip():
            raise ValueError(f"audit has more rows than source data: {source.audit}")
    finally:
        if audit_stream:
            audit_stream.close()


def normalize_template(prompt: str) -> str:
    text = prompt.lower().replace(PLACEHOLDER, " <ts> ")
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _question_count(text: str) -> int:
    numbered = re.findall(r"(?:^|\n)\s*\d+[.)]\s+", text)
    if numbered:
        return len(numbered)
    return 1


def rule_label(source: SourceSpec, prompt: str, output: str, audit: Optional[Mapping[str, Any]]) -> Decision:
    text = prompt.lower()
    source_task = str((audit or {}).get("task", "")).strip().lower()
    scores: Dict[str, float] = {}
    evidence: Dict[str, List[str]] = defaultdict(list)
    forced_fit: Optional[str] = None

    def add(label: str, score: float, reason: str) -> None:
        if score > scores.get(label, 0.0):
            scores[label] = score
        if reason not in evidence[label]:
            evidence[label].append(reason)

    # High-precision source metadata.  These rules intentionally do not force
    # imputation or generic classification to be exact TSRBench tasks.
    if source.name == "tsaqa":
        if source_task == "anomaly_detection":
            add("AD", 0.995, "TSAQA task=anomaly_detection")
        elif source_task == "comparison":
            add("CA", 0.985, "TSAQA task=comparison")
        elif source_task == "temporal_relationship":
            add("TR", 0.985, "TSAQA task=temporal_relationship")
        elif source_task == "classification":
            add("PR", 0.84, "TSAQA generic classification is closest to perception/PR")
            forced_fit = "closest"
        elif source_task == "data_transformation":
            add("DR", 0.80, "TSAQA transformation usually applies a stated operation/rule")
            forced_fit = "compatible"
    elif source.name == "time_mqa":
        if "anomaly" in source_task and source_task in {"anomaly detection", "anomaly_detection"}:
            add("AD", 0.995, "Time-MQA task=anomaly detection")
        elif source_task == "forecasting":
            add("TSF", 0.995, "Time-MQA task=forecasting")
        elif source_task == "imputation":
            add("TSF", 0.65, "imputation is only adjacent to numerical forecasting")
            forced_fit = "out_of_scope"
        elif source_task == "classification":
            add("PR", 0.80, "generic classification is only adjacent to pattern recognition")
            forced_fit = "closest"

    keyword_rules: List[Tuple[str, float, str, Sequence[str]]] = [
        (
            "QuantDM",
            0.96,
            "quantitatively compares operational strategies/outcomes",
            [
                r"backtest(?:ing|ed)?",
                r"maximum drawdown|best return|strategy return",
                r"optimal (?:strategy|procedure|policy|operation)",
                r"simulate.*(?:strategy|procedure|action).*(?:metric|outcome|return)",
            ],
        ),
        (
            "QualDM",
            0.94,
            "asks for an action/management decision",
            [
                r"most appropriate (?:clinical )?(?:management|treatment|action|intervention)",
                r"what should (?:the|a|we|you)",
                r"which (?:action|treatment|intervention|management plan)",
                r"recommend (?:an )?(?:action|treatment|intervention)",
            ],
        ),
        (
            "CD",
            0.97,
            "asks for directed causal relations between series",
            [r"causal (?:relationship|direction|graph)", r"cause nodes?", r"adjacency matrix", r"granger caus"],
        ),
        (
            "AR",
            0.94,
            "asks for a latent event explaining an observed change",
            [
                r"what might have happened",
                r"most plausible (?:latent )?event",
                r"event.*(?:explain|responsible for).*(?:change|shift|spike)",
                r"what happened (?:between|during|around)",
            ],
        ),
        (
            "ER",
            0.92,
            "asks for the generative source or underlying cause",
            [
                r"underlying (?:cause|factor|source)",
                r"generative source",
                r"what (?:could have )?(?:caused|generated|produced) (?:this|the) (?:series|pattern)",
                r"which physical activity.*(?:pattern|series)",
            ],
        ),
        (
            "TR",
            0.95,
            "asks for chronology or temporal ordering",
            [
                r"chronological (?:order|sequence)",
                r"temporal relationship",
                r"order(?:ing)? of (?:the )?(?:events|segments|patches)",
                r"which .* (?:happened|occurred) first",
            ],
        ),
        (
            "TSF",
            0.97,
            "asks for future numerical values",
            [
                r"forecast(?:ing)? (?:the )?(?:next|future)",
                r"predict (?:the )?next \w* ?(?:points?|values?|time series)",
                r"future numerical (?:values?|series)",
                r"most plausible .* (?:price|series) over the next",
            ],
        ),
        (
            "EP",
            0.94,
            "asks whether a future discrete event will occur",
            [
                r"predict whether .* will",
                r"will .* (?:occur|happen|rain|fail) (?:in|within|during|over)",
                r"what event(?:s)? will happen",
                r"event prediction",
            ],
        ),
        (
            "DR",
            0.91,
            "requires applying an explicit rule/equation/threshold",
            [
                r"supposing that",
                r"given (?:the following )?rule",
                r"according to (?:the|this) .* equation",
                r"apply (?:the|this) (?:formula|rule|operation)",
                r"(?:above|below) (?:the )?threshold",
                r"if .* then .*",
            ],
        ),
        (
            "IR",
            0.91,
            "infers a rule/principle and applies it",
            [
                r"infer (?:the )?(?:underlying )?(?:rule|principle)",
                r"identify (?:the )?rule.*predict",
                r"learned pattern.*(?:next|future)",
            ],
        ),
        (
            "NR",
            0.89,
            "requires quantitative calculation",
            [
                r"(?:calculate|compute|estimate|determine) (?:the )?(?:mean|average|variance|standard deviation|amplitude|duration|distance|range|sum|value|rate)",
                r"what is (?:the )?(?:mean|average|variance|standard deviation|amplitude|duration|distance|range|sum)",
                r"how many (?:points|steps|times|events)",
                r"numerical reasoning",
            ],
        ),
        (
            "CA",
            0.94,
            "compares two or more series",
            [
                r"compar(?:e|ing|ison|ative)",
                r"similar(?:ity)? (?:between|to)",
                r"different from (?:each other|time series)",
                r"relationship between .* and .*",
                r"correlation (?:between|of)|correlated with",
                r"find other metric\(s\).*related",
            ],
        ),
        (
            "AD",
            0.95,
            "detects/localizes abnormal observations",
            [
                r"anomal(?:y|ies|ous)",
                r"outlier",
                r"abnormal (?:point|segment|pattern|event)",
                r"structural break|change point",
                r"(?:upward|downward) spike",
            ],
        ),
        (
            "NU",
            0.95,
            "characterizes stochastic noise",
            [r"noise (?:characteristics|level|scale|magnitude|profile|standard deviation)", r"noisy", r"signal.to.noise|\bsnr\b", r"stochastic noise"],
        ),
        (
            "PR",
            0.91,
            "recognizes observed temporal patterns/properties",
            [
                r"trend|periodic|periodicity|seasonal|cyclic|stationar",
                r"pattern (?:recognition|identification)|identify .* pattern",
                r"local characteristic|frequency characteristic",
                r"describe (?:the )?(?:characteristics|behavior|series)",
                r"summari[sz]e (?:the )?(?:series|data|behavior)",
                r"classify the given time series",
            ],
        ),
    ]

    for label, score, reason, patterns in keyword_rules:
        if _has(text, patterns):
            add(label, score, reason)

    if not scores:
        return Decision(
            primary_label=None,
            secondary_labels=[],
            closest_label=None,
            taxonomy_fit="out_of_scope",
            confidence=0.0,
            status="review",
            method="rules",
            evidence=["no high-precision taxonomy rule matched"],
        )

    ranked = sorted(scores, key=lambda label: (-scores[label], label))
    primary = ranked[0]
    secondary = [label for label in ranked[1:] if scores[label] >= 0.90 and scores[primary] - scores[label] <= 0.08]
    compound = _question_count(prompt) > 1 and bool(secondary)
    fit = forced_fit or ("mixed" if compound else "exact")
    confidence = scores[primary]
    if fit == "out_of_scope":
        status = "review"
        primary_label: Optional[str] = None
        closest_label = primary
    else:
        primary_label = primary
        closest_label = primary
        status = "auto_accept" if fit == "exact" and confidence >= 0.94 and not compound else "review"

    combined_evidence: List[str] = []
    for label in [primary] + secondary:
        combined_evidence.extend(f"{label}: {reason}" for reason in evidence[label])
    return Decision(
        primary_label=primary_label,
        secondary_labels=secondary,
        closest_label=closest_label,
        taxonomy_fit=fit,
        confidence=round(confidence, 4),
        status=status,
        method="metadata+rules" if source_task else "rules",
        evidence=combined_evidence[:8],
    )


def _create_cluster_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute(
        """
        CREATE TABLE clusters (
            cluster_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            template_text TEXT NOT NULL,
            representative_input TEXT NOT NULL,
            representative_output TEXT NOT NULL,
            source_task TEXT NOT NULL,
            question_type TEXT NOT NULL,
            domain TEXT NOT NULL,
            provisional_json TEXT NOT NULL,
            first_sample_id TEXT NOT NULL,
            member_count INTEGER NOT NULL
        )
        """
    )
    return connection


def prepare_command(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry).resolve()
    sources = load_registry(registry_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "provisional_labels.jsonl"
    review_path = output_dir / "review_clusters.jsonl"
    invalid_path = output_dir / "invalid_source_rows.jsonl"
    manifest_path = output_dir / "prepare_manifest.json"
    db_path = output_dir / "annotation_state.sqlite"

    labels_temp, labels_stream = _atomic_writer(labels_path)
    invalid_temp, invalid_stream = _atomic_writer(invalid_path)
    descriptor, db_temp_name = tempfile.mkstemp(prefix="annotation_state.", suffix=".sqlite", dir=str(output_dir))
    os.close(descriptor)
    db_temp = Path(db_temp_name)
    db_temp.unlink()
    connection = _create_cluster_db(db_temp)

    total = 0
    counters: Dict[str, Counter] = {
        "source": Counter(),
        "primary": Counter(),
        "fit": Counter(),
        "status": Counter(),
        "invalid_source": Counter(),
    }

    def record_invalid(item: Dict[str, Any]) -> None:
        invalid_stream.write(_json_dump(item) + "\n")
        counters["invalid_source"][item["source"]] += 1

    try:
        for source in sources:
            for index, sample, audit, digest in iter_source(source, record_invalid):
                prompt = str(sample["input"])
                output = str(sample["output"])
                decision = rule_label(source, prompt, output, audit)
                sample_id = f"{source.name}:{index}:{digest[:16]}"
                template = normalize_template(prompt)
                cluster_payload = source.name + "\n" + template
                cluster_id = hashlib.sha256(cluster_payload.encode("utf-8")).hexdigest()[:24]
                source_task = str((audit or {}).get("task", ""))
                question_type = str((audit or {}).get("question_type", ""))
                domain = str((audit or {}).get("domain", ""))
                label_row = {
                    "sample_id": sample_id,
                    "source": source.name,
                    "split": source.split,
                    "source_index": index,
                    "source_sha256": digest,
                    "cluster_id": cluster_id,
                    "source_task": source_task,
                    "question_type": question_type,
                    "domain": domain,
                    "series_count": int((audit or {}).get("series_count", len(sample["timeseries"]))),
                    "provisional": decision.to_dict(),
                }
                labels_stream.write(_json_dump(label_row) + "\n")

                connection.execute(
                    """
                    INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(cluster_id) DO UPDATE SET member_count = member_count + 1
                    """,
                    (
                        cluster_id,
                        source.name,
                        template[: args.max_template_chars],
                        prompt[: args.max_prompt_chars],
                        output[: args.max_output_chars],
                        source_task,
                        question_type,
                        domain,
                        _json_dump(decision.to_dict()),
                        sample_id,
                    ),
                )
                total += 1
                counters["source"][source.name] += 1
                counters["primary"][decision.primary_label or "NONE"] += 1
                counters["fit"][decision.taxonomy_fit] += 1
                counters["status"][decision.status] += 1
                if total % args.commit_every == 0:
                    connection.commit()
                if args.limit and total >= args.limit:
                    break
            if args.limit and total >= args.limit:
                break
        connection.commit()
        _finish_atomic(labels_temp, labels_stream, labels_path)
        _finish_atomic(invalid_temp, invalid_stream, invalid_path)
        connection.close()
        os.replace(db_temp, db_path)
        os.chmod(db_path, 0o644)
    except Exception:
        if not labels_stream.closed:
            labels_stream.close()
        if not invalid_stream.closed:
            invalid_stream.close()
        labels_temp.unlink(missing_ok=True)
        invalid_temp.unlink(missing_ok=True)
        connection.close()
        db_temp.unlink(missing_ok=True)
        raise

    connection = sqlite3.connect(str(db_path))
    review_temp, review_stream = _atomic_writer(review_path)
    review_clusters = 0
    all_clusters = 0
    try:
        query = """
            SELECT cluster_id, source, template_text, representative_input,
                   representative_output, source_task, question_type, domain,
                   provisional_json, first_sample_id, member_count
            FROM clusters ORDER BY member_count DESC, cluster_id
        """
        for row in connection.execute(query):
            all_clusters += 1
            provisional = json.loads(row[8])
            if provisional["status"] == "auto_accept":
                continue
            review_stream.write(
                _json_dump(
                    {
                        "cluster_id": row[0],
                        "source": row[1],
                        "template_text": row[2],
                        "representative_input": row[3],
                        "representative_output": row[4],
                        "source_task": row[5],
                        "question_type": row[6],
                        "domain": row[7],
                        "provisional": provisional,
                        "first_sample_id": row[9],
                        "member_count": row[10],
                    }
                )
                + "\n"
            )
            review_clusters += 1
        _finish_atomic(review_temp, review_stream, review_path)
    finally:
        connection.close()

    manifest = {
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy": TAXONOMY,
        "registry": str(registry_path),
        "sources": [asdict(source) for source in sources],
        "total_samples": total,
        "total_template_clusters": all_clusters,
        "review_template_clusters": review_clusters,
        "counts": {name: dict(sorted(counter.items())) for name, counter in counters.items()},
        "outputs": {
            "provisional_labels": str(labels_path),
            "review_clusters": str(review_path),
            "state_database": str(db_path),
            "invalid_source_rows": str(invalid_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "taxonomy"}, ensure_ascii=False, indent=2))
    return 0


def taxonomy_prompt() -> str:
    lines = [
        "Classify the REQUIRED CAPABILITY of a time-series QA item into the TSRBench taxonomy.",
        "Use the operation needed to answer, not the domain/topic or superficial keywords.",
        "Labels:",
    ]
    for label, item in TAXONOMY.items():
        lines.append(f"- {label} ({item['major']}/{item['name']}): {item['definition']}")
    lines.extend(
        [
            "Boundaries:",
            "- PR describes an observed pattern; IR infers a rule and applies it to a new/future case.",
            "- CA compares association/similarity; CD asks directed causality.",
            "- ER asks the general generating cause; AR explains a localized change with a latent event.",
            "- NR computes a quantity; QuantDM numerically compares action outcomes and chooses an action.",
            "- TSF predicts future numbers; EP predicts a future event; decision labels choose an action.",
            "- Imputation and generic class-label prediction are not exact TSRBench tasks; use closest/out_of_scope.",
            "Return one JSON object only with keys: primary_label, secondary_labels, taxonomy_fit, confidence, rationale.",
            "primary_label is a taxonomy code or null. taxonomy_fit is exact, compatible, closest, mixed, or out_of_scope.",
            'Example JSON: {"primary_label":"AD","secondary_labels":[],"taxonomy_fit":"exact","confidence":0.95,"rationale":"The question asks to locate anomalies."}',
        ]
    )
    return "\n".join(lines)


def _parse_model_json(content: str) -> Dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response contains no JSON object")
    value = json.loads(stripped[start : end + 1])
    primary = value.get("primary_label")
    if primary is not None and primary not in VALID_LABELS:
        raise ValueError(f"invalid primary_label: {primary}")
    secondary = value.get("secondary_labels", [])
    if not isinstance(secondary, list) or any(label not in VALID_LABELS for label in secondary):
        raise ValueError("invalid secondary_labels")
    fit = value.get("taxonomy_fit")
    if fit not in VALID_FITS:
        raise ValueError(f"invalid taxonomy_fit: {fit}")
    confidence = float(value.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    return {
        "primary_label": primary,
        "secondary_labels": sorted(set(secondary) - {primary}),
        "taxonomy_fit": fit,
        "confidence": confidence,
        "rationale": str(value.get("rationale", ""))[:1000],
    }


def annotate_online_command(args: argparse.Namespace) -> int:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("annotate-online requires httpx") from exc
    key = os.environ.get(args.api_key_env, "")
    if not key and not args.allow_no_key:
        raise RuntimeError(f"environment variable {args.api_key_env} is not set")
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as stream:
            completed = {json.loads(line)["cluster_id"] for line in stream if line.strip()}

    items = []
    with input_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["cluster_id"] in completed:
                continue
            items.append(item)
            if args.limit and len(items) >= args.limit:
                break

    system = taxonomy_prompt()
    url = args.base_url.rstrip("/") + "/chat/completions"

    def classify(item: Mapping[str, Any]) -> Dict[str, Any]:
        user = _json_dump(
            {
                "source": item.get("source"),
                "source_task": item.get("source_task"),
                "question_type": item.get("question_type"),
                "question": item.get("representative_input"),
                "answer": item.get("representative_output"),
                "rule_proposal": item.get("provisional"),
            }
        )
        payload = {
            "model": args.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        if args.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if args.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        last_error: Optional[Exception] = None
        for attempt in range(args.retries + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=args.timeout)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                vote = _parse_model_json(content)
                return {
                    "cluster_id": item["cluster_id"],
                    "model": args.model,
                    "taxonomy_version": TAXONOMY_VERSION,
                    "vote": vote,
                }
            except Exception as exc:
                last_error = exc
                if attempt < args.retries:
                    time.sleep(min(8, 2**attempt))
        raise RuntimeError(f"{item['cluster_id']}: {last_error}")

    errors_path = output_path.with_name(output_path.stem + ".errors.jsonl")
    succeeded = failed = 0
    with output_path.open("a", encoding="utf-8") as output_stream, errors_path.open(
        "a", encoding="utf-8"
    ) as error_stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(classify, item): item for item in items}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    output_stream.write(_json_dump(result) + "\n")
                    output_stream.flush()
                    succeeded += 1
                except Exception as exc:
                    error_stream.write(_json_dump({"cluster_id": item["cluster_id"], "error": str(exc)}) + "\n")
                    error_stream.flush()
                    failed += 1
                if (succeeded + failed) % 100 == 0:
                    print(f"processed={succeeded + failed} succeeded={succeeded} failed={failed}", flush=True)
    print(json.dumps({"requested": len(items), "succeeded": succeeded, "failed": failed}, indent=2))
    return 0 if failed == 0 else 1


def _load_votes(paths: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    votes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw_path in paths:
        with Path(raw_path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                votes[item["cluster_id"]].append(item)
    return votes


def _load_human(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    result = {}
    if Path(path).suffix.lower() == ".csv":
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for item in csv.DictReader(stream):
                fit = str(item.get("human_taxonomy_fit", "")).strip()
                if not fit:
                    continue
                if fit not in VALID_FITS:
                    raise ValueError(f"invalid human taxonomy_fit: {fit}")
                raw_primary = str(item.get("human_primary_label", "")).strip()
                primary = None if raw_primary.lower() in {"", "none", "null"} else raw_primary
                if primary is not None and primary not in VALID_LABELS:
                    raise ValueError(f"invalid human primary_label: {primary}")
                raw_secondary = str(item.get("human_secondary_labels", "")).strip()
                secondary = [label for label in re.split(r"[\s,;|]+", raw_secondary) if label]
                if any(label not in VALID_LABELS for label in secondary):
                    raise ValueError(f"invalid human secondary_labels: {raw_secondary}")
                result[item["cluster_id"]] = {
                    "primary_label": primary,
                    "secondary_labels": sorted(set(secondary) - {primary}),
                    "taxonomy_fit": fit,
                    "rationale": str(item.get("human_rationale", "")),
                    "reviewer": str(item.get("reviewer", "")),
                }
        return result
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            primary = item.get("primary_label")
            if primary is not None and primary not in VALID_LABELS:
                raise ValueError(f"invalid human primary_label: {primary}")
            result[item["cluster_id"]] = item
    return result


def export_human_command(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    output_temp, output_stream = _atomic_writer(output_path)
    fields = [
        "cluster_id",
        "source",
        "source_task",
        "question_type",
        "domain",
        "member_count",
        "representative_input",
        "representative_output",
        "rule_primary_label",
        "rule_secondary_labels",
        "rule_taxonomy_fit",
        "human_primary_label",
        "human_secondary_labels",
        "human_taxonomy_fit",
        "human_rationale",
        "reviewer",
    ]
    writer = csv.DictWriter(output_stream, fieldnames=fields)
    writer.writeheader()
    count = 0
    try:
        with Path(args.input).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                provisional = item.get("provisional", {})
                writer.writerow(
                    {
                        "cluster_id": item["cluster_id"],
                        "source": item.get("source", ""),
                        "source_task": item.get("source_task", ""),
                        "question_type": item.get("question_type", ""),
                        "domain": item.get("domain", ""),
                        "member_count": item.get("member_count", ""),
                        "representative_input": item.get("representative_input", ""),
                        "representative_output": item.get("representative_output", ""),
                        "rule_primary_label": provisional.get("primary_label", ""),
                        "rule_secondary_labels": ",".join(provisional.get("secondary_labels", [])),
                        "rule_taxonomy_fit": provisional.get("taxonomy_fit", ""),
                    }
                )
                count += 1
        _finish_atomic(output_temp, output_stream, output_path)
    except Exception:
        if not output_stream.closed:
            output_stream.close()
        output_temp.unlink(missing_ok=True)
        raise
    print(json.dumps({"rows": count, "output": str(output_path.resolve())}, ensure_ascii=False, indent=2))
    return 0


def resolve_command(args: argparse.Namespace) -> int:
    votes = _load_votes(args.votes or [])
    human = _load_human(args.human)
    output_path = Path(args.output)
    output_temp, output_stream = _atomic_writer(output_path)
    unresolved_clusters = set()
    counts = Counter()

    try:
        with Path(args.provisional).open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                cluster_id = item["cluster_id"]
                provisional = item["provisional"]
                final: Dict[str, Any]
                if cluster_id in human:
                    label = human[cluster_id]
                    final = {
                        "primary_label": label.get("primary_label"),
                        "secondary_labels": label.get("secondary_labels", []),
                        "taxonomy_fit": label.get("taxonomy_fit", "exact"),
                        "confidence": 1.0,
                        "status": "accepted" if label.get("primary_label") else "excluded",
                        "method": "human",
                    }
                elif provisional["status"] == "auto_accept":
                    final = {
                        "primary_label": provisional["primary_label"],
                        "secondary_labels": provisional.get("secondary_labels", []),
                        "taxonomy_fit": provisional["taxonomy_fit"],
                        "confidence": provisional["confidence"],
                        "status": "accepted",
                        "method": "rules",
                    }
                else:
                    cluster_votes = [vote["vote"] for vote in votes.get(cluster_id, [])]
                    primary_counts = Counter(vote.get("primary_label") for vote in cluster_votes)
                    winner = primary_counts.most_common(1)[0] if primary_counts else (None, 0)
                    accepted = False
                    method = "unresolved"
                    if len(cluster_votes) >= 2 and winner[1] >= 2:
                        accepted = True
                        method = "model_consensus"
                    elif len(cluster_votes) == 1:
                        vote = cluster_votes[0]
                        proposed = provisional.get("primary_label") or provisional.get("closest_label")
                        if vote.get("primary_label") == proposed and float(vote.get("confidence", 0)) >= 0.85:
                            accepted = True
                            method = "rule_model_agreement"
                    if accepted:
                        agreeing = [vote for vote in cluster_votes if vote.get("primary_label") == winner[0]]
                        fit = Counter(vote.get("taxonomy_fit") for vote in agreeing).most_common(1)[0][0]
                        secondary = sorted({label for vote in agreeing for label in vote.get("secondary_labels", [])})
                        confidence = sum(float(vote.get("confidence", 0)) for vote in agreeing) / len(agreeing)
                        final = {
                            "primary_label": winner[0],
                            "secondary_labels": secondary,
                            "taxonomy_fit": fit,
                            "confidence": round(confidence, 4),
                            "status": "accepted" if winner[0] else "excluded",
                            "method": method,
                        }
                    else:
                        final = {
                            "primary_label": None,
                            "secondary_labels": [],
                            "taxonomy_fit": provisional.get("taxonomy_fit", "out_of_scope"),
                            "confidence": 0.0,
                            "status": "human_review",
                            "method": "unresolved",
                        }
                        unresolved_clusters.add(cluster_id)
                final_row = {key: item[key] for key in ("sample_id", "source", "split", "source_index", "cluster_id")}
                final_row["final"] = final
                output_stream.write(_json_dump(final_row) + "\n")
                counts[final["status"]] += 1
                counts[final.get("primary_label") or "NONE"] += 1
        _finish_atomic(output_temp, output_stream, output_path)
    except Exception:
        if not output_stream.closed:
            output_stream.close()
        output_temp.unlink(missing_ok=True)
        raise

    if args.clusters:
        review_path = output_path.with_name(output_path.stem + ".human_review.jsonl")
        review_temp, review_stream = _atomic_writer(review_path)
        with Path(args.clusters).open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if item["cluster_id"] in unresolved_clusters:
                    review_stream.write(_json_dump(item) + "\n")
        _finish_atomic(review_temp, review_stream, review_path)
    print(json.dumps({"counts": dict(sorted(counts.items())), "unresolved_clusters": len(unresolved_clusters)}, indent=2))
    return 0


def materialize_command(args: argparse.Namespace) -> int:
    sources = load_registry(Path(args.registry))
    allowed_splits = set(args.splits)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"materialize output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = {label: (output_dir / f"{label}_{TAXONOMY[label]['name']}.jsonl").open("w", encoding="utf-8") for label in TAXONOMY}
    counts = Counter()
    labels_stream = Path(args.labels).open(encoding="utf-8")
    try:
        for source in sources:
            with Path(source.path).open(encoding="utf-8") as data_stream:
                source_index = 0
                for data_line in data_stream:
                    if not data_line.strip():
                        source_index += 1
                        continue
                    try:
                        sample = json.loads(data_line)
                        if not isinstance(sample, dict) or set(sample) != {"input", "timeseries", "output"}:
                            raise ValueError("not exact ChatTS schema")
                    except (json.JSONDecodeError, ValueError):
                        counts["INVALID_SOURCE_ROW"] += 1
                        source_index += 1
                        continue
                    label_line = labels_stream.readline()
                    if not label_line:
                        raise ValueError("final label index ended before source data")
                    label_row = json.loads(label_line)
                    if label_row["source"] != source.name or int(label_row["source_index"]) != source_index:
                        raise ValueError(f"label/source mismatch at {source.name}:{source_index}")
                    final = label_row["final"]
                    primary = final.get("primary_label")
                    if source.split not in allowed_splits:
                        counts["EXCLUDED_SPLIT"] += 1
                    elif (
                        final.get("status") == "accepted"
                        and primary in TAXONOMY
                        and float(final.get("confidence", 0)) >= args.min_confidence
                        and final.get("taxonomy_fit") in set(args.include_fit)
                    ):
                        streams[primary].write(_json_dump(sample) + "\n")
                        counts[primary] += 1
                    else:
                        counts["EXCLUDED"] += 1
                    source_index += 1
        if labels_stream.readline().strip():
            raise ValueError("final label index has more rows than registry sources")
    finally:
        labels_stream.close()
        for stream in streams.values():
            stream.close()
    manifest = {
        "taxonomy_version": TAXONOMY_VERSION,
        "registry": args.registry,
        "labels": args.labels,
        "min_confidence": args.min_confidence,
        "include_fit": args.include_fit,
        "splits": args.splits,
        "counts": dict(sorted(counts.items())),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def taxonomy_command(args: argparse.Namespace) -> int:
    print(json.dumps({"version": TAXONOMY_VERSION, "taxonomy": TAXONOMY}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate ChatTS QA data with the TSRBench 4x15 taxonomy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    taxonomy = subparsers.add_parser("taxonomy", help="print the machine-readable taxonomy")
    taxonomy.set_defaults(func=taxonomy_command)

    prepare = subparsers.add_parser("prepare", help="rule-label all samples and export ambiguous template clusters")
    prepare.add_argument("--registry", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--commit-every", type=int, default=5000)
    prepare.add_argument("--max-template-chars", type=int, default=5000)
    prepare.add_argument("--max-prompt-chars", type=int, default=12000)
    prepare.add_argument("--max-output-chars", type=int, default=3000)
    prepare.set_defaults(func=prepare_command)

    annotate = subparsers.add_parser("annotate-online", help="classify review clusters through an OpenAI-compatible API")
    annotate.add_argument("--input", required=True)
    annotate.add_argument("--output", required=True)
    annotate.add_argument("--base-url", required=True)
    annotate.add_argument("--model", required=True)
    annotate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    annotate.add_argument("--allow-no-key", action="store_true")
    annotate.add_argument("--workers", type=int, default=8)
    annotate.add_argument("--limit", type=int, default=0)
    annotate.add_argument("--timeout", type=float, default=90.0)
    annotate.add_argument("--retries", type=int, default=3)
    annotate.add_argument("--max-tokens", type=int, default=500)
    annotate.add_argument(
        "--json-mode", action="store_true", help="request response_format=json_object from compatible APIs"
    )
    annotate.add_argument(
        "--disable-thinking", action="store_true", help="request non-thinking mode from compatible APIs"
    )
    annotate.set_defaults(func=annotate_online_command)

    export_human = subparsers.add_parser(
        "export-human", help="export a review-cluster JSONL file as an editable CSV"
    )
    export_human.add_argument("--input", required=True)
    export_human.add_argument("--output", required=True)
    export_human.set_defaults(func=export_human_command)

    resolve = subparsers.add_parser("resolve", help="resolve rules, model votes, and optional human overrides")
    resolve.add_argument("--provisional", required=True)
    resolve.add_argument("--votes", nargs="*", default=[])
    resolve.add_argument("--human", default=None)
    resolve.add_argument("--clusters", default=None)
    resolve.add_argument("--output", required=True)
    resolve.set_defaults(func=resolve_command)

    materialize = subparsers.add_parser("materialize", help="write 15 exact-schema ChatTS JSONL buckets")
    materialize.add_argument("--registry", required=True)
    materialize.add_argument("--labels", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--min-confidence", type=float, default=0.85)
    materialize.add_argument("--splits", nargs="+", default=["train"])
    materialize.add_argument(
        "--include-fit", nargs="+", choices=sorted(VALID_FITS), default=["exact", "compatible"]
    )
    materialize.set_defaults(func=materialize_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
