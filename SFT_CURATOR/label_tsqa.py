#!/usr/bin/env python3
"""按 YAML 配置用 DeepSeek 标注时间序列 QA 的质量和难度。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import httpx
import yaml

QUALITY_LEVELS = ("unusable", "weak", "acceptable", "good", "excellent")
DIFFICULTY_LEVELS = ("very_easy", "easy", "moderate", "hard", "very_hard")
PROMPT_VERSION = "deepseek-template-quality-difficulty-v1"

SYSTEM_PROMPT = """Label one time-series QA TEMPLATE for supervised training.

PLACEHOLDERS
- <ts> represents a real input or output time series whose values are intentionally hidden.
- <num> represents an instance-specific scalar number.
- <label> represents an instance-specific categorical answer.
The target model will receive the real series during training. Do not guess hidden values, penalize
their absence, or claim that you verified numeric correctness.

The user payload contains one normalized question template, a coarse answer_structure, and one
representative_answer_template from that group. QUALITY judges whether this training template is
clear, relevant, complete, internally consistent, and correctly formatted:
- unusable: empty, irrelevant, invalid, or fatally contradictory
- weak: a major defect needs substantial repair
- acceptable: usable after a small edit or human check
- good: ready for training with at most harmless wording defects
- excellent: clear, complete, consistent, and ready as-is

DIFFICULTY judges the operations requested by the question, not hidden series length or values:
- very_easy: direct observation or lookup
- easy: one standard operation on one series
- moderate: limited calculation, comparison, ordering, or a conventional composite operation
- hard: dependent operations, exact calculation, forecasting, cross-series alignment, or substantive reasoning
- very_hard: interacting hard demands, long-range multivariate reasoning, or a large constrained search

Return exactly one JSON object. The reason must briefly justify both labels using visible evidence;
do not reveal chain-of-thought or use Markdown:
{"quality":"unusable|weak|acceptable|good|excellent","difficulty":"very_easy|easy|moderate|hard|very_hard","reason":"brief justification for both labels"}"""

_NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?"
_TS_TOKEN_RE = re.compile(r"<ts>\s*(?:<ts\s*/>)?|<ts\s*/>", re.IGNORECASE)
_BRACKETED_SERIES_RE = re.compile(r"\[(?:\s*" + _NUMBER + r"\s*,){3,}\s*" + _NUMBER + r"\s*\]")
_UNBRACKETED_SERIES_RE = re.compile(r"(?<![\w.])" + _NUMBER + r"(?:\s*[,;|]\s*" + _NUMBER + r"){3,}(?![\w.])")
_NESTED_TS_RE = re.compile(r"\[(?:\s*<ts>\s*,?)+\]", re.IGNORECASE)
_SCALAR_RE = re.compile(r"(?<![\w.])" + _NUMBER + r"(?![\w.])")
_ANSWER_LABEL_RE = re.compile(r"^(?:<answer>\s*)?\(?[A-D]\)?(?:\s*</answer>)?$", re.IGNORECASE)
_ANSWER_TAG_LABEL_RE = re.compile(r"<answer>\s*\(?[A-D]\)?\s*</answer>", re.IGNORECASE)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def replace_series(text: str) -> str:
    """把问题或答案中的占位符、长数字数组统一变成 <ts>。"""
    value = _TS_TOKEN_RE.sub("<ts>", str(text))
    for _ in range(3):
        previous = value
        value = _BRACKETED_SERIES_RE.sub("<ts>", value)
        value = _UNBRACKETED_SERIES_RE.sub("<ts>", value)
        value = _NESTED_TS_RE.sub("<ts>", value)
        if value == previous:
            break
    return compact(value)


def extract_qa(record: Mapping[str, Any]) -> Tuple[str, str]:
    question = record.get("input") or record.get("question") or record.get("query")
    answer = record.get("output") or record.get("response") or record.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing question: expected input/question/query")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("missing answer: expected output/response/answer")
    context = record.get("context") or record.get("context_text")
    if isinstance(context, str) and context.strip() and not record.get("input"):
        question = "Context:\n%s\n\nQuestion:\n%s" % (context.strip(), question.strip())
    return question, answer


def answer_structure(answer: str) -> str:
    """只保留答案的训练结构，避免实例措辞阻止同题模板合并。"""
    normalized = replace_series(answer)
    lowered = normalized.lower()
    if "<ts>" in lowered:
        return "tagged_time_series" if "<answer>" in lowered else "time_series"
    if _ANSWER_LABEL_RE.fullmatch(normalized):
        return "choice_label"
    if _ANSWER_TAG_LABEL_RE.search(normalized):
        return "reasoning_with_label" if "<think>" in lowered else "choice_label"
    if "<think>" in lowered and "<answer>" in lowered:
        return "reasoning_with_answer"
    if "<answer>" in lowered:
        return "tagged_answer"
    if _SCALAR_RE.fullmatch(normalized):
        return "scalar"
    return "multiline_text" if "\n" in answer else "text"


def build_template(record: Mapping[str, Any]) -> Tuple[str, Dict[str, str], str]:
    """返回 template_id、发给 DeepSeek 的模板、当前样本的输入哈希。"""
    question, answer = extract_qa(record)
    # 与 DataTaste 的 cluster 规则一致：问题中的实例数值不参与模板 ID。
    question_template = _SCALAR_RE.sub("<num>", replace_series(question))
    representative_answer = replace_series(answer)
    if _ANSWER_LABEL_RE.fullmatch(representative_answer):
        representative_answer = "<label>"
    else:
        representative_answer = _ANSWER_TAG_LABEL_RE.sub(
            "<answer><label></answer>", representative_answer
        )
        representative_answer = _SCALAR_RE.sub("<num>", representative_answer)
    structure = answer_structure(answer)
    payload = {
        "question_template": question_template,
        "timeseries": "<ts>",
        "answer_structure": structure,
        "representative_answer_template": representative_answer,
    }
    template_key = canonical_json(
        {"question_template": question_template, "answer_structure": structure}
    )
    template_id = hashlib.sha256(template_key.encode("utf-8")).hexdigest()[:24]
    input_hash = hashlib.sha256(template_key.encode("utf-8")).hexdigest()
    return template_id, payload, input_hash


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_label(content: str) -> Dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("DeepSeek output is not a JSON object")
    if value.get("quality") not in QUALITY_LEVELS:
        raise ValueError("invalid quality label")
    if value.get("difficulty") not in DIFFICULTY_LEVELS:
        raise ValueError("invalid difficulty label")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("missing reason")
    return {
        "quality": str(value["quality"]),
        "difficulty": str(value["difficulty"]),
        "reason": compact(reason)[:500],
    }


def load_jsonl_index(path: Path, key: str, model: str) -> Dict[Any, Dict[str, Any]]:
    rows: Dict[Any, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if row.get("model") != model or row.get("prompt_version") != PROMPT_VERSION:
                    continue
                if key == "template_id":
                    row_key = str(row[key])
                else:
                    row_key = (str(row["record_id"]), int(row["line_number"]), str(row["input_hash"]))
                rows[row_key] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return rows


async def request_label(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    api_key: str,
    payload: Mapping[str, str],
    settings: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": canonical_json(payload)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": int(settings.get("max_tokens", 256)),
    }
    if bool(settings.get("thinking", True)):
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = str(settings.get("reasoning_effort", "max"))

    retries = int(settings.get("retries", 2))
    error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            response_json = response.json()
            label = parse_label(response_json["choices"][0]["message"]["content"])
            raw_usage = response_json.get("usage", {})
            usage = {
                name: int(raw_usage.get(name, 0) or 0)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            return label, usage
        except httpx.HTTPStatusError as exc:
            error = RuntimeError("HTTP %s: %s" % (exc.response.status_code, exc.response.text[:1000]))
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = exc
        if attempt < retries:
            await asyncio.sleep(min(2**attempt, 4))
    raise RuntimeError("DeepSeek request failed: %s: %s" % (type(error).__name__, error))


def derived_path(output: Path, suffix: str) -> Path:
    stem = output.name[:-6] if output.name.endswith(".jsonl") else output.name
    return output.with_name(stem + suffix + ".jsonl")


def scan_dataset(input_path: Path, done_rows: Mapping[Any, Any], limit: int) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """扫描输入并按问题模板 + 答案结构模板分组；完全忽略 timeseries 字段。"""
    templates: Dict[str, Any] = {}
    stats = {"input_rows": 0, "resume_skipped_rows": 0, "invalid_rows": 0}
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if limit and stats["input_rows"] >= limit:
                break
            stats["input_rows"] += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSONL row is not an object")
                template_id, payload, input_hash = build_template(record)
                record_id = str(record.get("sample_id") or record.get("id") or "line:%d" % line_number)
                row_key = (record_id, line_number, input_hash)
                if row_key in done_rows:
                    stats["resume_skipped_rows"] += 1
                    continue
                group = templates.setdefault(template_id, {"payload": payload, "members": []})
                group["members"].append((record_id, line_number, input_hash))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                stats["invalid_rows"] += 1
                templates.setdefault("__invalid__", {"members": []})["members"].append((line_number, str(exc)))
    return templates, stats


async def process_dataset(
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
    config_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    name = str(dataset.get("name") or Path(str(dataset["input"])).stem)
    input_path = resolve_path(config_dir, dataset["input"])
    output_path = resolve_path(config_dir, dataset["output"])
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_path = derived_path(output_path, ".templates")
    error_path = derived_path(output_path, ".errors")
    model = str(settings["model"])
    done_rows = load_jsonl_index(output_path, "rows", model)
    template_labels = load_jsonl_index(template_path, "template_id", model)
    templates, stats = scan_dataset(input_path, done_rows, int(dataset.get("limit", 0)))
    invalid = templates.pop("__invalid__", {"members": []})["members"]
    pending_rows = sum(len(group["members"]) for group in templates.values())
    missing = [(template_id, group) for template_id, group in templates.items() if template_id not in template_labels]
    result: Dict[str, Any] = {
        "dataset": name,
        **stats,
        "pending_rows": pending_rows,
        "unique_templates": len(templates),
        "cached_templates": len(templates) - len(missing),
        "api_requests_needed": len(missing),
        "template_compression_ratio": round(pending_rows / len(templates), 3) if templates else 0.0,
    }
    if dry_run:
        return result

    with error_path.open("a", encoding="utf-8") as errors:
        for line_number, message in invalid:
            errors.write(canonical_json({"line_number": line_number, "error": message}) + "\n")

    base_url = str(settings["base_url"]).rstrip("/")
    api_key = os.getenv(str(settings.get("api_key_env", "DEEPSEEK_API_KEY")), "")
    concurrency = max(1, int(settings.get("concurrency", 8)))
    timeout = float(settings.get("timeout_seconds", 120))
    url = base_url + "/chat/completions"
    progress_every = max(1, int(settings.get("progress_every_templates", 50)))
    api_requests = failed_templates = labels_written = 0

    async with httpx.AsyncClient(timeout=timeout, trust_env=bool(settings.get("trust_env", False))) as client:
        with template_path.open("a", encoding="utf-8") as template_stream, output_path.open(
            "a", encoding="utf-8"
        ) as output_stream, error_path.open("a", encoding="utf-8") as errors:
            for start in range(0, len(missing), concurrency):
                chunk = missing[start : start + concurrency]
                responses = await asyncio.gather(
                    *[request_label(client, url, model, api_key, group["payload"], settings) for _, group in chunk],
                    return_exceptions=True,
                )
                for (template_id, group), response in zip(chunk, responses):
                    api_requests += 1
                    if isinstance(response, Exception):
                        failed_templates += 1
                        errors.write(
                            canonical_json(
                                {"template_id": template_id, "member_count": len(group["members"]), "error": str(response)}
                            )
                            + "\n"
                        )
                        continue
                    label, usage = response
                    template_row = {
                        "template_id": template_id,
                        **group["payload"],
                        **label,
                        "member_count": len(group["members"]),
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        "usage": usage,
                    }
                    template_labels[template_id] = template_row
                    template_stream.write(canonical_json(template_row) + "\n")
                template_stream.flush()
                errors.flush()
                if api_requests % progress_every == 0 or start + concurrency >= len(missing):
                    print(
                        canonical_json(
                            {"event": "template_progress", "dataset": name, "finished": api_requests, "total": len(missing)}
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

            for template_id, group in templates.items():
                label = template_labels.get(template_id)
                if not label:
                    continue
                for record_id, line_number, input_hash in group["members"]:
                    output_stream.write(
                        canonical_json(
                            {
                                "record_id": record_id,
                                "line_number": line_number,
                                "input_hash": input_hash,
                                "template_id": template_id,
                                "quality": label["quality"],
                                "difficulty": label["difficulty"],
                                "reason": label["reason"],
                                "model": model,
                                "prompt_version": PROMPT_VERSION,
                            }
                        )
                        + "\n"
                    )
                    labels_written += 1
            output_stream.flush()

    result.update(
        {
            "api_requests": api_requests,
            "failed_templates": failed_templates,
            "labels_written": labels_written,
            "output": str(output_path),
            "template_cache": str(template_path),
            "errors": str(error_path),
        }
    )
    return result


def resolve_path(config_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("deepseek"), dict):
        raise ValueError("config must contain a deepseek mapping")
    if not isinstance(value.get("datasets"), list) or not value["datasets"]:
        raise ValueError("config must contain a non-empty datasets list")
    for key in ("base_url", "model"):
        if not value["deepseek"].get(key):
            raise ValueError("deepseek.%s is required" % key)
    for index, dataset in enumerate(value["datasets"]):
        if not isinstance(dataset, dict) or not dataset.get("input") or not dataset.get("output"):
            raise ValueError("datasets[%d] requires input and output" % index)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 DeepSeek 按模板标注时间序列 QA")
    parser.add_argument("--config", type=Path, default=Path("label_config.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="只统计模板数量和预计请求数")
    parser.add_argument("--dataset", action="append", help="只运行指定数据集；可重复传入")
    parser.add_argument("--limit", type=int, help="临时覆盖 YAML 中的单数据集行数上限")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        datasets = config["datasets"]
        if args.dataset:
            selected = set(args.dataset)
            datasets = [dataset for dataset in datasets if str(dataset.get("name")) in selected]
            missing = selected - {str(dataset.get("name")) for dataset in datasets}
            if missing:
                raise ValueError("unknown dataset: %s" % ", ".join(sorted(missing)))
        if args.limit is not None:
            if args.limit < 0:
                raise ValueError("--limit cannot be negative")
            datasets = [{**dataset, "limit": args.limit} for dataset in datasets]
        results = [
            asyncio.run(process_dataset(dataset, config["deepseek"], config_path.parent, args.dry_run))
            for dataset in datasets
        ]
    except Exception as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
