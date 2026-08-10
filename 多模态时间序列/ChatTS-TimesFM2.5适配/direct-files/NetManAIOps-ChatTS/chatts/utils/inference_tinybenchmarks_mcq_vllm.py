"""Evaluate local tinyBenchmarks MCQ data with ChatTS' existing vLLM backend.

This module intentionally has no lm-evaluation-harness or tinyBenchmarks package
dependency.  It reads already-downloaded JSON/JSONL/Parquet data, scores every
choice with vLLM prompt log-probabilities, and writes raw tiny-set metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Keep this at module scope. vLLM uses spawn; every worker must execute the
# ChatTS model/processor registration before it creates Qwen2/3TSForCausalLM.
import chatts.vllm.chatts_vllm as _chatts_vllm  # noqa: F401,E402


TASKS = (
    "tinyArc",
    "tinyHellaswag",
    "tinyMMLU",
    "tinyTruthfulQA",
    "tinyWinogrande",
)

TASK_ALIASES = {
    "tinyarc": "tinyArc",
    "tinyai2arc": "tinyArc",
    "arc": "tinyArc",
    "tinyhellaswag": "tinyHellaswag",
    "hellaswag": "tinyHellaswag",
    "tinymmlu": "tinyMMLU",
    "mmlu": "tinyMMLU",
    "tinytruthfulqa": "tinyTruthfulQA",
    "truthfulqa": "tinyTruthfulQA",
    "tinywinogrande": "tinyWinogrande",
    "winogrande": "tinyWinogrande",
}

TASK_DISCOVERY = {
    "tinyArc": {
        "tokens": ("tinyai2arc", "tinyarc", "arcchallenge"),
        "split": "test",
        "required": ("question", "choices", "answerKey"),
    },
    "tinyHellaswag": {
        "tokens": ("tinyhellaswag", "hellaswag"),
        "split": "validation",
        "required": ("endings", "label"),
    },
    "tinyMMLU": {
        "tokens": ("tinymmlu", "mmlu"),
        "split": "test",
        "required": ("question", "choices", "answer"),
    },
    "tinyTruthfulQA": {
        "tokens": ("tinytruthfulqa", "truthfulqa"),
        "split": "validation",
        "required": ("question", "mc2_targets"),
    },
    "tinyWinogrande": {
        "tokens": ("tinywinogrande", "winogrande"),
        "split": "validation",
        "required": ("sentence", "option1", "option2", "answer"),
    },
}

TRUTHFULQA_CONTEXT = (
    "Q: What is human life expectancy in the United States?\n"
    "A: Human life expectancy in the United States is 78 years.\n\n"
    "Q: Who was president of the United States in 1955?\n"
    "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
    "Q: Which party did he belong to?\n"
    "A: He belonged to the Republican Party.\n\n"
    "Q: What is the square root of banana?\n"
    "A: I have no comment.\n\n"
    "Q: How does a telescope work?\n"
    "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
    "Q: Where were the 1992 Olympics held?\n"
    "A: The 1992 Olympics were held in Barcelona, Spain."
)


@dataclass
class PreparedSample:
    doc_id: str
    context: str
    choices: list[str]
    gold_indices: list[int]
    metric: str


@dataclass
class EncodedChoice:
    doc_index: int
    choice_index: int
    token_ids: list[int]
    choice_start: int


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_task(value: str) -> str:
    key = _normalize(value)
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    if value in TASKS:
        return value
    raise ValueError(f"Unsupported task {value!r}; choose from {', '.join(TASKS)}")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            rows = []
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            return rows
        payload = json.load(stream)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Expected a JSON list (or data/rows/examples list) in {path}")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "This local dataset is Parquet, but pandas is absent. ChatTS normally "
            "already includes pandas; alternatively export only the 100-row eval "
            "split to JSONL on the machine where the dataset was downloaded."
        ) from exc
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot read {path}: the current environment has no Parquet engine. "
            "Do not upgrade ChatTS; export this eval split to JSONL elsewhere, then "
            "pass it with --task-file TASK=/path/file.jsonl."
        ) from exc
    return [_plain(row) for row in frame.to_dict(orient="records")]


def _read_saved_dataset(path: Path, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            f"{path} is a datasets.save_to_disk directory, but the optional "
            "'datasets' package is not already installed. No package was changed. "
            "Export its eval split to JSONL and use --task-file."
        ) from exc
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Saved dataset {path} has no {split!r} split: {list(dataset)}")
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Unsupported saved dataset object in {path}")
    return [_plain(row) for row in dataset]


def read_rows(path: Path, split: str) -> list[dict[str, Any]]:
    if path.is_dir():
        return _read_saved_dataset(path, split)
    suffix = path.suffix.lower()
    if suffix in (".json", ".jsonl"):
        return [_plain(row) for row in _read_json(path)]
    if suffix == ".parquet":
        return _read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def _candidate_score(task: str, path: Path) -> tuple[int, int, str]:
    spec = TASK_DISCOVERY[task]
    normalized = _normalize(str(path))
    name = _normalize(path.name)
    score = 0
    score += 100 if any(token in normalized for token in spec["tokens"]) else 0
    score += 80 if spec["split"] in name else 0
    score += 30 if spec["split"] in normalized else 0
    score -= 100 if "train" in name and spec["split"] != "train" else 0
    score -= 40 if "validation" in name and spec["split"] == "test" else 0
    return (-score, len(str(path)), str(path))


def _discover_candidates(root: Path, task: str) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".parquet")
    ]
    saved_dirs = {
        path.parent
        for marker in ("dataset_dict.json", "state.json")
        for path in root.rglob(marker)
    }
    candidates: list[Path] = []
    tokens = TASK_DISCOVERY[task]["tokens"]
    for path in [*files, *saved_dirs]:
        normalized = _normalize(str(path))
        if any(token in normalized for token in tokens):
            candidates.append(path)
    if not candidates and len(TASKS) == 1:
        candidates = [*files, *saved_dirs]
    return sorted(set(candidates), key=lambda path: _candidate_score(task, path))


def _validate_rows(task: str, rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows in {path}")
    missing = [key for key in TASK_DISCOVERY[task]["required"] if key not in rows[0]]
    if missing:
        raise ValueError(f"{path} does not look like {task}; missing {missing}")


def load_task_rows(
    root: Path,
    task: str,
    explicit_path: Path | None,
    allow_size_mismatch: bool,
) -> tuple[list[dict[str, Any]], Path]:
    split = TASK_DISCOVERY[task]["split"]
    candidates = [explicit_path] if explicit_path else _discover_candidates(root, task)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find local data for {task} under {root}. Use "
            f"--task-file {task}=/absolute/path/to/{split}.parquet (or JSONL)."
        )
    failures = []
    for path in candidates:
        assert path is not None
        path = path.expanduser().resolve()
        try:
            rows = read_rows(path, split)
            _validate_rows(task, rows, path)
            if len(rows) != 100 and not allow_size_mismatch:
                raise ValueError(
                    f"found {len(rows)} rows, expected the 100-row tiny eval split"
                )
            return rows, path
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            if explicit_path:
                break
    preview = "\n  ".join(failures[:6])
    raise RuntimeError(f"No valid local {task} eval split found. Tried:\n  {preview}")


def _as_list(value: Any) -> list[Any]:
    value = _plain(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise ValueError(f"Expected a list, got {type(value).__name__}")


def _hellaswag_preprocess(text: str) -> str:
    text = str(text).strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _choice_context(context: str, choice: str) -> tuple[str, str]:
    context = str(context)
    choice = str(choice).strip()
    trailing = context[len(context.rstrip()):]
    if trailing:
        context = context.rstrip()
        choice = trailing + choice
    elif choice and not choice[0].isspace():
        choice = " " + choice
    return context, choice


def prepare_sample(task: str, row: dict[str, Any], index: int) -> PreparedSample:
    formatted = str(row.get("input_formatted") or "").strip()
    doc_id = str(row.get("id", index))

    if task == "tinyArc":
        choices_value = _plain(row["choices"])
        choices = [str(item) for item in _as_list(choices_value["text"])]
        labels = [str(item) for item in _as_list(choices_value["label"])]
        answer = str(row["answerKey"])
        if answer not in labels:
            raise ValueError(f"answerKey {answer!r} is absent from labels {labels}")
        context = formatted or f"Question: {row['question']}\nAnswer:"
        return PreparedSample(doc_id, context, choices, [labels.index(answer)], "acc_norm")

    if task == "tinyHellaswag":
        choices = [_hellaswag_preprocess(item) for item in _as_list(row["endings"])]
        context = formatted
        if not context:
            ctx = str(row.get("ctx_a", "")) + " " + str(row.get("ctx_b", "")).capitalize()
            context = _hellaswag_preprocess(f"{row.get('activity_label', '')}: {ctx}")
        return PreparedSample(doc_id, context, choices, [int(row["label"])], "acc_norm")

    if task == "tinyMMLU":
        raw_choices = [str(item) for item in _as_list(row["choices"])]
        letters = [chr(ord("A") + offset) for offset in range(len(raw_choices))]
        if formatted:
            context = formatted
        else:
            rendered = " ".join(f"{letter}. {choice}" for letter, choice in zip(letters, raw_choices))
            context = f"Question: {row['question']}\n{rendered}\nAnswer:"
        return PreparedSample(doc_id, context, letters, [int(row["answer"])], "acc_norm")

    if task == "tinyTruthfulQA":
        targets = _plain(row["mc2_targets"])
        choices = [str(item) for item in _as_list(targets["choices"])]
        labels = [int(item) for item in _as_list(targets["labels"])]
        gold = [offset for offset, label in enumerate(labels) if label == 1]
        if not gold:
            raise ValueError("mc2_targets has no correct answer")
        context = formatted or f"{TRUTHFULQA_CONTEXT}\n\nQ: {row['question']}\nA:"
        return PreparedSample(doc_id, context, choices, gold, "mc2")

    if task == "tinyWinogrande":
        sentence = str(row["sentence"])
        if "_" not in sentence:
            raise ValueError("Winogrande sentence has no '_' placeholder")
        prefix, suffix = sentence.split("_", 1)
        options = [str(row["option1"]), str(row["option2"])]
        gold = int(row["answer"]) - 1
        context = prefix
        if formatted:
            # The official input_formatted leaks the gold option at its end.
            # Retain its five demonstrations but cut the current item back to
            # the blank prefix before scoring either option plus the suffix.
            position = formatted.rfind(prefix.rstrip())
            if position >= 0:
                context = formatted[:position] + prefix
        choices = [option + suffix for option in options]
        return PreparedSample(doc_id, context, choices, [gold], "acc_norm")

    raise AssertionError(task)


def _encode_choice(tokenizer: Any, sample: PreparedSample, doc_index: int, choice_index: int) -> EncodedChoice:
    context, continuation = _choice_context(sample.context, sample.choices[choice_index])
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not context_ids:
        raise ValueError(f"Document {sample.doc_id} has an empty tokenized context")
    if full_ids[: len(context_ids)] != context_ids:
        mismatch = next(
            (offset for offset, pair in enumerate(zip(context_ids, full_ids)) if pair[0] != pair[1]),
            min(len(context_ids), len(full_ids)),
        )
        raise ValueError(
            f"Tokenizer boundary changed at token {mismatch} for document {sample.doc_id}, "
            f"choice {choice_index}; cannot calculate a trustworthy conditional score"
        )
    if len(full_ids) <= len(context_ids):
        raise ValueError(f"Document {sample.doc_id}, choice {choice_index} has no continuation tokens")
    return EncodedChoice(doc_index, choice_index, full_ids, len(context_ids))


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        value = value.logprob
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite prompt log-probability: {result}")
    return result


def _score_output(output: Any, encoded: EncodedChoice) -> tuple[float, float, int]:
    token_ids = list(output.prompt_token_ids or [])
    prompt_logprobs = output.prompt_logprobs
    if token_ids != encoded.token_ids:
        raise RuntimeError("vLLM returned prompt_token_ids different from the submitted token IDs")
    if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
        raise RuntimeError("vLLM did not return aligned prompt_logprobs")
    values = []
    for position in range(encoded.choice_start, len(token_ids)):
        entry = prompt_logprobs[position]
        token_id = token_ids[position]
        if entry is None or token_id not in entry:
            raise RuntimeError(f"Missing prompt logprob for position={position}, token_id={token_id}")
        values.append(_logprob_value(entry[token_id]))
    total = math.fsum(values)
    return total, total / len(values), len(values)


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def protocol_hash(task: str, samples: list[PreparedSample]) -> str:
    payload = [
        {
            "doc_id": sample.doc_id,
            "context": sample.context,
            "choices": sample.choices,
            "gold_indices": sample.gold_indices,
            "metric": sample.metric,
        }
        for sample in samples
    ]
    serialized = json.dumps(
        {"task": task, "samples": payload}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _is_chatts_config(config: Any) -> bool:
    architectures = getattr(config, "architectures", None) or []
    return any("TSForCausalLM" in str(name) for name in architectures)


def evaluate_task(
    llm: Any,
    tokenizer: Any,
    task: str,
    rows: list[dict[str, Any]],
    source: Path,
    output_dir: Path,
    request_chunk_size: int,
    max_model_len: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    samples = [prepare_sample(task, row, index) for index, row in enumerate(rows)]
    encoded = [
        _encode_choice(tokenizer, sample, doc_index, choice_index)
        for doc_index, sample in enumerate(samples)
        for choice_index in range(len(sample.choices))
    ]
    longest = max(len(item.token_ids) for item in encoded)
    if longest + 1 > max_model_len:
        raise ValueError(
            f"{task} needs {longest + 1} tokens including the one-token generation, "
            f"but --max-model-len={max_model_len}. Raise CHATTS_VLLM_MAX_MODEL_LEN."
        )

    scores: dict[tuple[int, int], tuple[float, float, int]] = {}
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
        seed=seed,
    )
    for start in range(0, len(encoded), request_chunk_size):
        batch = encoded[start : start + request_chunk_size]
        prompts = [{"prompt_token_ids": item.token_ids} for item in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM returned a different number of outputs than prompts")
        for item, output in zip(batch, outputs):
            scores[(item.doc_index, item.choice_index)] = _score_output(output, item)
        print(f"[tinyBenchmarks] {task}: scored {min(start + len(batch), len(encoded))}/{len(encoded)} choices")

    sample_rows = []
    metric_values = []
    for doc_index, sample in enumerate(samples):
        raw_scores = [scores[(doc_index, choice_index)][0] for choice_index in range(len(sample.choices))]
        normalized_scores = [scores[(doc_index, choice_index)][1] for choice_index in range(len(sample.choices))]
        token_counts = [scores[(doc_index, choice_index)][2] for choice_index in range(len(sample.choices))]
        if sample.metric == "mc2":
            denominator = _logsumexp(raw_scores)
            probabilities = [math.exp(value - denominator) for value in raw_scores]
            value = math.fsum(probabilities[index] for index in sample.gold_indices)
            predicted = max(range(len(raw_scores)), key=raw_scores.__getitem__)
        else:
            probabilities = None
            predicted = max(range(len(normalized_scores)), key=normalized_scores.__getitem__)
            value = float(predicted in sample.gold_indices)
        metric_values.append(value)
        sample_rows.append(
            {
                "doc_index": doc_index,
                "doc_id": sample.doc_id,
                "metric": sample.metric,
                "score": value,
                "predicted_index": predicted,
                "gold_indices": sample.gold_indices,
                "raw_loglikelihoods": raw_scores,
                "normalized_loglikelihoods": normalized_scores,
                "continuation_token_counts": token_counts,
                "mc2_probabilities": probabilities,
            }
        )

    task_hash = protocol_hash(task, samples)
    result = {
        "task": task,
        "metric": "mc2_probability_mass" if samples[0].metric == "mc2" else "accuracy_norm",
        "score": math.fsum(metric_values) / len(metric_values),
        "num_samples": len(samples),
        "num_choices_scored": len(encoded),
        "dataset_source": str(source),
        "protocol_hash": task_hash,
        "max_prompt_tokens": longest,
    }
    _write_jsonl(output_dir / f"samples_{task}.jsonl", sample_rows)
    _write_json(output_dir / f"metrics_{task}.json", result)
    print(f"[tinyBenchmarks] {task}: {result['metric']}={result['score']:.6f}, n={len(samples)}")
    return result


def parse_task_files(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --task-file {value!r}; expected TASK=PATH")
        task_value, path_value = value.split("=", 1)
        task = canonical_task(task_value)
        if task in result:
            raise ValueError(f"Duplicate --task-file for {task}")
        result[task] = Path(path_value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--task-file", action="append", default=[], metavar="TASK=PATH")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--request-chunk-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--allow-size-mismatch", action="store_true")
    parser.add_argument("--inspect-data-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_gpus < 1 or args.request_chunk_size < 1 or args.max_model_len < 2:
        raise SystemExit("GPU count, request chunk size, and max model length must be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    task_names = [canonical_task(item.strip()) for item in args.tasks.split(",") if item.strip()]
    if not task_names:
        raise SystemExit("No tasks selected")
    if len(task_names) != len(set(task_names)):
        raise SystemExit("Duplicate tasks are not allowed")

    root = Path(args.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Local dataset root does not exist: {root}")
    explicit = parse_task_files(args.task_file)
    datasets: dict[str, tuple[list[dict[str, Any]], Path]] = {}
    for task in task_names:
        rows, source = load_task_rows(
            root,
            task,
            explicit.get(task),
            allow_size_mismatch=args.allow_size_mismatch,
        )
        if args.max_samples > 0:
            rows = rows[: args.max_samples]
        datasets[task] = (rows, source)
        print(f"[tinyBenchmarks] {task}: local_source={source}, rows={len(rows)}")
        # Validate every row before allocating GPU memory.
        for index, row in enumerate(rows):
            prepare_sample(task, row, index)

    if args.inspect_data_only:
        print("[tinyBenchmarks] Local dataset inspection passed; no model was loaded.")
        return 0

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise SystemExit(f"Local model directory/config.json not found: {model_path}")

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    llm_kwargs = {
        "model": str(model_path),
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.num_gpus,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "enable_prefix_caching": False,
        "seed": args.seed,
    }
    if _is_chatts_config(config):
        llm_kwargs["limit_mm_per_prompt"] = {"timeseries": 50}
    print(
        f"[tinyBenchmarks] loading model={model_path}, chatts={_is_chatts_config(config)}, "
        f"tensor_parallel_size={args.num_gpus}, max_model_len={args.max_model_len}"
    )
    llm = LLM(**llm_kwargs)

    output_dir = Path(args.output_dir).expanduser().resolve()
    task_results = {}
    for task in task_names:
        rows, source = datasets[task]
        task_results[task] = evaluate_task(
            llm,
            tokenizer,
            task,
            rows,
            source,
            output_dir,
            args.request_chunk_size,
            args.max_model_len,
            args.seed,
        )

    macro = math.fsum(item["score"] for item in task_results.values()) / len(task_results)
    summary = {
        "evaluator": "ChatTS vLLM prompt_logprobs",
        "evaluator_version": 1,
        "model_name": args.model_name or model_path.name,
        "model_path": str(model_path),
        "tasks": task_results,
        "macro_score": macro,
        "num_tasks": len(task_results),
        "seed": args.seed,
        "note": (
            "Raw 100-item tiny-set metrics only; no GPIRT/IRT++ extrapolation is applied. "
            "tinyTruthfulQA is MC2 probability mass; other tasks are length-normalized accuracy."
        ),
    }
    _write_json(output_dir / "metrics.json", summary)
    print(f"[tinyBenchmarks] macro_score={macro:.6f}; saved={output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
