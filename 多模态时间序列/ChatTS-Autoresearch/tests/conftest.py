from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

TRAIN_SCRIPT = r"""#!/usr/bin/env bash
set -Eeuo pipefail
mode="${PIPELINE_MODE:-full}"
mkdir -p "$(dirname "$STAGE1_OUT")" "$(dirname "$FINAL_MODEL_PATH")"
if [[ "$mode" == "full" || "$mode" == "stage1" ]]; then
  mkdir -p "$STAGE1_OUT"
  printf '{"model_type":"mock"}\n' > "$STAGE1_OUT/config.json"
  printf 'mock-stage1-weights\n' > "$STAGE1_OUT/pytorch_model.bin"
  printf '{"selected_checkpoint":"mock-s1","best_metric":0.4,"exported_model_dir":"%s"}\n' "$STAGE1_OUT" > "$STAGE1_OUT/best_model_manifest.json"
fi
if [[ "$mode" == "full" || "$mode" == "stage2" ]]; then
  test -f "${STAGE2_FROM:-$STAGE1_OUT}/config.json"
  mkdir -p "$FINAL_MODEL_PATH"
  printf '{"model_type":"mock"}\n' > "$FINAL_MODEL_PATH/config.json"
  printf 'mock-weights-%s\n' "$TRIAL_ID" > "$FINAL_MODEL_PATH/pytorch_model.bin"
  printf '{"selected_checkpoint":"mock-s2","best_metric":0.3,"exported_model_dir":"%s"}\n' "$FINAL_MODEL_PATH" > "$FINAL_MODEL_PATH/best_model_manifest.json"
  printf '{"status":"complete","stage2_best_eval_loss":0.3,"trial":"%s"}\n' "$TRIAL_ID" > "$FINAL_MODEL_PATH/TRAINING_COMPLETE.json"
fi
"""


EVAL_SCRIPT = r"""#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p "$OUTPUT_ROOT/tsrbench" "$OUTPUT_ROOT/timeseriesexam" "$OUTPUT_ROOT/ts_haystack" "$OUTPUT_ROOT/tinybenchmarks"
MODEL_NAME="$MODEL_NAME" OUTPUT_ROOT="$OUTPUT_ROOT" python3 - <<'PY'
import json, os
from pathlib import Path
name=os.environ['MODEL_NAME']
if name.startswith('full-'):
    score=0.66
elif name.startswith('proxy-01'):
    score=0.63
elif name.startswith('proxy-02'):
    score=0.61
elif name.startswith('final-champion'):
    score=0.67
else:
    score=0.60
root=Path(os.environ['OUTPUT_ROOT'])
payload={"suites":{
 "tsrbench":{"strict_accuracy":score,"parsed_accuracy":score+0.01,"coverage":0.99},
 "timeseriesexam":{"strict_accuracy":score-0.02,"flexible_accuracy":score,"coverage":0.98},
 "ts_haystack":{"mean_iou":0.75},
 "tinybenchmarks":{"average_accuracy":0.70,"tasks":{"tinyArc":{"accuracy":0.70},"tinyMMLU":{"accuracy":0.69}}}
}}
(root/'metrics.json').write_text(json.dumps(payload))
rows=[
 {"question":"good","prediction":"A","gold":"A","correct":True,"task":"shape"},
 {"question":"bad","prediction":"B","gold":"A","correct":False,"task":"trend","difficulty":"hard"},
]
with (root/'tsrbench'/'predictions.jsonl').open('w') as f:
    for row in rows: f.write(json.dumps(row)+'\n')
PY
"""


@pytest.fixture()
def project(tmp_path: Path) -> dict[str, Path]:
    train = tmp_path / "train-project"
    evaluate = tmp_path / "eval-project"
    base = tmp_path / "base-model"
    chronos = tmp_path / "chronos2"
    datav2 = tmp_path / "datav2"
    tsrbench = tmp_path / "benchmarks" / "tsrbench"
    timeseriesexam = tmp_path / "benchmarks" / "timeseriesexam"
    ts_haystack = tmp_path / "benchmarks" / "ts-haystack"
    tinybench = tmp_path / "benchmarks" / "tinybench"
    for directory in (
        train,
        evaluate,
        base,
        chronos,
        datav2 / "files",
        tsrbench / "task",
        timeseriesexam,
        ts_haystack / "src" / "datasets",
        ts_haystack / "data",
        tinybench,
    ):
        directory.mkdir(parents=True)
    (base / "config.json").write_text("{}\n")
    (base / "pytorch_model.bin").write_text("mock-base-weights\n")
    (chronos / "model.safetensors").write_text("mock-chronos2-weights\n")
    train_script = train / "train.sh"
    eval_script = evaluate / "eval.sh"
    train_script.write_text(TRAIN_SCRIPT)
    eval_script.write_text(EVAL_SCRIPT)
    train_script.chmod(0o755)
    eval_script.chmod(0o755)
    rows_a = [
        {"input": "describe", "timeseries": [[1, 2, 3]], "output": "rising"},
        {"input": "find spike", "timeseries": [[0, 5, 0]], "output": "at 1"},
    ]
    rows_b = [
        rows_a[0],
        {"input": "compare", "timeseries": [[3, 2, 1]], "output": "falling"},
    ]
    for name, rows in (("chatts_align_256", rows_a), ("chatts_sft", rows_b)):
        with (datav2 / "files" / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    registry = {
        "sources": [
            {
                "name": "chatts_align_256",
                "path": "files/chatts_align_256.jsonl",
                "split": "train",
                "family": "chatts",
                "training_role": "stage1_alignment",
            },
            {
                "name": "chatts_sft",
                "path": "files/chatts_sft.jsonl",
                "split": "train",
                "family": "chatts",
                "training_role": "sft",
            },
        ]
    }
    (datav2 / "sources.json").write_text(json.dumps(registry))
    (datav2 / "manifest.json").write_text(
        json.dumps({"name": "fixture", "content_sha256": "fixture-content"})
    )
    (tsrbench / "task" / "perception.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": f"tsr-{index}",
                    "category": f"category-{index % 2}",
                    "difficulty": "medium",
                    "answer": "A",
                }
            )
            + "\n"
            for index in range(20)
        )
    )
    exam_file = timeseriesexam / "qa_dataset.json"
    exam_file.write_text(
        json.dumps(
            [
                {
                    "id": f"exam-{index}",
                    "category": f"category-{index % 2}",
                    "difficulty": "medium",
                }
                for index in range(20)
            ]
        )
    )
    (timeseriesexam / "concepts.py").write_text("CONCEPTS = {}\n")
    (ts_haystack / "src" / "datasets" / "registry.py").write_text("# mock registry\n")
    config = {
        "schema_version": "chatts-autoresearch-v1",
        "paths": {
            "train_project": str(train),
            "train_script": str(train_script),
            "eval_project": str(evaluate),
            "eval_script": str(eval_script),
            "datav2_root": str(datav2),
            "datav2_registry": "sources.json",
            "datav2_manifest": "manifest.json",
            "chronos2_model": str(chronos),
            "base_model": str(base),
            "tsrbench_root": str(tsrbench),
            "timeseriesexam_root": str(timeseriesexam),
            "timeseriesexam_data_file": str(exam_file),
            "ts_haystack_root": str(ts_haystack),
            "tinybench_root": str(tinybench),
        },
        "runtime": {
            "output_root": str(tmp_path / "artifacts"),
            "seed": 42,
            "gpu_ids": "0,1,2,3,4,5,6,7",
        },
        "deepseek": {
            "enabled": True,
            "base_url": "http://localhost:30000/v1",
            "model": "mock-model",
            "api_key_env": "MOCK_API_KEY",
            "concurrency": 2,
            "timeout_seconds": 2,
            "max_retries": 0,
            "prompt_version": "test-v1",
        },
        "labeling": {"max_samples": 1, "sources": ["chatts_align_256"]},
        "data": {
            "snapshot_name": "filtered",
            "baseline_snapshot": "raw",
            "minimum_quality": 0.0,
            "missing_label_policy": "keep",
            "drop_exact_duplicates": True,
            "drop_cross_source_duplicates": True,
            "drop_near_duplicates": False,
            "near_duplicate_hamming": 3,
            "source_weights": {},
            "difficulty_weights": {"easy": 1.0, "medium": 1.0, "hard": 1.0},
            "aliases": {"chatts_align_256": "align_256", "chatts_sft": "sft"},
        },
        "training": {
            "stage1_learning_rate": 1e-5,
            "stage2_learning_rate": 1e-5,
            "stage1_timeseries_learning_rate": 1e-5,
            "stage2_timeseries_learning_rate": 1e-5,
            "stage1_datasets": "align_256",
            "stage2_datasets": "sft",
            "stage1_mix_strategy": "concat",
            "stage2_mix_strategy": "concat",
            "stage1_interleave_probs": "",
            "stage2_interleave_probs": "",
            "stage1_epochs": 1,
            "stage2_epochs": 1,
            "stage2_warmup_ratio": 0.02,
            "stage2_scheduler": "cosine",
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 32,
            "cutoff_len": 2048,
            "val_size": 0.05,
        },
        "search": {
            "proxy_trials": 2,
            "proxy_max_steps": 3,
            "full_finalists": 1,
            "proposal_mode": "deterministic",
            "learning_rates": [5e-6, 1e-5, 2e-5],
            "projector_lr_ratios": [0.5, 1.0, 2.0],
            "warmup_ratios": [0.01, 0.02, 0.05],
            "schedulers": ["cosine", "linear"],
            "epochs": [1, 2],
            "source_weight_range": [0.5, 2.0],
        },
        "evaluation": {
            "search_benchmarks": "tsrbench,timeseriesexam",
            "guard_benchmarks": "ts_haystack,tinybenchmarks",
            "final_benchmarks": "tsrbench,timeseriesexam,ts_haystack,tinybenchmarks",
            "search_split": "search-dev",
            "final_split": "final-test",
            "max_samples": 0,
            "split_sources": [],
        },
        "gates": {
            "tiny_expected_tasks": ["tinyArc", "tinyMMLU"],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return {
        "config": config_path,
        "artifacts": tmp_path / "artifacts",
        "datav2": datav2,
    }
