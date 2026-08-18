from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_all_chatts_benchmarks.sh"
ARTIFACT_HELPER = REPO_ROOT / "scripts" / "chatts_benchmark_artifacts.py"


def _write(path: Path, text: str = "# fixture\n", *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _fake_project(root: Path) -> Path:
    project = root / "ChatTS"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(RUNNER, project / "scripts" / RUNNER.name)
    shutil.copyfile(ARTIFACT_HELPER, project / "scripts" / ARTIFACT_HELPER.name)

    capture_helper = project / "scripts" / "capture_suite.py"
    _write(
        capture_helper,
        """\
import json
import os
import sys
from pathlib import Path

suite, output_root, model_name = sys.argv[1:4]
keys = {
    "tsrbench": [
        "TS_ENCODER_TYPE", "CHATTS_TS_ENCODER_TYPE", "TEMPERATURE",
        "MAX_RETRIES", "MAX_INPUT_TOKENS", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ],
    "tinybenchmarks": [
        "CHATTS_TS_ENCODER_TYPE", "SUMMARY_ONLY", "DTYPE",
        "ALLOW_SIZE_MISMATCH", "FORGETTING_THRESHOLD_PP",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    ],
    "ts_haystack": [
        "TS_ENCODER_TYPE", "CHATTS_TS_ENCODER_TYPE", "TEMPERATURE",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    ],
    "timeseriesexam": [
        "TS_ENCODER_TYPE", "CHATTS_TS_ENCODER_TYPE", "TEMPERATURE",
        "MAX_CONCEPTS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    ],
}[suite]
with open(os.environ["CAPTURE_PATH"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"suite": suite, **{key: os.environ.get(key) for key in keys}}, sort_keys=True) + "\\n")

root = Path(output_root)
if suite == "tsrbench":
    summary = root / f"tsrbench_summary_{model_name}.json"
    payload = {"overall": {"accuracy_strict": 1.0}}
elif suite == "tinybenchmarks":
    summary = root / model_name / "metrics.json"
    payload = {"tasks": {}, "macro_score": 1.0, "num_tasks": 0}
elif suite == "ts_haystack":
    summary = root / f"ts_haystack_summary_{model_name}.json"
    payload = {"overall": {"mean_iou": 1.0}}
else:
    summary = root / f"{model_name}_query_hint_concepts_examples" / f"timeseriesexam_summary_{model_name}.json"
    payload = {"overall": {"official_strict_accuracy": 1.0}}
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text(json.dumps(payload) + "\\n", encoding="utf-8")
""",
    )

    suite_scripts = {
        "run_chatts_tsrbench.sh": "tsrbench",
        "run_chatts_tinybenchmarks_mcq.sh": "tinybenchmarks",
        "run_chatts_ts_haystack.sh": "ts_haystack",
        "run_chatts_timeseriesexam.sh": "timeseriesexam",
    }
    for name, suite in suite_scripts.items():
        _write(
            project / "scripts" / name,
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            f'"$PYTHON_BIN" "$PROJECT_ROOT/scripts/capture_suite.py" {suite} '
            '"$OUTPUT_ROOT" "$MODEL_NAME"\n',
            executable=True,
        )

    for name in (
        "inspect_chatts_ts_encoder_checkpoints.py",
        "evaluate_tsrbench.py",
        "summarize_tinybenchmarks_mcq.py",
        "evaluate_ts_haystack.py",
        "evaluate_timeseriesexam.py",
    ):
        _write(project / "scripts" / name)
    for name in (
        "llm_utils.py",
        "inference_tsrbench_vllm.py",
        "inference_tinybenchmarks_mcq_vllm.py",
        "inference_ts_haystack_vllm.py",
        "inference_timeseriesexam_vllm.py",
    ):
        _write(project / "chatts" / "utils" / name)
    _write(project / "chatts" / "vllm" / "chatts_vllm.py")
    return project


def _run_fixture(root: Path, *, prompt_mode: str = "answer_only") -> tuple[dict[str, dict[str, str | None]], Path]:
    project = _fake_project(root)
    model = root / "model"
    chronos = root / "chronos2"
    tsrbench = root / "tsrbench"
    tiny = root / "tiny"
    haystack = root / "haystack"
    exam = root / "exam"
    output = root / "output"
    capture = root / "capture.jsonl"
    for directory in (
        model,
        chronos,
        tsrbench,
        tiny,
        haystack / "src" / "datasets",
        haystack / "data",
        exam / "evaluate",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _write(model / "config.json", "{}\n")
    _write(model / "TRAINING_COMPLETE.json", "{}\n")
    (model / "model.safetensors").write_bytes(b"weights")
    _write(chronos / "config.json", "{}\n")
    _write(tsrbench / "perception.jsonl", "{}\n")
    _write(tiny / "fixture.json", "[]\n")
    _write(haystack / "src" / "datasets" / "registry.py")
    _write(haystack / "data" / "fixture.json", "[]\n")
    _write(exam / "evaluate" / "concepts.py")
    exam_data = exam / "qa_dataset.json"
    _write(exam_data, "[]\n")

    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(project),
            "MODEL_PATH": str(model),
            "MODEL_NAME": "candidate",
            "CHRONOS2_MODEL_PATH": str(chronos),
            "TSRBENCH_ROOT": str(tsrbench),
            "TSRBENCH_DATASET_ROOT": str(tsrbench),
            "TINYBENCH_DATASET_ROOT": str(tiny),
            "TS_HAYSTACK_ROOT": str(haystack),
            "TIMESERIESEXAM_ROOT": str(exam),
            "TIMESERIESEXAM_DATA_FILE": str(exam_data),
            "OUTPUT_ROOT": str(output),
            "BENCHMARKS": "tsrbench,tinybenchmarks,ts_haystack,timeseriesexam",
            "TSR_PROMPT_MODE": prompt_mode,
            "FORCE_EVAL": "1",
            "OFFLINE": "0",
            "AVAILABLE_GPUS_OVERRIDE": "8",
            "PYTHON_BIN": sys.executable,
            "CAPTURE_PATH": str(capture),
            "CHATTS_FINGERPRINT_CACHE": "0",
            # Simulate a dirty, long-lived container environment. None of
            # these undeclared values may alter the frozen top-level protocol.
            "TS_ENCODER_TYPE": "timesfm2_5",
            "CHATTS_TS_ENCODER_TYPE": "zeus",
            "TEMPERATURE": "9.9",
            "MAX_RETRIES": "99",
            "MAX_INPUT_TOKENS": "9999",
            "TSR_TEMPERATURE": "8.8",
            "TSR_MAX_RETRIES": "88",
            "TSR_MAX_INPUT_TOKENS": "8888",
            "SUMMARY_ONLY": "1",
            "DTYPE": "float16",
            "ALLOW_SIZE_MISMATCH": "1",
            "FORGETTING_THRESHOLD_PP": "99",
            "TINY_DTYPE": "float32",
            "TINY_FORGETTING_THRESHOLD_PP": "88",
            "MAX_CONCEPTS": "77",
            "EXAM_MAX_CONCEPTS": "66",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    events = {
        event["suite"]: event
        for event in (
            json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()
        )
    }
    return events, output


def test_top_level_runner_closes_polluted_child_environment(tmp_path: Path) -> None:
    events, output = _run_fixture(tmp_path)

    assert events["tsrbench"] == {
        "suite": "tsrbench",
        "TS_ENCODER_TYPE": "chronos2",
        "CHATTS_TS_ENCODER_TYPE": "chronos2",
        "TEMPERATURE": "0.0",
        "MAX_RETRIES": "0",
        "MAX_INPUT_TOKENS": "0",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
    }
    assert events["tinybenchmarks"]["CHATTS_TS_ENCODER_TYPE"] == "chronos2"
    assert events["tinybenchmarks"]["SUMMARY_ONLY"] == "0"
    assert events["tinybenchmarks"]["DTYPE"] == "auto"
    assert events["tinybenchmarks"]["ALLOW_SIZE_MISMATCH"] == "0"
    assert events["tinybenchmarks"]["FORGETTING_THRESHOLD_PP"] == "5.0"
    assert events["ts_haystack"]["TS_ENCODER_TYPE"] == "chronos2"
    assert events["timeseriesexam"]["TS_ENCODER_TYPE"] == "chronos2"
    assert events["timeseriesexam"]["MAX_CONCEPTS"] == "3"
    for event in events.values():
        assert event["HF_HUB_OFFLINE"] == "0"
        assert event["TRANSFORMERS_OFFLINE"] == "0"

    tiny_manifest = json.loads(
        (output / "tinybenchmarks" / ".chatts_benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "summary_only=0" in tiny_manifest["protocol_items"]
    assert "dtype=auto" in tiny_manifest["protocol_items"]
    assert "allow_size_mismatch=0" in tiny_manifest["protocol_items"]
    assert "forgetting_threshold_pp=5.0" in tiny_manifest["protocol_items"]


def test_tsr_declared_prompt_changes_fixed_protocol_fingerprint(tmp_path: Path) -> None:
    answer_events, answer_output = _run_fixture(tmp_path / "answer")
    official_events, official_output = _run_fixture(
        tmp_path / "official", prompt_mode="official"
    )

    answer_manifest = json.loads(
        (answer_output / "tsrbench" / ".chatts_benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    official_manifest = json.loads(
        (official_output / "tsrbench" / ".chatts_benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert answer_manifest["protocol_fingerprint"] != official_manifest[
        "protocol_fingerprint"
    ]
    assert answer_events["tsrbench"]["TEMPERATURE"] == "0.0"
    assert answer_events["tsrbench"]["MAX_RETRIES"] == "0"
    assert official_events["tsrbench"]["TEMPERATURE"] == "1.0"
    assert official_events["tsrbench"]["MAX_RETRIES"] == "10"
    assert official_events["tsrbench"]["MAX_INPUT_TOKENS"] == "8000"


def test_standalone_preflight_requires_selected_model_immediately(tmp_path: Path) -> None:
    project = _fake_project(tmp_path)
    chronos = tmp_path / "chronos2"
    tsrbench = tmp_path / "tsrbench"
    chronos.mkdir()
    tsrbench.mkdir()
    _write(tsrbench / "perception.jsonl", "{}\n")
    missing_model = tmp_path / "missing-model"
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(project),
            "MODEL_PATH": str(missing_model),
            "MODEL_NAME": "external-candidate",
            "CHRONOS2_MODEL_PATH": str(chronos),
            "TSRBENCH_ROOT": str(tsrbench),
            "TSRBENCH_DATASET_ROOT": str(tsrbench),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "BENCHMARKS": "tsrbench",
            "PREFLIGHT_ONLY": "1",
            "REQUIRE_TRAINING_MARKER": "0",
            "REQUIRE_MODEL_ON_PREFLIGHT": "1",
            "AVAILABLE_GPUS_OVERRIDE": "8",
            "PYTHON_BIN": sys.executable,
        }
    )

    strict = subprocess.run(
        ["bash", str(RUNNER)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    environment["REQUIRE_MODEL_ON_PREFLIGHT"] = "0"
    deferred = subprocess.run(
        ["bash", str(RUNNER)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert strict.returncode != 0
    assert "Final model config not found" in strict.stderr
    assert deferred.returncode == 0, deferred.stderr
    assert not (tmp_path / "output").exists()
