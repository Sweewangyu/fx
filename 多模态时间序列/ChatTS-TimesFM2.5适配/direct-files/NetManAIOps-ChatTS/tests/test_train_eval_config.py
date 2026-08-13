from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_LOADER = REPO_ROOT / "scripts" / "load_train_eval_config.py"
HOST_RUNNER = REPO_ROOT / "scripts" / "run_train_then_eval.sh"

ADVANCED_EVAL_DEFAULTS = {
    "TSR_PROMPT_MODE": "answer_only",
    "TSR_MAX_MODEL_LEN": "12288",
    "TSR_MAX_NEW_TOKENS": "8",
    "TSR_BATCH_SIZE": "16",
    "TSR_REQUEST_CHUNK_SIZE": "128",
    "TINY_MAX_MODEL_LEN": "6000",
    "TINY_REQUEST_CHUNK_SIZE": "16",
    "TINY_GPU_MEMORY_UTILIZATION": "0.70",
    "HAYSTACK_MAX_MODEL_LEN": "40960",
    "HAYSTACK_MAX_NEW_TOKENS": "500",
    "HAYSTACK_BATCH_SIZE": "1",
    "HAYSTACK_REQUEST_CHUNK_SIZE": "8",
    "EXAM_MAX_MODEL_LEN": "8192",
    "EXAM_MAX_NEW_TOKENS": "1024",
    "EXAM_BATCH_SIZE": "8",
    "EXAM_REQUEST_CHUNK_SIZE": "64",
}


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in {
        *ADVANCED_EVAL_DEFAULTS,
        "DATA_VERSION",
        "DATASET_SNAPSHOT_HASH",
        "TRAINING_RECIPE_HASH",
        "TRIAL_ID",
        "TRIAL_CONFIG_HASH",
        "CONFIG_FILE",
        "SHARED_ROOT",
        "HOST_PYTHON_BIN",
        "CAPTURE_PATH",
        "CAPTURE_PYTHON",
    }:
        environment.pop(name, None)
    return environment


def _loader_assignments(config: Path) -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, str(CONFIG_LOADER), str(config)],
        env=_clean_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in completed.stdout.splitlines() if line)


def test_loader_maps_dataset_identity_and_safe_advanced_evaluation(tmp_path: Path) -> None:
    snapshot_hash = "a" * 64
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        textwrap.dedent(
            f"""
            pipeline:
              data_version: datav4
              dataset_snapshot_hash: {snapshot_hash}
              training_recipe_hash: {"c" * 64}
              trial_id: studio-job-7
              trial_config_hash: {"b" * 64}
            evaluation:
              tsr_prompt_mode: official
              tsr_max_model_len: 13000
              tsr_max_new_tokens: 512
              tsr_batch_size: 2
              tsr_request_chunk_size: 64
              tiny_max_model_len: 7000
              tiny_request_chunk_size: 8
              tiny_gpu_memory_utilization: 0.65
              haystack_max_model_len: 42000
              haystack_max_new_tokens: 600
              haystack_batch_size: 2
              haystack_request_chunk_size: 4
              exam_max_model_len: 9000
              exam_max_new_tokens: 1200
              exam_batch_size: 4
              exam_request_chunk_size: 32
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assignments = _loader_assignments(config)

    assert assignments == {
        "DATASET_SNAPSHOT_HASH": snapshot_hash,
        "DATA_VERSION": "datav4",
        "TRAINING_RECIPE_HASH": "c" * 64,
        "TRIAL_CONFIG_HASH": "b" * 64,
        "TRIAL_ID": "studio-job-7",
        "EXAM_BATCH_SIZE": "4",
        "EXAM_MAX_MODEL_LEN": "9000",
        "EXAM_MAX_NEW_TOKENS": "1200",
        "EXAM_REQUEST_CHUNK_SIZE": "32",
        "HAYSTACK_BATCH_SIZE": "2",
        "HAYSTACK_MAX_MODEL_LEN": "42000",
        "HAYSTACK_MAX_NEW_TOKENS": "600",
        "HAYSTACK_REQUEST_CHUNK_SIZE": "4",
        "TINY_GPU_MEMORY_UTILIZATION": "0.65",
        "TINY_MAX_MODEL_LEN": "7000",
        "TINY_REQUEST_CHUNK_SIZE": "8",
        "TSR_BATCH_SIZE": "2",
        "TSR_MAX_MODEL_LEN": "13000",
        "TSR_MAX_NEW_TOKENS": "512",
        "TSR_PROMPT_MODE": "official",
        "TSR_REQUEST_CHUNK_SIZE": "64",
    }


def test_loader_keeps_old_minimal_config_free_of_new_assignments(tmp_path: Path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text("pipeline:\n  seed: 42\nevaluation:\n  benchmarks: tsrbench\n", encoding="utf-8")

    assignments = _loader_assignments(config)

    assert assignments == {"BENCHMARKS": "tsrbench", "SEED": "42"}


def _write_capture_script(path: Path, stage: str) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -Eeuo pipefail
            "$CAPTURE_PYTHON" - "$CAPTURE_PATH" <<'PY'
            import json
            import os
            import sys

            keys = ["DATA_VERSION", "DATASET_SNAPSHOT_HASH", "TRAINING_RECIPE_HASH"]
            if {stage!r} == "training":
                keys.extend(["TRIAL_ID", "TRIAL_CONFIG_HASH"])
            else:
                keys.extend({list(ADVANCED_EVAL_DEFAULTS)!r})
            payload = {{"stage": {stage!r}, **{{key: os.environ.get(key) for key in keys}}}}
            with open(sys.argv[1], "a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\\n")
            PY
            """
        ),
        encoding="utf-8",
    )
    if stage == "training":
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                'mkdir -p "$FINAL_MODEL_PATH"\n'
                'printf \'%s\\n\' \'{}\' > "$FINAL_MODEL_PATH/config.json"\n'
                'printf \'%s\\n\' \'{"status":"complete"}\' > "$FINAL_MODEL_PATH/TRAINING_COMPLETE.json"\n'
            )
    else:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                'mkdir -p "$OUTPUT_ROOT"\n'
                'printf \'suite\\tstatus\\nall\\tPASS\\n\' > "$OUTPUT_ROOT/benchmark_status.tsv"\n'
                'printf \'# summary\\n\' > "$OUTPUT_ROOT/all_benchmarks_summary.md"\n'
                'printf \'%s\\n\' \'{"status":"pass"}\' > "$OUTPUT_ROOT/metrics.json"\n'
            )
    path.chmod(0o755)


def _run_host_pipeline(tmp_path: Path, pipeline_fields: str, evaluation_fields: str) -> list[dict[str, str | None]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        f"""#!{sys.executable}
import os
import subprocess
import sys

arguments = sys.argv[1:]
if arguments[:1] == ["inspect"]:
    print("true")
    raise SystemExit(0)
if arguments[:1] != ["exec"]:
    raise SystemExit("unsupported docker invocation: " + repr(arguments))
cursor = 1
environment = os.environ.copy()
while cursor < len(arguments) and arguments[cursor] == "-e":
    key, value = arguments[cursor + 1].split("=", 1)
    environment[key] = value
    cursor += 2
cursor += 1  # container name
command = arguments[cursor:]
if command[:2] == ["python", "-c"] and "torch.cuda.device_count" in command[2]:
    print("8")
    raise SystemExit(0)
raise SystemExit(subprocess.run(command, env=environment, check=False).returncode)
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    shared = tmp_path / "shared"
    dataset = shared / "dataset"
    base_model = shared / "base-model"
    chronos = shared / "chronos2"
    train_project = tmp_path / "training-project"
    eval_project = tmp_path / "evaluation-project"
    for directory in (dataset, base_model, chronos, train_project, eval_project):
        directory.mkdir(parents=True)
    train_script = train_project / "train.sh"
    eval_script = eval_project / "eval.sh"
    _write_capture_script(train_script, "training")
    _write_capture_script(eval_script, "evaluation")

    final_model = shared / "models" / "candidate"
    train_output = shared / "training"
    eval_output = shared / "evaluation"
    config = tmp_path / "pipeline.yaml"
    config_lines = [
        "pipeline:",
        "  seed: 42",
        "  force_train: false",
        "  force_eval: false",
        "  preflight_only: false",
        "  max_samples: 0",
        "  offline: true",
    ]
    config_lines.extend(
        f"  {line}" for line in textwrap.dedent(pipeline_fields).strip().splitlines()
    )
    config_lines.extend(
        [
            "containers:",
            "  training: training",
            "  evaluation: evaluation",
            "training:",
            f"  project_root: {train_project}",
            f"  script: {train_script}",
            f"  base_model_path: {base_model}",
            f"  output_root: {train_output}",
            f"  final_model_path: {final_model}",
            f"  chronos2_model_path: {chronos}",
            f"  dataset_dir: {dataset}",
            "evaluation:",
            f"  project_root: {eval_project}",
            f"  script: {eval_script}",
            f"  output_root: {eval_output}",
            f"  chronos2_model_path: {chronos}",
            f"  tsrbench_root: {shared}",
            f"  tinybench_dataset_root: {shared}",
            f"  ts_haystack_root: {shared}",
            f"  timeseriesexam_root: {shared}",
            f"  timeseriesexam_data_file: {shared / 'exam.json'}",
            "  benchmarks: tsrbench",
            "  run_id: test-run",
        ]
    )
    config_lines.extend(
        f"  {line}" for line in textwrap.dedent(evaluation_fields).strip().splitlines()
    )
    config.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    (shared / "exam.json").write_text("[]\n", encoding="utf-8")
    capture = tmp_path / "capture.jsonl"
    environment = _clean_environment()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "HOST_PYTHON_BIN": sys.executable,
            "CONFIG_FILE": str(config),
            "SHARED_ROOT": str(shared),
            "CAPTURE_PATH": str(capture),
            "CAPTURE_PYTHON": sys.executable,
        }
    )
    completed = subprocess.run(
        ["bash", str(HOST_RUNNER)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    return [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]


def test_host_forwards_dataset_identity_and_advanced_evaluation(tmp_path: Path) -> None:
    snapshot_hash = "b" * 64
    recipe_hash = "d" * 64
    events = _run_host_pipeline(
        tmp_path,
        f"data_version: datav4\ndataset_snapshot_hash: {snapshot_hash}\n"
        f"training_recipe_hash: {recipe_hash}\n"
        f"trial_id: studio-job-7\ntrial_config_hash: {'c' * 64}",
        """
        tsr_prompt_mode: official
        tsr_max_model_len: 13000
        tsr_max_new_tokens: 512
        tsr_batch_size: 2
        tsr_request_chunk_size: 64
        tiny_max_model_len: 7000
        tiny_request_chunk_size: 8
        tiny_gpu_memory_utilization: 0.65
        haystack_max_model_len: 42000
        haystack_max_new_tokens: 600
        haystack_batch_size: 2
        haystack_request_chunk_size: 4
        exam_max_model_len: 9000
        exam_max_new_tokens: 1200
        exam_batch_size: 4
        exam_request_chunk_size: 32
        """,
    )

    assert events[0] == {
        "stage": "training",
        "DATA_VERSION": "datav4",
        "DATASET_SNAPSHOT_HASH": snapshot_hash,
        "TRAINING_RECIPE_HASH": recipe_hash,
        "TRIAL_ID": "studio-job-7",
        "TRIAL_CONFIG_HASH": "c" * 64,
    }
    assert events[1] == {
        "stage": "evaluation",
        "DATA_VERSION": "datav4",
        "DATASET_SNAPSHOT_HASH": snapshot_hash,
        "TRAINING_RECIPE_HASH": None,
        "TSR_PROMPT_MODE": "official",
        "TSR_MAX_MODEL_LEN": "13000",
        "TSR_MAX_NEW_TOKENS": "512",
        "TSR_BATCH_SIZE": "2",
        "TSR_REQUEST_CHUNK_SIZE": "64",
        "TINY_MAX_MODEL_LEN": "7000",
        "TINY_REQUEST_CHUNK_SIZE": "8",
        "TINY_GPU_MEMORY_UTILIZATION": "0.65",
        "HAYSTACK_MAX_MODEL_LEN": "42000",
        "HAYSTACK_MAX_NEW_TOKENS": "600",
        "HAYSTACK_BATCH_SIZE": "2",
        "HAYSTACK_REQUEST_CHUNK_SIZE": "4",
        "EXAM_MAX_MODEL_LEN": "9000",
        "EXAM_MAX_NEW_TOKENS": "1200",
        "EXAM_BATCH_SIZE": "4",
        "EXAM_REQUEST_CHUNK_SIZE": "32",
    }


def test_host_legacy_config_retains_advanced_evaluation_defaults(tmp_path: Path) -> None:
    events = _run_host_pipeline(tmp_path, "", "")

    assert events[0]["DATA_VERSION"] == ""
    assert events[0]["DATASET_SNAPSHOT_HASH"] == ""
    assert events[0]["TRAINING_RECIPE_HASH"] == ""
    assert events[0]["TRIAL_ID"] == ""
    assert events[0]["TRIAL_CONFIG_HASH"] == ""
    assert events[1] == {
        "stage": "evaluation",
        "DATA_VERSION": "",
        "DATASET_SNAPSHOT_HASH": "",
        "TRAINING_RECIPE_HASH": None,
        **ADVANCED_EVAL_DEFAULTS,
    }
