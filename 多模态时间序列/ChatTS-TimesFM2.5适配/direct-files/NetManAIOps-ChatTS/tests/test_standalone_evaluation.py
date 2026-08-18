from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOADER = REPO_ROOT / "scripts" / "load_studio_evaluation_config.py"
HOST_RUNNER = REPO_ROOT / "scripts" / "run_eval_only.sh"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def frozen_payload(
    *,
    job_id: str = "eval-job-1",
    project_root: str = "/workspace/ChatTS/ChatTS-main",
    model_path: str = "/share/models/candidate",
    output_root: str = "/share/evaluation/candidate/protocol-aaaaaaaaaaaaaaaa",
    include_slurm: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pipeline": {
            "task_type": "standalone_evaluation",
            "seed": 42,
            "force_eval": False,
            "preflight_only": False,
            "max_samples": 0,
            "offline": True,
            "trial_id": job_id,
            "batch_id": "batch-1",
        },
        "containers": {"evaluation": "ragas"},
        "evaluation": {
            "project_root": project_root,
            "script": f"{project_root}/scripts/run_all_chatts_benchmarks.sh",
            "model_path": model_path,
            "model_name": "candidate-0123456789ab",
            "output_root": output_root,
            "chronos2_model_path": "/workspace/chronos2",
            "tsrbench_root": "/share/TSRBench-dataset",
            "tinybench_dataset_root": "/share/tyb",
            "ts_haystack_root": "/workspace/TS-Haystack",
            "timeseriesexam_root": "/workspace/TimeSeriesExam",
            "timeseriesexam_data_file": "/workspace/TimeSeriesExam/output/qa.json",
            "benchmarks": "tsrbench,timeseriesexam",
            "run_id": "candidate-protocol-aaaaaaaaaaaaaaaa-eval",
            "protocol_hash": "a" * 64,
            "haystack_split": "test",
            "tiny_data_partition": "all",
            "tiny_partition_seed": 42,
            "tsr_prompt_mode": "answer_only",
            "tsr_max_model_len": 12288,
            "tsr_max_new_tokens": 8,
            "tsr_batch_size": 16,
            "tsr_request_chunk_size": 128,
            "tiny_max_model_len": 6000,
            "tiny_request_chunk_size": 16,
            "tiny_gpu_memory_utilization": 0.7,
            "haystack_max_model_len": 40960,
            "haystack_max_new_tokens": 500,
            "haystack_batch_size": 1,
            "haystack_request_chunk_size": 8,
            "exam_max_model_len": 8192,
            "exam_max_new_tokens": 1024,
            "exam_batch_size": 8,
            "exam_request_chunk_size": 64,
        },
    }
    if include_slurm:
        payload["slurm"] = {
            "evaluation_host_root": "/host/ChatTS",
            "evaluation_sif_image": "/host/images/ragas.sif",
        }
    payload["pipeline"]["trial_config_hash"] = canonical_hash(payload)
    return payload


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load(path: Path, job_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LOADER), str(path), "--expected-job-id", job_id],
        check=False,
        capture_output=True,
        text=True,
    )


def assignments(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


def test_loader_emits_frozen_standalone_contract(tmp_path: Path) -> None:
    config = tmp_path / "resolved.yaml"
    payload = frozen_payload()
    write_payload(config, payload)

    completed = load(config, "eval-job-1")

    assert completed.returncode == 0, completed.stderr
    values = assignments(completed.stdout)
    assert values["TASK_TYPE"] == "standalone_evaluation"
    assert values["EVAL_CONTAINER"] == "ragas"
    assert values["EVAL_MODEL_PATH"] == "/share/models/candidate"
    assert values["EVAL_OUTPUT_ROOT"].endswith("/protocol-aaaaaaaaaaaaaaaa")
    assert values["RUN_ID"].endswith("-protocol-aaaaaaaaaaaaaaaa-eval")
    assert values["TRIAL_CONFIG_HASH"] == payload["pipeline"]["trial_config_hash"]
    assert values["CHATTS_EVAL_SIF_IMAGE"] == "/host/images/ragas.sif"


def test_loader_accepts_docker_config_without_slurm_block(tmp_path: Path) -> None:
    config = tmp_path / "docker.json"
    write_payload(config, frozen_payload(include_slurm=False))

    completed = load(config, "eval-job-1")

    assert completed.returncode == 0, completed.stderr
    values = assignments(completed.stdout)
    assert "CHATTS_EVALUATION_DIR" not in values
    assert values["EVAL_CONTAINER"] == "ragas"


def test_dependency_free_loader_accepts_safe_dump_empty_slurm() -> None:
    spec = importlib.util.spec_from_file_location("standalone_eval_loader", LOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # PyYAML safe_dump emits an empty mapping exactly in this form for Docker
    # jobs.  The no-PyYAML path must preserve its mapping type.
    assert module.fallback_yaml_load("slurm: {}\n") == {"slurm": {}}


def test_loader_rejects_tampering_and_wrong_job(tmp_path: Path) -> None:
    config = tmp_path / "tampered.json"
    payload = frozen_payload()
    payload["evaluation"]["model_path"] = "/share/models/other"
    write_payload(config, payload)

    tampered = load(config, "eval-job-1")
    wrong_job = load(config, "eval-job-2")

    assert tampered.returncode != 0
    assert "trial_config_hash does not match" in tampered.stderr
    assert wrong_job.returncode != 0
    assert "does not match submitted Studio job" in wrong_job.stderr


def test_docker_runner_uses_eval_container_without_training_marker(tmp_path: Path) -> None:
    project = tmp_path / "ChatTS"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    evaluator = scripts / "run_all_chatts_benchmarks.sh"
    evaluator.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
[[ "$REQUIRE_TRAINING_MARKER" == "0" ]]
[[ "$REQUIRE_MODEL_ON_PREFLIGHT" == "1" ]]
[[ -z "$DATA_VERSION" && -z "$DATASET_SNAPSHOT_HASH" ]]
[[ "$MODEL_COMPLETION_MARKER" == "TRAINING_COMPLETE.json" ]]
[[ "$EVAL_GPUS" == "0,1,2,3,4,5,6,7" ]]
[[ "$EVAL_NUM_GPUS" == "8" && "$TS_GPUS_PER_PROCESS" == "2" ]]
[[ "$METRICS_FILE" == "$OUTPUT_ROOT/metrics.json" ]]
[[ "$TSR_PROMPT_MODE" == "json_reasoning" ]]
[[ "$TSR_MAX_MODEL_LEN" == "12288" ]]
[[ "$TSR_MAX_NEW_TOKENS" == "256" ]]
[[ "$TSR_BATCH_SIZE" == "1" ]]
[[ "$TSR_REQUEST_CHUNK_SIZE" == "128" ]]
mkdir -p "$OUTPUT_ROOT"
printf 'suite\\tstatus\\nall\\tPASS\\n' > "$OUTPUT_ROOT/benchmark_status.tsv"
printf '# summary\\n' > "$OUTPUT_ROOT/all_benchmarks_summary.md"
printf '{"status":"pass"}\\n' > "$OUTPUT_ROOT/metrics.json"
""",
        encoding="utf-8",
    )
    evaluator.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    protocol = "a" * 64
    output = tmp_path / "evaluation" / "candidate" / f"protocol-{protocol[:16]}"
    config = tmp_path / "docker-eval.json"
    payload = frozen_payload(
        project_root=str(project),
        model_path=str(model),
        output_root=str(output),
        include_slurm=False,
    )
    payload["evaluation"]["script"] = str(evaluator)
    payload["evaluation"]["tsr_prompt_mode"] = "json_reasoning"
    payload["evaluation"]["tsr_max_new_tokens"] = 256
    payload["evaluation"]["tsr_batch_size"] = 1
    payload["pipeline"]["trial_config_hash"] = canonical_hash(
        {
            **payload,
            "pipeline": {
                key: value
                for key, value in payload["pipeline"].items()
                if key != "trial_config_hash"
            },
        }
    )
    write_payload(config, payload)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        f"""#!{sys.executable}
import os
import subprocess
import sys

args = sys.argv[1:]
if args[:1] == ["inspect"]:
    print("true")
    raise SystemExit(0)
if args[:1] != ["exec"]:
    raise SystemExit("unsupported docker call: " + repr(args))
cursor = 1
environment = os.environ.copy()
working_directory = None
while cursor < len(args):
    if args[cursor] == "--workdir":
        working_directory = args[cursor + 1]
        cursor += 2
    elif args[cursor] == "-e":
        key, value = args[cursor + 1].split("=", 1)
        environment[key] = value
        cursor += 2
    else:
        break
cursor += 1  # container
command = args[cursor:]
raise SystemExit(subprocess.run(command, env=environment, cwd=working_directory).returncode)
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    flock = fake_bin / "flock"
    flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    flock.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}",
            "CONFIG_FILE": str(config),
            "HOST_PYTHON_BIN": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(HOST_RUNNER)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "metrics.json").is_file()
    assert not (model / "TRAINING_COMPLETE.json").exists()
    assert Path(f"{output}.chatts-evaluation.lock").is_file()
