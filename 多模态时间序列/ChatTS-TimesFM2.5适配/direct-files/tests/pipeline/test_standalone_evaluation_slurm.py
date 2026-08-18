from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


TRAINING_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_CHATTS_ROOT = TRAINING_ROOT / "NetManAIOps-ChatTS"
CHATTS_ROOT = (
    PACKAGED_CHATTS_ROOT
    if PACKAGED_CHATTS_ROOT.is_dir()
    else TRAINING_ROOT.parent / "ChatTS"
)
LOADER = CHATTS_ROOT / "scripts" / "load_studio_evaluation_config.py"
SBATCH = TRAINING_ROOT / "slurm" / "run_chatts_studio_evaluation.sbatch"


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


def payload(
    *,
    job_id: str,
    evaluation_root: Path,
    shared_root: Path,
    chronos_root: Path,
    preflight: bool,
) -> dict[str, Any]:
    protocol = "e" * 64
    value: dict[str, Any] = {
        "pipeline": {
            "task_type": "standalone_evaluation",
            "seed": 42,
            "force_eval": False,
            "preflight_only": preflight,
            "max_samples": 0,
            "offline": True,
            "trial_id": job_id,
            "batch_id": "batch-eval",
        },
        "containers": {"evaluation": "ragas"},
        "evaluation": {
            "project_root": str(evaluation_root),
            "script": str(evaluation_root / "scripts" / "run_all_chatts_benchmarks.sh"),
            "model_path": str(shared_root / "models" / "external"),
            "model_name": "external-abcdef012345",
            "output_root": str(
                shared_root
                / "evaluation"
                / "external-abcdef012345"
                / f"protocol-{protocol[:16]}"
            ),
            "chronos2_model_path": str(chronos_root),
            "tsrbench_root": str(shared_root / "TSRBench"),
            "tinybench_dataset_root": str(shared_root / "tiny"),
            "ts_haystack_root": str(shared_root / "haystack"),
            "timeseriesexam_root": str(shared_root / "exam"),
            "timeseriesexam_data_file": str(shared_root / "exam" / "qa.json"),
            "benchmarks": "tsrbench",
            "run_id": f"external-{job_id}-protocol-{protocol[:16]}-eval",
            "protocol_hash": protocol,
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
        "slurm": {
            "evaluation_host_root": str(evaluation_root),
            "chronos2_host_root": str(chronos_root),
            "tsrbench_host_root": str(shared_root / "TSRBench"),
        },
    }
    value["pipeline"]["trial_config_hash"] = canonical_hash(value)
    return value


def test_standalone_launcher_has_distinct_contract_and_no_trainer() -> None:
    text = SBATCH.read_text(encoding="utf-8")

    assert "# CHATTS_STUDIO_SBATCH_API=1" in text
    assert "# CHATTS_STUDIO_EVALUATION_SBATCH_API=1" in text
    assert "run_chronos2_best_two_stage.sh" not in text
    assert '$(dirname "$CHATTS_SIF_IMAGE")/ragas.sif' in text
    assert "REQUIRE_TRAINING_MARKER=0" in text
    assert "REQUIRE_MODEL_ON_PREFLIGHT=1" in text


@pytest.mark.parametrize("preflight", [True, False])
def test_standalone_launcher_runs_ragas_only_and_locks_real_output(
    tmp_path: Path, preflight: bool
) -> None:
    assert LOADER.is_file(), "ChatTS standalone loader must be delivered with Training"
    evaluation_root = tmp_path / "ChatTS"
    scripts = evaluation_root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(LOADER, scripts / LOADER.name)
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
test -s "$MODEL_PATH/config.json"
test -s "$MODEL_PATH/model.safetensors"
if [[ "$PREFLIGHT_ONLY" != "1" ]]; then
  mkdir -p "$OUTPUT_ROOT"
  printf 'ok\\n' > "$OUTPUT_ROOT/benchmark_status.tsv"
  printf 'ok\\n' > "$OUTPUT_ROOT/all_benchmarks_summary.md"
  printf '{}\\n' > "$OUTPUT_ROOT/metrics.json"
fi
""",
        encoding="utf-8",
    )
    evaluator.chmod(0o755)

    shared_root = tmp_path / "share"
    model = shared_root / "models" / "external"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (shared_root / "TSRBench").mkdir()
    chronos = tmp_path / "chronos2"
    chronos.mkdir()

    image_root = tmp_path / "images"
    image_root.mkdir()
    training_sif = image_root / "chatts_v1.sif"
    ragas_sif = image_root / "ragas.sif"
    training_sif.write_bytes(b"training image is not executed")
    ragas_sif.write_bytes(b"evaluation image")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    singularity_log = tmp_path / "singularity.json"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text("#!/usr/bin/env bash\nexec \"$@\"\n", encoding="utf-8")
    fake_srun.chmod(0o755)
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_flock.chmod(0o755)
    fake_singularity = fake_bin / "singularity"
    fake_singularity.write_text(
        f"""#!{sys.executable}
import json
import os
import subprocess
import sys

args = sys.argv[1:]
if args[:1] != ["run"]:
    raise SystemExit("expected singularity run")
cursor = 1
environment = os.environ.copy()
working_directory = None
while cursor < len(args):
    token = args[cursor]
    if token in {{"--nv", "--cleanenv"}}:
        cursor += 1
    elif token == "--env":
        key, value = args[cursor + 1].split("=", 1)
        environment[key] = value
        cursor += 2
    elif token in {{"--bind", "--home"}}:
        cursor += 2
    elif token == "--pwd":
        working_directory = args[cursor + 1]
        cursor += 2
    else:
        break
image = args[cursor]
command = args[cursor + 1:]
with open({str(singularity_log)!r}, "w", encoding="utf-8") as stream:
    json.dump({{"image": image, "command": command}}, stream)
raise SystemExit(subprocess.run(command, env=environment, cwd=working_directory).returncode)
""",
        encoding="utf-8",
    )
    fake_singularity.chmod(0o755)

    job_id = "eval-preflight" if preflight else "eval-real"
    frozen = tmp_path / f"{job_id}.json"
    resolved = payload(
        job_id=job_id,
        evaluation_root=evaluation_root,
        shared_root=shared_root,
        chronos_root=chronos,
        preflight=preflight,
    )
    frozen.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}",
            "SLURM_JOB_ID": "31415" if preflight else "31416",
            "SLURM_SUBMIT_DIR": str(TRAINING_ROOT),
            "CHATTS_EVALUATION_DIR": str(evaluation_root),
            "CHATTS_SIF_IMAGE": str(training_sif),
            "CHATTS_SHARED_HOST_PATH": str(shared_root),
            "CHATTS_SHARED_CONTAINER_PATH": str(shared_root),
            "CHATTS_HOST_PYTHON_BIN": sys.executable,
            "CHATTS_SRUN_BIN": str(fake_srun),
            "CHATTS_SINGULARITY_BIN": str(fake_singularity),
            "CHATTS_FLOCK_BIN": str(fake_flock),
            "CHATTS_JOB_TMP_ROOT": str(tmp_path / "job-tmp"),
        }
    )

    completed = subprocess.run(
        ["bash", str(SBATCH), str(frozen), job_id],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    invocation = json.loads(singularity_log.read_text(encoding="utf-8"))
    assert invocation["image"] == str(ragas_sif)
    assert invocation["image"] != str(training_sif)
    output = Path(resolved["evaluation"]["output_root"])
    lock = Path(f"{output}.chatts-evaluation.lock")
    if preflight:
        assert not output.exists()
        assert not lock.exists()
    else:
        assert (output / "metrics.json").is_file()
        assert lock.is_file()
    assert not (model / "TRAINING_COMPLETE.json").exists()
