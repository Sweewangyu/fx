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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOADER = PROJECT_ROOT / "scripts" / "slurm" / "load_studio_pipeline_config.py"
SBATCH = PROJECT_ROOT / "slurm" / "run_chatts_studio_pipeline.sbatch"
LEGACY_SBATCH = PROJECT_ROOT / "slurm" / "run_chronos2_all_data_one_stage.sbatch"


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


def stage(*, datasets: str) -> dict[str, Any]:
    return {
        "learning_rate": "1e-5",
        "timeseries_learning_rate": "2e-5",
        "datasets": datasets,
        "mix_strategy": "concat",
        "interleave_probs": "",
        "num_train_epochs": 1,
        "max_steps": 0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.02,
        "logging_steps": 1,
        "save_steps": 100,
        "eval_steps": 100,
        "val_size": 0.05,
        "per_device_eval_batch_size": 2,
        "cutoff_len": 2048,
        "preprocessing_num_workers": 32,
    }


def frozen_payload(*, mode: str = "full", job_id: str = "a" * 32) -> dict[str, Any]:
    recipe_hash = "b" * 64
    output_root = f"/share/output/ChatTS/experiments/recipe-{recipe_hash[:16]}"
    payload: dict[str, Any] = {
        "pipeline": {
            "seed": 42,
            "data_version": "datav3",
            "dataset_snapshot_hash": "c" * 64,
            "training_recipe_hash": recipe_hash,
            "force_train": False,
            "force_eval": False,
            "preflight_only": False,
            "max_samples": 0,
            "offline": True,
            "trial_id": job_id,
            "training_mode": mode,
        },
        "containers": {"training": "chatts", "evaluation": "ragas"},
        "training": {
            "project_root": "/workspace/ChatTS-Training",
            "script": (
                "/workspace/ChatTS-Training/scripts/full/"
                "run_chronos2_best_two_stage.sh"
            ),
            "base_model_path": "/share/models/ChatTS-Qwen3-8B",
            "output_root": output_root,
            "final_model_path": (
                f"{output_root}/best_stage1_seed42"
                if mode == "stage1"
                else f"{output_root}/best_seed42"
            ),
            "chronos2_model_path": "/workspace/chronos2",
            "dataset_dir": "/share/data/chatts-data-versions/datav3",
            "keep_stage1": False,
            "deepspeed_include": "localhost:0,1,2,3,4,5,6,7",
            "master_port": 19901,
            "stage1": stage(datasets="source_a_stage1,source_b_stage1"),
            "stage2": stage(datasets="source_a_stage2,source_c_stage2"),
        },
        "evaluation": {
            "project_root": "/workspace/ChatTS/ChatTS-main",
            "script": "/workspace/ChatTS/ChatTS-main/scripts/run_all_chatts_benchmarks.sh",
            "model_path": (
                f"{output_root}/best_stage1_seed42"
                if mode == "stage1"
                else f"{output_root}/best_seed42"
            ),
            "model_name": "chatts-datav3-fixture",
            "output_root": (
                "/share/evaluation/chatts-datav3-fixture/"
                "protocol-dddddddddddddddd"
            ),
            "chronos2_model_path": "/workspace/chronos2",
            "tsrbench_root": "/share/TSRBench-dataset",
            "tinybench_dataset_root": "/share/tyb",
            "ts_haystack_root": "/workspace/TS-Haystack",
            "timeseriesexam_root": "/workspace/TimeSeriesExam",
            "timeseriesexam_data_file": "/workspace/TimeSeriesExam/output/qa_dataset.json",
            "benchmarks": "tsrbench,timeseriesexam",
            "run_id": (
                "chronos2-datav3-fixture-protocol-dddddddddddddddd-"
                f"{mode}"
            ),
            "protocol_hash": "d" * 64,
            "model_completion_marker": (
                "STAGE1_COMPLETE.json"
                if mode == "stage1"
                else "TRAINING_COMPLETE.json"
            ),
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
            "tiny_gpu_memory_utilization": "0.70",
            "haystack_max_model_len": 40960,
            "haystack_max_new_tokens": 500,
            "haystack_batch_size": 1,
            "haystack_request_chunk_size": 8,
            "exam_max_model_len": 8192,
            "exam_max_new_tokens": 1024,
            "exam_batch_size": 8,
            "exam_request_chunk_size": 64,
        },
        "slurm": {"evaluation_host_root": "/host/ChatTS"},
    }
    payload["pipeline"]["trial_config_hash"] = canonical_hash(payload)
    return payload


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    # JSON is a valid YAML subset and exercises the loader's dependency-free path.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_environment(path: Path, job_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LOADER), str(path), "--expected-job-id", job_id],
        check=False,
        capture_output=True,
        text=True,
    )


def assignments(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


def test_loader_emits_full_frozen_training_contract(tmp_path: Path) -> None:
    job_id = "1" * 32
    config = tmp_path / "resolved.yaml"
    payload = frozen_payload(job_id=job_id)
    write_payload(config, payload)

    completed = load_environment(config, job_id)

    assert completed.returncode == 0, completed.stderr
    values = assignments(completed.stdout)
    assert values["PIPELINE_MODE"] == "full"
    assert values["TRIAL_ID"] == job_id
    assert values["TRIAL_CONFIG_HASH"] == payload["pipeline"]["trial_config_hash"]
    assert values["DATASET_SNAPSHOT_HASH"] == "c" * 64
    assert values["TRAINING_RECIPE_HASH"] == "b" * 64
    assert values["OUTPUT_ROOT"].endswith("/experiments/recipe-bbbbbbbbbbbbbbbb")
    assert values["FINAL_MODEL_PATH"].endswith("/recipe-bbbbbbbbbbbbbbbb/best_seed42")
    assert values["STAGE1_DATASETS"] == "source_a_stage1,source_b_stage1"
    assert values["STAGE2_DATASETS"] == "source_a_stage2,source_c_stage2"
    assert values["STAGE1_TIMESERIES_SFT_LR"] == "2e-5"
    assert values["STAGE1_INTERLEAVE_PROBS"] == ""
    assert values["EVAL_MODEL_PATH"] == values["FINAL_MODEL_PATH"]
    assert values["MODEL_COMPLETION_MARKER"] == "TRAINING_COMPLETE.json"
    assert values["EVAL_PROTOCOL_HASH"] == "d" * 64
    assert values["EVAL_OUTPUT_ROOT"].endswith("/protocol-dddddddddddddddd")
    assert values["RUN_ID"].endswith("-protocol-dddddddddddddddd-full")
    assert values["BENCHMARKS"] == "tsrbench,timeseriesexam"
    assert values["CHATTS_EVALUATION_DIR"] == "/host/ChatTS"
    assert "STAGE1_OUT" not in values


def test_stage1_mode_saves_directly_to_recipe_final_model(tmp_path: Path) -> None:
    job_id = "2" * 32
    config = tmp_path / "stage1.yaml"
    write_payload(config, frozen_payload(mode="stage1", job_id=job_id))

    completed = load_environment(config, job_id)

    assert completed.returncode == 0, completed.stderr
    values = assignments(completed.stdout)
    assert values["PIPELINE_MODE"] == "stage1"
    assert values["STAGE1_OUT"] == values["FINAL_MODEL_PATH"]
    assert values["KEEP_STAGE1"] == "1"
    assert values["MODEL_COMPLETION_MARKER"] == "STAGE1_COMPLETE.json"


def test_loader_rejects_tampering_wrong_job_and_non_recipe_output(tmp_path: Path) -> None:
    job_id = "3" * 32
    original = frozen_payload(job_id=job_id)
    config = tmp_path / "invalid.yaml"

    tampered = json.loads(json.dumps(original))
    tampered["training"]["stage1"]["learning_rate"] = "9e-5"
    write_payload(config, tampered)
    result = load_environment(config, job_id)
    assert result.returncode != 0
    assert "trial_config_hash does not match" in result.stderr

    write_payload(config, original)
    result = load_environment(config, "4" * 32)
    assert result.returncode != 0
    assert "does not match submitted Studio job" in result.stderr

    invalid_output = json.loads(json.dumps(original))
    invalid_output["training"]["output_root"] = "/share/output/not-the-recipe"
    invalid_output["training"]["final_model_path"] = (
        "/share/output/not-the-recipe/best_seed42"
    )
    invalid_output["evaluation"]["model_path"] = invalid_output["training"][
        "final_model_path"
    ]
    del invalid_output["pipeline"]["trial_config_hash"]
    invalid_output["pipeline"]["trial_config_hash"] = canonical_hash(invalid_output)
    write_payload(config, invalid_output)
    result = load_environment(config, job_id)
    assert result.returncode != 0
    assert "must end in the frozen recipe id" in result.stderr


def test_loader_rejects_wrong_marker_unknown_eval_field_and_invalid_protocol(
    tmp_path: Path,
) -> None:
    job_id = "6" * 32
    config = tmp_path / "invalid-evaluation.yaml"

    wrong_marker = frozen_payload(mode="stage1", job_id=job_id)
    wrong_marker["evaluation"]["model_completion_marker"] = "TRAINING_COMPLETE.json"
    del wrong_marker["pipeline"]["trial_config_hash"]
    wrong_marker["pipeline"]["trial_config_hash"] = canonical_hash(wrong_marker)
    write_payload(config, wrong_marker)
    result = load_environment(config, job_id)
    assert result.returncode != 0
    assert "must be STAGE1_COMPLETE.json" in result.stderr

    unknown_field = frozen_payload(job_id=job_id)
    unknown_field["evaluation"]["silently_ignored"] = True
    del unknown_field["pipeline"]["trial_config_hash"]
    unknown_field["pipeline"]["trial_config_hash"] = canonical_hash(unknown_field)
    write_payload(config, unknown_field)
    result = load_environment(config, job_id)
    assert result.returncode != 0
    assert "unknown evaluation fields" in result.stderr

    invalid_protocol = frozen_payload(job_id=job_id)
    invalid_protocol["evaluation"]["protocol_hash"] = "not-a-hash"
    del invalid_protocol["pipeline"]["trial_config_hash"]
    invalid_protocol["pipeline"]["trial_config_hash"] = canonical_hash(invalid_protocol)
    write_payload(config, invalid_protocol)
    result = load_environment(config, job_id)
    assert result.returncode != 0
    assert "protocol_hash must be a 64-character SHA256" in result.stderr

    unscoped_output = frozen_payload(job_id=job_id)
    unscoped_output["evaluation"]["output_root"] = "/share/evaluation/unscoped"
    del unscoped_output["pipeline"]["trial_config_hash"]
    unscoped_output["pipeline"]["trial_config_hash"] = canonical_hash(
        unscoped_output
    )
    write_payload(config, unscoped_output)
    result = load_environment(config, job_id)
    assert result.returncode != 0
    assert "must end in the frozen protocol id" in result.stderr


@pytest.mark.parametrize(
    ("mode", "marker", "preflight"),
    [
        ("stage1", "STAGE1_COMPLETE.json", False),
        ("full", "TRAINING_COMPLETE.json", False),
        ("stage1", "STAGE1_COMPLETE.json", True),
    ],
)
def test_sbatch_passes_training_then_evaluation_contract_to_singularity(
    tmp_path: Path,
    mode: str,
    marker: str,
    preflight: bool,
) -> None:
    job_id = "5" * 32
    config = tmp_path / "resolved.yaml"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "srun.argv"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        "printf '%s\\n' '=== CALL ===' \"$@\" >> \"$MOCK_SRUN_LOG\"\n",
        encoding="utf-8",
    )
    fake_singularity = fake_bin / "singularity"
    fake_singularity.write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    flock_log = tmp_path / "flock.argv"
    fake_flock = fake_bin / "flock"
    fake_flock.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$MOCK_FLOCK_LOG\"\n",
        encoding="utf-8",
    )
    fake_srun.chmod(0o755)
    fake_singularity.chmod(0o755)
    fake_flock.chmod(0o755)

    image = tmp_path / "chatts.sif"
    image.write_bytes(b"mock-sif")
    chronos = tmp_path / "chronos2"
    shared = tmp_path / "share"
    job_tmp = tmp_path / "job-tmp"
    evaluation_repo = tmp_path / "ChatTS"
    tsrbench = shared / "TSRBench-dataset"
    timeseriesexam = tmp_path / "TimeSeriesExam"
    chronos.mkdir()
    shared.mkdir()
    job_tmp.mkdir()
    (evaluation_repo / "scripts").mkdir(parents=True)
    (evaluation_repo / "scripts" / "run_all_chatts_benchmarks.sh").write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )
    tsrbench.mkdir()
    timeseriesexam.mkdir()

    payload = frozen_payload(mode=mode, job_id=job_id)
    payload["pipeline"]["offline"] = False
    payload["pipeline"]["preflight_only"] = preflight
    payload["slurm"].update(
        {
            "evaluation_host_root": str(evaluation_repo),
            "tsrbench_host_root": str(tsrbench),
            "timeseriesexam_host_root": str(timeseriesexam),
        }
    )
    del payload["pipeline"]["trial_config_hash"]
    payload["pipeline"]["trial_config_hash"] = canonical_hash(payload)
    write_payload(config, payload)

    env = os.environ.copy()
    env.update(
        {
            "SLURM_JOB_ID": "12345",
            # Slurm executes a spool copy rather than the repository file.
            # The launcher must locate Training through SLURM_SUBMIT_DIR.
            "SLURM_SUBMIT_DIR": str(PROJECT_ROOT),
            "CHATTS_SIF_IMAGE": str(image),
            "CHATTS_HOST_CHRONOS2_PATH": str(chronos),
            "CHATTS_SHARED_HOST_PATH": str(shared),
            "CHATTS_SHARED_CONTAINER_PATH": "/share",
            "CHATTS_HOST_PYTHON_BIN": sys.executable,
            "CHATTS_SRUN_BIN": str(fake_srun),
            "CHATTS_SINGULARITY_BIN": str(fake_singularity),
            "CHATTS_FLOCK_BIN": str(fake_flock),
            "CHATTS_JOB_TMP_ROOT": str(job_tmp),
            "MOCK_SRUN_LOG": str(invocation_log),
            "MOCK_FLOCK_LOG": str(flock_log),
        }
    )
    spooled_sbatch = tmp_path / "slurm-spool" / "run_chatts_studio_pipeline.sbatch"
    spooled_sbatch.parent.mkdir()
    shutil.copyfile(SBATCH, spooled_sbatch)
    completed = subprocess.run(
        ["bash", str(spooled_sbatch), str(config), job_id],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    argv_text = invocation_log.read_text(encoding="utf-8")
    assert str(fake_singularity) in argv_text
    assert f"PIPELINE_MODE={mode}" in argv_text
    if mode == "stage1":
        assert "STAGE1_OUT=/share/output/ChatTS/experiments/recipe-" in argv_text
    else:
        assert "STAGE1_OUT=/share/output/ChatTS/experiments/recipe-" not in argv_text
    assert "DATASET_DIR=/share/data/chatts-data-versions/datav3" in argv_text
    assert "TRAINING_RECIPE_HASH=" + "b" * 64 in argv_text
    assert "run_chronos2_best_two_stage.sh" in argv_text
    assert "run_all_chatts_benchmarks.sh" in argv_text
    assert argv_text.count("=== CALL ===") == 2
    assert f"MODEL_COMPLETION_MARKER={marker}" in argv_text
    assert f"PREFLIGHT_ONLY={int(preflight)}" in argv_text
    assert "EVAL_PROTOCOL_HASH=" + "d" * 64 in argv_text
    assert "Config file SHA256:" in completed.stdout
    calls = argv_text.split("=== CALL ===")[1:]
    assert len(calls) == 2
    assert "HF_HUB_OFFLINE=1" in calls[0]
    assert "TRANSFORMERS_OFFLINE=1" in calls[0]
    assert "HF_HUB_OFFLINE=0" in calls[1]
    assert "TRANSFORMERS_OFFLINE=0" in calls[1]
    assert flock_log.read_text(encoding="utf-8").splitlines() == ["-n", "9"]
    assert "Recipe lock:" in completed.stdout
    if preflight:
        assert "no training or evaluation was started" in completed.stdout
        assert not (
            shared
            / "output"
            / "ChatTS"
            / "experiments"
            / "recipe-bbbbbbbbbbbbbbbb"
            / ("best_stage1_seed42" if mode == "stage1" else "best_seed42")
        ).exists()
        assert not (shared / "evaluation" / "chatts-datav3-fixture").exists()


def test_sbatch_training_failure_blocks_evaluation_step(tmp_path: Path) -> None:
    job_id = "7" * 32
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "srun.calls"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        "printf 'training-step\\n' >> \"$MOCK_SRUN_LOG\"\n"
        "exit 9\n",
        encoding="utf-8",
    )
    fake_singularity = fake_bin / "singularity"
    fake_singularity.write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_srun.chmod(0o755)
    fake_singularity.chmod(0o755)
    fake_flock.chmod(0o755)

    image = tmp_path / "chatts.sif"
    image.write_bytes(b"mock-sif")
    chronos = tmp_path / "chronos2"
    shared = tmp_path / "share"
    eval_repo = tmp_path / "ChatTS"
    tsrbench = shared / "TSRBench-dataset"
    for directory in (chronos, shared, eval_repo / "scripts", tsrbench):
        directory.mkdir(parents=True, exist_ok=True)
    (eval_repo / "scripts" / "run_all_chatts_benchmarks.sh").write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )

    payload = frozen_payload(job_id=job_id)
    payload["evaluation"]["benchmarks"] = "tsrbench"
    payload["slurm"] = {
        "evaluation_host_root": str(eval_repo),
        "tsrbench_host_root": str(tsrbench),
    }
    del payload["pipeline"]["trial_config_hash"]
    payload["pipeline"]["trial_config_hash"] = canonical_hash(payload)
    config = tmp_path / "resolved.yaml"
    write_payload(config, payload)

    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_JOB_ID": "45678",
            "SLURM_SUBMIT_DIR": str(PROJECT_ROOT),
            "CHATTS_SIF_IMAGE": str(image),
            "CHATTS_HOST_CHRONOS2_PATH": str(chronos),
            "CHATTS_SHARED_HOST_PATH": str(shared),
            "CHATTS_SHARED_CONTAINER_PATH": "/share",
            "CHATTS_HOST_PYTHON_BIN": sys.executable,
            "CHATTS_SRUN_BIN": str(fake_srun),
            "CHATTS_SINGULARITY_BIN": str(fake_singularity),
            "CHATTS_FLOCK_BIN": str(fake_flock),
            "CHATTS_JOB_TMP_ROOT": str(tmp_path / "job-tmp"),
            "MOCK_SRUN_LOG": str(invocation_log),
        }
    )
    completed = subprocess.run(
        ["bash", str(SBATCH), str(config), job_id],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 9
    assert invocation_log.read_text(encoding="utf-8").splitlines() == [
        "training-step"
    ]
    assert "Studio Slurm training completed" not in completed.stdout


def test_sbatch_nonblocking_recipe_lock_rejects_concurrent_writer(
    tmp_path: Path,
) -> None:
    job_id = "8" * 32
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_called = tmp_path / "srun-called"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/usr/bin/env bash\ntouch \"$MOCK_SRUN_CALLED\"\nexit 99\n",
        encoding="utf-8",
    )
    fake_singularity = fake_bin / "singularity"
    fake_singularity.write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    fake_flock = fake_bin / "flock"
    fake_flock.write_text(
        "#!/usr/bin/env bash\n[[ \"$1\" == -n && \"$2\" == 9 ]]\nexit 1\n",
        encoding="utf-8",
    )
    for executable in (fake_srun, fake_singularity, fake_flock):
        executable.chmod(0o755)

    image = tmp_path / "chatts.sif"
    image.write_bytes(b"mock-sif")
    chronos = tmp_path / "chronos2"
    shared = tmp_path / "share"
    eval_repo = tmp_path / "ChatTS"
    tsrbench = shared / "TSRBench-dataset"
    for directory in (chronos, shared, eval_repo / "scripts", tsrbench):
        directory.mkdir(parents=True, exist_ok=True)
    (eval_repo / "scripts" / "run_all_chatts_benchmarks.sh").write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )

    payload = frozen_payload(job_id=job_id)
    payload["evaluation"]["benchmarks"] = "tsrbench"
    payload["slurm"] = {
        "evaluation_host_root": str(eval_repo),
        "tsrbench_host_root": str(tsrbench),
    }
    del payload["pipeline"]["trial_config_hash"]
    payload["pipeline"]["trial_config_hash"] = canonical_hash(payload)
    config = tmp_path / "resolved.yaml"
    write_payload(config, payload)

    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_JOB_ID": "56789",
            "SLURM_SUBMIT_DIR": str(PROJECT_ROOT),
            "CHATTS_SIF_IMAGE": str(image),
            "CHATTS_HOST_CHRONOS2_PATH": str(chronos),
            "CHATTS_SHARED_HOST_PATH": str(shared),
            "CHATTS_SHARED_CONTAINER_PATH": "/share",
            "CHATTS_HOST_PYTHON_BIN": sys.executable,
            "CHATTS_SRUN_BIN": str(fake_srun),
            "CHATTS_SINGULARITY_BIN": str(fake_singularity),
            "CHATTS_FLOCK_BIN": str(fake_flock),
            "CHATTS_JOB_TMP_ROOT": str(tmp_path / "job-tmp"),
            "MOCK_SRUN_CALLED": str(srun_called),
        }
    )
    completed = subprocess.run(
        ["bash", str(SBATCH), str(config), job_id],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75
    assert not srun_called.exists()
    assert "already using this training recipe" in completed.stderr
    assert ".chatts-studio.lock" in completed.stderr


def test_sbatch_contract_marker_and_shell_syntax() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "# CHATTS_STUDIO_SBATCH_API=1" in text.splitlines()[:5]
    assert LEGACY_SBATCH.is_file()
    for path in (SBATCH, LEGACY_SBATCH):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
