from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


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
            "final_model_path": f"{output_root}/best_seed42",
            "chronos2_model_path": "/workspace/chronos2",
            "dataset_dir": "/share/data/chatts-data-versions/datav3",
            "keep_stage1": False,
            "deepspeed_include": "localhost:0,1,2,3,4,5,6,7",
            "master_port": 19901,
            "stage1": stage(datasets="source_a_stage1,source_b_stage1"),
            "stage2": stage(datasets="source_a_stage2,source_c_stage2"),
        },
        "evaluation": {
            "benchmarks": "tsrbench,timeseriesexam",
            "model_name": "ignored-by-training-sbatch",
        },
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
    del invalid_output["pipeline"]["trial_config_hash"]
    invalid_output["pipeline"]["trial_config_hash"] = canonical_hash(invalid_output)
    write_payload(config, invalid_output)
    result = load_environment(config, job_id)
    assert result.returncode != 0
    assert "must end in the frozen recipe id" in result.stderr


def test_sbatch_passes_validated_contract_to_standard_singularity_runner(
    tmp_path: Path,
) -> None:
    job_id = "5" * 32
    config = tmp_path / "resolved.yaml"
    write_payload(config, frozen_payload(mode="stage1", job_id=job_id))

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "srun.argv"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\nprintf '%s\\n' \"$@\" > \"$MOCK_SRUN_LOG\"\n",
        encoding="utf-8",
    )
    fake_singularity = fake_bin / "singularity"
    fake_singularity.write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    fake_srun.chmod(0o755)
    fake_singularity.chmod(0o755)

    image = tmp_path / "chatts.sif"
    image.write_bytes(b"mock-sif")
    chronos = tmp_path / "chronos2"
    shared = tmp_path / "share"
    job_tmp = tmp_path / "job-tmp"
    chronos.mkdir()
    shared.mkdir()
    job_tmp.mkdir()

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
            "CHATTS_JOB_TMP_ROOT": str(job_tmp),
            "MOCK_SRUN_LOG": str(invocation_log),
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
    assert "PIPELINE_MODE=stage1" in argv_text
    assert "STAGE1_OUT=/share/output/ChatTS/experiments/recipe-" in argv_text
    assert "DATASET_DIR=/share/data/chatts-data-versions/datav3" in argv_text
    assert "TRAINING_RECIPE_HASH=" + "b" * 64 in argv_text
    assert "run_chronos2_best_two_stage.sh" in argv_text
    assert "Config file SHA256:" in completed.stdout


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
