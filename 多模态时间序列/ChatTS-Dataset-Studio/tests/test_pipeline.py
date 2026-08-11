from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from chatts_dataset_studio.models import StudioError
from chatts_dataset_studio.pipeline import PipelineJobs, resolve_pipeline_request


def _integration(tmp_path: Path) -> dict[str, object]:
    return {
        "pipeline_script": str(tmp_path / "ChatTS" / "scripts" / "run_train_then_eval.sh"),
        "training_root": str(tmp_path / "ChatTS-Training"),
        "evaluation_root": str(tmp_path / "ChatTS"),
        "train_project_root": "/workspace/ChatTS-Training",
        "eval_project_root": "/workspace/ChatTS/ChatTS-main",
        "training_script": "/workspace/ChatTS-Training/scripts/full/run_chronos2_best_two_stage.sh",
        "evaluation_script": "/workspace/ChatTS/ChatTS-main/scripts/run_all_chatts_benchmarks.sh",
        "training_container": "chatts",
        "evaluation_container": "ragas",
        "base_model_path": "/share/model/ChatTS-Qwen3-8B",
        "model_output_base": "/share/output/ChatTS-msxf-8B-datav1",
        "evaluation_output_base": "/share/evaluation/all-benchmarks",
        "model_name_base": "chatts-msxf-8B-datav1",
        "train_chronos2_model_path": "/workspace/chronos2",
        "eval_chronos2_model_path": "/workspace/chronos2",
        "tsrbench_root": "/share/TSRBench-dataset",
        "tinybench_dataset_root": "/share/tyb",
        "ts_haystack_root": "/workspace/TS-Haystack",
        "timeseriesexam_root": "/workspace/TimeSeriesExam",
        "timeseriesexam_data_file": "/workspace/TimeSeriesExam/output/qa_dataset.json",
    }


def _version(tmp_path: Path) -> dict[str, object]:
    return {
        "version": "datav3",
        "snapshot_dir": str(tmp_path / "versions" / "datav3"),
        "dataset_snapshot_hash": "a" * 64,
        "dataset_names": {
            "stage1": ["datav3__stage1__alpha", "datav3__stage1__beta"],
            "stage2": ["datav3__stage2__alpha", "datav3__stage2__beta"],
        },
    }


def test_resolve_pipeline_derives_versioned_paths_and_snapshot_datasets(
    tmp_path: Path,
) -> None:
    resolved = resolve_pipeline_request(
        {
            "mode": "train_eval",
            "version": "datav3",
            "training": {
                "seed": 7,
                "stage1": {"learning_rate": "5e-6"},
                "stage2": {"learning_rate": "2e-5", "num_train_epochs": 2},
            },
            "evaluation": {"benchmarks": ["tsrbench", "timeseriesexam"]},
        },
        _version(tmp_path),
        _integration(tmp_path),
    )

    training = resolved["config"]["training"]
    evaluation = resolved["config"]["evaluation"]
    assert resolved["config"]["pipeline"]["data_version"] == "datav3"
    assert resolved["config"]["pipeline"]["dataset_snapshot_hash"] == "a" * 64
    assert training["dataset_dir"].endswith("/versions/datav3")
    assert training["stage1"]["datasets"] == (
        "datav3__stage1__alpha,datav3__stage1__beta"
    )
    assert training["output_root"] == "/share/output/ChatTS-msxf-8B-datav3"
    assert training["final_model_path"].endswith("ChatTS-msxf-8B-datav3/best_seed7")
    assert evaluation["model_name"] == "chatts-msxf-8B-datav3-seed7"
    assert evaluation["output_root"].endswith("/chatts-msxf-8B-datav3-seed7")
    assert evaluation["benchmarks"] == "tsrbench,timeseriesexam"
    assert resolved["config_hash"]


def test_resolve_pipeline_rejects_path_override_and_invalid_stage_mix(
    tmp_path: Path,
) -> None:
    with pytest.raises(StudioError, match="fixed by the server"):
        resolve_pipeline_request(
            {"training": {"project_root": "/tmp/injected"}},
            _version(tmp_path),
            _integration(tmp_path),
        )

    with pytest.raises(StudioError, match="must contain 2 probabilities"):
        resolve_pipeline_request(
            {
                "training": {
                    "stage1": {
                        "mix_strategy": "interleave_over",
                        "interleave_probs": "1.0",
                    }
                }
            },
            _version(tmp_path),
            _integration(tmp_path),
        )


def test_pipeline_job_persists_resolved_yaml_and_log(tmp_path: Path) -> None:
    integration = _integration(tmp_path)
    script = Path(str(integration["pipeline_script"]))
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        "test -f \"$CONFIG_FILE\"\n"
        "echo pipeline-ok\n",
        encoding="utf-8",
    )
    resolved = resolve_pipeline_request({}, _version(tmp_path), integration)
    jobs = PipelineJobs(tmp_path / "state", script)

    started = jobs.start(resolved, preflight=True)
    deadline = time.monotonic() + 5
    while True:
        job = jobs.get(started["job_id"])
        if job["status"] in {"completed", "failed"}:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert job["status"] == "completed"
    assert job["exit_code"] == 0
    assert "pipeline-ok" in job["log_tail"]
    saved_config = yaml.safe_load(Path(job["config_path"]).read_text(encoding="utf-8"))
    assert saved_config["pipeline"]["preflight_only"] is True
    assert saved_config["pipeline"]["data_version"] == "datav3"
    assert saved_config["pipeline"]["trial_id"] == job["job_id"]
    on_disk = json.loads(
        (tmp_path / "state" / "jobs" / f"{job['job_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert on_disk["status"] == "completed"
    hash_payload = json.loads(json.dumps(saved_config))
    del hash_payload["pipeline"]["trial_config_hash"]
    expected_hash = hashlib.sha256(
        json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert job["config_hash"] == expected_hash
    assert saved_config["pipeline"]["trial_config_hash"] == expected_hash


def test_pipeline_jobs_recovers_finished_worker_status(tmp_path: Path) -> None:
    state = tmp_path / "state"
    jobs_root = state / "jobs"
    status_root = state / "worker-status"
    jobs_root.mkdir(parents=True)
    status_root.mkdir(parents=True)
    job_id = "1" * 32
    job = {
        "job_id": job_id,
        "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00",
        "log_path": str(state / "logs" / f"{job_id}.log"),
        "pid": os.getpid(),
    }
    (jobs_root / f"{job_id}.json").write_text(json.dumps(job), encoding="utf-8")
    worker_status = {
        "job_id": job_id,
        "status": "completed",
        "pid": 123,
        "exit_code": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:03+00:00",
        "duration_seconds": 3.0,
    }
    (status_root / f"{job_id}.json").write_text(
        json.dumps(worker_status), encoding="utf-8"
    )

    recovered = PipelineJobs(state, None).get(job_id, include_log=False)

    assert recovered["status"] == "completed"
    assert recovered["exit_code"] == 0
    assert recovered["duration_seconds"] == 3.0


def test_pipeline_jobs_marks_stale_running_job_failed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    jobs_root = state / "jobs"
    jobs_root.mkdir(parents=True)
    job_id = "2" * 32
    (jobs_root / f"{job_id}.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "running",
                "created_at": "2026-01-01T00:00:00+00:00",
                "log_path": str(state / "logs" / f"{job_id}.log"),
                "pid": 999_999_999,
            }
        ),
        encoding="utf-8",
    )

    recovered = PipelineJobs(state, None).get(job_id, include_log=False)

    assert recovered["status"] == "failed"
    assert recovered["exit_code"] == 125


def test_resolved_yaml_matches_sibling_chatts_loader_contract(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    loader = project_root.parent / "ChatTS" / "scripts" / "load_train_eval_config.py"
    if not loader.is_file():
        pytest.skip("Sibling ChatTS integration repository is not present")
    resolved = resolve_pipeline_request({}, _version(tmp_path), _integration(tmp_path))
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(resolved["config"], allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(loader), str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DATA_VERSION=datav3" in completed.stdout
    assert f"DATASET_SNAPSHOT_HASH={'a' * 64}" in completed.stdout
    assert "TSR_MAX_MODEL_LEN=12288" in completed.stdout
