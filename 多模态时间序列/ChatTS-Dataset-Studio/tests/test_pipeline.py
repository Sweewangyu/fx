from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from chatts_dataset_studio.models import StudioError
from chatts_dataset_studio.pipeline import (
    PipelineJobs,
    public_pipeline_defaults,
    resolve_pipeline_request,
)


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


def _trusted_slurm_integration(tmp_path: Path) -> dict[str, object]:
    integration = _integration(tmp_path)
    training_root = Path(str(integration["training_root"]))
    slurm_root = training_root / "slurm"
    slurm_root.mkdir(parents=True, exist_ok=True)
    launcher = slurm_root / "run_chatts_studio_pipeline.sbatch"
    launcher.write_text(
        "#!/usr/bin/env bash\n# CHATTS_STUDIO_SBATCH_API=1\nexit 0\n",
        encoding="utf-8",
    )
    integration.update(
        {
            "execution_mode": "slurm",
            "slurm_root": str(slurm_root),
            "slurm_sbatch": launcher.name,
            "slurm_poll_seconds": 0.01,
            "slurm_accounting_grace_seconds": 0.2,
        }
    )
    return integration


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
    recipe_id = resolved["derived"]["training_recipe_id"]
    recipe_hash = resolved["derived"]["training_recipe_hash"]
    assert recipe_id == f"recipe-{recipe_hash[:16]}"
    assert training["output_root"] == (
        f"/share/output/ChatTS-msxf-8B-datav3/experiments/{recipe_id}"
    )
    assert training["final_model_path"].endswith(
        f"ChatTS-msxf-8B-datav3/experiments/{recipe_id}/best_seed7"
    )
    assert evaluation["model_name"] == (
        f"chatts-msxf-8B-datav3-seed7-{recipe_id}"
    )
    assert evaluation["output_root"].endswith(
        f"/chatts-msxf-8B-datav3-seed7-{recipe_id}"
    )
    assert resolved["config"]["pipeline"]["training_recipe_hash"] == recipe_hash
    assert evaluation["benchmarks"] == "tsrbench,timeseriesexam"
    assert resolved["config_hash"]


def test_resolve_pipeline_accepts_user_base_model_path_without_server_default(
    tmp_path: Path,
) -> None:
    integration = _integration(tmp_path)
    del integration["base_model_path"]

    resolved = resolve_pipeline_request(
        {"training": {"base_model_path": "/share/custom models/Qwen3-8B"}},
        _version(tmp_path),
        integration,
    )

    assert resolved["config"]["training"]["base_model_path"] == (
        "/share/custom models/Qwen3-8B"
    )


@pytest.mark.parametrize(
    ("base_model", "scale", "output", "model_name"),
    [
        (
            "/share/model/ChatTS-Qwen3-4B",
            "4B",
            "/share/output/ChatTS-msxf-4B-datav3",
            "chatts-msxf-4B-datav3-seed42",
        ),
        (
            "/share/model/ChatTS-Qwen3-1.7b-Instruct",
            "1.7B",
            "/share/output/ChatTS-msxf-1.7B-datav3",
            "chatts-msxf-1.7B-datav3-seed42",
        ),
        (
            "/share/model/custom-no-scale",
            None,
            "/share/output/ChatTS-msxf-8B-datav3",
            "chatts-msxf-8B-datav3-seed42",
        ),
    ],
)
def test_resolve_pipeline_syncs_output_scale_from_base_model(
    tmp_path: Path,
    base_model: str,
    scale: str | None,
    output: str,
    model_name: str,
) -> None:
    resolved = resolve_pipeline_request(
        {"training": {"base_model_path": base_model}},
        _version(tmp_path),
        _integration(tmp_path),
    )

    recipe_id = resolved["derived"]["training_recipe_id"]
    assert resolved["config"]["training"]["output_root"] == (
        f"{output}/experiments/{recipe_id}"
    )
    assert resolved["config"]["evaluation"]["model_name"] == (
        f"{model_name}-{recipe_id}"
    )
    assert resolved["derived"]["model_scale"] == scale


def test_training_recipe_hash_is_stable_for_retries_and_isolates_parameter_changes(
    tmp_path: Path,
) -> None:
    baseline = resolve_pipeline_request({}, _version(tmp_path), _integration(tmp_path))
    retry = resolve_pipeline_request(
        {
            "training": {"force_train": True},
            "evaluation": {
                "force_eval": True,
                "benchmarks": ["tsrbench"],
                "max_samples": 16,
            },
        },
        _version(tmp_path),
        _integration(tmp_path),
    )
    changed = resolve_pipeline_request(
        {"training": {"stage2": {"learning_rate": "2e-5"}}},
        _version(tmp_path),
        _integration(tmp_path),
    )

    assert retry["derived"]["training_recipe_hash"] == baseline["derived"][
        "training_recipe_hash"
    ]
    assert retry["derived"]["final_model_path"] == baseline["derived"][
        "final_model_path"
    ]
    assert changed["derived"]["training_recipe_hash"] != baseline["derived"][
        "training_recipe_hash"
    ]
    assert changed["derived"]["final_model_path"] != baseline["derived"][
        "final_model_path"
    ]


def test_stage1_only_saves_a_final_recipe_model_and_ignores_stage2_and_eval(
    tmp_path: Path,
) -> None:
    base_payload = {
        "mode": "train",
        "training": {
            "profile": "chronos2-stage1",
            "stage1": {"learning_rate": "5e-6"},
        },
    }
    baseline = resolve_pipeline_request(
        base_payload, _version(tmp_path), _integration(tmp_path)
    )
    changed_hidden_fields = resolve_pipeline_request(
        {
            **base_payload,
            "training": {
                **base_payload["training"],
                "stage2": {"learning_rate": "9e-5", "num_train_epochs": 9},
            },
            "evaluation": {"benchmarks": ["timeseriesexam"], "max_samples": 99},
        },
        _version(tmp_path),
        _integration(tmp_path),
    )

    assert baseline["pipeline_mode"] == "stage1"
    assert baseline["mode"] == "train"
    assert baseline["config"]["pipeline"]["pipeline_mode"] == "stage1"
    assert baseline["config"]["training"]["keep_stage1"] is True
    assert baseline["config"]["training"]["stage1_model_path"] == baseline["derived"][
        "final_model_path"
    ]
    assert baseline["derived"]["final_model_path"].endswith("/best_stage1_seed42")
    assert "evaluation" not in baseline["config"]
    assert "stage2" not in baseline["derived"]["training_recipe"]
    assert changed_hidden_fields["derived"]["training_recipe_hash"] == baseline[
        "derived"
    ]["training_recipe_hash"]
    assert changed_hidden_fields["derived"]["final_model_path"] == baseline["derived"][
        "final_model_path"
    ]


def test_stage1_recipe_path_changes_when_an_executed_parameter_changes(
    tmp_path: Path,
) -> None:
    baseline = resolve_pipeline_request(
        {"mode": "train", "training": {"profile": "chronos2-stage1"}},
        _version(tmp_path),
        _integration(tmp_path),
    )
    changed = resolve_pipeline_request(
        {
            "mode": "train",
            "training": {
                "profile": "chronos2-stage1",
                "stage1": {"warmup_ratio": 0.05},
            },
        },
        _version(tmp_path),
        _integration(tmp_path),
    )

    assert changed["derived"]["training_recipe_hash"] != baseline["derived"][
        "training_recipe_hash"
    ]
    assert changed["derived"]["final_model_path"] != baseline["derived"][
        "final_model_path"
    ]


@pytest.mark.parametrize(
    "value, message",
    [
        (None, "non-empty absolute POSIX path"),
        ("", "non-empty absolute POSIX path"),
        ("relative/model", "non-empty absolute POSIX path"),
        (r"C:\\models\\qwen", "non-empty absolute POSIX path"),
        ("/share/model\x00bad", "must not contain NUL or newline"),
        ("/share/model\nbad", "must not contain NUL or newline"),
        ("/share/model\rbad", "must not contain NUL or newline"),
    ],
)
def test_resolve_pipeline_rejects_invalid_user_base_model_path(
    tmp_path: Path, value: object, message: str
) -> None:
    integration = _integration(tmp_path)
    del integration["base_model_path"]

    with pytest.raises(StudioError, match=message):
        resolve_pipeline_request(
            {"training": {"base_model_path": value}},
            _version(tmp_path),
            integration,
        )


def test_public_pipeline_defaults_reports_all_host_readiness_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chatts_dataset_studio.pipeline.shutil.which", lambda _: "/bin/docker")
    missing_script = tmp_path / "ChatTS" / "scripts" / "run_train_then_eval.sh"
    training_root = tmp_path / "ChatTS-Training"
    defaults = public_pipeline_defaults(
        {
            "pipeline_script": str(missing_script),
            "training_root": str(training_root),
        }
    )
    status = defaults["integration"]

    assert status["enabled"] is False
    assert status["disabled_reasons"] == [
        (
            "integration.pipeline_script does not exist or is not a file: "
            f"{missing_script}"
        ),
        (
            "integration.training_root does not exist or is not a directory: "
            f"{training_root}"
        ),
        "integration.evaluation_root is not configured",
        "integration.train_project_root is not configured",
        "integration.eval_project_root is not configured",
        "integration.training_script is not configured",
        "integration.evaluation_script is not configured",
        "integration.model_output_base is not configured",
        "integration.evaluation_output_base is not configured",
        "integration.train_chronos2_model_path is not configured",
        "integration.eval_chronos2_model_path is not configured",
        "integration.tsrbench_root is not configured",
        "integration.tinybench_dataset_root is not configured",
        "integration.ts_haystack_root is not configured",
        "integration.timeseriesexam_root is not configured",
        "integration.timeseriesexam_data_file is not configured",
    ]


def test_public_pipeline_defaults_enabled_matches_empty_disabled_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chatts_dataset_studio.pipeline.shutil.which", lambda _: "/bin/docker")
    integration = _integration(tmp_path)
    script = Path(str(integration["pipeline_script"]))
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    Path(str(integration["training_root"])).mkdir(parents=True)
    Path(str(integration["evaluation_root"])).mkdir(parents=True, exist_ok=True)

    status = public_pipeline_defaults(integration)["integration"]

    assert status["enabled"] is True
    assert status["disabled_reasons"] == []
    assert status["execution_mode"] == "docker_host"


def test_public_pipeline_defaults_rejects_container_control_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = _integration(tmp_path)
    script = Path(str(integration["pipeline_script"]))
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    Path(str(integration["training_root"])).mkdir(parents=True)
    Path(str(integration["evaluation_root"])).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("chatts_dataset_studio.pipeline.shutil.which", lambda _: None)

    status = public_pipeline_defaults(integration)["integration"]

    assert status["enabled"] is False
    assert status["disabled_reasons"] == [
        (
            "Docker CLI is unavailable to Dataset Studio; run the Studio control plane "
            "on the Docker host, not inside the training/evaluation container"
        )
    ]


def test_public_pipeline_defaults_uses_optional_base_model_default(
    tmp_path: Path,
) -> None:
    integration = _integration(tmp_path)
    integration["base_model_path"] = "/share/default/model"
    assert public_pipeline_defaults(integration)["training"]["base_model_path"] == (
        "/share/default/model"
    )
    del integration["base_model_path"]
    assert public_pipeline_defaults(integration)["training"]["base_model_path"] is None


def test_slurm_readiness_does_not_require_docker_or_evaluation_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = _trusted_slurm_integration(tmp_path)
    integration.pop("evaluation_root")
    monkeypatch.setattr(
        "chatts_dataset_studio.slurm.shutil.which",
        lambda command: f"/mock/bin/{command}",
    )
    monkeypatch.setattr(
        "chatts_dataset_studio.pipeline.shutil.which",
        lambda command: None if command == "docker" else f"/mock/bin/{command}",
    )

    defaults = public_pipeline_defaults(integration)

    assert defaults["integration"]["execution_mode"] == "slurm"
    assert defaults["integration"]["enabled"] is True
    assert defaults["integration"]["backends"]["slurm"]["default_sbatch"] == (
        "run_chatts_studio_pipeline.sbatch"
    )
    assert defaults["integration"]["backends"]["docker_host"]["enabled"] is False


def test_resolve_slurm_rejects_launcher_outside_trusted_root(tmp_path: Path) -> None:
    integration = _trusted_slurm_integration(tmp_path)
    outside = tmp_path / "outside.sbatch"
    outside.write_text(
        "#!/usr/bin/env bash\n# CHATTS_STUDIO_SBATCH_API=1\n",
        encoding="utf-8",
    )

    with pytest.raises(StudioError, match="trusted root"):
        resolve_pipeline_request(
            {
                "mode": "train",
                "execution": {"backend": "slurm", "sbatch_path": str(outside)},
                "training": {"profile": "chronos2-stage1"},
            },
            _version(tmp_path),
            integration,
        )


def test_slurm_job_submits_frozen_config_and_tracks_scheduler_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = _trusted_slurm_integration(tmp_path)
    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    submitted_args = tmp_path / "sbatch-args.txt"
    scripts = {
        "sbatch": (
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$MOCK_SBATCH_ARGS\"\n"
            "printf '12345\\n'\n"
        ),
        "squeue": "#!/bin/sh\nexit 0\n",
        "sacct": (
            "#!/bin/sh\n"
            "printf '12345|COMPLETED|0:0|2026-01-01|2026-01-01|1|\\n'\n"
        ),
    }
    for name, text in scripts.items():
        path = mock_bin / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{mock_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("MOCK_SBATCH_ARGS", str(submitted_args))

    resolved = resolve_pipeline_request(
        {
            "mode": "train",
            "execution": {"backend": "slurm"},
            "training": {"profile": "chronos2-stage1"},
        },
        _version(tmp_path),
        integration,
    )
    jobs = PipelineJobs(tmp_path / "state", None, integration)
    started = jobs.start(resolved)
    deadline = time.monotonic() + 5
    while True:
        job = jobs.get(started["job_id"])
        if job["status"] in {"completed", "failed", "canceled"}:
            break
        assert time.monotonic() < deadline, job
        time.sleep(0.01)

    assert job["status"] == "completed"
    assert job["kind"] == "train_stage1"
    assert job["execution_backend"] == "slurm"
    assert job["scheduler_job_id"] == "12345"
    assert len(job["sbatch_sha256"]) == 64
    argv = submitted_args.read_text(encoding="utf-8").splitlines()
    assert "--parsable" in argv
    assert str(Path(job["config_path"])) in argv
    assert job["job_id"] in argv
    assert job["sbatch_path"] in argv


def test_docker_fifo_does_not_block_immediate_parallel_slurm_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = _trusted_slurm_integration(tmp_path)
    docker_script = Path(str(integration["pipeline_script"]))
    docker_script.parent.mkdir(parents=True)
    docker_state = tmp_path / "docker-state"
    docker_state.mkdir()
    docker_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f"DOCKER_STATE={shlex.quote(str(docker_state))}\n"
        "if mkdir \"$DOCKER_STATE/first.once\" 2>/dev/null; then\n"
        "  touch \"$DOCKER_STATE/started\"\n"
        "  while [[ ! -f \"$DOCKER_STATE/release\" ]]; do sleep 0.02; done\n"
        "fi\n",
        encoding="utf-8",
    )

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    counter = tmp_path / "slurm-counter.txt"
    release_slurm = tmp_path / "release-slurm"
    scripts = {
        "sbatch": f"""#!{sys.executable}
import fcntl
import os
from pathlib import Path

path = Path(os.environ["MOCK_SLURM_COUNTER"])
path.touch(exist_ok=True)
with path.open("r+", encoding="utf-8") as stream:
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    raw = stream.read().strip()
    value = int(raw or "0") + 1
    stream.seek(0)
    stream.write(str(value))
    stream.truncate()
print(30000000 + value)
""",
        "squeue": f"""#!{sys.executable}
import os
from pathlib import Path

if not Path(os.environ["MOCK_RELEASE_SLURM"]).is_file():
    print("PENDING")
""",
        "sacct": f"""#!{sys.executable}
import os
import sys
from pathlib import Path

if Path(os.environ["MOCK_RELEASE_SLURM"]).is_file():
    job_id = sys.argv[sys.argv.index("-j") + 1]
    print(f"{{job_id}}|COMPLETED|0:0|2026-01-01|2026-01-01|1|")
""",
    }
    for name, source in scripts.items():
        path = mock_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{mock_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("MOCK_SLURM_COUNTER", str(counter))
    monkeypatch.setenv("MOCK_RELEASE_SLURM", str(release_slurm))

    jobs = PipelineJobs(tmp_path / "state", docker_script, integration)
    docker_first = jobs.start(
        resolve_pipeline_request(
            {"mode": "train_eval", "execution": {"backend": "docker_host"}},
            _version(tmp_path),
            integration,
        )
    )
    deadline = time.monotonic() + 5
    while not (docker_state / "started").is_file():
        assert time.monotonic() < deadline
        time.sleep(0.01)

    docker_second = jobs.start(
        resolve_pipeline_request(
            {
                "mode": "train_eval",
                "execution": {"backend": "docker_host"},
                "training": {"stage2": {"warmup_ratio": 0.05}},
            },
            _version(tmp_path),
            integration,
        )
    )
    slurm_first = jobs.start(
        resolve_pipeline_request(
            {
                "mode": "train",
                "execution": {"backend": "slurm"},
                "training": {
                    "profile": "chronos2-stage1",
                    "stage1": {"learning_rate": "5e-6"},
                },
            },
            _version(tmp_path),
            integration,
        )
    )
    slurm_second = jobs.start(
        resolve_pipeline_request(
            {
                "mode": "train",
                "execution": {"backend": "slurm"},
                "training": {
                    "profile": "chronos2-stage1",
                    "stage1": {"learning_rate": "2e-5"},
                },
            },
            _version(tmp_path),
            integration,
        )
    )

    try:
        deadline = time.monotonic() + 8
        while True:
            slurm_jobs = [
                jobs.get(job_id, include_log=False)
                for job_id in (slurm_first["job_id"], slurm_second["job_id"])
            ]
            if all(
                job.get("scheduler_job_id")
                and job["status"] in {"scheduled", "running"}
                for job in slurm_jobs
            ):
                break
            assert time.monotonic() < deadline, slurm_jobs
            time.sleep(0.01)

        assert counter.read_text(encoding="utf-8") == "2"
        assert len({job["scheduler_job_id"] for job in slurm_jobs}) == 2
        assert all("queue_position" not in job for job in slurm_jobs)
        assert jobs.get(docker_first["job_id"], include_log=False)["status"] == "running"
        queued_docker = jobs.get(docker_second["job_id"], include_log=False)
        assert queued_docker["status"] == "queued"
        assert queued_docker["queue_position"] == 1

        release_slurm.touch()
        deadline = time.monotonic() + 8
        while True:
            states = [
                jobs.get(job_id, include_log=False)["status"]
                for job_id in (slurm_first["job_id"], slurm_second["job_id"])
            ]
            if states == ["completed", "completed"]:
                break
            assert time.monotonic() < deadline, states
            time.sleep(0.01)
        assert jobs.get(docker_first["job_id"], include_log=False)["status"] == "running"
    finally:
        release_slurm.touch(exist_ok=True)
        (docker_state / "release").touch(exist_ok=True)

    deadline = time.monotonic() + 8
    while True:
        docker_states = [
            jobs.get(job_id, include_log=False)["status"]
            for job_id in (docker_first["job_id"], docker_second["job_id"])
        ]
        if docker_states == ["completed", "completed"]:
            break
        assert time.monotonic() < deadline, docker_states
        time.sleep(0.01)


def test_stage1_resolved_yaml_is_accepted_by_training_slurm_contract(
    tmp_path: Path,
) -> None:
    loader = (
        Path(__file__).resolve().parents[2]
        / "ChatTS-Training"
        / "scripts"
        / "slurm"
        / "load_studio_pipeline_config.py"
    )
    if not loader.is_file():
        pytest.skip("sibling ChatTS-Training checkout is not available")
    integration = _trusted_slurm_integration(tmp_path)
    resolved = resolve_pipeline_request(
        {
            "mode": "train",
            "execution": {"backend": "slurm"},
            "training": {"profile": "chronos2-stage1"},
        },
        _version(tmp_path),
        integration,
    )
    job_id = "7" * 32
    config = json.loads(json.dumps(resolved["config"]))
    config["pipeline"]["trial_id"] = job_id
    config["pipeline"]["preflight_only"] = False
    config["pipeline"]["trial_config_hash"] = hashlib.sha256(
        json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path = tmp_path / "studio-stage1.resolved.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(loader), str(path), "--expected-job-id", job_id],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    environment = dict(line.split("=", 1) for line in completed.stdout.splitlines())
    assert environment["PIPELINE_MODE"] == "stage1"
    assert environment["STAGE1_OUT"] == resolved["derived"]["final_model_path"]
    assert environment["KEEP_STAGE1"] == "1"


def test_resolve_pipeline_rejects_path_override_and_invalid_stage_mix(
    tmp_path: Path,
) -> None:
    with pytest.raises(StudioError, match="fixed by the server"):
        resolve_pipeline_request(
            {"training": {"project_root": "/tmp/injected"}},
            _version(tmp_path),
            _integration(tmp_path),
        )

    custom_model = resolve_pipeline_request(
        {"training": {"base_model_path": "/different/valid/model"}},
        _version(tmp_path),
        _integration(tmp_path),
    )
    assert custom_model["config"]["training"]["base_model_path"] == (
        "/different/valid/model"
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


def test_pipeline_jobs_queue_fifo_freezes_configs_and_continues_after_failure(
    tmp_path: Path,
) -> None:
    integration = _integration(tmp_path)
    script = Path(str(integration["pipeline_script"]))
    script.parent.mkdir(parents=True)
    run_state = tmp_path / "fake-pipeline"
    run_state.mkdir()
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f"RUN_STATE={shlex.quote(str(run_state))}\n"
        "printf '%s\\n' \"$(basename \"$CONFIG_FILE\" .yaml)\" >> \"$RUN_STATE/order.txt\"\n"
        "if mkdir \"$RUN_STATE/first.once\" 2>/dev/null; then\n"
        "  touch \"$RUN_STATE/first-started\"\n"
        "  while [[ ! -f \"$RUN_STATE/release-first\" ]]; do sleep 0.02; done\n"
        "  exit 9\n"
        "fi\n"
        "sleep 0.05\n",
        encoding="utf-8",
    )
    jobs = PipelineJobs(tmp_path / "state", script)
    version3 = _version(tmp_path)
    version4 = json.loads(json.dumps(version3))
    version4.update(
        {
            "version": "datav4",
            "snapshot_dir": str(tmp_path / "versions" / "datav4"),
            "dataset_snapshot_hash": "b" * 64,
        }
    )

    first = jobs.start(resolve_pipeline_request({}, version3, integration))
    deadline = time.monotonic() + 5
    while not (run_state / "first-started").is_file():
        assert time.monotonic() < deadline
        time.sleep(0.01)

    second_resolved = resolve_pipeline_request(
        {"training": {"stage2": {"learning_rate": "2e-5"}}},
        version4,
        integration,
    )
    second = jobs.start(second_resolved)
    second_config = Path(second["config_path"])
    second_config_before = second_config.read_bytes()
    # Mutating the caller's object after submission must not change the queued
    # experiment; the resolved YAML is the queue's immutable execution input.
    second_resolved["config"]["training"]["stage2"]["learning_rate"] = "9e-5"
    third = jobs.start(
        resolve_pipeline_request(
            {"training": {"stage2": {"warmup_ratio": 0.05}}},
            version4,
            integration,
        )
    )

    assert jobs.get(first["job_id"], include_log=False)["status"] == "running"
    assert jobs.get(second["job_id"], include_log=False)["queue_position"] == 1
    assert jobs.get(third["job_id"], include_log=False)["queue_position"] == 2
    assert second_config.read_bytes() == second_config_before

    (run_state / "release-first").touch()
    deadline = time.monotonic() + 8
    while True:
        states = {
            item["job_id"]: item["status"]
            for item in jobs.list()
            if item["job_id"] in {first["job_id"], second["job_id"], third["job_id"]}
        }
        if all(value in {"completed", "failed"} for value in states.values()):
            break
        assert time.monotonic() < deadline, states
        time.sleep(0.01)

    assert states == {
        first["job_id"]: "failed",
        second["job_id"]: "completed",
        third["job_id"]: "completed",
    }
    assert (run_state / "order.txt").read_text(encoding="utf-8").splitlines() == [
        first["job_id"],
        second["job_id"],
        third["job_id"],
    ]
    saved_second = yaml.safe_load(second_config.read_text(encoding="utf-8"))
    assert saved_second["pipeline"]["data_version"] == "datav4"
    assert saved_second["pipeline"]["dataset_snapshot_hash"] == "b" * 64
    assert saved_second["training"]["stage2"]["learning_rate"] == "2e-05"


def test_pipeline_jobs_restart_preserves_and_dispatches_queued_job(tmp_path: Path) -> None:
    state = tmp_path / "state"
    jobs_root = state / "jobs"
    configs_root = state / "configs"
    logs_root = state / "logs"
    jobs_root.mkdir(parents=True)
    configs_root.mkdir(parents=True)
    logs_root.mkdir(parents=True)
    script = tmp_path / "ChatTS" / "scripts" / "run_train_then_eval.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\necho restored-ok\n", encoding="utf-8")
    job_id = "3" * 32
    config_path = configs_root / f"{job_id}.yaml"
    config_path.write_text("pipeline:\n  data_version: datav4\n", encoding="utf-8")
    (jobs_root / f"{job_id}.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "kind": "train_eval",
                "status": "queued",
                "created_at": "2026-01-01T00:00:00+00:00",
                "config_path": str(config_path),
                "log_path": str(logs_root / f"{job_id}.log"),
            }
        ),
        encoding="utf-8",
    )

    # A temporarily incomplete host integration must leave durable queue state
    # untouched instead of converting it to a failed/interrupted run.
    unavailable = PipelineJobs(state, None)
    assert unavailable.get(job_id, include_log=False)["status"] == "queued"
    assert unavailable.get(job_id, include_log=False)["queue_position"] == 1

    restored = PipelineJobs(state, script)
    deadline = time.monotonic() + 5
    while restored.get(job_id, include_log=False)["status"] not in {
        "completed",
        "failed",
    }:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    recovered = restored.get(job_id)
    assert recovered["status"] == "completed"
    assert recovered["exit_code"] == 0
    assert "restored-ok" in recovered["log_tail"]


def test_pipeline_jobs_restart_tracks_running_then_dispatches_queued(tmp_path: Path) -> None:
    state = tmp_path / "state"
    jobs_root = state / "jobs"
    configs_root = state / "configs"
    logs_root = state / "logs"
    status_root = state / "worker-status"
    for path in (jobs_root, configs_root, logs_root, status_root):
        path.mkdir(parents=True)
    script = tmp_path / "ChatTS" / "scripts" / "run_train_then_eval.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\necho next-ok\n", encoding="utf-8")
    running_id = "4" * 32
    queued_id = "5" * 32
    queued_config = configs_root / f"{queued_id}.yaml"
    queued_config.write_text("pipeline:\n  data_version: datav4\n", encoding="utf-8")
    running_job = {
        "job_id": running_id,
        "kind": "train_eval",
        "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00",
        "queue_sequence": 1,
        "log_path": str(logs_root / f"{running_id}.log"),
        "pid": os.getpid(),
    }
    queued_job = {
        "job_id": queued_id,
        "kind": "train_eval",
        "status": "queued",
        "created_at": "2026-01-01T00:00:01+00:00",
        "queue_sequence": 2,
        "config_path": str(queued_config),
        "log_path": str(logs_root / f"{queued_id}.log"),
    }
    (jobs_root / f"{running_id}.json").write_text(
        json.dumps(running_job), encoding="utf-8"
    )
    (jobs_root / f"{queued_id}.json").write_text(
        json.dumps(queued_job), encoding="utf-8"
    )
    running_status_path = status_root / f"{running_id}.json"
    running_status_path.write_text(
        json.dumps(
            {
                "job_id": running_id,
                "status": "running",
                "pid": os.getpid(),
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    restored = PipelineJobs(state, script)
    assert restored.get(running_id, include_log=False)["status"] == "running"
    assert restored.get(queued_id, include_log=False)["queue_position"] == 1
    # Simulate the detached worker publishing its terminal status after the
    # Studio process has restarted and attached a recovery watcher.
    running_status_path.write_text(
        json.dumps(
            {
                "job_id": running_id,
                "status": "completed",
                "pid": os.getpid(),
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:01:00+00:00",
                "duration_seconds": 60.0,
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )

    deadline = time.monotonic() + 5
    while restored.get(queued_id, include_log=False)["status"] not in {
        "completed",
        "failed",
    }:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert restored.get(running_id, include_log=False)["status"] == "completed"
    recovered_queued = restored.get(queued_id)
    assert recovered_queued["status"] == "completed"
    assert "next-ok" in recovered_queued["log_tail"]


def test_training_run_exports_data_config_and_diff_from_previous(tmp_path: Path) -> None:
    integration = _integration(tmp_path)
    script = Path(str(integration["pipeline_script"]))
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n", encoding="utf-8")
    version = _version(tmp_path)
    snapshot = Path(str(version["snapshot_dir"]))
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps({"schema_version": "snapshot-v1", "row_count": 12}), encoding="utf-8"
    )
    jobs = PipelineJobs(tmp_path / "state", script)

    first = jobs.start(
        resolve_pipeline_request(
            {"training": {"base_model_path": "/share/model/ChatTS-Qwen3-8B"}},
            version,
            integration,
        )
    )
    deadline = time.monotonic() + 5
    while jobs.get(first["job_id"])["status"] not in {"completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    first_job = jobs.get(first["job_id"], include_log=False)
    first_diff = json.loads(
        Path(first_job["artifacts"]["与上次差异 JSON"]).read_text(encoding="utf-8")
    )
    assert first_diff["has_previous_run"] is False
    assert first_diff["changes"] == []
    assert first_job["diff_from_previous"]["changes"] == first_diff["changes"]
    assert first_job["diff_from_previous"]["change_count"] == 0
    first_data = json.loads(
        Path(first_job["artifacts"]["训练数据清单"]).read_text(encoding="utf-8")
    )
    assert first_data["version"] == "datav3"
    assert first_data["dataset_snapshot_hash"] == "a" * 64
    assert first_data["dataset_names"] == version["dataset_names"]
    assert first_data["snapshot_manifest"]["row_count"] == 12
    assert Path(first_job["artifacts"]["训练参数配置"]).is_file()

    second = jobs.start(
        resolve_pipeline_request(
            {
                "training": {
                    "base_model_path": "/share/model/ChatTS-Qwen3-4B",
                    "stage2": {"learning_rate": "2e-5"},
                }
            },
            version,
            integration,
        )
    )
    deadline = time.monotonic() + 5
    while jobs.get(second["job_id"])["status"] not in {"completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    second_job = jobs.get(second["job_id"], include_log=False)
    second_diff = json.loads(
        Path(second_job["artifacts"]["与上次差异 JSON"]).read_text(encoding="utf-8")
    )
    assert second_diff["has_previous_run"] is True
    assert second_diff["previous_job_id"] == first["job_id"]
    assert second_job["diff_from_previous"]["changes"] == second_diff["changes"][:100]
    assert second_job["diff_from_previous"]["change_count"] == second_diff["change_count"]
    changed = {item["path"]: item for item in second_diff["changes"]}
    assert changed["config.training.base_model_path"]["after"].endswith("Qwen3-4B")
    assert "/ChatTS-msxf-4B-datav3/experiments/recipe-" in changed[
        "config.training.output_root"
    ]["after"]
    assert changed["config.training.stage2.learning_rate"] == {
        "path": "config.training.stage2.learning_rate",
        "before": "1e-05",
        "after": "2e-05",
    }
    diff_report = Path(second_job["artifacts"]["与上次差异报告"]).read_text(
        encoding="utf-8"
    )
    assert "与上一次训练的差异" in diff_report
    assert "config.training.stage2.learning_rate" in diff_report


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
