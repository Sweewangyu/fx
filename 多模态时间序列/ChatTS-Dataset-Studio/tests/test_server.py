from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from chatts_dataset_studio.catalog import CatalogCache
from chatts_dataset_studio.exporter import export_selection
from chatts_dataset_studio.server import StudioHTTPServer, StudioService


def _json_request(base_url: str, path: str, payload: dict[str, Any] | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=2) as response:  # noqa: S310 - loopback fixture server.
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_service_job(service: StudioService, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while True:
        job = service.get_job(job_id)
        if job["status"] in {"completed", "failed"}:
            return job
        assert time.monotonic() < deadline, job
        time.sleep(0.01)


def _versioned_orphan(
    labeled_corpus: dict[str, Any], selection: dict[str, Any], version: str = "datav3"
) -> Path:
    sources, catalog = CatalogCache().get(
        labeled_corpus["registry_path"],
        labeled_corpus["annotations_root"],
        labeled_corpus["data_root"],
    )
    result = export_selection(
        {
            **selection,
            "run_name": version,
            "data_version": version,
            "output_root": str(labeled_corpus["output_root"]),
        },
        sources,
        catalog,
    )
    return Path(result["output_dir"])


def test_real_http_api_catalog_preview_and_background_export(
    labeled_corpus: dict[str, Any],
    default_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatts_dataset_studio.pipeline.shutil.which", lambda _: "/bin/docker")
    defaults = {
        "registry_path": str(labeled_corpus["registry_path"]),
        "annotations_root": str(labeled_corpus["annotations_root"]),
        "data_root": str(labeled_corpus["data_root"]),
        "output_root": str(labeled_corpus["output_root"]),
    }
    server = StudioHTTPServer(("127.0.0.1", 0), StudioService(defaults))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, health = _json_request(base_url, "/api/health")
        assert status == 200
        assert health == {"status": "ok"}

        status, public_defaults = _json_request(base_url, "/api/defaults")
        assert status == 200
        assert public_defaults["target_sources"] == list(labeled_corpus["sources"])
        assert public_defaults["pipeline"]["integration"]["enabled"] is False
        disabled_reasons = public_defaults["pipeline"]["integration"][
            "disabled_reasons"
        ]
        assert disabled_reasons[:3] == [
            "integration.pipeline_script is not configured",
            "integration.training_root is not configured",
            "integration.evaluation_root is not configured",
        ]
        assert "integration.train_project_root is not configured" in disabled_reasons
        assert "integration.timeseriesexam_data_file is not configured" in disabled_reasons
        assert not any("base_model_path" in reason for reason in disabled_reasons)
        assert public_defaults["presets"]["stage1"]["difficulties"] == [
            "very_easy",
            "easy",
            "moderate",
        ]
        assert public_defaults["presets"]["stage2"]["difficulties"] == [
            "moderate",
            "hard",
            "very_hard",
        ]

        status, catalog = _json_request(base_url, "/api/catalog", {})
        assert status == 200
        assert catalog["available_sources"] == 6

        status, preview = _json_request(base_url, "/api/preview", default_selection)
        assert status == 200
        assert preview["counts"] == {"stage1": 12, "stage2": 18, "overlap": 6}

        export_payload = {
            **default_selection,
            "run_name": "api-export",
        }
        status, started = _json_request(base_url, "/api/export", export_payload)
        assert status == 200
        assert started["status"] == "queued"

        deadline = time.monotonic() + 5
        while True:
            status, job = _json_request(base_url, f"/api/jobs/{started['job_id']}")
            assert status == 200
            if job["status"] in {"completed", "failed"}:
                break
            assert time.monotonic() < deadline, job
            time.sleep(0.01)
        assert job["status"] == "completed", job
        assert job["result"]["counts"] == {"stage1": 12, "stage2": 18, "overlap": 6}
        assert Path(job["result"]["output_dir"]).is_dir()

        status, error = _json_request(base_url, "/api/jobs/not-a-job")
        assert status == 400
        assert "Unknown export job" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_public_defaults_returns_empty_legacy_sources_when_catalog_is_unavailable(
    tmp_path: Path,
) -> None:
    service = StudioService({"state_root": str(tmp_path / "state")})

    assert service.public_defaults()["target_sources"] == []


def test_publish_adopts_matching_verified_orphan_idempotently(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    recipe = json.loads(json.dumps(default_selection))
    source = labeled_corpus["sources"][0]
    recipe["stage1"]["source_rules"] = {
        source: {
            "qualities": ["good"],
            "difficulties": ["hard"],
            "abilities": ["anomaly_detection"],
        }
    }
    orphan = _versioned_orphan(labeled_corpus, recipe)
    manifest_before = (orphan / "manifest.json").read_bytes()
    service = StudioService(
        {
            "registry_path": str(labeled_corpus["registry_path"]),
            "annotations_root": str(labeled_corpus["annotations_root"]),
            "data_root": str(labeled_corpus["data_root"]),
            "output_root": str(labeled_corpus["output_root"]),
            "state_root": str(labeled_corpus["tmp_path"] / "orphan-state"),
        }
    )

    started = service.start_publish(
        {
            "version": "datav3",
            "recipe": recipe,
            "register": False,
            "activate": False,
        }
    )
    job = _wait_service_job(service, started["job_id"])

    assert job["status"] == "completed", job
    assert job["result"]["idempotent"] is True
    assert job["result"]["version"]["adopted_orphan"] is True
    assert (orphan / "manifest.json").read_bytes() == manifest_before
    assert [entry["version"] for entry in service.versions()["versions"]] == ["datav3"]

    repeated = service.start_publish(
        {
            "version": "datav3",
            "recipe": recipe,
            "register": False,
            "activate": False,
        }
    )
    repeated_job = _wait_service_job(service, repeated["job_id"])
    assert repeated_job["status"] == "completed", repeated_job
    assert repeated_job["result"]["idempotent"] is True
    assert (orphan / "manifest.json").read_bytes() == manifest_before


def test_publish_preserves_valid_orphan_with_different_recipe(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    changed = json.loads(json.dumps(default_selection))
    changed["stage1"]["qualities"] = ["acceptable", "good", "excellent"]
    orphan = _versioned_orphan(labeled_corpus, changed)
    manifest_before = (orphan / "manifest.json").read_bytes()
    service = StudioService(
        {
            "registry_path": str(labeled_corpus["registry_path"]),
            "annotations_root": str(labeled_corpus["annotations_root"]),
            "data_root": str(labeled_corpus["data_root"]),
            "output_root": str(labeled_corpus["output_root"]),
            "state_root": str(labeled_corpus["tmp_path"] / "mismatch-state"),
        }
    )

    started = service.start_publish(
        {
            "version": "datav3",
            "recipe": default_selection,
            "register": False,
            "activate": False,
        }
    )
    job = _wait_service_job(service, started["job_id"])

    assert job["status"] == "failed"
    assert "different recipe" in job["error"]
    assert orphan.is_dir()
    assert (orphan / "manifest.json").read_bytes() == manifest_before
    assert service.versions()["versions"] == []


def test_publish_preserves_tampered_orphan_and_refuses_adoption(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
    orphan = _versioned_orphan(labeled_corpus, default_selection)
    tampered = orphan / "stage1" / "chatts_sft.jsonl"
    tampered.write_bytes(tampered.read_bytes() + b"{}\n")
    tampered_before = tampered.read_bytes()
    service = StudioService(
        {
            "registry_path": str(labeled_corpus["registry_path"]),
            "annotations_root": str(labeled_corpus["annotations_root"]),
            "data_root": str(labeled_corpus["data_root"]),
            "output_root": str(labeled_corpus["output_root"]),
            "state_root": str(labeled_corpus["tmp_path"] / "tampered-state"),
        }
    )

    started = service.start_publish(
        {
            "version": "datav3",
            "recipe": default_selection,
            "register": False,
            "activate": False,
        }
    )
    job = _wait_service_job(service, started["job_id"])

    assert job["status"] == "failed"
    assert "SHA256 mismatch" in job["error"]
    assert orphan.is_dir()
    assert tampered.read_bytes() == tampered_before
    assert service.versions()["versions"] == []


def test_version_publish_register_activate_and_pipeline_preflight_api(
    labeled_corpus: dict[str, Any],
    default_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatts_dataset_studio.pipeline.shutil.which", lambda _: "/bin/docker")
    training_root = labeled_corpus["tmp_path"] / "ChatTS-Training"
    (training_root / "data").mkdir(parents=True)
    evaluation_root = labeled_corpus["tmp_path"] / "ChatTS"
    pipeline_script = evaluation_root / "scripts" / "run_train_then_eval.sh"
    pipeline_script.parent.mkdir(parents=True)
    pipeline_script.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        "test -f \"$CONFIG_FILE\"\n"
        "echo server-preflight-ok\n",
        encoding="utf-8",
    )
    defaults = {
        "registry_path": str(labeled_corpus["registry_path"]),
        "annotations_root": str(labeled_corpus["annotations_root"]),
        "data_root": str(labeled_corpus["data_root"]),
        "output_root": str(labeled_corpus["output_root"]),
        "state_root": str(labeled_corpus["tmp_path"] / "studio-state"),
        "version_start": 3,
        "integration": {
            "training_root": str(training_root),
            "evaluation_root": str(evaluation_root),
            "pipeline_script": str(pipeline_script),
            "train_project_root": "/workspace/ChatTS-Training",
            "eval_project_root": "/workspace/ChatTS/ChatTS-main",
            "training_script": (
                "/workspace/ChatTS-Training/scripts/full/"
                "run_chronos2_best_two_stage.sh"
            ),
            "evaluation_script": (
                "/workspace/ChatTS/ChatTS-main/scripts/run_all_chatts_benchmarks.sh"
            ),
            "base_model_path": "/share/model/ChatTS-Qwen3-8B",
            "model_output_base": "/share/output/ChatTS-msxf-8B-datav1",
            "evaluation_output_base": "/share/evaluation/all-benchmarks",
            "train_chronos2_model_path": "/workspace/chronos2",
            "eval_chronos2_model_path": "/workspace/chronos2",
            "tsrbench_root": "/share/TSRBench-dataset",
            "tinybench_dataset_root": "/share/tyb",
            "ts_haystack_root": "/workspace/TS-Haystack",
            "timeseriesexam_root": "/workspace/TimeSeriesExam",
            "timeseriesexam_data_file": "/workspace/TimeSeriesExam/output/qa.json",
        },
    }
    server = StudioHTTPServer(("127.0.0.1", 0), StudioService(defaults))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, started = _json_request(
            base_url,
            "/api/versions/publish",
            {
                "version": "data-v3",
                "notes": "first all-source recipe",
                "register": True,
                "activate": True,
                "recipe": default_selection,
            },
        )
        assert status == 200
        deadline = time.monotonic() + 5
        while True:
            status, publish_job = _json_request(
                base_url, f"/api/jobs/{started['job_id']}"
            )
            assert status == 200
            if publish_job["status"] in {"completed", "failed"}:
                break
            assert time.monotonic() < deadline, publish_job
            time.sleep(0.01)
        assert publish_job["status"] == "completed", publish_job
        assert publish_job["version"] == "datav3"

        status, versions = _json_request(base_url, "/api/versions")
        assert status == 200
        assert versions["active_version"] == "datav3"
        assert versions["next_version"] == "datav4"
        assert versions["versions"][0]["notes"] == "first all-source recipe"
        profile = training_root / "data" / "studio_versions" / "datav3.json"
        assert profile.is_file()

        status, public_defaults = _json_request(base_url, "/api/defaults")
        assert status == 200
        pipeline_integration = public_defaults["pipeline"]["integration"]
        assert pipeline_integration["enabled"] is True
        assert pipeline_integration["disabled_reasons"] == []

        status, duplicate_started = _json_request(
            base_url,
            "/api/versions/publish",
            {
                "version": "datav4",
                "notes": "same recipe must not create a fake version",
                "register": True,
                "activate": True,
                "recipe": default_selection,
            },
        )
        assert status == 200
        deadline = time.monotonic() + 5
        while True:
            status, duplicate_job = _json_request(
                base_url, f"/api/jobs/{duplicate_started['job_id']}"
            )
            assert status == 200
            if duplicate_job["status"] in {"completed", "failed"}:
                break
            assert time.monotonic() < deadline, duplicate_job
            time.sleep(0.01)
        assert duplicate_job["status"] == "completed", duplicate_job
        assert duplicate_job["result"]["version"]["version"] == "datav3"
        assert duplicate_job["result"]["idempotent"] is True
        assert not (labeled_corpus["output_root"] / "datav4").exists()

        changed_recipe = json.loads(json.dumps(default_selection))
        changed_recipe["stage1"]["difficulties"] = [
            "very_easy",
            "easy",
            "moderate",
            "hard",
        ]
        status, second_started = _json_request(
            base_url,
            "/api/versions/publish",
            {
                "version": "datav4",
                "notes": "historical run target",
                "register": False,
                "activate": False,
                "recipe": changed_recipe,
            },
        )
        assert status == 200
        deadline = time.monotonic() + 5
        while True:
            status, second_job = _json_request(
                base_url, f"/api/jobs/{second_started['job_id']}"
            )
            assert status == 200
            if second_job["status"] in {"completed", "failed"}:
                break
            assert time.monotonic() < deadline, second_job
            time.sleep(0.01)
        assert second_job["status"] == "completed", second_job
        assert server.service.versions()["active_version"] == "datav3"

        status, preflight = _json_request(
            base_url,
            "/api/runs/preflight",
            {
                "mode": "train_eval",
                "version": "datav4",
                "training": {"seed": 42},
                "evaluation": {"benchmarks": ["tsrbench"]},
            },
        )
        assert status == 200
        deadline = time.monotonic() + 5
        while True:
            status, run_job = _json_request(base_url, f"/api/jobs/{preflight['job_id']}")
            assert status == 200
            if run_job["status"] in {"completed", "failed"}:
                break
            assert time.monotonic() < deadline, run_job
            time.sleep(0.01)
        assert run_job["status"] == "completed", run_job
        assert "server-preflight-ok" in run_job["log_tail"]
        assert server.service.versions()["active_version"] == "datav3"
        assert (
            training_root / "data" / "studio_versions" / "datav4.json"
        ).is_file()
        status, jobs = _json_request(base_url, "/api/jobs")
        assert status == 200
        assert {item["kind"] for item in jobs["jobs"]} >= {"publish", "preflight"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
