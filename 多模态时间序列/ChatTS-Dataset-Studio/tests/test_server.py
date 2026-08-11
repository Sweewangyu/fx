from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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


def test_real_http_api_catalog_preview_and_background_export(
    labeled_corpus: dict[str, Any], default_selection: dict[str, Any]
) -> None:
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
