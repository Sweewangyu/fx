from __future__ import annotations

import json
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .catalog import CatalogCache
from .exporter import export_selection, parse_rules, preview_selection
from .models import (
    DEFAULT_TARGET_SOURCES,
    DIFFICULTY_LABELS_ZH,
    DIFFICULTY_LEVELS,
    QUALITY_LABELS_ZH,
    QUALITY_LEVELS,
    StudioError,
)

STATIC_ROOT = Path(__file__).resolve().parent / "static"


class StudioService:
    def __init__(self, defaults: dict[str, Any]):
        self.defaults = defaults
        self.cache = CatalogCache()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def _paths(self, payload: dict[str, Any]) -> tuple[str, str, str | None]:
        registry = payload.get("registry_path") or self.defaults.get("registry_path")
        annotations = payload.get("annotations_root") or self.defaults.get("annotations_root")
        data_root = payload.get("data_root") or self.defaults.get("data_root")
        if not isinstance(registry, str) or not registry:
            raise StudioError("registry_path is required")
        if not isinstance(annotations, str) or not annotations:
            raise StudioError("annotations_root is required")
        if data_root is not None and not isinstance(data_root, str):
            raise StudioError("data_root must be a path string or null")
        return registry, annotations, data_root

    def catalog(self, payload: dict[str, Any]) -> dict[str, Any]:
        registry, annotations, data_root = self._paths(payload)
        _, catalog = self.cache.get(registry, annotations, data_root)
        return catalog

    def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        registry, annotations, data_root = self._paths(payload)
        sources, catalog = self.cache.get(registry, annotations, data_root)
        stage1, stage2 = parse_rules(payload, sources)
        return preview_selection(catalog, stage1, stage2)

    def start_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        registry, annotations, data_root = self._paths(payload)
        sources, catalog = self.cache.get(registry, annotations, data_root)
        # Validate the complete request before returning a job id.
        stage1, stage2 = parse_rules(payload, sources)
        preview_selection(catalog, stage1, stage2)
        effective = dict(payload)
        effective["output_root"] = payload.get("output_root") or self.defaults.get("output_root")
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "phase": "queued",
                "processed_rows": 0,
                "total_rows": 0,
            }

        def update(event: dict[str, Any]) -> None:
            with self.lock:
                self.jobs[job_id].update(event)
                if event.get("phase") == "completed":
                    self.jobs[job_id]["status"] = "completed"
                elif self.jobs[job_id]["status"] == "queued":
                    self.jobs[job_id]["status"] = "running"

        def worker() -> None:
            try:
                update({"status": "running", "phase": "preparing"})
                result = export_selection(effective, sources, catalog, update)
                update({"status": "completed", "result": result})
            except Exception as exc:  # noqa: BLE001 - background jobs must expose failures.
                update({"status": "failed", "phase": "failed", "error": str(exc)})

        threading.Thread(target=worker, name=f"dataset-export-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id, "status": "queued"}

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise StudioError(f"Unknown export job: {job_id}")
            return dict(self.jobs[job_id])

    def public_defaults(self) -> dict[str, Any]:
        return {
            **self.defaults,
            "quality_levels": list(QUALITY_LEVELS),
            "difficulty_levels": list(DIFFICULTY_LEVELS),
            "quality_labels_zh": QUALITY_LABELS_ZH,
            "difficulty_labels_zh": DIFFICULTY_LABELS_ZH,
            "target_sources": list(DEFAULT_TARGET_SOURCES),
            "presets": {
                "stage1": {
                    "qualities": list(QUALITY_LEVELS[1:]),
                    "difficulties": list(DIFFICULTY_LEVELS[:3]),
                },
                "stage2": {
                    "qualities": list(QUALITY_LEVELS[1:]),
                    "difficulties": list(DIFFICULTY_LEVELS[2:]),
                },
            },
        }


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdigit():
            raise StudioError("A JSON request body is required")
        length = int(raw_length)
        if length > 2_000_000:
            raise StudioError("Request body is too large")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StudioError(f"Invalid request JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise StudioError("Request body must be one JSON object")
        return value

    def _static(self, name: str) -> None:
        content_types = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }
        if name not in content_types:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = STATIC_ROOT / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        encoded = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types[name])
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._static("index.html")
            elif parsed.path == "/styles.css":
                self._static("styles.css")
            elif parsed.path == "/app.js":
                self._static("app.js")
            elif parsed.path == "/api/health":
                self._json({"status": "ok"})
            elif parsed.path == "/api/defaults":
                self._json(self.server.service.public_defaults())
            elif parsed.path.startswith("/api/jobs/"):
                self._json(self.server.service.get_job(parsed.path.rsplit("/", 1)[-1]))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except StudioError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/catalog":
                result = self.server.service.catalog(payload)
            elif parsed.path == "/api/preview":
                result = self.server.service.preview(payload)
            elif parsed.path == "/api/export":
                result = self.server.service.start_export(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(result)
        except StudioError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except OSError as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)


class StudioHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        service: StudioService,
        *,
        verbose: bool = False,
    ):
        super().__init__(address, StudioHandler)
        self.service = service
        self.verbose = verbose


def serve(
    defaults: dict[str, Any],
    host: str = "127.0.0.1",
    port: int = 7865,
    *,
    open_browser: bool = False,
    verbose: bool = False,
) -> None:
    service = StudioService(defaults)
    server = StudioHTTPServer((host, port), service, verbose=verbose)
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{display_host}:{actual_port}"
    print(f"ChatTS Dataset Studio: {url}", flush=True)
    print("Press Ctrl-C to stop. Exports continue only while this process is running.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
