from __future__ import annotations

import fcntl
import json
import shutil
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .catalog import CatalogCache
from .exporter import canonical_selection, export_selection, parse_rules, preview_selection
from .models import (
    DIFFICULTY_LABELS_ZH,
    DIFFICULTY_LEVELS,
    QUALITY_LABELS_ZH,
    QUALITY_LEVELS,
    StudioError,
)
from .pipeline import PipelineJobs, public_pipeline_defaults, resolve_pipeline_request
from .registry_builder import build_registry
from .training_registry import register_training_version
from .versioning import (
    VersionLedger,
    atomic_write_json,
    normalize_data_version,
    verify_snapshot,
)

STATIC_ROOT = Path(__file__).resolve().parent / "static"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StudioService:
    def __init__(self, defaults: dict[str, Any]):
        self.defaults = defaults
        self.cache = CatalogCache()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        output_root = defaults.get("output_root")
        self.ledger = (
            VersionLedger(output_root, start_number=int(defaults.get("version_start", 3)))
            if isinstance(output_root, str) and output_root
            else None
        )
        state_root = defaults.get("state_root")
        if not isinstance(state_root, str) or not state_root:
            state_root = str(Path(output_root or ".") / ".studio-state")
        self.state_root = Path(state_root).expanduser().resolve()
        self.export_jobs_root = self.state_root / "export-jobs"
        integration = defaults.get("integration")
        self.integration = integration if isinstance(integration, dict) else {}
        self.pipeline_jobs = PipelineJobs(
            self.state_root / "pipeline",
            self.integration.get("pipeline_script"),
            self.integration,
        )
        if defaults.get("registry_auto_build") is True:
            self.rebuild_registry()
        self._restore_export_jobs()

    def rebuild_registry(self) -> dict[str, Any]:
        registry_path = self.defaults.get("registry_path")
        annotations_root = self.defaults.get("annotations_root")
        data_root = self.defaults.get("data_root")
        if not isinstance(registry_path, str) or not registry_path:
            raise StudioError("registry_path is required to build the all-source registry")
        if not isinstance(annotations_root, str) or not annotations_root:
            raise StudioError("annotations_root is required to build the all-source registry")
        result = build_registry(
            annotations_root,
            registry_path,
            data_root=data_root,
            metadata_registry=self.defaults.get("metadata_registry"),
            force=Path(registry_path).exists(),
        )
        self.cache = CatalogCache()
        return result

    def _restore_export_jobs(self) -> None:
        if not self.export_jobs_root.is_dir():
            return
        for path in self.export_jobs_root.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
                continue
            if job.get("status") in {"queued", "running"}:
                job.update(
                    {
                        "status": "failed",
                        "phase": "interrupted",
                        "error": "Dataset Studio restarted before this export completed",
                    }
                )
                atomic_write_json(path, job)
            self.jobs[job["job_id"]] = job

    def _save_export_job(self, job: dict[str, Any]) -> None:
        self.export_jobs_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.export_jobs_root / f"{job['job_id']}.json", job)

    def _ledger(self) -> VersionLedger:
        if self.ledger is None:
            raise StudioError("output_root is required for dataset version management")
        return self.ledger

    def _training_root(self, supplied: Any = None) -> str:
        configured = self.integration.get("training_root")
        if not isinstance(configured, str) or not configured:
            raise StudioError("integration.training_root is not configured")
        if supplied not in (None, "") and str(Path(str(supplied)).expanduser().resolve()) != str(
            Path(configured).expanduser().resolve()
        ):
            raise StudioError("training_root is fixed by the server and cannot be overridden")
        return configured

    def _assert_publish_paths(self, payload: dict[str, Any]) -> None:
        paths = payload.get("paths")
        if paths is None:
            return
        if not isinstance(paths, dict):
            raise StudioError("paths must be an object")
        allowed = {
            "registry_path",
            "annotations_root",
            "data_root",
            "output_root",
            "training_root",
        }
        unknown = sorted(set(paths) - allowed)
        if unknown:
            raise StudioError(f"Unknown publish paths: {unknown}")
        for key in ("registry_path", "annotations_root", "data_root", "output_root"):
            supplied = paths.get(key)
            configured = self.defaults.get(key)
            if supplied in (None, ""):
                continue
            if configured in (None, "") or Path(str(supplied)).expanduser().resolve() != Path(
                str(configured)
            ).expanduser().resolve():
                raise StudioError(f"{key} is fixed by the server and cannot be overridden")
        if "training_root" in paths:
            self._training_root(paths.get("training_root"))

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
        stage1, stage2 = parse_rules(payload, sources, catalog)
        return preview_selection(catalog, stage1, stage2)

    def start_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        registry, annotations, data_root = self._paths(payload)
        sources, catalog = self.cache.get(registry, annotations, data_root)
        # Validate the complete request before returning a job id.
        stage1, stage2 = parse_rules(payload, sources, catalog)
        preview_selection(catalog, stage1, stage2)
        effective = dict(payload)
        effective["output_root"] = payload.get("output_root") or self.defaults.get("output_root")
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "kind": "export",
                "created_at": _utc_now(),
                "status": "queued",
                "phase": "queued",
                "processed_rows": 0,
                "total_rows": 0,
            }
            self._save_export_job(self.jobs[job_id])

        def update(event: dict[str, Any]) -> None:
            with self.lock:
                self.jobs[job_id].update(event)
                if event.get("phase") == "completed":
                    self.jobs[job_id]["status"] = "completed"
                elif self.jobs[job_id]["status"] == "queued":
                    self.jobs[job_id]["status"] = "running"
                self._save_export_job(self.jobs[job_id])

        def worker() -> None:
            try:
                update({"status": "running", "phase": "preparing"})
                result = export_selection(effective, sources, catalog, update)
                update({"status": "completed", "result": result})
            except Exception as exc:  # noqa: BLE001 - background jobs must expose failures.
                update({"status": "failed", "phase": "failed", "error": str(exc)})

        threading.Thread(target=worker, name=f"dataset-export-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id, "status": "queued"}

    def start_publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._assert_publish_paths(payload)
        recipe = payload.get("recipe")
        if not isinstance(recipe, dict):
            recipe = {"stage1": payload.get("stage1"), "stage2": payload.get("stage2")}
        effective = {"stage1": recipe.get("stage1"), "stage2": recipe.get("stage2")}
        registry, annotations, data_root = self._paths({})
        sources, catalog = self.cache.get(registry, annotations, data_root)
        stage1, stage2 = parse_rules(effective, sources, catalog)
        preview_selection(catalog, stage1, stage2)
        expected_selection = canonical_selection(stage1, stage2)

        requested = payload.get("version")
        automatic = requested in (None, "", "auto")
        canonical = None if automatic else normalize_data_version(requested)
        notes = payload.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 4_000:
            raise StudioError("notes must be a string up to 4000 characters")
        parent = payload.get("parent")
        if parent in (None, ""):
            parent = self._ledger().state().get("active_version")
        elif parent is not None:
            parent = normalize_data_version(parent)
        if parent is not None:
            self._ledger().get(parent)

        def flag(name: str, default: bool) -> bool:
            value = payload.get(name, default)
            if isinstance(value, bool):
                return value
            if value in (0, 1, "0", "1"):
                return bool(int(value))
            raise StudioError(f"{name} must be true/false or 0/1")

        register = flag("register", True)
        activate = flag("activate", True)
        if activate:
            register = True

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "kind": "publish",
            "created_at": _utc_now(),
            "status": "queued",
            "phase": "queued",
            "processed_rows": 0,
            "total_rows": 0,
            "requested_version": canonical or "auto",
        }
        with self.lock:
            self.jobs[job_id] = job
            self._save_export_job(job)

        def update(event: dict[str, Any]) -> None:
            with self.lock:
                self.jobs[job_id].update(event)
                if event.get("phase") == "completed":
                    self.jobs[job_id]["status"] = "completed"
                elif self.jobs[job_id]["status"] == "queued":
                    self.jobs[job_id]["status"] = "running"
                self._save_export_job(self.jobs[job_id])

        def worker() -> None:
            publish_lock = self._ledger().root / ".publish.lock"
            self._ledger().root.mkdir(parents=True, exist_ok=True)
            try:
                update({"status": "running", "phase": "preparing"})

                def export_update(event: dict[str, Any]) -> None:
                    # export_selection's own "completed" only means files are
                    # written. Publication still has verification, ledger and
                    # optional Training registration gates to pass.
                    if event.get("phase") == "completed":
                        update({"phase": "verifying", "processed_rows": event.get("processed_rows", 0)})
                    else:
                        update(event)

                with publish_lock.open("a+b") as lock_stream:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                    try:
                        ledger = self._ledger()
                        version = canonical or ledger.next()
                        export_payload = {
                            **effective,
                            "run_name": version,
                            "data_version": version,
                            "output_root": str(ledger.root),
                        }
                        target = ledger.root / version
                        prior_entries = ledger.list()
                        prior_version = next(
                            (item for item in prior_entries if item["version"] == version),
                            None,
                        )
                        if target.exists() or target.is_symlink():
                            # A finalized export can outlive a crash before the
                            # ledger write.  Never replace or remove such a
                            # directory: verify it completely, then adopt it
                            # only when it is exactly this version and recipe.
                            if target.is_symlink():
                                raise StudioError(
                                    f"Existing publish destination is a symbolic link; "
                                    f"left untouched: {target}"
                                )
                            verified = verify_snapshot(target)
                            if (
                                verified.get("run_name") != version
                                or verified.get("data_version") != version
                            ):
                                raise StudioError(
                                    f"Existing publish destination has different version "
                                    f"metadata; left untouched: {target}"
                                )
                            if verified.get("selection") != expected_selection:
                                raise StudioError(
                                    f"Existing publish destination has a different recipe; "
                                    f"left untouched: {target}"
                                )
                            if prior_version is None:
                                entry = ledger.record(
                                    target,
                                    version=version,
                                    notes=notes,
                                    parent=parent,
                                    verified_snapshot=verified,
                                )
                                entry = {
                                    **entry,
                                    "idempotent": True,
                                    "adopted_orphan": True,
                                }
                            else:
                                entry = {
                                    **ledger.verify(
                                        version, verified_snapshot=verified
                                    ),
                                    "idempotent": True,
                                }
                        else:
                            result = export_selection(
                                export_payload, sources, catalog, export_update
                            )
                            verified = verify_snapshot(result["output_dir"])
                            duplicate = next(
                                (
                                    item
                                    for item in prior_entries
                                    if item["dataset_snapshot_hash"]
                                    == verified["dataset_snapshot_hash"]
                                ),
                                None,
                            )
                            if duplicate is not None:
                                # This directory was created by this worker, so
                                # it is safe to remove.  Pre-existing/orphaned
                                # destinations are never deleted above.
                                shutil.rmtree(Path(result["output_dir"]))
                                verified = ledger.verify(duplicate["version"])
                                entry = {**duplicate, "idempotent": True}
                            else:
                                entry = ledger.record(
                                    result["output_dir"],
                                    version=version,
                                    notes=notes,
                                    parent=parent,
                                    verified_snapshot=verified,
                                )
                        registration = None
                        if register:
                            registration = register_training_version(
                                self._training_root(),
                                ledger.root,
                                entry["version"],
                                model_output_base=self.integration.get("model_output_base"),
                                activate=activate,
                                verified_snapshot=verified,
                            )
                    finally:
                        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
                update(
                    {
                        "status": "completed",
                        "phase": "completed",
                        "version": entry["version"],
                        "result": {
                            "version": entry,
                            "registration": registration,
                            "idempotent": entry.get("idempotent", False),
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001 - persisted background failure.
                update({"status": "failed", "phase": "failed", "error": str(exc)})

        threading.Thread(target=worker, name=f"publish-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id, "status": "queued"}

    def versions(self) -> dict[str, Any]:
        state = self._ledger().state()
        training_root = self.integration.get("training_root")
        registry_root = (
            Path(training_root).expanduser().resolve() / "data" / "studio_versions"
            if isinstance(training_root, str) and training_root
            else None
        )
        versions = []
        for entry in reversed(state["versions"]):
            version = entry["version"]
            registered = bool(
                registry_root
                and (registry_root / f"{version}.json").is_file()
                and (registry_root / f"{version}.env").is_file()
            )
            versions.append(
                {
                    **entry,
                    "registered": registered,
                    "active": version == state.get("active_version"),
                }
            )
        return {
            "versions": versions,
            "active_version": state.get("active_version"),
            "next_version": self._ledger().next(),
        }

    def version(self, version: str) -> dict[str, Any]:
        return self._ledger().get(version)

    def register_version(self, payload: dict[str, Any], *, activate: bool = False) -> dict[str, Any]:
        version = normalize_data_version(payload.get("version"))
        supplied_root = payload.get("training_root")
        requested_activate = payload.get("activate", False)
        if not isinstance(requested_activate, bool):
            if requested_activate in (0, 1, "0", "1"):
                requested_activate = bool(int(requested_activate))
            else:
                raise StudioError("activate must be true/false or 0/1")
        return register_training_version(
            self._training_root(supplied_root),
            self._ledger().root,
            version,
            model_output_base=self.integration.get("model_output_base"),
            activate=activate or requested_activate,
        )

    def start_pipeline(self, payload: dict[str, Any], *, preflight: bool) -> dict[str, Any]:
        version_value = payload.get("version") or self._ledger().state().get("active_version")
        if version_value is None:
            raise StudioError("Select or activate a dataset version before launching")
        version = normalize_data_version(version_value)
        entry = self._ledger().verify(version)
        register_training_version(
            self._training_root(),
            self._ledger().root,
            version,
            model_output_base=self.integration.get("model_output_base"),
            activate=True,
            verified_snapshot=entry,
        )
        resolved = resolve_pipeline_request({**payload, "version": version}, entry, self.integration)
        return self.pipeline_jobs.start(resolved, preflight=preflight)

    def list_jobs(self) -> dict[str, Any]:
        with self.lock:
            exports = [dict(job) for job in self.jobs.values()]
        jobs = exports + self.pipeline_jobs.list()
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return {"jobs": jobs}

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id in self.jobs:
                return dict(self.jobs[job_id])
        try:
            return self.pipeline_jobs.get(job_id)
        except StudioError as exc:
            raise StudioError(f"Unknown export job: {job_id}") from exc

    def public_defaults(self) -> dict[str, Any]:
        safe_defaults = {
            key: value
            for key, value in self.defaults.items()
            if key not in {"integration"}
        }
        available_sources: list[str] = []
        try:
            registry, annotations, data_root = self._paths({})
            _, catalog = self.cache.get(registry, annotations, data_root)
            available_sources = [
                item["name"]
                for item in catalog.get("sources", [])
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and item.get("available") is True
            ]
        except Exception:  # noqa: BLE001 - legacy defaults must remain available.
            available_sources = []
        return {
            **safe_defaults,
            "quality_levels": list(QUALITY_LEVELS),
            "difficulty_levels": list(DIFFICULTY_LEVELS),
            "quality_labels_zh": QUALITY_LABELS_ZH,
            "difficulty_labels_zh": DIFFICULTY_LABELS_ZH,
            "target_sources": available_sources,
            "pipeline": public_pipeline_defaults(self.integration),
            "versions": self.versions() if self.ledger is not None else None,
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
            elif parsed.path == "/api/versions":
                self._json(self.server.service.versions())
            elif parsed.path == "/api/versions/next":
                self._json({"version": self.server.service._ledger().next()})
            elif parsed.path.startswith("/api/versions/"):
                self._json(self.server.service.version(parsed.path.rsplit("/", 1)[-1]))
            elif parsed.path == "/api/jobs":
                self._json(self.server.service.list_jobs())
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
            elif parsed.path == "/api/registry/rebuild":
                result = self.server.service.rebuild_registry()
            elif parsed.path == "/api/preview":
                result = self.server.service.preview(payload)
            elif parsed.path == "/api/export":
                result = self.server.service.start_export(payload)
            elif parsed.path == "/api/versions/publish":
                result = self.server.service.start_publish(payload)
            elif parsed.path == "/api/versions/register":
                result = self.server.service.register_version(payload)
            elif parsed.path == "/api/versions/activate":
                result = self.server.service.register_version(payload, activate=True)
            elif parsed.path == "/api/versions/verify":
                result = self.server.service._ledger().verify(payload.get("version"))
            elif parsed.path == "/api/runs/preflight":
                result = self.server.service.start_pipeline(payload, preflight=True)
            elif parsed.path == "/api/runs":
                result = self.server.service.start_pipeline(payload, preflight=False)
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
