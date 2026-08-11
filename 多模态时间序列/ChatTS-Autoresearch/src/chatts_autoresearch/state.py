from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import canonical_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


EXPERIMENT_COLUMNS = (
    "id",
    "kind",
    "phase",
    "status",
    "parent_id",
    "config_hash",
    "dataset_hash",
    "protocol_hash",
    "command_hash",
    "config_json",
    "command_json",
    "metrics_json",
    "model_path",
    "output_dir",
    "error",
    "created_at",
    "started_at",
    "completed_at",
)


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
                    parent_id TEXT,
                    config_hash TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    command_hash TEXT,
                    config_json TEXT NOT NULL,
                    command_json TEXT,
                    metrics_json TEXT,
                    model_path TEXT,
                    output_dir TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY(parent_id) REFERENCES experiments(id)
                );
                CREATE INDEX IF NOT EXISTS experiments_status_idx ON experiments(status, phase);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    event TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    purpose TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS labels (
                    sample_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    record_hash TEXT NOT NULL,
                    quality REAL NOT NULL,
                    difficulty TEXT NOT NULL,
                    taxonomy TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS labels_source_idx ON labels(source);
                CREATE TABLE IF NOT EXISTS template_labels (
                    template_id TEXT PRIMARY KEY,
                    quality REAL NOT NULL,
                    difficulty TEXT NOT NULL,
                    taxonomy TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(labels)")}
            if "template_id" not in columns:
                db.execute("ALTER TABLE labels ADD COLUMN template_id TEXT NOT NULL DEFAULT ''")
            db.execute("CREATE INDEX IF NOT EXISTS labels_template_idx ON labels(template_id)")

    def create_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = {
            "status": "pending",
            "created_at": utc_now(),
            "parent_id": None,
            "command_hash": None,
            "command_json": None,
            "metrics_json": None,
            "model_path": None,
            "error": None,
            "started_at": None,
            "completed_at": None,
            **payload,
        }
        values["config_json"] = canonical_json(values["config_json"])
        if values["command_json"] is not None:
            values["command_json"] = canonical_json(values["command_json"])
        with self.connect() as db:
            db.execute(
                f"INSERT OR IGNORE INTO experiments ({','.join(EXPERIMENT_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in EXPERIMENT_COLUMNS)})",
                [values[column] for column in EXPERIMENT_COLUMNS],
            )
        return self.get_experiment(values["id"])

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return self._decode_experiment(row)

    def list_experiments(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM experiments ORDER BY created_at, id").fetchall()
        return [self._decode_experiment(row) for row in rows]

    @staticmethod
    def _decode_experiment(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        for key in ("config_json", "command_json", "metrics_json"):
            if payload.get(key):
                payload[key.removesuffix("_json")] = json.loads(payload[key])
            payload.pop(key, None)
        return payload

    def mark_running(self, experiment_id: str, command_hash: str, command: dict[str, Any]) -> None:
        with self.connect() as db:
            current = db.execute("SELECT status FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if current is None:
                raise KeyError(experiment_id)
            if current["status"] == "completed":
                return
            db.execute(
                "UPDATE experiments SET status='running', command_hash=?, command_json=?, "
                "started_at=?, completed_at=NULL, error=NULL WHERE id=?",
                (command_hash, canonical_json(command), utc_now(), experiment_id),
            )
            self._event(db, experiment_id, "running", {"command_hash": command_hash})

    def mark_completed(
        self, experiment_id: str, metrics: dict[str, Any], model_path: str | None = None
    ) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE experiments SET status='completed', metrics_json=?, model_path=?, "
                "completed_at=?, error=NULL WHERE id=?",
                (canonical_json(metrics), model_path, utc_now(), experiment_id),
            )
            self._event(db, experiment_id, "completed", {"model_path": model_path})

    def mark_failed(self, experiment_id: str, error: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE experiments SET status='failed', error=?, completed_at=? WHERE id=?",
                (error[-10000:], utc_now(), experiment_id),
            )
            self._event(db, experiment_id, "failed", {"error": error[-2000:]})

    @staticmethod
    def _event(
        db: sqlite3.Connection, experiment_id: str | None, event: str, payload: Any
    ) -> None:
        db.execute(
            "INSERT INTO events(experiment_id,event,payload_json,created_at) VALUES(?,?,?,?)",
            (experiment_id, event, canonical_json(payload), utc_now()),
        )

    def cache_get(self, key: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT response_json FROM llm_cache WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, purpose: str, request_hash: str, response: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES(?,?,?,?,?)",
                (key, purpose, request_hash, canonical_json(response), utc_now()),
            )

    def label_get(
        self, sample_id: str, prompt_version: str, model: str
    ) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM labels WHERE sample_id=? AND prompt_version=? AND model=?",
                (sample_id, prompt_version, model),
            ).fetchone()
        return dict(row) if row else None

    def label_put(self, label: dict[str, Any]) -> None:
        self.label_put_many([label])

    def label_put_many(self, labels: list[dict[str, Any]]) -> None:
        if not labels:
            return
        columns = (
            "sample_id",
            "template_id",
            "source",
            "record_hash",
            "quality",
            "difficulty",
            "taxonomy",
            "rationale",
            "prompt_version",
            "model",
            "created_at",
        )
        now = utc_now()
        payloads = [{"created_at": now, **label} for label in labels]
        with self.connect() as db:
            db.executemany(
                f"INSERT OR REPLACE INTO labels({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                [[payload[key] for key in columns] for payload in payloads],
            )

    def template_label_get(
        self, template_id: str, prompt_version: str, model: str
    ) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM template_labels WHERE template_id=? AND prompt_version=? AND model=?",
                (template_id, prompt_version, model),
            ).fetchone()
        return dict(row) if row else None

    def label_fingerprint(self, prompt_version: str, model: str) -> dict[str, Any]:
        digest = hashlib.sha256()
        count = 0
        with self.connect() as db:
            rows = db.execute(
                "SELECT sample_id,quality,difficulty,taxonomy FROM labels "
                "WHERE prompt_version=? AND model=? ORDER BY sample_id",
                (prompt_version, model),
            )
            for row in rows:
                payload = {
                    "sample_id": row["sample_id"],
                    "quality": row["quality"],
                    "difficulty": row["difficulty"],
                    "taxonomy": row["taxonomy"],
                }
                digest.update(canonical_json(payload).encode("utf-8"))
                digest.update(b"\n")
                count += 1
        return {
            "sha256": digest.hexdigest(),
            "count": count,
            "prompt_version": prompt_version,
            "model": model,
        }

    def template_label_put(self, label: dict[str, Any]) -> None:
        columns = (
            "template_id",
            "quality",
            "difficulty",
            "taxonomy",
            "rationale",
            "prompt_version",
            "model",
            "created_at",
        )
        payload = {"created_at": utc_now(), **label}
        with self.connect() as db:
            db.execute(
                f"INSERT OR REPLACE INTO template_labels({','.join(columns)}) "
                f"VALUES({','.join('?' for _ in columns)})",
                [payload[key] for key in columns],
            )

    def export_labels(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        count = 0
        with self.connect() as db, temporary.open("w", encoding="utf-8") as stream:
            for row in db.execute("SELECT * FROM labels ORDER BY source,sample_id"):
                stream.write(canonical_json(dict(row)) + "\n")
                count += 1
        temporary.replace(path)
        return count

    def metadata_get(self, key: str, default: Any = None) -> Any:
        with self.connect() as db:
            row = db.execute("SELECT value_json FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def metadata_put(self, key: str, value: Any) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?,?)",
                (key, canonical_json(value), utc_now()),
            )

    def export(self, output_root: Path) -> None:
        experiments = self.list_experiments()
        jsonl = output_root / "experiments.jsonl"
        csv_path = output_root / "leaderboard.csv"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        json_tmp = jsonl.with_suffix(".jsonl.tmp")
        with json_tmp.open("w", encoding="utf-8") as stream:
            for row in experiments:
                stream.write(canonical_json(row) + "\n")
        json_tmp.replace(jsonl)
        fields = [
            "id",
            "kind",
            "phase",
            "status",
            "primary_score",
            "gate_pass",
            "gpu_hours",
            "validation_loss",
            "model_path",
            "config_hash",
            "dataset_hash",
            "protocol_hash",
            "error",
        ]
        csv_tmp = csv_path.with_suffix(".csv.tmp")
        with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in experiments:
                metrics = row.get("metrics") or {}
                writer.writerow(
                    {
                        **{key: row.get(key) for key in fields},
                        "primary_score": metrics.get("primary_score"),
                        "gate_pass": metrics.get("gate_pass"),
                        "gpu_hours": metrics.get("gpu_hours"),
                        "validation_loss": metrics.get("validation_loss"),
                    }
                )
        csv_tmp.replace(csv_path)
