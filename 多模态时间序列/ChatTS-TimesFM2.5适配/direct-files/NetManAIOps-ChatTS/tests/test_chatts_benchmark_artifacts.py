from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "chatts_benchmark_artifacts.py"
SPEC = importlib.util.spec_from_file_location("chatts_benchmark_artifacts", MODULE_PATH)
assert SPEC and SPEC.loader
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _request(
    tmp_path: Path,
    *,
    protocol_items: list[str] | None = None,
    large_model_file: bool = False,
):
    model = tmp_path / "model"
    backbone = tmp_path / "backbone"
    data = tmp_path / "data"
    protocol = tmp_path / "runner.py"
    for directory in (model, backbone, data):
        directory.mkdir(parents=True, exist_ok=True)
    initial_files = {
        model / "config.json": '{"model": 1}',
        backbone / "config.json": '{"encoder": 1}',
        data / "test.jsonl": '{"id": 1}\n',
        protocol: "print('evaluate')\n",
    }
    for path, content in initial_files.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    if large_model_file:
        weights = model / "model.safetensors"
        if not weights.exists():
            weights.write_bytes(b"A" * (4 * 1024 * 1024 + 17))
    return artifacts.build_request(
        suite="tsrbench",
        model_path=str(model),
        model_name="candidate",
        model_components=[str(backbone)],
        data_paths=[str(data)],
        protocol_files=[str(protocol)],
        protocol_items=protocol_items or ["seed=42", "prompt=answer_only"],
        eval_protocol_hash="external-protocol-v1",
    )


def test_large_file_content_change_cannot_hide_behind_size_and_restored_mtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        artifacts.FILE_DIGEST_CACHE_ENV,
        str(tmp_path / "fingerprints.sqlite3"),
    )
    request = _request(tmp_path, large_model_file=True)
    output = tmp_path / "output"
    summary = output / "tsrbench_summary_candidate.json"
    _write_json(summary, {"overall": {"accuracy_strict": 0.5}})
    manifest_path = artifacts.write_suite_manifest(
        request=request,
        output_dir=str(output),
        summary_file=str(summary),
        run_id="before-content-change",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    weights = tmp_path / "model" / "model.safetensors"
    original_stat = weights.stat()
    with weights.open("r+b") as stream:
        stream.seek(2 * 1024 * 1024)
        stream.write(b"tampered-but-same-size")
    os.utime(
        weights,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    changed_stat = weights.stat()
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns

    changed_request = _request(tmp_path, large_model_file=True)
    assert changed_request["model_fingerprint"] != request["model_fingerprint"]
    assert artifacts.cache_matches(manifest, changed_request) == (
        False,
        "model_fingerprint changed",
    )


def test_nested_directory_symlink_tracks_target_contents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        artifacts.FILE_DIGEST_CACHE_ENV,
        str(tmp_path / "fingerprints.sqlite3"),
    )
    linked_target = tmp_path / "linked-model-component"
    linked_target.mkdir()
    linked_weights = linked_target / "weights.bin"
    linked_weights.write_bytes(b"A" * 4096)
    model = tmp_path / "model"
    model.mkdir()
    (model / "nested-component").symlink_to(linked_target, target_is_directory=True)

    original_fingerprint = artifacts.fingerprint_paths([model])
    original_stat = linked_weights.stat()
    linked_weights.write_bytes(b"B" * 4096)
    os.utime(
        linked_weights,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert linked_weights.stat().st_size == original_stat.st_size
    assert linked_weights.stat().st_mtime_ns == original_stat.st_mtime_ns

    assert artifacts.fingerprint_paths([model]) != original_fingerprint


def test_directory_symlink_cycle_fails_closed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    nested = data / "nested"
    nested.mkdir(parents=True)
    (nested / "back-to-root").symlink_to(data, target_is_directory=True)

    with pytest.raises(ValueError, match="Directory symlink cycle"):
        artifacts.fingerprint_paths([data])


def test_cache_binds_raw_outputs_but_ignores_logs_and_temporary_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        artifacts.FILE_DIGEST_CACHE_ENV,
        str(tmp_path / "fingerprints.sqlite3"),
    )
    request = _request(tmp_path)
    output = tmp_path / "output"
    summary = output / "tsrbench_summary_candidate.json"
    prediction = output / "predictions" / "part-000.jsonl"
    _write_json(summary, {"overall": {"accuracy_strict": 0.5}})
    prediction.parent.mkdir(parents=True, exist_ok=True)
    original_prediction = '{"id":1,"answer":"A"}\n'
    prediction.write_text(original_prediction, encoding="utf-8")
    manifest_path = artifacts.write_suite_manifest(
        request=request,
        output_dir=str(output),
        summary_file=str(summary),
        run_id="raw-output-integrity",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["output_artifact_count"] == 2
    assert artifacts.cache_matches(manifest, request) == (
        True,
        "all fingerprints match",
    )

    (output / "logs").mkdir()
    (output / "logs" / "runner.log").write_text("new log\n", encoding="utf-8")
    (output / "worker.tmp").write_text("temporary\n", encoding="utf-8")
    assert artifacts.cache_matches(manifest, request) == (
        True,
        "all fingerprints match",
    )

    prediction.unlink()
    assert artifacts.cache_matches(manifest, request) == (
        False,
        "output artifacts changed",
    )
    prediction.write_text(original_prediction, encoding="utf-8")
    assert artifacts.cache_matches(manifest, request) == (
        True,
        "all fingerprints match",
    )

    original_stat = prediction.stat()
    prediction.write_text(
        '{"id":1,"answer":"B"}\n',
        encoding="utf-8",
    )
    os.utime(
        prediction,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert prediction.stat().st_size == original_stat.st_size
    assert prediction.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert artifacts.cache_matches(manifest, request) == (
        False,
        "output artifacts changed",
    )


def test_persistent_digest_cache_avoids_rehashing_shared_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        artifacts.FILE_DIGEST_CACHE_ENV,
        str(tmp_path / "fingerprints.sqlite3"),
    )
    hash_calls: list[Path] = []
    original_sha256_file = artifacts._sha256_file

    def counted_sha256_file(path: Path) -> str:
        hash_calls.append(path.resolve())
        return original_sha256_file(path)

    monkeypatch.setattr(artifacts, "_sha256_file", counted_sha256_file)
    first = _request(tmp_path, large_model_file=True)
    calls_after_first = list(hash_calls)

    second = artifacts.build_request(
        suite="timeseriesexam",
        model_path=first["model_path"],
        model_name="candidate",
        model_components=first["model_components"],
        data_paths=first["data_paths"],
        protocol_files=first["protocol_files"],
        protocol_items=["seed=42", "prompt=official"],
        eval_protocol_hash="external-protocol-v1",
    )

    assert second["model_fingerprint"] == first["model_fingerprint"]
    assert second["data_fingerprint"] == first["data_fingerprint"]
    assert hash_calls == calls_after_first
    weights = (tmp_path / "model" / "model.safetensors").resolve()
    assert hash_calls.count(weights) == 1


def test_cache_requires_all_fingerprints_and_untouched_summary(tmp_path: Path) -> None:
    request = _request(tmp_path)
    output = tmp_path / "output"
    summary = output / "tsrbench_summary_candidate.json"
    _write_json(summary, {"overall": {"accuracy_strict": 0.5}})
    manifest_path = artifacts.write_suite_manifest(
        request=request,
        output_dir=str(output),
        summary_file=str(summary),
        run_id="trial-1",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert artifacts.cache_matches(manifest, request) == (True, "all fingerprints match")

    summary.write_text('{"overall": {"accuracy_strict": 1.0}}', encoding="utf-8")
    assert artifacts.cache_matches(manifest, request) == (False, "source summary changed")
    _write_json(summary, {"overall": {"accuracy_strict": 0.5}})

    (tmp_path / "data" / "test.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
    data_changed = _request(tmp_path)
    assert artifacts.cache_matches(manifest, data_changed)[0] is False
    assert artifacts.cache_matches(manifest, data_changed)[1] == "data_fingerprint changed"

    protocol_root = tmp_path / "protocol-case"
    protocol_request = _request(protocol_root)
    protocol_summary = protocol_root / "output" / "tsrbench_summary_candidate.json"
    _write_json(protocol_summary, {"overall": {"accuracy_strict": 0.5}})
    protocol_manifest_path = artifacts.write_suite_manifest(
        request=protocol_request,
        output_dir=str(protocol_summary.parent),
        summary_file=str(protocol_summary),
        run_id="trial-2",
    )
    protocol_manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    protocol_changed = _request(protocol_root, protocol_items=["seed=42", "prompt=official"])
    assert artifacts.cache_matches(protocol_manifest, protocol_changed)[0] is False
    assert artifacts.cache_matches(protocol_manifest, protocol_changed)[1] == "protocol_fingerprint changed"

    model_root = tmp_path / "model-case"
    model_request = _request(model_root)
    model_summary = model_root / "output" / "tsrbench_summary_candidate.json"
    _write_json(model_summary, {"overall": {"accuracy_strict": 0.5}})
    model_manifest_path = artifacts.write_suite_manifest(
        request=model_request,
        output_dir=str(model_summary.parent),
        summary_file=str(model_summary),
        run_id="trial-3",
    )
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    (model_root / "backbone" / "config.json").write_text('{"encoder": 2}', encoding="utf-8")
    model_changed = _request(model_root)
    assert artifacts.cache_matches(model_manifest, model_changed)[0] is False
    assert artifacts.cache_matches(model_manifest, model_changed)[1] == "model_fingerprint changed"


def test_cache_identity_binds_nonempty_dataset_version_and_snapshot_hash(tmp_path: Path) -> None:
    legacy = _request(tmp_path)
    explicit_empty = artifacts.build_request(
        suite="tsrbench",
        model_path=legacy["model_path"],
        model_name="candidate",
        model_components=legacy["model_components"],
        data_paths=legacy["data_paths"],
        protocol_files=legacy["protocol_files"],
        protocol_items=["seed=42", "prompt=answer_only"],
        eval_protocol_hash="external-protocol-v1",
        data_version="",
        dataset_snapshot_hash="",
    )
    assert explicit_empty["command_fingerprint"] == legacy["command_fingerprint"]
    assert explicit_empty["protocol_fingerprint"] == legacy["protocol_fingerprint"]

    snapshot_hash = "a" * 64
    versioned = artifacts.build_request(
        suite="tsrbench",
        model_path=legacy["model_path"],
        model_name="candidate",
        model_components=legacy["model_components"],
        data_paths=legacy["data_paths"],
        protocol_files=legacy["protocol_files"],
        protocol_items=["seed=42", "prompt=answer_only"],
        eval_protocol_hash="external-protocol-v1",
        data_version="datav3",
        dataset_snapshot_hash=snapshot_hash,
    )
    output = tmp_path / "versioned-output"
    summary = output / "tsrbench_summary_candidate.json"
    _write_json(summary, {"overall": {"accuracy_strict": 0.5}})
    manifest_path = artifacts.write_suite_manifest(
        request=versioned,
        output_dir=str(output),
        summary_file=str(summary),
        run_id="datav3",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert artifacts.cache_matches(manifest, versioned)[0] is True

    for changed_version, changed_hash in (("datav4", snapshot_hash), ("datav3", "b" * 64)):
        changed = artifacts.build_request(
            suite="tsrbench",
            model_path=legacy["model_path"],
            model_name="candidate",
            model_components=legacy["model_components"],
            data_paths=legacy["data_paths"],
            protocol_files=legacy["protocol_files"],
            protocol_items=["seed=42", "prompt=answer_only"],
            eval_protocol_hash="external-protocol-v1",
            data_version=changed_version,
            dataset_snapshot_hash=changed_hash,
        )
        assert artifacts.cache_matches(manifest, changed) == (
            False,
            "protocol_fingerprint changed",
        )


def test_normalizes_existing_suite_summaries_without_rescoring() -> None:
    tsr = artifacts.normalized_suite_metrics(
        "tsrbench",
        {
            "overall": {
                "dataset_size": 10,
                "generated": 9,
                "parsed": 8,
                "correct": 7,
                "coverage": 0.9,
                "parse_rate": 8 / 9,
                "accuracy_strict": 0.7,
                "accuracy_parsed": 0.875,
            }
        },
    )
    assert tsr["strict_accuracy"] == 0.7
    assert tsr["parsed_accuracy"] == 0.875

    exam = artifacts.normalized_suite_metrics(
        "timeseriesexam",
        {
            "overall": {
                "total": 10,
                "official_flexible_accuracy": 0.8,
                "official_strict_accuracy": 0.7,
                "letter_accuracy": 0.6,
            }
        },
    )
    assert exam["flexible_accuracy"] == 0.8
    assert exam["strict_accuracy"] == 0.7

    haystack = artifacts.normalized_suite_metrics(
        "ts_haystack",
        {"overall": {"accuracy_strict": 0.4, "mean_iou": 0.55}},
    )
    assert haystack["strict_accuracy"] == 0.4
    assert haystack["mean_iou"] == 0.55

    tiny = artifacts.normalized_suite_metrics(
        "tinybenchmarks",
        {"macro_score": 0.3, "num_tasks": 2, "tasks": {"a": {"score": 0.2}, "b": {"score": 0.4}}},
    )
    assert tiny == {"macro_score": 0.3, "num_tasks": 2, "task_scores": {"a": 0.2, "b": 0.4}}

    # Historical/raw evaluator schema: only per-task scores are guaranteed.
    tiny_raw = artifacts.normalized_suite_metrics(
        "tinybenchmarks",
        {
            "evaluator": "ChatTS vLLM prompt_logprobs",
            "tasks": {
                "tinyArc": {"metric": "accuracy_norm", "score": 0.5, "num_samples": 100},
                "tinyHellaswag": {"metric": "accuracy_norm", "score": 0.25, "num_samples": 100},
                "tinyMMLU": {"metric": "accuracy_norm", "score": 0.75, "num_samples": 100},
                "tinyTruthfulQA": {"metric": "mc2_probability_mass", "score": 0.4, "num_samples": 100},
                "tinyWinogrande": {"metric": "accuracy_norm", "score": 0.6, "num_samples": 100},
            },
        },
    )
    assert tiny_raw["num_tasks"] == 5
    assert tiny_raw["macro_score"] == 0.5
    assert tiny_raw["task_scores"]["tinyTruthfulQA"] == 0.4


def test_aggregate_writes_metrics_and_run_manifest(tmp_path: Path) -> None:
    request = _request(tmp_path)
    output = tmp_path / "run" / "tsrbench"
    summary = output / "tsrbench_summary_candidate.json"
    _write_json(
        summary,
        {
            "model": "candidate",
            "overall": {
                "dataset_size": 4,
                "generated": 4,
                "parsed": 4,
                "correct": 3,
                "coverage": 1.0,
                "parse_rate": 1.0,
                "accuracy_strict": 0.75,
                "accuracy_parsed": 0.75,
            },
        },
    )
    suite_manifest = artifacts.write_suite_manifest(
        request=request,
        output_dir=str(output),
        summary_file=str(summary),
        run_id="trial-1",
    )
    status_file = tmp_path / "run" / "benchmark_status.tsv"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["suite", "gpus", "status", "exit_code", "output_dir", "log_file"])
        writer.writerow(["tsrbench", "0,1,2,3,4,5,6,7", "PASS", "0", output, output / "run.log"])

    metrics_file = tmp_path / "run" / "metrics.json"
    run_manifest_file = tmp_path / "run" / "run_manifest.json"
    metrics, manifest = artifacts.aggregate_run(
        status_file=str(status_file),
        suite_manifests={"tsrbench": suite_manifest},
        metrics_file=str(metrics_file),
        run_manifest_file=str(run_manifest_file),
        run_id="trial-1",
        model_path=str(tmp_path / "model"),
        model_name="candidate",
        seed=42,
        max_samples=0,
        force_eval=False,
        output_root=str(tmp_path / "run"),
        eval_protocol_hash="external-protocol-v1",
        data_version="datav3",
        dataset_snapshot_hash="c" * 64,
    )

    assert metrics["status"] == "pass"
    assert metrics["suites"]["tsrbench"]["metrics"]["strict_accuracy"] == 0.75
    assert metrics["suites"]["tsrbench"]["summary"]["overall"]["correct"] == 3
    assert manifest["eval_protocol_hash"] == "external-protocol-v1"
    assert json.loads(metrics_file.read_text(encoding="utf-8"))["run_id"] == "trial-1"
    written_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    written_manifest = json.loads(run_manifest_file.read_text(encoding="utf-8"))
    assert written_metrics["data_version"] == "datav3"
    assert written_metrics["dataset_snapshot_hash"] == "c" * 64
    assert written_manifest["status"] == "pass"
    assert written_manifest["data_version"] == "datav3"
    assert written_manifest["dataset_snapshot_hash"] == "c" * 64


def test_preflight_accepts_one_selected_suite_without_other_dataset_roots(tmp_path: Path) -> None:
    backbone = tmp_path / "chronos2"
    tiny_data = tmp_path / "tiny"
    backbone.mkdir()
    tiny_data.mkdir()
    output = tmp_path / "must-not-be-created"
    env = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "PROJECT_ROOT": str(REPO_ROOT),
        "CHRONOS2_MODEL_PATH": str(backbone),
        "TINYBENCH_DATASET_ROOT": str(tiny_data),
        "MODEL_PATH": str(tmp_path / "not-trained-yet"),
        "OUTPUT_ROOT": str(output),
        "BENCHMARKS": "tinybenchmarks",
        "PREFLIGHT_ONLY": "1",
        "AVAILABLE_GPUS_OVERRIDE": "8",
        "PYTHON_BIN": sys.executable,
    }
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_all_chatts_benchmarks.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Benchmarks:     tinybenchmarks" in completed.stdout
    assert not output.exists()
