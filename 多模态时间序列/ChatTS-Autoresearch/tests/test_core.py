from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from chatts_autoresearch.config import ConfigError, load_config
from chatts_autoresearch.data import (
    DataCatalog,
    DataError,
    _source_taxonomy,
    create_eval_dataset_views,
    label_catalog,
    prepare_snapshot,
    record_hash,
    sample_id,
)
from chatts_autoresearch.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    proposal_validator,
    round_analysis_validator,
    validate_label,
)
from chatts_autoresearch.hashing import command_fingerprint, hash_object
from chatts_autoresearch.metrics import apply_gates, extract_badcases, load_metrics
from chatts_autoresearch.orchestrator import Autoresearch, OrchestrationError
from chatts_autoresearch.report import generate_report
from chatts_autoresearch.state import StateStore


def test_config_and_hashes_are_stable(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    assert config.get("runtime.seed") == 42
    assert config.fingerprint == load_config(project["config"]).fingerprint
    assert DataCatalog(config).fingerprint
    first = command_fingerprint(["bash", "x.sh"], "/tmp", {"B": "2", "A": "1"})
    second = command_fingerprint(["bash", "x.sh"], "/tmp", {"A": "1", "B": "2"})
    assert first == second


def test_catalog_fingerprint_binds_actual_source_bytes(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    first = DataCatalog(config).fingerprint
    source = project["datav2"] / "files" / "chatts_sft.jsonl"
    source.write_text(source.read_text() + "\n")
    second = DataCatalog(config).fingerprint
    assert second != first


def test_seed_is_fixed(project: dict[str, Path]) -> None:
    raw = project["config"].read_text()
    project["config"].write_text(raw.replace("seed: 42", "seed: 7", 1))
    with pytest.raises(ConfigError, match="seed"):
        load_config(project["config"])


def test_deepseek_strict_json_cache_and_label_resume(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    state = StateStore(project["artifacts"] / "state.sqlite3")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        request_payload = json.loads(request.content)
        response_format = request_payload["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert (
            response_format["json_schema"]["schema"]["additionalProperties"]
            is False
        )
        content = json.dumps(
            {
                "quality_score": 0.8,
                "difficulty": "medium",
                "taxonomy": "trend description",
                "rationale": "Question and answer are aligned.",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = DeepSeekClient(config.data["deepseek"], state, httpx.MockTransport(handler))
    first = label_catalog(config, state, client)
    second = label_catalog(config, state, client)
    client.close()
    assert first["completed_templates"] == 1
    assert second["completed_templates"] == 0
    assert calls == 1
    sidecar = project["artifacts"] / "labels" / "quality_difficulty_taxonomy.jsonl"
    assert len(sidecar.read_text().splitlines()) == 1


def test_same_normalized_template_calls_deepseek_once(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    config.data["labeling"]["max_samples"] = 0
    source = project["datav2"] / "files" / "chatts_align_256.jsonl"
    rows = [
        {"input": "Where is spike 17?", "timeseries": [[0, 7, 0]], "output": "Spike is at 17."},
        {"input": "Where is spike 93?", "timeseries": [[0, 2, 0]], "output": "Spike is at 93."},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    state = StateStore(project["artifacts"] / "state.sqlite3")
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = json.dumps(
            {
                "quality_score": 0.9,
                "difficulty": "easy",
                "taxonomy": "spike localization",
                "rationale": "Template is coherent.",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = DeepSeekClient(config.data["deepseek"], state, httpx.MockTransport(handler))
    result = label_catalog(config, state, client)
    client.close()
    assert calls == 1
    assert result["submitted_templates"] == 1
    assert result["expanded_samples"] == 2


def test_ecg_source_gets_stable_domain_taxonomy() -> None:
    assert _source_taxonomy("ltaf_ecg", "rhythm classification") == (
        "ECG / rhythm classification"
    )
    assert _source_taxonomy("ltaf_ecg", "ECG rhythm classification") == (
        "ECG rhythm classification"
    )


def test_deepseek_rejects_non_json(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    state = StateStore(project["artifacts"] / "state.sqlite3")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "```json\n{}\n```"}}]})

    client = DeepSeekClient(config.data["deepseek"], state, httpx.MockTransport(handler))
    with pytest.raises(DeepSeekError):
        client.complete_json("label", "system", "user", validate_label, "v1")
    client.close()


def test_proposal_rejects_unknown_weight_keys(project: dict[str, Path]) -> None:
    search = load_config(project["config"]).data["search"]
    validate = proposal_validator(search, {"chatts_sft"})
    with pytest.raises(DeepSeekError, match="Unknown source_weights"):
        validate(
            {
                "family": "source_weights",
                "patch": {"source_weights": {"made_up_source": 1.2}},
                "rationale": "test",
            }
        )
    with pytest.raises(DeepSeekError, match="Unknown difficulty_weights"):
        validate(
            {
                "family": "difficulty_weights",
                "patch": {"difficulty_weights": {"extreme": 1.2}},
                "rationale": "test",
            }
        )


def test_label_cache_is_bound_to_prompt_and_model(project: dict[str, Path]) -> None:
    state = StateStore(project["artifacts"] / "state.sqlite3")
    label = {
        "sample_id": "sample",
        "template_id": "template",
        "source": "source",
        "record_hash": "record",
        "quality": 0.8,
        "difficulty": "medium",
        "taxonomy": "trend",
        "rationale": "visible evidence",
        "prompt_version": "v1",
        "model": "m1",
    }
    state.label_put(label)
    state.template_label_put({key: value for key, value in label.items() if key not in {"sample_id", "source", "record_hash"}})
    assert state.label_get("sample", "v1", "m1") is not None
    assert state.label_get("sample", "v2", "m1") is None
    assert state.template_label_get("template", "v1", "m1") is not None
    assert state.template_label_get("template", "v1", "m2") is None


def test_legacy_label_schema_migrates_before_index(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE labels(sample_id TEXT PRIMARY KEY, source TEXT, record_hash TEXT, "
            "quality REAL, difficulty TEXT, taxonomy TEXT, rationale TEXT, "
            "prompt_version TEXT, model TEXT, created_at TEXT)"
        )
    StateStore(path)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(labels)")}
        indexes = {row[1] for row in db.execute("PRAGMA index_list(labels)")}
    assert "template_id" in columns
    assert "labels_template_idx" in indexes


def test_prepare_snapshot_marks_duplicate_and_writes_dataset_info(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    state = StateStore(project["artifacts"] / "state.sqlite3")
    manifest = prepare_snapshot(config, state)
    root = project["artifacts"] / "datasets" / "filtered"
    assert manifest["snapshot_hash"]
    assert (root / "dataset_info.json").is_file()
    info = json.loads((root / "dataset_info.json").read_text())
    assert set(info) == {"align_256", "sft"}
    assert manifest["stats"]["chatts_sft"]["duplicate"] == 1
    audits = [json.loads(line) for line in (root / "duplicate_labels.jsonl").read_text().splitlines()]
    assert any(item["cross_source_duplicate"] for item in audits)
    assert (project["artifacts"] / "datasets" / "raw" / "data" / "chatts_sft.jsonl").is_symlink()


def test_data_patch_starts_from_raw_equivalent_baseline_policy(
    project: dict[str, Path],
) -> None:
    config = load_config(project["config"])
    config.data["data"]["source_weights"] = {"chatts_align_256": 0.8}
    controller = Autoresearch(config)
    controller.prepare_data()
    dataset_path, _ = controller._dataset_for(
        False, {"source_weights": {"chatts_sft": 1.5}}
    )
    manifest = json.loads((dataset_path / "manifest.json").read_text())
    expected = dict(config.data["data"])
    expected.update(
        {
            "minimum_quality": 0.0,
            "missing_label_policy": "keep",
            "drop_exact_duplicates": False,
            "drop_cross_source_duplicates": False,
            "drop_near_duplicates": False,
            "source_weights": {"chatts_sft": 1.5},
        }
    )
    expected["difficulty_weights"] = {
        "easy": 1.0,
        "medium": 1.0,
        "hard": 1.0,
    }
    expected["snapshot_name"] = dataset_path.name
    assert manifest["snapshot_config_hash"] == hash_object(expected)
    controller.close()


def test_snapshot_cache_is_bound_to_current_labels(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    state = StateStore(project["artifacts"] / "state.sqlite3")
    first = prepare_snapshot(config, state)
    assert first["label_fingerprint"]["count"] == 0
    record = json.loads(
        (project["datav2"] / "files" / "chatts_align_256.jsonl").read_text().splitlines()[0]
    )
    digest = record_hash(record)
    state.label_put(
        {
            "sample_id": sample_id("chatts_align_256", digest),
            "template_id": "template",
            "source": "chatts_align_256",
            "record_hash": digest,
            "quality": 0.9,
            "difficulty": "medium",
            "taxonomy": "trend",
            "rationale": "visible evidence",
            "prompt_version": config.get("deepseek.prompt_version"),
            "model": config.get("deepseek.model"),
        }
    )
    with pytest.raises(DataError, match="different data/config fingerprint"):
        prepare_snapshot(config, state)


def test_state_exports_and_hash_collision_guard(project: dict[str, Path]) -> None:
    state = StateStore(project["artifacts"] / "state.sqlite3")
    payload = {
        "id": "x",
        "kind": "candidate",
        "phase": "proxy",
        "config_hash": hash_object({"x": 1}),
        "dataset_hash": "data",
        "protocol_hash": "protocol",
        "config_json": {"x": 1},
        "output_dir": "/tmp/out",
    }
    state.create_experiment(payload)
    state.mark_running("x", "command", {"argv": ["true"]})
    state.mark_completed("x", {"primary_score": 0.5}, "/tmp/model")
    state.export(project["artifacts"])
    assert len((project["artifacts"] / "experiments.jsonl").read_text().splitlines()) == 1
    with (project["artifacts"] / "leaderboard.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["primary_score"] == "0.5"


def test_full_mock_lifecycle_resume_freeze_and_report(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    assert controller.preflight()["passed"]
    controller.prepare_data()
    with pytest.raises(OrchestrationError, match="locked"):
        controller.final_eval()
    baseline = controller.baseline()
    assert baseline["status"] == "completed"
    baseline_command = json.loads(
        (project["artifacts"] / "commands" / "baseline.json").read_text()
    )
    assert "STAGE2_FROM" not in baseline_command["train"]["env"]
    controller.state.mark_failed("baseline", "simulated interruption")
    assert controller.baseline()["status"] == "completed"
    resumed_command = json.loads(
        (project["artifacts"] / "commands" / "baseline.json").read_text()
    )
    assert resumed_command["train"]["env"]["FORCE_TRAIN"] == "1"
    search = controller.search()
    assert len(search["proxies"]) == 2
    assert len(search["finalists"]) == 1
    before = len(controller.state.list_experiments())
    controller.resume()
    assert len(controller.state.list_experiments()) == before
    frozen = controller.freeze()
    assert frozen["champion"]["experiment_id"].startswith("full-")
    final = controller.final_eval()
    assert final["baseline"].startswith("final-baseline")
    report = controller.report()
    controller.close()
    assert report.is_file()
    assert (project["artifacts"] / "figures" / "leaderboard.svg").is_file()
    assert "单 seed" in report.read_text()
    assert "正式测试：baseline vs champion" in report.read_text()
    assert "final-champion" not in (project["artifacts"] / "figures" / "leaderboard.svg").read_text()
    assert (project["artifacts"] / "FROZEN.json").is_file()
    champion_weights = Path(frozen["champion"]["model_path"]) / "pytorch_model.bin"
    champion_weights.write_text("tampered\n")
    with pytest.raises(OrchestrationError, match="identity has changed"):
        controller.final_eval()


def test_report_uses_frozen_search_scope_rank_order_and_observed_guard_metrics(
    tmp_path: Path,
) -> None:
    state = StateStore(tmp_path / "state.sqlite3")

    def metrics(
        score: float,
        gpu: float,
        loss: float,
        badcases: int,
        *,
        haystack: float = 0.75,
        tiny: float = 0.70,
        gate: bool = True,
    ) -> dict[str, object]:
        return {
            "primary_score": score,
            "coverage": 0.98,
            "gpu_hours": gpu,
            "validation_loss": loss,
            "gate_pass": gate,
            "gate_reasons": [],
            "badcase_summary": {"badcases": badcases, "scored_records": 10},
            "suites": {
                "tsrbench": {
                    "strict_accuracy": score + 0.01,
                    "flexible_accuracy": score + 0.02,
                    "coverage": 0.99,
                },
                "timeseriesexam": {
                    "strict_accuracy": score - 0.01,
                    "flexible_accuracy": score,
                    "coverage": 0.97,
                },
                "ts_haystack": {"mean_iou": haystack},
                "tinybenchmarks": {
                    "average_accuracy": tiny,
                    "tasks": {"tinyArc": tiny, "tinyMMLU": tiny - 0.01},
                },
            },
        }

    def completed(
        experiment_id: str,
        phase: str,
        observed: dict[str, object],
        *,
        parent_id: str | None = None,
        role: str | None = None,
    ) -> None:
        state.create_experiment(
            {
                "id": experiment_id,
                "kind": "candidate",
                "phase": phase,
                "parent_id": parent_id,
                "config_hash": f"config-{experiment_id}",
                "dataset_hash": "dataset",
                "protocol_hash": "protocol",
                "config_json": {"role": role} if role else {},
                "output_dir": str(tmp_path / "evaluations" / experiment_id),
            }
        )
        state.mark_running(experiment_id, "command", {"argv": ["true"]})
        state.mark_completed(experiment_id, observed, str(tmp_path / "models" / experiment_id))

    completed("baseline", "baseline", metrics(0.60, 8.0, 0.30, 5))
    completed("proxy-slow", "proxy", metrics(0.65, 2.0, 0.01, 4), parent_id="baseline")
    completed("proxy-gpu", "proxy", metrics(0.65, 1.0, 0.30, 3), parent_id="baseline")
    completed("proxy-loss", "proxy", metrics(0.65, 1.0, 0.10, 2), parent_id="baseline")
    completed("full-loss", "full", metrics(0.66, 4.0, 0.20, 2), parent_id="proxy-loss")
    completed("stale-best", "full", metrics(0.99, 0.1, 0.01, 0), parent_id="baseline")
    completed(
        "final-baseline",
        "final-test",
        metrics(0.60, 1.0, 0.30, 5, haystack=0.75, tiny=0.70),
        parent_id="baseline",
        role="formal-baseline",
    )
    completed(
        "final-champion",
        "final-test",
        metrics(0.67, 1.0, 0.20, 3, haystack=0.74, tiny=0.71),
        parent_id="full-loss",
        role="formal-champion",
    )
    manifest = {
        "baseline": {"id": "baseline"},
        "proxies": [
            {"id": "proxy-slow"},
            {"id": "proxy-gpu"},
            {"id": "proxy-loss"},
        ],
        "ranking": ["proxy-loss", "proxy-gpu", "proxy-slow"],
        "selected_proxy_ids": ["proxy-loss"],
        "finalists": [{"id": "full-loss"}],
        "analysis_hashes": {"round-01.json": "observed-hash"},
        "search_hash": "search-hash",
    }
    (tmp_path / "SEARCH_COMPLETE.json").write_text(json.dumps(manifest))
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "round-01.json").write_text(
        json.dumps(
            {
                "source_experiment_id": "proxy-loss",
                "sampled_badcases": 2,
                "recommended_family": "scheduler",
            }
        )
    )

    report_path = generate_report(state, tmp_path, None)
    report = report_path.read_text()
    search_table = report.split("## 实验排行榜（search-dev）", 1)[1].split(
        "## 正式测试：baseline vs champion", 1
    )[0]
    assert "stale-best" not in search_table
    assert "已排除 1 个" in search_table
    assert search_table.index("full-loss") < search_table.index("proxy-loss")
    assert search_table.index("proxy-loss") < search_table.index("proxy-gpu")
    assert search_table.index("proxy-gpu") < search_table.index("proxy-slow")
    assert "| proxy-loss | proxy | completed" in search_table
    assert "main-only" in search_table
    assert "TSR flexible" in search_table
    assert "TSE flexible" in search_table
    assert "tinyArc=0.7000; tinyMMLU=0.6900" in search_table
    assert "[SEARCH_COMPLETE.json](SEARCH_COMPLETE.json)" in report
    assert "`proxy-loss` → `proxy-gpu` → `proxy-slow`" in report
    assert "[round-01.json](analysis/round-01.json)" in report
    assert "manifest-bound" in report
    assert "Δ champion-baseline" in report
    assert "| tinyArc | 0.7000 | 0.7100 | +0.0100 |" in report
    assert "| tinyMMLU | 0.6900 | 0.7000 | +0.0100 |" in report
    assert "0.7400" in report
    assert "-0.0100" in report
    assert "| 3 | guard-pass |" in report
    svg = (tmp_path / "figures" / "leaderboard.svg").read_text()
    assert "stale-best" not in svg
    assert svg.index("full-loss") < svg.index("proxy-loss")
    assert svg.index("proxy-loss") < svg.index("proxy-gpu")
    summary = json.loads((tmp_path / "report_summary.json").read_text())
    assert summary["official_search_experiments"] == 5
    assert summary["stale_search_experiments"] == 1
    assert summary["analysis_rounds"] == 1
    assert summary["scored"] == 7


def test_metrics_and_badcase_accounting(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    (root / "tsrbench").mkdir(parents=True)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "suites": {
                    "tsrbench": {"strict_accuracy": 0.5, "coverage": 1.0},
                    "timeseriesexam": {"strict_accuracy": 0.7, "coverage": 0.9},
                }
            }
        )
    )
    rows = [
        {"prediction": "A", "gold": "A", "correct": True},
        {"prediction": "B", "gold": "A", "correct": False},
    ]
    with (root / "tsrbench" / "rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    metrics = load_metrics(root)
    summary = extract_badcases(root, tmp_path / "badcases.jsonl")
    assert metrics["primary_score"] == pytest.approx(0.6)
    assert summary["scored_records"] == 2
    assert summary["badcases"] == 1


def test_tsrbench_badcases_join_predictions_to_locked_dataset(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    result_dir = root / "tsrbench" / "perception_model"
    result_dir.mkdir(parents=True)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    samples = [
        {"question": "q1", "answer": "A"},
        {"question": "q2", "answer": "B"},
    ]
    (dataset_root / "perception.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples)
    )
    (result_dir / "generated_answer.json").write_text(
        json.dumps(
            [
                {"idx": 0, "answer": "A", "response": '{"answer":"A"}'},
                {"idx": 1, "answer": "C", "response": '{"answer":"C"}'},
            ]
        )
    )
    destination = tmp_path / "badcases.jsonl"
    summary = extract_badcases(root, destination, {"tsrbench": dataset_root})
    assert summary["scored_records"] == 2
    assert summary["badcases"] == 1
    badcase = json.loads(destination.read_text())
    assert badcase["prediction"] == "C"
    assert badcase["gold"] == "B"


def test_real_tiny_nested_metrics_shape(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    root.mkdir()
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "suites": {
                    "tinybenchmarks": {
                        "status": "PASS",
                        "metrics": {
                            "macro_score": 0.71,
                            "task_scores": {
                                "tinyArc": {"score": 0.72},
                                "tinyMMLU": {"accuracy": 0.70},
                            },
                        },
                    }
                }
            }
        )
    )
    tiny = load_metrics(root)["suites"]["tinybenchmarks"]
    assert tiny["average_accuracy"] == pytest.approx(0.71)
    assert tiny["tasks"] == {"tinyArc": pytest.approx(0.72), "tinyMMLU": pytest.approx(0.70)}


def test_gates_fail_closed_when_required_metrics_are_missing() -> None:
    main_only = {
        "primary_score": 0.6,
        "coverage": 0.9,
        "suites": {
            "tsrbench": {"strict_accuracy": 0.6, "coverage": 0.9},
            "timeseriesexam": {"strict_accuracy": 0.6, "coverage": 0.9},
        },
    }
    thresholds = {
        "tiny_average_max_drop": 0.01,
        "tiny_task_max_drop": 0.02,
        "haystack_iou_max_drop": 0.02,
        "coverage_max_drop": 0.01,
    }
    full = apply_gates(main_only, None, thresholds, require_guards=True)
    proxy = apply_gates(main_only, None, thresholds, require_guards=False)
    assert full["gate_pass"] is False
    assert proxy["gate_pass"] is True
    no_coverage = {**main_only, "coverage": None}
    assert apply_gates(no_coverage, None, thresholds, require_guards=False)["gate_pass"] is False
    baseline = {
        **main_only,
        "suites": {
            **main_only["suites"],
            "ts_haystack": {"mean_iou": 0.7},
            "tinybenchmarks": {
                "average_accuracy": 0.7,
                "tasks": {"tinyArc": 0.7, "tinyMMLU": 0.7},
            },
        },
    }
    candidate = {
        **baseline,
        "suites": {
            **baseline["suites"],
            "tinybenchmarks": {"average_accuracy": 0.7, "tasks": {"tinyArc": 0.7}},
        },
    }
    gated = apply_gates(candidate, baseline, thresholds, require_guards=True)
    assert gated["gate_pass"] is False
    assert any("tinyMMLU" in reason for reason in gated["gate_reasons"])


def test_baseline_gate_requires_every_expected_tiny_task() -> None:
    metrics = {
        "primary_score": 0.6,
        "coverage": 1.0,
        "suites": {
            "tsrbench": {"strict_accuracy": 0.6, "coverage": 1.0},
            "timeseriesexam": {"strict_accuracy": 0.6, "coverage": 1.0},
            "ts_haystack": {"mean_iou": 0.7},
            "tinybenchmarks": {
                "average_accuracy": 0.7,
                "tasks": {"tinyArc": 0.7},
            },
        },
    }
    thresholds = {
        "tiny_average_max_drop": 0.01,
        "tiny_task_max_drop": 0.02,
        "haystack_iou_max_drop": 0.02,
        "coverage_max_drop": 0.01,
        "tiny_expected_tasks": ["tinyArc", "tinyMMLU"],
    }
    gated = apply_gates(metrics, None, thresholds, require_guards=True)
    assert gated["gate_pass"] is False
    assert "missing expected tinyBench task score: tinyMMLU" in gated["gate_reasons"]


def test_completed_experiment_rejects_changed_script_and_command_hash(
    project: dict[str, Path],
) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    controller.baseline()
    train_script = Path(config.require("paths.train_script"))
    train_script.write_text(train_script.read_text() + "\n# changed\n")
    with pytest.raises(OrchestrationError, match="different config_hash"):
        controller.baseline()
    controller.close()


def test_completed_experiment_compares_command_hash(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    controller.baseline()
    with controller.state.connect() as db:
        db.execute("UPDATE experiments SET command_hash='tampered' WHERE id='baseline'")
    with pytest.raises(OrchestrationError, match="command hash no longer matches"):
        controller.baseline()
    controller.close()


def test_completed_experiment_binds_global_training_config(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    controller.baseline()
    config.data["training"]["stage2_warmup_ratio"] = 0.05
    with pytest.raises(OrchestrationError, match="different config_hash"):
        controller.baseline()
    controller.close()


def test_completed_experiment_rehashes_changed_base_model(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    controller.baseline()
    base_weights = Path(config.require("paths.base_model")) / "pytorch_model.bin"
    base_weights.write_text("changed-base-weights\n")
    with pytest.raises(OrchestrationError, match="different config_hash"):
        controller.baseline()
    controller.close()


def test_baseline_honors_explicit_projector_learning_rate(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    config.data["training"]["stage2_timeseries_learning_rate"] = 2e-5
    controller = Autoresearch(config)
    controller.prepare_data()
    controller.baseline()
    command = json.loads((project["artifacts"] / "commands" / "baseline.json").read_text())
    assert float(command["train"]["env"]["STAGE2_TIMESERIES_SFT_LR"]) == pytest.approx(2e-5)
    controller.close()


def test_projector_search_uses_actual_baseline_ratio(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    config.data["training"]["stage2_timeseries_learning_rate"] = 2e-5
    controller = Autoresearch(config)
    projector_values = [
        proposal["patch"]["projector_lr_ratio"]
        for proposal in controller._deterministic_proposals()
        if proposal["family"] == "projector_lr_ratio"
    ]
    assert 2.0 not in projector_values
    assert 1.0 in projector_values
    assert controller._is_baseline_equivalent_patch({"projector_lr_ratio": 2.0})
    assert not controller._is_baseline_equivalent_patch({"projector_lr_ratio": 1.0})
    assert controller._is_baseline_equivalent_patch(
        {"difficulty_weights": {"hard": 1.0}}
    )
    controller.close()


def test_completed_reuse_rejects_changed_final_or_stage1_model(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    baseline = controller.baseline()
    stage1_weights = project["artifacts"] / "models" / "shared-stage1" / "pytorch_model.bin"
    stage1_weights.write_text("changed-stage1\n")
    with pytest.raises(OrchestrationError, match="Stage1 identity changed"):
        controller.baseline()
    stage1_weights.write_text("mock-stage1-weights\n")
    final_weights = Path(baseline["model_path"]) / "pytorch_model.bin"
    final_weights.write_text("changed-final\n")
    with pytest.raises(OrchestrationError, match="final model identity changed"):
        controller.baseline()
    controller.close()


def test_freeze_requires_search_and_model_weights(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    controller.baseline()
    with pytest.raises(OrchestrationError, match="proxy trials"):
        controller.freeze()
    empty_model = project["artifacts"] / "models" / "empty"
    empty_model.mkdir()
    (empty_model / "config.json").write_text("{}\n")
    with pytest.raises(OrchestrationError, match="no weight files"):
        controller._model_identity(empty_model, require_weights=True)
    controller.close()


def test_frozen_document_schema_and_hash_are_verified(project: dict[str, Path]) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    controller.baseline()
    controller.search()
    controller.freeze()
    path = project["artifacts"] / "FROZEN.json"
    original = json.loads(path.read_text())

    wrong_schema = dict(original)
    wrong_schema["schema_version"] = "unknown"
    path.write_text(json.dumps(wrong_schema))
    with pytest.raises(OrchestrationError, match="schema_version"):
        controller._load_freeze()
    with pytest.raises(OrchestrationError, match="schema_version"):
        controller.report()

    tampered = json.loads(json.dumps(original))
    tampered["champion"]["primary_score"] = -1
    path.write_text(json.dumps(tampered))
    with pytest.raises(OrchestrationError, match="freeze_hash"):
        controller._load_freeze()
    with pytest.raises(OrchestrationError, match="freeze_hash"):
        controller.report()
    controller.close()


def test_python_module_cli_propagates_failure_exit_code(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    process = subprocess.run(
        [sys.executable, "-m", "chatts_autoresearch", "preflight", "-c", str(tmp_path / "missing.yaml")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2


def test_eval_views_are_disjoint_and_cover_full_data(project: dict[str, Path]) -> None:
    tsr_root = project["datav2"].parent / "tsrbench"
    (tsr_root / "task").mkdir(parents=True)
    tsr_rows = [
        {"id": f"tsr-{index}", "category": f"c{index % 3}", "difficulty": "hard", "x": index}
        for index in range(40)
    ]
    tsr_rows[0]["official_split"] = "dev"
    tsr_rows[1]["official_split"] = "test"
    (tsr_root / "task" / "perception.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tsr_rows)
    )
    exam_file = project["datav2"].parent / "qa_dataset.json"
    exam_rows = [
        {"id": f"exam-{index}", "category": f"c{index % 2}", "difficulty": "medium"}
        for index in range(30)
    ]
    exam_file.write_text(json.dumps(exam_rows))
    config = load_config(project["config"])
    config.data["paths"]["tsrbench_root"] = str(tsr_root)
    config.data["paths"]["timeseriesexam_data_file"] = str(exam_file)
    manifest = create_eval_dataset_views(config)
    assert manifest["counts"]["tsrbench"]["search-dev"] > 0
    views = project["artifacts"] / "eval_views"

    def ids(path: Path) -> set[str]:
        return {json.loads(line)["id"] for line in path.read_text().splitlines()}

    tsr_search = ids(views / "search-dev" / "tsrbench" / "task" / "perception.jsonl")
    tsr_final = ids(views / "final-test" / "tsrbench" / "task" / "perception.jsonl")
    assert tsr_search.isdisjoint(tsr_final)
    assert tsr_search | tsr_final == {row["id"] for row in tsr_rows}
    # A partial official annotation is not a valid official dev/test protocol;
    # all rows fall back to one exact, deterministic stratified split.
    assert manifest["modes"]["tsrbench"] == "hash-stratified-20-80"
    assert len(tsr_search) == 8
    assert len(tsr_final) == 32
    exam_search = {row["id"] for row in json.loads((views / "search-dev" / "timeseriesexam" / "qa_dataset.json").read_text())}
    exam_final = {row["id"] for row in json.loads((views / "final-test" / "timeseriesexam" / "qa_dataset.json").read_text())}
    assert exam_search.isdisjoint(exam_final)
    assert exam_search | exam_final == {row["id"] for row in exam_rows}


def test_search_config_bounds_and_full_final_eval_are_fail_closed(
    project: dict[str, Path],
) -> None:
    original = yaml.safe_load(project["config"].read_text())
    mutations = [
        (("search", "proxy_max_steps"), 0, "proxy_max_steps"),
        (("search", "full_finalists"), 0, "full_finalists"),
        (("search", "full_finalists"), 3, "full_finalists"),
        (("search", "learning_rates"), [], "learning_rates"),
        (("search", "source_weight_range"), [0.5], "source_weight_range"),
        (("evaluation", "final_max_samples"), 1, "final_max_samples"),
        (("data", "baseline_snapshot"), "unknown", "baseline_snapshot"),
        (("runtime", "gpu_ids"), "0,1", "eight distinct"),
        (("runtime", "master_port"), 70000, "master_port"),
    ]
    for (section, key), value, message in mutations:
        candidate = json.loads(json.dumps(original))
        candidate.setdefault(section, {})[key] = value
        project["config"].write_text(yaml.safe_dump(candidate))
        with pytest.raises(ConfigError, match=message):
            load_config(project["config"])


def test_non_data_trials_reuse_baseline_snapshot_and_eval_split_env(
    project: dict[str, Path],
) -> None:
    config = load_config(project["config"])
    config.data["evaluation"]["search_max_samples"] = 7
    controller = Autoresearch(config)
    controller.prepare_data()
    baseline_path, baseline_hash = controller._dataset_for(True)
    lr_path, lr_hash = controller._dataset_for(False, {"learning_rate": 5e-6})
    assert (baseline_path, baseline_hash) == (lr_path, lr_hash)
    search_env = controller._evaluation_env(
        "search", Path(config.require("paths.base_model")), project["artifacts"] / "eval-search",
        "search-dev", "tsrbench,timeseriesexam", "protocol", False,
    )
    final_env = controller._evaluation_env(
        "final", Path(config.require("paths.base_model")), project["artifacts"] / "eval-final",
        "final-test", "tsrbench,timeseriesexam", "protocol", False,
    )
    assert search_env["MAX_SAMPLES"] == "7"
    assert search_env["HAYSTACK_SPLIT"] == "validation"
    assert search_env["TINY_DATA_PARTITION"] == "search-dev"
    assert final_env["MAX_SAMPLES"] == "0"
    assert final_env["HAYSTACK_SPLIT"] == "test"
    assert final_env["TINY_DATA_PARTITION"] == "final-test"
    controller.close()


def test_quality_and_difficulty_search_require_complete_labels(
    project: dict[str, Path],
) -> None:
    controller = Autoresearch(load_config(project["config"]))
    controller.prepare_data()
    with pytest.raises(OrchestrationError, match="complete quality/difficulty labels"):
        controller._dataset_for(False, {"minimum_quality": 0.5})
    with pytest.raises(OrchestrationError, match="complete quality/difficulty labels"):
        controller._dataset_for(False, {"difficulty_weights": {"hard": 1.5}})
    controller.close()


def test_catalog_cache_rehashes_same_size_edit_with_restored_mtime(
    project: dict[str, Path],
) -> None:
    config = load_config(project["config"])
    source = project["datav2"] / "files" / "chatts_sft.jsonl"
    original = source.read_text()
    before_stat = source.stat()
    first = DataCatalog(config).fingerprint
    changed = original.replace("compare", "COMPARE", 1)
    assert len(changed) == len(original)
    source.write_text(changed)
    os.utime(source, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    assert DataCatalog(config).fingerprint != first


def test_snapshot_reuse_validates_actual_data_and_metadata_files(
    project: dict[str, Path],
) -> None:
    config = load_config(project["config"])
    controller = Autoresearch(config)
    controller.prepare_data()
    raw_info = project["artifacts"] / "datasets" / "raw" / "dataset_info.json"
    original_info = raw_info.read_text()
    raw_info.write_text(original_info + " ")
    with pytest.raises(DataError, match="dataset_info.json changed"):
        controller.baseline()
    raw_info.write_text(original_info)

    filtered_data = (
        project["artifacts"]
        / "datasets"
        / "filtered"
        / "data"
        / "chatts_sft.jsonl"
    )
    filtered_data.write_text(filtered_data.read_text() + "\n")
    with pytest.raises(DataError, match="Snapshot data file changed"):
        controller.prepare_data()
    controller.close()


def test_round_analysis_validator_rejects_unobserved_or_duplicated_evidence(
    project: dict[str, Path],
) -> None:
    search = load_config(project["config"]).data["search"]
    validate = round_analysis_validator(
        search,
        {"chatts_sft"},
        {"bad-1", "bad-2"},
        disallowed_patch_hashes=set(),
        reject_patch=lambda _: False,
    )
    proposal = {
        "family": "learning_rate",
        "patch": {"learning_rate": 5e-6},
        "rationale": "observed regression",
    }
    with pytest.raises(DeepSeekError, match="supplied sample"):
        validate(
            {
                "error_groups": [
                    {
                        "error_type": "parse",
                        "likely_data_cause": "format",
                        "badcase_ids": ["invented"],
                    }
                ],
                "recommended_family": "learning_rate",
                "proposal": proposal,
            }
        )
    with pytest.raises(DeepSeekError, match="only one error group"):
        validate(
            {
                "error_groups": [
                    {
                        "error_type": "parse",
                        "likely_data_cause": "format",
                        "badcase_ids": ["bad-1"],
                    },
                    {
                        "error_type": "reasoning",
                        "likely_data_cause": "difficulty",
                        "badcase_ids": ["bad-1"],
                    },
                ],
                "recommended_family": "learning_rate",
                "proposal": proposal,
            }
        )
    final_validate = round_analysis_validator(
        search,
        {"chatts_sft"},
        set(),
        disallowed_patch_hashes=set(),
        reject_patch=lambda _: False,
        require_proposal=False,
    )
    with pytest.raises(DeepSeekError, match="recommended_family"):
        final_validate(
            {"error_groups": [], "recommended_family": [], "proposal": None}
        )


def test_deepseek_search_writes_observed_rounds_and_fast_resume(
    project: dict[str, Path],
) -> None:
    config = load_config(project["config"])
    config.data["search"]["proposal_mode"] = "deepseek"
    state = StateStore(project["artifacts"] / "state.sqlite3")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        prompt = json.loads(body["messages"][1]["content"])
        round_index = int(prompt["round_index"])
        calls.append(round_index)
        ids = [item["badcase_id"] for item in prompt["badcases"]]
        groups = (
            [
                {
                    "error_type": "trend confusion",
                    "likely_data_cause": "hard examples are under-represented",
                    "badcase_ids": [ids[0]],
                }
            ]
            if ids
            else []
        )
        proposals = {
            0: {
                "family": "learning_rate",
                "patch": {"learning_rate": 5e-6},
                "rationale": "stabilize Stage2",
            },
            1: {
                "family": "projector_lr_ratio",
                "patch": {"projector_lr_ratio": 0.5},
                "rationale": "reduce projector drift",
            },
            2: None,
        }
        proposal = proposals[round_index]
        content = json.dumps(
            {
                "error_groups": groups,
                "recommended_family": (
                    proposal["family"] if proposal is not None else "scheduler"
                ),
                "proposal": proposal,
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = DeepSeekClient(
        config.data["deepseek"], state, httpx.MockTransport(handler)
    )
    controller = Autoresearch(config, deepseek_client=client)
    controller.prepare_data()
    controller.baseline()
    result = controller.search()
    assert calls == [0, 1, 2]
    assert len(result["proxies"]) == 2
    search_manifest = json.loads(
        (project["artifacts"] / "SEARCH_COMPLETE.json").read_text()
    )
    assert set(search_manifest["analysis_hashes"]) == {
        "round-00.json",
        "round-01.json",
        "round-02.json",
    }
    final_round = json.loads(
        (project["artifacts"] / "analysis" / "round-02.json").read_text()
    )
    assert final_round["proposal"] is None
    assert final_round["sampled_badcase_ids"]
    assert set(final_round["error_groups"][0]["badcase_ids"]).issubset(
        final_round["sampled_badcase_ids"]
    )
    controller.search()
    assert calls == [0, 1, 2]
    (project["artifacts"] / "analysis" / "round-02.json").unlink()
    with pytest.raises(OrchestrationError, match="deleted or changed"):
        controller.freeze()
    controller.close()


def test_search_manifest_excludes_stale_full_and_checks_rank_lineage(
    project: dict[str, Path],
) -> None:
    controller = Autoresearch(load_config(project["config"]))
    controller.prepare_data()
    controller.baseline()
    controller.search()
    controller.state.create_experiment(
        {
            "id": "stale-full",
            "kind": "candidate",
            "phase": "full",
            "parent_id": "baseline",
            "config_hash": "stale-config",
            "dataset_hash": "stale-data",
            "protocol_hash": "stale-protocol",
            "config_json": {"role": "stale"},
            "output_dir": str(project["artifacts"] / "evaluations" / "stale"),
        }
    )
    controller.state.mark_running("stale-full", "stale-command", {"argv": ["true"]})
    controller.state.mark_completed(
        "stale-full", {"primary_score": 0.99, "gate_pass": True}, "/tmp/stale"
    )
    frozen = controller.freeze()
    assert frozen["champion"]["experiment_id"].startswith("full-")
    assert frozen["champion"]["experiment_id"] != "stale-full"

    search_path = project["artifacts"] / "SEARCH_COMPLETE.json"
    search_manifest = json.loads(search_path.read_text())
    search_manifest["ranking"] = list(reversed(search_manifest["ranking"]))
    unsigned = {key: value for key, value in search_manifest.items() if key != "search_hash"}
    search_manifest["search_hash"] = hash_object(unsigned)
    search_path.write_text(json.dumps(search_manifest))
    with pytest.raises(OrchestrationError, match="ranked top"):
        controller._load_search_manifest()
    controller.close()


def test_completed_eval_reuse_requires_untampered_raw_artifacts(
    project: dict[str, Path],
) -> None:
    controller = Autoresearch(load_config(project["config"]))
    controller.prepare_data()
    baseline = controller.baseline()
    predictions_path = Path(baseline["output_dir"]) / "tsrbench" / "predictions.jsonl"
    original_predictions = predictions_path.read_text()
    predictions_path.write_text(original_predictions.replace('"prediction": "A"', '"prediction": "C"', 1))
    with pytest.raises(OrchestrationError, match="evaluation artifacts changed"):
        controller.baseline()
    predictions_path.write_text(original_predictions)
    metrics_path = Path(baseline["output_dir"]) / "metrics.json"
    metrics_path.unlink()
    with pytest.raises(OrchestrationError, match="metrics.json is missing"):
        controller.baseline()
    controller.close()


def test_official_split_file_names_and_passthrough_assets(project: dict[str, Path]) -> None:
    tsr_root = project["datav2"].parent / "official-tsr"
    tsr_root.mkdir()
    (tsr_root / "dev.jsonl").write_text(
        "".join(json.dumps({"id": f"dev-{index}"}) + "\n" for index in range(4))
    )
    (tsr_root / "test.jsonl").write_text(
        "".join(json.dumps({"id": f"test-{index}"}) + "\n" for index in range(4))
    )
    (tsr_root / "README.txt").write_text("shared benchmark metadata\n")
    config = load_config(project["config"])
    config.data["paths"]["tsrbench_root"] = str(tsr_root)
    manifest = create_eval_dataset_views(config)
    assert manifest["modes"]["tsrbench"] == "official-dev-test"
    assert manifest["counts"]["tsrbench"] == {"search-dev": 4, "final-test": 4}
    for split in ("search-dev", "final-test"):
        asset = project["artifacts"] / "eval_views" / split / "tsrbench" / "README.txt"
        assert asset.is_symlink()
        assert asset.read_text() == "shared benchmark metadata\n"


def test_tiny_badcases_keep_binary_and_mc2_diagnostic_accounting(
    tmp_path: Path,
) -> None:
    tiny = tmp_path / "evaluation" / "tinybenchmarks"
    tiny.mkdir(parents=True)
    rows = [
        {"metric": "acc_norm", "score": 1.0, "predicted_index": 0, "gold_indices": [0]},
        {"metric": "acc_norm", "score": 0.0, "predicted_index": 1, "gold_indices": [0]},
    ]
    (tiny / "samples_tinyArc.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    truth = [
        {"metric": "mc2", "score": 0.2, "predicted_index": 2, "gold_indices": [0, 1]},
        {"metric": "mc2", "score": 0.8, "predicted_index": 0, "gold_indices": [0, 1]},
    ]
    (tiny / "samples_tinyTruthfulQA.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in truth)
    )
    destination = tmp_path / "badcases.jsonl"
    summary = extract_badcases(tmp_path / "evaluation", destination)
    assert summary["scored_records"] == 2
    assert summary["badcases"] == 1
    assert summary["diagnostic_badcases"] == 1
    assert summary["by_suite"]["tinybenchmarks"] == {
        "scored_records": 2,
        "badcases": 1,
        "diagnostic_badcases": 1,
    }
    badcases = [json.loads(line) for line in destination.read_text().splitlines()]
    assert {item["accounting"] for item in badcases} == {
        "official_binary_correctness",
        "diagnostic_top1_only",
    }
