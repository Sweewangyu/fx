import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

import server


class InspectorServerTest(unittest.TestCase):
    def test_normalization_preserves_task_but_replaces_numbers(self):
        prompt = "Forecast the next 32 points from <ts><ts/> at 125 Hz."
        self.assertEqual(
            server.normalize_template(prompt),
            "forecast the next <num> points from <ts> at <num> hz.",
        )

    def test_json_array_offsets_support_pretty_printed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(json.dumps([{"text": "a,b"}, {"nested": [1, {"x": 2}]}], indent=2), encoding="utf-8")
            slices = list(server.line_offsets(path))
            self.assertEqual(len(slices), 2)
            rows = [server.read_slice(path, offset, length) for _, offset, length in slices]
            self.assertEqual(rows, [{"text": "a,b"}, {"nested": [1, {"x": 2}]}])

    def test_answer_leakage_and_visual_mismatch(self):
        issues = server.issue_flags(
            'You MUST end your response with "Answer: sitting".',
            "The graph shows a stable series.",
            {"verifier": {"reasoning_status": "source_rationale_unverified"}},
        )
        self.assertEqual(
            issues,
            ["answer_leakage", "visual_grounding_mismatch", "unverified_reasoning"],
        )

    def test_fixture_dataset_random_access_and_template_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            files = data_root / "files"
            audit = data_root / "audit"
            files.mkdir(parents=True)
            audit.mkdir()
            rows = [
                {"input": "Classify <ts><ts/> into A or B.", "timeseries": [[1, 2, 3]], "output": "A"},
                {"input": "Classify <ts><ts/> into A or B.", "timeseries": [[4, 5, 6]], "output": "B"},
            ]
            (files / "fixture.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (audit / "fixture.audit.jsonl").write_text(
                "".join(json.dumps({"sample_index": i, "task": "classification"}) + "\n" for i in range(2)),
                encoding="utf-8",
            )
            registry = {
                "sources": [
                    {
                        "name": "fixture",
                        "family": "test",
                        "path": "files/fixture.jsonl",
                        "audit": "audit/fixture.audit.jsonl",
                        "split": "train",
                        "training_role": "sft",
                    }
                ]
            }
            (data_root / "sources.json").write_text(json.dumps(registry), encoding="utf-8")
            config = {
                "data": {"root": "data", "registry": "data/sources.json", "template_stats": "missing.json"},
                "qwen": {"base_url": "http://localhost:1/v1", "model": "fixture", "allow_no_key": True},
                "server": {"cache_dir": "cache", "max_chart_points": 100},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            store = server.DatasetStore(config_path)
            payload = store.record("fixture", 1)
            self.assertEqual(payload["answer_class"], "B")
            self.assertEqual(payload["series"][0]["values"], [4.0, 5.0, 6.0])
            self.assertEqual(payload["template"]["members"], 2)
            members = store.template_members("fixture", payload["taxonomy_template_id"], 0, 10)
            self.assertEqual([item["answer_class"] for item in members["members"]], ["A", "B"])
            self.assertEqual(store.datasets()["datasets"][0]["rows"], 2)

    def test_tsrbench_standard_and_abductive_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "sources.json").write_text('{"sources": []}', encoding="utf-8")
            benchmark = root / "TSRBench" / "dataset"
            reasoning = benchmark / "reasoning"
            reasoning.mkdir(parents=True)
            standard = {
                "question": "Which series rises fastest?",
                "answer": "B",
                "domain": "finance",
                "name_of_series": ["Alpha", "Beta"],
                "timeseries": [[1, 2, 3], [1, 4, 9]],
                "choices": ["Alpha", "Beta", "Neither"],
            }
            abductive = {
                "context": {
                    "history_events": ["Team A scores"],
                    "history_times": ["Q4 01:00"],
                    "future_events": ["Team B takes the lead"],
                    "future_times": ["Q4 00:30"],
                },
                "numerical_time_series": {
                    "wp_Team A": {"history": [0.7], "future": [0.4]},
                    "wp_Team B": {"history": [0.3], "future": [0.6]},
                },
                "multiple_choice_question": {
                    "question": "What most likely happened?",
                    "choices": ["Turnover", "Timeout"],
                    "answer": "A",
                },
            }
            (reasoning / "causal_reasoning.jsonl").write_text(json.dumps(standard) + "\n", encoding="utf-8")
            (reasoning / "abductive_reasoning.jsonl").write_text(json.dumps(abductive) + "\n", encoding="utf-8")
            config = {
                "data": {
                    "root": "data",
                    "registry": "data/sources.json",
                    "tsrbench_root": "TSRBench/dataset",
                },
                "qwen": {"base_url": "http://localhost:1/v1", "model": "fixture", "allow_no_key": True},
                "server": {"cache_dir": "cache", "max_chart_points": 100},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            store = server.DatasetStore(config_path)
            self.assertEqual(set(store.sources), {"tsrbench_causal_reasoning", "tsrbench_abductive_reasoning"})
            causal = store.record("tsrbench_causal_reasoning", 0)
            self.assertEqual(causal["schema"], "tsrbench")
            self.assertEqual(causal["input"], standard["question"])
            self.assertEqual(causal["choices"], standard["choices"])
            self.assertEqual(causal["series_names"], ["Alpha", "Beta"])
            self.assertEqual(causal["benchmark"]["major"], "Reasoning")
            self.assertEqual(causal["benchmark"]["domain"], "finance")

            missing_event = store.record("tsrbench_abductive_reasoning", 0)
            self.assertIn("A CRITICAL EVENT HAPPENED HERE", missing_event["input"])
            self.assertEqual(missing_event["output"], "A")
            self.assertEqual(missing_event["series_count"], 2)
            self.assertEqual(missing_event["series_names"], ["wp_Team A", "wp_Team B"])
            self.assertEqual(missing_event["choices"], ["Turnover", "Timeout"])
            self.assertTrue(store.tsrbench_status["found"])
            self.assertEqual(store.tsrbench_status["tasks_found"], 2)

    def test_tsrbench_missing_root_has_explicit_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "sources.json").write_text('{"sources": []}', encoding="utf-8")
            config = {
                "data": {
                    "root": "data",
                    "registry": "data/sources.json",
                    "tsrbench_root": ["missing-one/dataset", "missing-two/dataset"],
                },
                "qwen": {"base_url": "http://localhost:1/v1", "model": "fixture", "allow_no_key": True},
                "server": {"cache_dir": "cache"},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            store = server.DatasetStore(config_path)
            payload = store.datasets()
            self.assertFalse(payload["tsrbench"]["found"])
            self.assertEqual(payload["tsrbench"]["tasks_found"], 0)
            self.assertIn(str((root / "missing-one" / "dataset").resolve()), payload["tsrbench"]["checked_paths"])

    def test_imports_multi_model_json_and_jsonl_shards_by_task_and_idx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "sources.json").write_text('{"sources": []}', encoding="utf-8")
            benchmark = root / "TSRBench" / "dataset" / "reasoning"
            benchmark.mkdir(parents=True)
            samples = [
                {
                    "question": "First?",
                    "answer": "A",
                    "timeseries": [[1, 2]],
                    "choices": ["yes", "no"],
                },
                {
                    "question": "Second?",
                    "answer": "B",
                    "timeseries": [[2, 1]],
                    "choices": ["yes", "no"],
                },
            ]
            (benchmark / "causal_reasoning.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
            )
            results = root / "results"
            alpha = results / "causal_reasoning_model_alpha"
            alpha.mkdir(parents=True)
            (alpha / "generated_answer.json").write_text(
                json.dumps(
                    [
                        {"idx": 0, "response": "A", "answer": "A", "prompt_mode": "answer_only"},
                        {"idx": 1, "response": "A", "answer": "A", "prompt_mode": "answer_only"},
                        {"response": "missing idx"},
                    ]
                ),
                encoding="utf-8",
            )
            beta = results / "causal_reasoning_model_beta"
            beta.mkdir(parents=True)
            (beta / "generated_answer_2_0.jsonl").write_text(
                json.dumps({"idx": 0, "response": "B", "answer": "B", "prompt_mode": "official"}) + "\n",
                encoding="utf-8",
            )
            (beta / "generated_answer_2_1.json").write_text(
                json.dumps(
                    [
                        {"idx": 0, "response": "A", "answer": "A", "prompt_mode": "official"},
                        {"idx": 1, "response": "B", "answer": "B", "prompt_mode": "official"},
                    ]
                ),
                encoding="utf-8",
            )
            config = {
                "data": {
                    "root": "data",
                    "registry": "data/sources.json",
                    "tsrbench_root": "TSRBench/dataset",
                },
                "qwen": {"base_url": "http://localhost:1/v1", "model": "fixture", "allow_no_key": True},
                "evaluation_results": {"root": "results"},
                "review": {"state_db": "state/inspector-state.sqlite3"},
                "server": {"cache_dir": "cache"},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            store = server.DatasetStore(config_path)
            imported = store.import_model_results()

            self.assertEqual(len(imported["runs"]), 2)
            record0 = store.record("tsrbench_causal_reasoning", 0)
            self.assertEqual(len(record0["model_responses"]), 2)
            self.assertEqual(
                {row["model_name"]: row["status"] for row in record0["model_responses"]},
                {"model_alpha": "correct", "model_beta": "error"},
            )
            alpha_run = next(row for row in store.model_runs() if row["model_name"] == "model_alpha")
            self.assertEqual(alpha_run["invalid"], 1)
            beta_run = next(row for row in store.model_runs() if row["model_name"] == "model_beta")
            self.assertTrue(
                any(item["type"] == "duplicate_index_conflict" for item in beta_run["diagnostics"])
            )
            badcase = store.next_badcase(alpha_run["run_id"], "tsrbench_causal_reasoning", -1)
            self.assertEqual(badcase["index"], 1)
            page = store.badcases(alpha_run["run_id"], None, "incorrect", 0, 10)
            self.assertEqual([item["index"] for item in page["items"]], [1])

            # Re-importing an unchanged root is idempotent.
            store.import_model_results()
            self.assertEqual(len(store.model_runs()), 2)
            self.assertEqual(len(store.record("tsrbench_causal_reasoning", 0)["model_responses"]), 2)

    def test_model_identity_prompt_modes_missing_and_mutually_exclusive_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "sources.json").write_text('{"sources": []}', encoding="utf-8")
            benchmark = root / "TSRBench" / "dataset" / "reasoning"
            benchmark.mkdir(parents=True)
            samples = [
                {"question": f"Question {index}?", "answer": answer, "timeseries": [[index]]}
                for index, answer in enumerate(("A", "B", "C"))
            ]
            (benchmark / "causal_reasoning.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
            )

            results = root / "results"
            model3 = results / "text" / "causal_reasoning_OpenGVLab" / "InternVL3-8B"
            model4 = results / "text" / "causal_reasoning_OpenGVLab" / "InternVL4-8B"
            model3.mkdir(parents=True)
            model4.mkdir(parents=True)
            (model3 / "generated_answer.json").write_text(
                json.dumps(
                    [
                        {"idx": 0, "raw_response": "A", "prompt_mode": "answer_only"},
                        {
                            "idx": 1,
                            "response": "B",
                            "answer": "B",
                            "error": "generation failed after partial output",
                            "prompt_mode": "official",
                        },
                        {"idx": True, "response": "B", "prompt_mode": "official"},
                    ]
                ),
                encoding="utf-8",
            )
            model4_file = model4 / "generated_answer.jsonl"
            model4_file.write_text(
                json.dumps({"idx": 0, "completion": "A", "prompt_mode": "official"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "data": {
                    "root": "data",
                    "registry": "data/sources.json",
                    "tsrbench_root": "TSRBench/dataset",
                },
                "qwen": {"base_url": "http://localhost:1/v1", "model": "fixture", "allow_no_key": True},
                "evaluation_results": {"root": "results"},
                "review": {"state_db": "state/inspector-state.sqlite3"},
                "server": {"cache_dir": "cache"},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            store = server.DatasetStore(config_path)
            store.import_model_results()

            runs = store.model_runs()
            self.assertEqual(
                {(run["model_name"], run["prompt_mode"]) for run in runs},
                {
                    ("OpenGVLab/InternVL3-8B", "answer_only"),
                    ("OpenGVLab/InternVL3-8B", "official"),
                    ("OpenGVLab/InternVL4-8B", "official"),
                },
            )
            record0 = store.record("tsrbench_causal_reasoning", 0)["model_responses"]
            statuses0 = {
                (row["model_name"], row["prompt_mode"]): row["status"] for row in record0
            }
            self.assertEqual(statuses0[("OpenGVLab/InternVL3-8B", "answer_only")], "correct")
            self.assertEqual(statuses0[("OpenGVLab/InternVL3-8B", "official")], "missing")
            record1 = store.record("tsrbench_causal_reasoning", 1)["model_responses"]
            statuses1 = {
                (row["model_name"], row["prompt_mode"]): row["status"] for row in record1
            }
            self.assertEqual(statuses1[("OpenGVLab/InternVL3-8B", "official")], "error")
            official3 = next(
                run for run in runs
                if run["model_name"] == "OpenGVLab/InternVL3-8B" and run["prompt_mode"] == "official"
            )
            self.assertEqual((official3["correct"], official3["errors"]), (0, 1))
            official4 = next(run for run in runs if run["model_name"] == "OpenGVLab/InternVL4-8B")
            self.assertEqual(
                store.next_badcase(official4["run_id"], "tsrbench_causal_reasoning", 0)["index"],
                1,
            )

            # A successful full refresh removes runs whose source disappeared.
            model4_file.unlink()
            store.import_model_results()
            self.assertNotIn(
                "OpenGVLab/InternVL4-8B", {run["model_name"] for run in store.model_runs()}
            )

    def test_human_good_bad_annotations_persist_progress_next_export_and_reject_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            files = data_root / "files"
            files.mkdir(parents=True)
            rows = [
                {"input": f"Question {index}", "timeseries": [[index]], "output": "answer"}
                for index in range(3)
            ]
            (files / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (data_root / "sources.json").write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "name": "train_fixture",
                                "family": "test",
                                "path": "files/train.jsonl",
                                "split": "train",
                                "training_role": "sft",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            benchmark = root / "TSRBench" / "dataset" / "perception"
            benchmark.mkdir(parents=True)
            (benchmark / "perception.jsonl").write_text(
                json.dumps({"question": "Benchmark?", "answer": "A", "timeseries": [[1]]}) + "\n",
                encoding="utf-8",
            )
            config = {
                "data": {
                    "root": "data",
                    "registry": "data/sources.json",
                    "tsrbench_root": "TSRBench/dataset",
                },
                "qwen": {"base_url": "http://localhost:1/v1", "model": "fixture", "allow_no_key": True},
                "review": {"state_db": "state/inspector-state.sqlite3"},
                "server": {"cache_dir": "cache"},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            store = server.DatasetStore(config_path)

            first = store.save_human_annotation("train_fixture", 0, "good", annotator="alice", expected_revision=0)
            self.assertEqual(first["label"], "good")
            self.assertEqual(first["revision"], 1)
            with self.assertRaises(server.AnnotationConflict):
                store.save_human_annotation("train_fixture", 0, "bad", expected_revision=0)
            changed = store.save_human_annotation(
                "train_fixture", 0, "bad", comment="bad rationale", expected_revision=1
            )
            self.assertEqual(changed["revision"], 2)
            store.save_human_annotation("train_fixture", 1, "good", expected_revision=0)
            progress = store.human_progress("train_fixture")
            self.assertEqual((progress["labeled"], progress["good"], progress["bad"]), (2, 1, 1))
            self.assertEqual(store.next_unlabeled("train_fixture", -1)["index"], 2)
            exported, _, _ = store.export_human_annotations("jsonl", "train_fixture")
            self.assertEqual(len([line for line in exported.decode().splitlines() if line]), 2)
            self.assertNotIn(".cache", str(store.state_db))

            recreated = server.DatasetStore(config_path)
            persisted = recreated.record("train_fixture", 0)["human_review"]
            self.assertEqual((persisted["label"], persisted["revision"]), ("bad", 2))
            with self.assertRaises(server.ApiError) as context:
                recreated.save_human_annotation("tsrbench_perception", 0, "good")
            self.assertEqual(context.exception.status, 403)
            cleared = recreated.delete_human_annotation("train_fixture", 0, expected_revision=2)
            self.assertIsNone(cleared["label"])
            self.assertEqual(cleared["revision"], 3)
            self.assertEqual(cleared["progress"]["labeled"], 1)
            with self.assertRaises(server.AnnotationConflict):
                recreated.save_human_annotation(
                    "train_fixture", 0, "good", expected_revision=2
                )
            restored = recreated.save_human_annotation(
                "train_fixture", 0, "good", expected_revision=3
            )
            self.assertEqual(restored["revision"], 4)

            # Source changes conservatively invalidate old labels: they no
            # longer count as progress, are revisited, or appear in the default export.
            changed_rows = [
                {"input": f"Changed question {index}", "timeseries": [[index]], "output": "answer"}
                for index in range(3)
            ]
            (files / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in changed_rows), encoding="utf-8"
            )
            self.assertEqual(recreated.human_progress("train_fixture")["labeled"], 0)
            self.assertEqual(recreated.next_unlabeled("train_fixture", -1)["index"], 0)
            fresh_export, _, _ = recreated.export_human_annotations("jsonl", "train_fixture")
            self.assertEqual(fresh_export, b"")
            stale_export, _, _ = recreated.export_human_annotations(
                "jsonl", "train_fixture", include_stale=True
            )
            self.assertTrue(all(json.loads(line)["stale"] for line in stale_export.splitlines()))
            self.assertTrue(recreated.record("train_fixture", 0)["human_review"]["stale"])

    def test_human_label_http_alias_validates_required_fields_and_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            files = data_root / "files"
            files.mkdir(parents=True)
            (files / "train.jsonl").write_text(
                json.dumps({"input": "Question", "timeseries": [[1]], "output": "Answer"}) + "\n",
                encoding="utf-8",
            )
            (data_root / "sources.json").write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "name": "train_fixture",
                                "family": "test",
                                "path": "files/train.jsonl",
                                "split": "train",
                                "training_role": "sft",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "data": {"root": "data", "registry": "data/sources.json"},
                "qwen": {"base_url": "http://localhost:1/v1", "model": "fixture", "allow_no_key": True},
                "review": {"state_db": "state/inspector-state.sqlite3"},
                "server": {"cache_dir": "cache"},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            server.ApiHandler.store = server.DatasetStore(config_path)
            server.ApiHandler.allowed_origin = "*"
            api = ThreadingHTTPServer(("127.0.0.1", 0), server.ApiHandler)
            thread = threading.Thread(target=api.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{api.server_port}"

            def post(payload):
                request = urllib.request.Request(
                    base + "/api/human-label",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    return response.status, json.loads(response.read())

            try:
                with self.assertRaises(urllib.error.HTTPError) as missing_label:
                    post({"dataset": "train_fixture", "index": 0})
                self.assertEqual(missing_label.exception.code, 400)
                with self.assertRaises(urllib.error.HTTPError) as unsafe_index:
                    post({"dataset": "train_fixture", "index": True, "label": "good"})
                self.assertEqual(unsafe_index.exception.code, 400)

                status, saved = post(
                    {
                        "dataset": "train_fixture",
                        "index": 0,
                        "label": "good",
                        "annotator": "http-test",
                        "expected_revision": 0,
                    }
                )
                self.assertEqual(status, 200)
                self.assertEqual(saved["human_review"]["label"], "good")
                with urllib.request.urlopen(
                    base + "/api/human-labels/next?dataset=train_fixture&after=-1", timeout=5
                ) as response:
                    self.assertTrue(json.loads(response.read())["complete"])
            finally:
                api.shutdown()
                api.server_close()
                thread.join(timeout=2)

    def test_qwen_translation_proxy_and_cache(self):
        requests = []

        class FakeQwenHandler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                payload = {
                    "choices": [
                        {"message": {"content": json.dumps({"input": "准确的中文译文"}, ensure_ascii=False)}}
                    ]
                }
                encoded = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        qwen_server = ThreadingHTTPServer(("127.0.0.1", 0), FakeQwenHandler)
        thread = threading.Thread(target=qwen_server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data_root = root / "data"
                data_root.mkdir()
                (data_root / "sources.json").write_text('{"sources": []}', encoding="utf-8")
                config = {
                    "data": {"root": "data", "registry": "data/sources.json"},
                    "qwen": {
                        "base_url": f"http://127.0.0.1:{qwen_server.server_port}/v1",
                        "model": "fake-qwen",
                        "allow_no_key": True,
                    },
                    "server": {"cache_dir": "cache"},
                }
                config_path = root / "config.yaml"
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                store = server.DatasetStore(config_path)

                first = store.translate({"input": "Translate this."})
                second = store.translate({"input": "Translate this."})

                self.assertEqual(first["translations"]["input"], "准确的中文译文")
                self.assertFalse(first["cached"])
                self.assertTrue(second["cached"])
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        finally:
            qwen_server.shutdown()
            qwen_server.server_close()
            thread.join(timeout=2)

    def test_qwen_http_400_uses_compatibility_retry(self):
        requests = []

        class StrictQwenHandler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                if "response_format" in body:
                    encoded = b'{"error":{"message":"response_format is unsupported"}}'
                    self.send_response(400)
                else:
                    encoded = json.dumps(
                        {"choices": [{"message": {"content": '{"input":"兼容译文"}'}}]}
                    ).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        qwen_server = ThreadingHTTPServer(("127.0.0.1", 0), StrictQwenHandler)
        thread = threading.Thread(target=qwen_server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data_root = root / "data"
                data_root.mkdir()
                (data_root / "sources.json").write_text('{"sources": []}', encoding="utf-8")
                config = {
                    "data": {"root": "data", "registry": "data/sources.json"},
                    "qwen": {
                        "base_url": f"http://127.0.0.1:{qwen_server.server_port}/v1",
                        "model": "strict-qwen",
                        "allow_no_key": True,
                        "max_tokens": 8192,
                    },
                    "server": {"cache_dir": "cache"},
                }
                config_path = root / "config.yaml"
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                result = server.DatasetStore(config_path).translate({"input": "Translate."})

                self.assertEqual(result["translations"]["input"], "兼容译文")
                self.assertEqual(len(requests), 2)
                self.assertIn("response_format", requests[0])
                self.assertNotIn("response_format", requests[1])
                self.assertEqual(requests[1]["max_tokens"], 1024)
        finally:
            qwen_server.shutdown()
            qwen_server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
