import json
import tempfile
import threading
import unittest
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


if __name__ == "__main__":
    unittest.main()
