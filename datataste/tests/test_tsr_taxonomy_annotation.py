import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.annotate_tsr_taxonomy import (
    SourceSpec,
    _build_annotation_payload,
    annotate_online_command,
    build_distribution_report,
    _load_human,
    _parse_model_json,
    iter_source,
    materialize_command,
    normalize_template,
    resolve_command,
    rule_label,
)


class RuleBoundaryTests(unittest.TestCase):
    def label(self, question, source="chatts_sft", task=""):
        audit = {"task": task} if task else None
        return rule_label(SourceSpec(source, "unused"), question, "answer", audit)

    def test_perception_labels(self):
        self.assertEqual(self.label("Describe the trend and periodicity.").primary_label, "PR")
        self.assertEqual(self.label("What are the noise characteristics and noise magnitude?").primary_label, "NU")
        self.assertEqual(self.label("Locate the anomalous segment.").primary_label, "AD")
        self.assertEqual(self.label("Compare the trend between series A and B.").primary_label, "CA")

    def test_reasoning_labels(self):
        self.assertEqual(self.label("What underlying cause generated this observed series?").primary_label, "ER")
        self.assertEqual(self.label("Find the directed causal relationship between the rivers.").primary_label, "CD")
        self.assertEqual(self.label("What might have happened during this sudden change?").primary_label, "AR")
        self.assertEqual(self.label("Put these events in chronological order.").primary_label, "TR")
        self.assertEqual(self.label("Calculate the mean value of the series.").primary_label, "NR")
        self.assertEqual(self.label("According to the stated equation, derive the correct result.").primary_label, "DR")
        self.assertEqual(self.label("Infer the underlying rule and predict the next symbol.").primary_label, "IR")

    def test_prediction_and_decision_labels(self):
        self.assertEqual(self.label("Predict the next 12 time series values.").primary_label, "TSF")
        self.assertEqual(self.label("Predict whether the machine will fail within 24 hours.").primary_label, "EP")
        self.assertEqual(self.label("Which treatment is the most appropriate management action?").primary_label, "QualDM")
        self.assertEqual(self.label("Backtest each strategy and choose the best return.").primary_label, "QuantDM")

    def test_metadata_and_out_of_scope(self):
        tsaqa = self.label("Classify the given time series.", source="tsaqa", task="classification")
        self.assertEqual(tsaqa.primary_label, "PR")
        self.assertEqual(tsaqa.taxonomy_fit, "closest")
        imputation = self.label("Fill the missing entries.", source="time_mqa", task="imputation")
        self.assertIsNone(imputation.primary_label)
        self.assertEqual(imputation.closest_label, "TSF")
        self.assertEqual(imputation.taxonomy_fit, "out_of_scope")

    def test_template_normalization_masks_values(self):
        first = normalize_template("Series 12.5 is <ts><ts/>. Predict next 8 values")
        second = normalize_template("Series 99.1 is <ts><ts/>. Predict next 20 values")
        self.assertEqual(first, second)

    def test_model_json_validation(self):
        parsed = _parse_model_json(
            '```json\n{"primary_label":"AD","secondary_labels":[],"taxonomy_fit":"exact",'
            '"confidence":0.9,"rationale":"anomaly"}\n```'
        )
        self.assertEqual(parsed["primary_label"], "AD")

    def test_deepseek_thinking_payload_supports_effort(self):
        payload, config = _build_annotation_payload(
            SimpleNamespace(
                model="/models",
                max_tokens=300,
                json_mode=True,
                thinking_mode="enabled",
                disable_thinking=False,
                reasoning_effort="max",
            ),
            "system",
            "user",
        )
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(config["thinking_mode"], "enabled")

    def test_reasoning_effort_implicitly_enables_thinking(self):
        payload, _ = _build_annotation_payload(
            SimpleNamespace(
                model="/models",
                max_tokens=300,
                json_mode=False,
                thinking_mode=None,
                disable_thinking=False,
                reasoning_effort="high",
            ),
            "system",
            "user",
        )
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertNotIn("temperature", payload)

    def test_reasoning_effort_rejects_disabled_thinking(self):
        with self.assertRaisesRegex(ValueError, "requires thinking mode"):
            _build_annotation_payload(
                SimpleNamespace(
                    model="/models",
                    max_tokens=300,
                    json_mode=False,
                    thinking_mode="disabled",
                    disable_thinking=False,
                    reasoning_effort="max",
                ),
                "system",
                "user",
            )

    @mock.patch("httpx.post")
    def test_annotate_online_keeps_reasoning_private_and_parses_final_content(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "private chain of thought",
                        "content": json.dumps(
                            {
                                "primary_label": "AD",
                                "secondary_labels": [],
                                "taxonomy_fit": "exact",
                                "confidence": 0.94,
                                "rationale": "locates anomalies",
                            }
                        ),
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "review.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "cluster_id": "cluster",
                        "source": "time_mqa",
                        "source_task": "anomaly detection",
                        "question_type": "unknown",
                        "representative_input": "Locate anomalies in <ts><ts/>.",
                        "representative_output": "Point 5 is anomalous.",
                        "provisional": {"primary_label": "AD"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "votes.jsonl"
            result = annotate_online_command(
                SimpleNamespace(
                    input=str(input_path),
                    output=str(output_path),
                    base_url="http://localhost:30000/v1",
                    model="/models",
                    api_key_env="UNSET_TEST_API_KEY",
                    allow_no_key=True,
                    workers=1,
                    limit=0,
                    timeout=30.0,
                    retries=0,
                    max_tokens=2048,
                    json_mode=True,
                    thinking_mode="enabled",
                    disable_thinking=False,
                    reasoning_effort="max",
                )
            )
            self.assertEqual(result, 0)
            request_payload = post.call_args.kwargs["json"]
            self.assertEqual(request_payload["thinking"], {"type": "enabled"})
            self.assertEqual(request_payload["reasoning_effort"], "max")
            vote = json.loads(output_path.read_text())
            self.assertTrue(vote["inference"]["reasoning_content_present"])
            self.assertNotIn("reasoning_content", vote)
            self.assertNotIn("private chain of thought", output_path.read_text())
            self.assertEqual(vote["vote"]["primary_label"], "AD")

    def test_authoritative_vote_overrides_uncertain_model_but_not_auto_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provisional_path = root / "provisional.jsonl"
            rows = [
                {
                    "sample_id": "auto",
                    "source": "source",
                    "split": "train",
                    "source_index": 0,
                    "cluster_id": "auto-cluster",
                    "provisional": {
                        "primary_label": "PR",
                        "secondary_labels": [],
                        "closest_label": "PR",
                        "taxonomy_fit": "exact",
                        "confidence": 0.99,
                        "status": "auto_accept",
                    },
                },
                {
                    "sample_id": "review",
                    "source": "source",
                    "split": "train",
                    "source_index": 1,
                    "cluster_id": "review-cluster",
                    "provisional": {
                        "primary_label": "PR",
                        "secondary_labels": [],
                        "closest_label": "PR",
                        "taxonomy_fit": "exact",
                        "confidence": 0.8,
                        "status": "review",
                    },
                },
                {
                    "sample_id": "excluded",
                    "source": "source",
                    "split": "train",
                    "source_index": 2,
                    "cluster_id": "excluded-cluster",
                    "provisional": {
                        "primary_label": None,
                        "secondary_labels": [],
                        "closest_label": "TSF",
                        "taxonomy_fit": "out_of_scope",
                        "confidence": 0.65,
                        "status": "review",
                    },
                },
            ]
            provisional_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            qwen_path = root / "qwen.jsonl"
            qwen_path.write_text(
                json.dumps(
                    {
                        "cluster_id": "review-cluster",
                        "model": "qwen",
                        "vote": {
                            "primary_label": "PR",
                            "secondary_labels": [],
                            "taxonomy_fit": "exact",
                            "confidence": 0.9,
                            "rationale": "qwen",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            authoritative_path = root / "deepseek.jsonl"
            authoritative_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "cluster_id": "auto-cluster",
                            "model": "/models",
                            "inference": {
                                "thinking_mode": "enabled",
                                "reasoning_effort": "max",
                            },
                            "vote": {
                                "primary_label": "AD",
                                "secondary_labels": [],
                                "taxonomy_fit": "exact",
                                "confidence": 0.91,
                                "rationale": "should not override auto rule",
                            },
                        },
                        {
                            "cluster_id": "review-cluster",
                            "model": "/models",
                            "inference": {
                                "thinking_mode": "enabled",
                                "reasoning_effort": "max",
                            },
                            "vote": {
                                "primary_label": "IR",
                                "secondary_labels": ["PR"],
                                "taxonomy_fit": "exact",
                                "confidence": 0.93,
                                "rationale": "DeepSeek final decision",
                            },
                        },
                        {
                            "cluster_id": "excluded-cluster",
                            "model": "/models",
                            "inference": {
                                "thinking_mode": "enabled",
                                "reasoning_effort": "high",
                            },
                            "vote": {
                                "primary_label": None,
                                "secondary_labels": [],
                                "taxonomy_fit": "out_of_scope",
                                "confidence": 0.97,
                                "rationale": "not one of the fifteen tasks",
                            },
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "final.jsonl"
            resolve_command(
                SimpleNamespace(
                    provisional=str(provisional_path),
                    votes=[str(qwen_path)],
                    authoritative_votes=[str(authoritative_path)],
                    human=None,
                    clusters=None,
                    output=str(output_path),
                )
            )
            final_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(final_rows[0]["final"]["primary_label"], "PR")
            self.assertEqual(final_rows[0]["final"]["method"], "rules")
            self.assertEqual(final_rows[1]["final"]["primary_label"], "IR")
            self.assertEqual(final_rows[1]["final"]["method"], "authoritative_model")
            self.assertEqual(final_rows[1]["final"]["model"], "/models")
            self.assertEqual(final_rows[1]["final"]["confidence"], 0.93)
            self.assertEqual(final_rows[1]["final"]["inference"]["reasoning_effort"], "max")
            self.assertEqual(final_rows[2]["final"]["status"], "excluded")
            self.assertEqual(final_rows[2]["final"]["method"], "authoritative_model")

    def test_invalid_jsonl_row_can_be_audited_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            valid = {"input": "Describe the trend.", "timeseries": [[1, 2]], "output": "up"}
            path.write_text(json.dumps(valid) + "\n{broken json\n", encoding="utf-8")
            invalid = []
            rows = list(iter_source(SourceSpec("source", str(path)), invalid.append))
            self.assertEqual(len(rows), 1)
            self.assertEqual(invalid[0]["source_index"], 1)
            self.assertEqual(invalid[0]["reason"], "JSONDecodeError")

    def test_human_csv_ignores_blank_rows_and_loads_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.csv"
            path.write_text(
                "cluster_id,human_primary_label,human_secondary_labels,human_taxonomy_fit\n"
                "todo,,,\n"
                "done,AD,TR,exact\n"
                "excluded,,,out_of_scope\n",
                encoding="utf-8",
            )
            labels = _load_human(str(path))
            self.assertNotIn("todo", labels)
            self.assertEqual(labels["done"]["primary_label"], "AD")
            self.assertEqual(labels["done"]["secondary_labels"], ["TR"])
            self.assertIsNone(labels["excluded"]["primary_label"])

    def test_materialize_defaults_can_exclude_non_train_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {"input": "trend?", "timeseries": [[1, 2]], "output": "up"}
            train_path = root / "train.jsonl"
            dev_path = root / "dev.jsonl"
            train_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            dev_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "sources": [
                            {"name": "train", "path": str(train_path), "split": "train"},
                            {"name": "dev", "path": str(dev_path), "split": "dev"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            labels = root / "labels.jsonl"
            final = {
                "primary_label": "PR",
                "secondary_labels": [],
                "taxonomy_fit": "exact",
                "confidence": 1.0,
                "status": "accepted",
                "method": "human",
            }
            labels.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "sample_id": name,
                            "source": name,
                            "split": split,
                            "source_index": 0,
                            "cluster_id": name,
                            "final": final,
                        }
                    )
                    for name, split in (("train", "train"), ("dev", "dev"))
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out"
            materialize_command(
                SimpleNamespace(
                    registry=str(registry),
                    labels=str(labels),
                    output_dir=str(output),
                    min_confidence=0.85,
                    include_fit=["exact", "compatible"],
                    splits=["train"],
                )
            )
            self.assertEqual((output / "PR_pattern_recognition.jsonl").read_text().count("\n"), 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["EXCLUDED_SPLIT"], 1)

    def test_materialize_deterministically_caps_each_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            samples = [
                {"input": f"question-{index}", "timeseries": [[index, index + 1]], "output": "answer"}
                for index in range(6)
            ]
            source_path.write_text(
                "\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8"
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {"sources": [{"name": "time_mqa", "path": str(source_path), "split": "train"}]}
                ),
                encoding="utf-8",
            )
            labels = root / "labels.jsonl"
            final = {
                "primary_label": "PR",
                "secondary_labels": [],
                "taxonomy_fit": "exact",
                "confidence": 0.96,
                "status": "accepted",
                "method": "authoritative_model",
            }
            label_rows = [
                {
                    "sample_id": f"sample-{index}",
                    "source": "time_mqa",
                    "split": "train",
                    "source_index": index,
                    "cluster_id": "large-template" if index < 5 else "single-template",
                    "final": final,
                }
                for index in range(6)
            ]
            labels.write_text(
                "\n".join(json.dumps(row) for row in label_rows) + "\n", encoding="utf-8"
            )

            def run(output: Path) -> None:
                materialize_command(
                    SimpleNamespace(
                        registry=str(registry),
                        labels=str(labels),
                        output_dir=str(output),
                        min_confidence=0.85,
                        include_fit=["exact", "compatible"],
                        splits=["train"],
                        max_per_template=2,
                        template_cap_sources=[],
                        template_sample_seed=17,
                    )
                )

            first_output = root / "first"
            second_output = root / "second"
            run(first_output)
            run(second_output)
            first_text = (first_output / "PR_pattern_recognition.jsonl").read_text()
            self.assertEqual(
                first_text,
                (second_output / "PR_pattern_recognition.jsonl").read_text(),
            )
            written = [json.loads(line)["input"] for line in first_text.splitlines()]
            chosen_large = sorted(
                range(5),
                key=lambda index: hashlib.sha256(f"17\0sample-{index}".encode()).hexdigest(),
            )[:2]
            expected = {f"question-{index}" for index in chosen_large} | {"question-5"}
            self.assertEqual(set(written), expected)

            manifest = json.loads((first_output / "manifest.json").read_text())
            sampling = manifest["template_sampling"]
            self.assertTrue(sampling["enabled"])
            self.assertEqual(sampling["candidate_samples"], 6)
            self.assertEqual(sampling["selected_samples"], 3)
            self.assertEqual(sampling["filtered_samples"], 3)
            self.assertEqual(sampling["template_clusters"], 2)
            self.assertEqual(sampling["capped_template_clusters"], 1)
            self.assertEqual(sampling["largest_template_cluster"], 5)
            self.assertEqual(manifest["counts"]["FILTERED_TEMPLATE_CAP"], 3)
            self.assertEqual(manifest["counts"]["PR"], 3)

    def test_materialize_template_cap_can_target_selected_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_specs = []
            labels_rows = []
            for source_name in ("chatts_sft", "tsaqa"):
                source_path = root / f"{source_name}.jsonl"
                samples = [
                    {
                        "input": f"{source_name}-{index}",
                        "timeseries": [[index, index + 1]],
                        "output": "answer",
                    }
                    for index in range(3)
                ]
                source_path.write_text(
                    "\n".join(json.dumps(sample) for sample in samples) + "\n",
                    encoding="utf-8",
                )
                source_specs.append(
                    {"name": source_name, "path": str(source_path), "split": "train"}
                )
                labels_rows.extend(
                    {
                        "sample_id": f"{source_name}-{index}",
                        "source": source_name,
                        "split": "train",
                        "source_index": index,
                        "cluster_id": f"{source_name}-template",
                        "final": {
                            "primary_label": "TR",
                            "secondary_labels": [],
                            "taxonomy_fit": "exact",
                            "confidence": 0.95,
                            "status": "accepted",
                            "method": "authoritative_model",
                        },
                    }
                    for index in range(3)
                )
            registry = root / "registry.json"
            registry.write_text(json.dumps({"sources": source_specs}), encoding="utf-8")
            labels = root / "labels.jsonl"
            labels.write_text(
                "\n".join(json.dumps(row) for row in labels_rows) + "\n", encoding="utf-8"
            )
            output = root / "out"
            materialize_command(
                SimpleNamespace(
                    registry=str(registry),
                    labels=str(labels),
                    output_dir=str(output),
                    min_confidence=0.85,
                    include_fit=["exact", "compatible"],
                    splits=["train"],
                    max_per_template=1,
                    template_cap_sources=["tsaqa"],
                    template_sample_seed=42,
                )
            )
            rows = [
                json.loads(line)
                for line in (output / "TR_temporal_relation_reasoning.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["input"].startswith("chatts_sft") for row in rows), 3)
            self.assertEqual(sum(row["input"].startswith("tsaqa") for row in rows), 1)
            sampling = json.loads((output / "manifest.json").read_text())["template_sampling"]
            self.assertEqual(sampling["cap_sources"], ["tsaqa"])
            self.assertEqual(sampling["candidate_samples"], 3)
            self.assertEqual(sampling["selected_samples"], 1)

    def test_distribution_report_groups_chatts_and_excludes_dev_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.jsonl"
            rows = [
                ("chatts_sft", "train", "PR", "accepted"),
                ("chatts_ift", "train", "TSF", "accepted"),
                ("chatts_dev", "dev", "AD", "accepted"),
                ("time_mqa", "train", "AD", "accepted"),
                ("time_mqa", "train", None, "human_review"),
                ("tsaqa", "train", "CA", "accepted"),
            ]
            labels.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "sample_id": str(index),
                            "source": source,
                            "split": split,
                            "source_index": index,
                            "cluster_id": str(index),
                            "final": {
                                "primary_label": label,
                                "status": status,
                            },
                        }
                    )
                    for index, (source, split, label, status) in enumerate(rows)
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_distribution_report(labels, ["train"])
            self.assertEqual(report["datasets"]["chatts"]["total"], 2)
            self.assertEqual(report["datasets"]["chatts"]["dimensions"]["PR"], 1)
            self.assertEqual(report["datasets"]["chatts"]["dimensions"]["TSF"], 1)
            self.assertEqual(report["datasets"]["chatts"]["dimensions"]["AD"], 0)
            self.assertEqual(report["datasets"]["time_mqa"]["statuses"]["human_review"], 1)
            self.assertEqual(report["datasets"]["tsaqa"]["dimensions"]["CA"], 1)
            self.assertEqual(report["datasets"]["chatts"]["percent_of_accepted"]["PR"], 50.0)


if __name__ == "__main__":
    unittest.main()
