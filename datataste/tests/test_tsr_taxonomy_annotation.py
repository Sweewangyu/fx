import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.annotate_tsr_taxonomy import (
    SourceSpec,
    _load_human,
    _parse_model_json,
    iter_source,
    materialize_command,
    normalize_template,
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


if __name__ == "__main__":
    unittest.main()
