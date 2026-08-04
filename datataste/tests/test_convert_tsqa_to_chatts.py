import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_tsqa_to_chatts import (
    PLACEHOLDER,
    ConvertOptions,
    _contains_sunspot,
    _is_nontrain_file,
    _is_time_mqa_classification_file,
    convert_time_mqa,
    convert_tsaqa,
    main,
)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.options = ConvertOptions()

    def test_tsaqa_keeps_primary_then_inline_candidate_order(self):
        row = {
            "task": "data_transformation",
            "question_type": "multiple_choices",
            "input_ts": "[0, 1, 2, 3, 4, 5, 6, 7]",
            "meta_info": "Choose the transformed series.",
            "question": ("Which is correct? A. [10, 11, 12, 13, 14, 15, 16, 17] B. [20, 21, 22, 23, 24, 25, 26, 27]"),
            "answer": "B",
            "raw_ts": "[[999, 999], [888, 888]]",
        }
        sample, used_mask = convert_tsaqa(row, self.options)
        self.assertFalse(used_mask)
        self.assertEqual(sample["output"], "B")
        self.assertEqual(len(sample["timeseries"]), 3)
        self.assertEqual(sample["timeseries"][0], list(map(float, range(8))))
        self.assertEqual(sample["timeseries"][1][0], 10.0)
        self.assertEqual(sample["timeseries"][2][0], 20.0)
        self.assertEqual(sample["input"].count(PLACEHOLDER), 3)
        self.assertNotIn("999", sample["input"])

    def test_tsaqa_multiseries_input_is_row_major(self):
        row = {
            "input_ts": "[[1,2,3,4], [10,20,30,40]]",
            "question": "Which series increases faster?",
            "answer": "The second series.",
        }
        sample, _ = convert_tsaqa(row, self.options)
        self.assertEqual(sample["timeseries"], [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        self.assertEqual(sample["input"].count(PLACEHOLDER), 2)

    def test_time_mqa_missing_values_get_value_and_mask_series(self):
        row = {
            "question": "Impute [1, 2, X, 4, 5, 6, 7, 8].",
            "answer": "[1, 2, 3, 4, 5, 6, 7, 8]",
        }
        sample, used_mask = convert_time_mqa(row, self.options)
        self.assertTrue(used_mask)
        self.assertEqual(sample["timeseries"][0], [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(sample["timeseries"][1], [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(sample["input"].count(PLACEHOLDER), 2)

    def test_time_major_nested_array_is_transposed(self):
        row = {
            "question": ("Classify [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]."),
            "answer": "walking",
        }
        sample, _ = convert_time_mqa(row, self.options)
        self.assertEqual(sample["timeseries"][0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(sample["timeseries"][1], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

    def test_time_mqa_combined_training_text_is_supported(self):
        row = {"text": ("<QUE> Analyze [1,2,3,4,5,6,7,8]. <ANS> The series is increasing. </END>")}
        sample, _ = convert_time_mqa(row, self.options)
        self.assertEqual(sample["output"], "The series is increasing.")
        self.assertEqual(sample["input"].count(PLACEHOLDER), 1)

    def test_official_time_mqa_qa_list_fragment_is_supported(self):
        row = {
            "application_domain": "Nature",
            "task_type": "anomaly_detection",
            "QA_list": (
                '"question": "Analyze [1,2,3,4,5,6,7,8].", '
                '"answer": "The series increases."'
            ),
        }
        sample, _ = convert_time_mqa(row, self.options)
        self.assertEqual(sample["output"], "The series increases.")
        self.assertEqual(sample["timeseries"], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        self.assertEqual(sample["input"].count(PLACEHOLDER), 1)

    def test_apostrophe_in_prose_does_not_hide_inline_array(self):
        row = {
            "QA_list": (
                '"question": "Yahoo\'s metric is [1,2,3,4,5,6,7,8].", '
                '"answer": "Normal."'
            )
        }
        sample, _ = convert_time_mqa(row, self.options)
        self.assertEqual(sample["input"].count(PLACEHOLDER), 1)

    def test_unescaped_quotes_in_official_qa_fragment_use_safe_delimiters(self):
        row = {
            "QA_list": (
                '"question": "The CFO said: "demand improved". '
                'Series [1,2,3,4].", "answer": "Increasing."'
            )
        }
        sample, _ = convert_time_mqa(row, self.options)
        self.assertEqual(sample["output"], "Increasing.")
        self.assertEqual(sample["timeseries"], [[1.0, 2.0, 3.0, 4.0]])

    def test_contamination_and_split_filters(self):
        self.assertTrue(_contains_sunspot({"dataset": '["Nature_sunspot", "weather"]'}))
        self.assertTrue(_is_time_mqa_classification_file(Path("raw/Classification/classification.csv")))
        self.assertTrue(_is_nontrain_file(Path("test.parquet")))
        self.assertFalse(_is_nontrain_file(Path("train.parquet")))


class EndToEndTests(unittest.TestCase):
    def test_jsonl_conversion_writes_exact_training_schema_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            output = root / "chatts.jsonl"
            rows = [
                {
                    "task": "classification",
                    "input_ts": "[1,2,3,4,5,6,7,8]",
                    "question": "A or B?",
                    "answer": "A",
                    "dataset": "safe_source",
                },
                {
                    "task": "temporal",
                    "input_ts": "[1,2,3,4,5,6,7,8]",
                    "question": "True or false?",
                    "answer": "True",
                    "dataset": "Nature_sunspot",
                },
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            result = main(
                [
                    "convert",
                    "--dataset",
                    "tsaqa",
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--preview",
                    "0",
                ]
            )
            self.assertEqual(result, 0)
            converted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(converted), 1)
            self.assertEqual(set(converted[0]), {"input", "timeseries", "output"})
            self.assertTrue((root / "chatts.audit.jsonl").exists())
            manifest = json.loads((root / "chatts.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stats"]["rows_filtered_contamination"], 1)


if __name__ == "__main__":
    unittest.main()
