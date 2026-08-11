from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


DIRECT_FILES = Path(__file__).resolve().parents[2]
CHATTS_REPO = Path(
    os.environ.get("CHATTS_REPO_ROOT", DIRECT_FILES.parent / "ChatTS")
).resolve()
FINALIZER = DIRECT_FILES / "scripts" / "finalize_chatts_best_checkpoint.py"
TRAIN_RUNNER = DIRECT_FILES / "scripts" / "full" / "run_chronos2_best_two_stage.sh"
EVAL_RUNNER = (
    CHATTS_REPO
    / "scripts"
    / "run_all_chatts_benchmarks.sh"
)
CONFIG_LOADER = (
    CHATTS_REPO
    / "scripts"
    / "load_train_eval_config.py"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FinalizeCheckpointTest(unittest.TestCase):
    def test_finalizer_stamps_metadata_and_removes_only_checkpoint_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "stage2"
            stage1 = temporary_root / "stage1"
            stage1.mkdir()
            write_json(stage1 / "config.json", {})
            write_json(
                stage1 / "best_model_manifest.json",
                {
                    "stage": "stage1",
                    "exported_model_dir": str(stage1),
                    "selected_checkpoint": "checkpoint-50",
                    "best_metric": 0.25,
                },
            )
            selected = root / "checkpoint-100"
            selected.mkdir(parents=True)
            (root / "pytorch_model-00001-of-00001.bin").write_bytes(b"root-best-weights")
            (selected / "pytorch_model.bin").write_bytes(b"selected-weights")
            write_json(root / "config.json", {"architectures": ["Qwen3TSForCausalLM"]})
            write_json(
                root / "trainer_state.json",
                {"best_model_checkpoint": str(selected), "best_metric": 0.125},
            )

            subprocess.run(
                [
                    sys.executable,
                    str(FINALIZER),
                    "--checkpoint-dir",
                    str(root),
                    "--stage",
                    "stage2",
                    "--seed",
                    "42",
                    "--learning-rate",
                    "1e-5",
                    "--chronos2-model-path",
                    "/workspace/chronos2",
                    "--input-model-dir",
                    str(stage1),
                    "--input-best-model-manifest",
                    str(stage1 / "best_model_manifest.json"),
                    "--cleanup-checkpoints",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertFalse(selected.exists())
            self.assertTrue((root / "pytorch_model-00001-of-00001.bin").is_file())
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["ts_encoder_type"], "chronos2")
            self.assertEqual(config["ts"]["patch_size"], 16)
            manifest = json.loads(
                (root / "best_model_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["selected_checkpoint"], "checkpoint-100")
            self.assertEqual(manifest["best_metric"], 0.125)
            self.assertEqual(manifest["input_model_dir"], str(stage1.resolve()))
            self.assertEqual(
                manifest["input_best_model"]["selected_checkpoint"], "checkpoint-50"
            )
            self.assertEqual(
                manifest["model_files"][0]["sha256"],
                "d531fd39d473a326ce8c632facd6ade7b75cceae53b480d6044a41d0faa4d27c",
            )

    def test_finalizer_rejects_missing_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage1"
            root.mkdir()
            (root / "pytorch_model.bin").write_bytes(b"weights")
            write_json(root / "config.json", {})
            write_json(
                root / "trainer_state.json",
                {
                    "best_model_checkpoint": str(root / "checkpoint-200"),
                    "best_metric": 1.0,
                },
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(FINALIZER),
                    "--checkpoint-dir",
                    str(root),
                    "--stage",
                    "stage1",
                    "--seed",
                    "42",
                    "--learning-rate",
                    "1e-5",
                    "--chronos2-model-path",
                    "/workspace/chronos2",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not exist", result.stderr + result.stdout)


class PipelineShellTest(unittest.TestCase):
    def test_yaml_loader_maps_training_data_and_environment_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "experiment.yaml"
            config.write_text(
                """
pipeline:
  seed: 7
training:
  base_model_path: /models/qwen
  stage1:
    learning_rate: "3e-5"
    datasets: [align_256, ift]
  stage2:
    learning_rate: "8e-6"
    datasets: "sft,my_private_data"
evaluation:
  model_name: yaml-name
""".strip()
                + "\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["S1_LR"] = "9e-5"
            for name in ("SEED", "BASE_MODEL_PATH", "STAGE1_DATASETS", "S2_LR", "STAGE2_DATASETS", "MODEL_NAME"):
                env.pop(name, None)
            env["S1_LR"] = "9e-5"
            result = subprocess.run(
                [sys.executable, str(CONFIG_LOADER), str(config)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            assignments = dict(
                line.split("=", 1) for line in result.stdout.splitlines() if line
            )
            self.assertEqual(assignments["SEED"], "7")
            self.assertEqual(assignments["BASE_MODEL_PATH"], "/models/qwen")
            self.assertEqual(assignments["STAGE1_DATASETS"], "align_256,ift")
            self.assertNotIn("S1_LR", assignments)
            self.assertEqual(assignments["S2_LR"], "8e-6")
            self.assertEqual(assignments["STAGE2_DATASETS"], "sft,my_private_data")
            self.assertEqual(assignments["MODEL_NAME"], "yaml-name")

    def test_training_preflight_is_non_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "training"
            model = root / "base"
            chronos = root / "chronos2"
            output = root / "output"
            project.mkdir()
            (project / "data").mkdir()
            chronos.mkdir()
            write_json(model / "config.json", {})
            env = os.environ.copy()
            env.update(
                {
                    "PROJECT_ROOT": str(project),
                    "MODEL_PATH": str(model),
                    "CHRONOS2_MODEL_PATH": str(chronos),
                    "OUTPUT_ROOT": str(output),
                    "STAGE1_SCRIPT": str(
                        DIRECT_FILES
                        / "scripts"
                        / "full"
                        / "train_chronos2_best_stage1.sh"
                    ),
                    "STAGE2_SCRIPT": str(
                        DIRECT_FILES
                        / "scripts"
                        / "full"
                        / "train_chronos2_best_stage2.sh"
                    ),
                    "FINALIZER": str(FINALIZER),
                    "PREFLIGHT_ONLY": "1",
                    "AVAILABLE_GPUS_OVERRIDE": "8",
                    "PYTHON_BIN": sys.executable,
                }
            )
            subprocess.run(
                ["bash", str(TRAIN_RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output.exists())

    def _evaluation_fixture(self, root: Path, *, fail_runner: str = "") -> dict[str, str]:
        project = root / "chatts"
        scripts = project / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(
            CHATTS_REPO / "scripts" / "chatts_benchmark_artifacts.py",
            scripts / "chatts_benchmark_artifacts.py",
        )
        shutil.copy2(EVAL_RUNNER, scripts / "run_all_chatts_benchmarks.sh")

        # The top-level runner fingerprints these files as part of the
        # evaluation protocol.  The fixture deliberately uses inert stubs: the
        # child benchmark runners below emit representative summaries without
        # importing either ChatTS or a benchmark implementation.
        protocol_files = (
            scripts / "evaluate_tsrbench.py",
            scripts / "evaluate_ts_haystack.py",
            scripts / "evaluate_timeseriesexam.py",
            scripts / "summarize_tinybenchmarks_mcq.py",
            project / "chatts" / "vllm" / "chatts_vllm.py",
            project / "chatts" / "utils" / "llm_utils.py",
            project / "chatts" / "utils" / "inference_tsrbench_vllm.py",
            project / "chatts" / "utils" / "inference_tinybenchmarks_mcq_vllm.py",
            project / "chatts" / "utils" / "inference_ts_haystack_vllm.py",
            project / "chatts" / "utils" / "inference_timeseriesexam_vllm.py",
        )
        for path in protocol_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# offline protocol fixture\n", encoding="utf-8")

        sync_dir = root / "sync"
        sync_dir.mkdir()
        fake_runner = """#!/usr/bin/env bash
set -euo pipefail
name="$(basename "$0")"
if ! mkdir "$SYNC_DIR/active" 2>/dev/null; then
    echo "another benchmark suite is already active" >&2
    exit 88
fi
trap 'rmdir "$SYNC_DIR/active"' EXIT
printf '%s\n' "$name" >> "$SYNC_DIR/order.txt"
printf '%s|%s|%s\n' "$CUDA_VISIBLE_DEVICES" "$NUM_GPUS" "${NUM_GPUS_PER_PROCESS:-}" > "$SYNC_DIR/$name.env"
sleep 0.02
if [[ "${FAIL_RUNNER:-}" == "$name" ]]; then
    exit 7
fi
case "$name" in
    run_chatts_tsrbench.sh)
        mkdir -p "$OUTPUT_ROOT"
        printf '%s\n' '{"overall":{"dataset_size":2,"generated":2,"parsed":2,"correct":1,"coverage":1.0,"parse_rate":1.0,"accuracy_strict":0.5,"accuracy_parsed":0.5}}' > "$OUTPUT_ROOT/tsrbench_summary_${MODEL_NAME}.json"
        ;;
    run_chatts_tinybenchmarks_mcq.sh)
        mkdir -p "$OUTPUT_ROOT/$MODEL_NAME"
        printf '%s\n' '{"tasks":{"tinyArc":{"score":0.50},"tinyHellaswag":{"score":0.51},"tinyMMLU":{"score":0.52},"tinyTruthfulQA":{"score":0.53},"tinyWinogrande":{"score":0.54}}}' > "$OUTPUT_ROOT/$MODEL_NAME/metrics.json"
        ;;
    run_chatts_ts_haystack.sh)
        mkdir -p "$OUTPUT_ROOT"
        printf '%s\n' '{"overall":{"total":2,"generated":2,"parsed":2,"correct":1,"coverage":1.0,"parse_rate":1.0,"accuracy_strict":0.5,"accuracy_generated":0.5,"mean_iou":0.75,"mean_timestamp_error_s":0.25}}' > "$OUTPUT_ROOT/ts_haystack_summary_${MODEL_NAME}.json"
        ;;
    run_chatts_timeseriesexam.sh)
        summary_dir="$OUTPUT_ROOT/${MODEL_NAME}_query_hint_concepts_examples"
        mkdir -p "$summary_dir"
        printf '%s\n' '{"overall":{"total":2,"generated":2,"parsed":2,"coverage":1.0,"parse_rate":1.0,"official_flexible_accuracy":0.5,"official_strict_accuracy":0.5,"letter_accuracy":0.5,"letter_accuracy_parsed":0.5}}' > "$summary_dir/timeseriesexam_summary_${MODEL_NAME}.json"
        ;;
esac
"""
        runner_names = (
            "run_chatts_tsrbench.sh",
            "run_chatts_tinybenchmarks_mcq.sh",
            "run_chatts_ts_haystack.sh",
            "run_chatts_timeseriesexam.sh",
        )
        for name in runner_names:
            path = scripts / name
            path.write_text(fake_runner, encoding="utf-8")
            path.chmod(0o755)
        (scripts / "inspect_chatts_ts_encoder_checkpoints.py").write_text(
            "print('chronos2')\n", encoding="utf-8"
        )

        model = root / "model"
        write_json(model / "config.json", {})
        write_json(model / "TRAINING_COMPLETE.json", {"status": "complete"})
        chronos = root / "chronos2"
        chronos.mkdir()
        tsr = root / "tsr"
        write_json(tsr / "perception.jsonl", {})
        tiny = root / "tiny"
        tiny.mkdir()
        haystack = root / "haystack"
        (haystack / "src" / "datasets").mkdir(parents=True)
        (haystack / "src" / "datasets" / "registry.py").write_text("", encoding="utf-8")
        (haystack / "data").mkdir()
        exam = root / "exam"
        (exam / "evaluate").mkdir(parents=True)
        (exam / "evaluate" / "concepts.py").write_text("", encoding="utf-8")
        write_json(exam / "dataset.json", [])
        output = root / "evaluation"

        env = os.environ.copy()
        env.update(
            {
                "PROJECT_ROOT": str(project),
                "MODEL_PATH": str(model),
                "MODEL_NAME": "pipeline-test",
                "CHRONOS2_MODEL_PATH": str(chronos),
                "TSRBENCH_ROOT": str(tsr),
                "TSRBENCH_DATASET_ROOT": str(tsr),
                "TINYBENCH_DATASET_ROOT": str(tiny),
                "TS_HAYSTACK_ROOT": str(haystack),
                "TIMESERIESEXAM_ROOT": str(exam),
                "TIMESERIESEXAM_DATA_FILE": str(exam / "dataset.json"),
                "OUTPUT_ROOT": str(output),
                "AVAILABLE_GPUS_OVERRIDE": "8",
                "MAX_SAMPLES": "2",
                "PYTHON_BIN": sys.executable,
                "SYNC_DIR": str(sync_dir),
                "FAIL_RUNNER": fail_runner,
            }
        )
        return env

    def test_four_benchmarks_run_sequentially_on_all_eight_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self._evaluation_fixture(root)
            subprocess.run(
                ["bash", str(EVAL_RUNNER)],
                env=env,
                check=True,
                timeout=20,
                capture_output=True,
                text=True,
            )
            with (root / "evaluation" / "benchmark_status.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["status"] for row in rows}, {"PASS"})
            self.assertEqual({row["gpus"] for row in rows}, {"0,1,2,3,4,5,6,7"})
            order = (root / "sync" / "order.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                order,
                [
                    "run_chatts_tsrbench.sh",
                    "run_chatts_tinybenchmarks_mcq.sh",
                    "run_chatts_ts_haystack.sh",
                    "run_chatts_timeseriesexam.sh",
                ],
            )
            for runner_name in order:
                gpu_env = (root / "sync" / f"{runner_name}.env").read_text(
                    encoding="utf-8"
                ).strip()
                visible, num_gpus, gpus_per_process = gpu_env.split("|")
                self.assertEqual(visible, "0,1,2,3,4,5,6,7")
                self.assertEqual(num_gpus, "8")
                if runner_name == "run_chatts_tinybenchmarks_mcq.sh":
                    self.assertEqual(gpus_per_process, "")
                else:
                    self.assertEqual(gpus_per_process, "2")

    def test_one_failure_does_not_prevent_other_suites_finishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self._evaluation_fixture(
                root, fail_runner="run_chatts_ts_haystack.sh"
            )
            result = subprocess.run(
                ["bash", str(EVAL_RUNNER)],
                env=env,
                timeout=20,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            with (root / "evaluation" / "benchmark_status.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 4)
            status = {row["suite"]: row["status"] for row in rows}
            self.assertEqual(status["ts_haystack"], "FAIL")
            self.assertEqual(sum(value == "PASS" for value in status.values()), 3)


if __name__ == "__main__":
    unittest.main()
