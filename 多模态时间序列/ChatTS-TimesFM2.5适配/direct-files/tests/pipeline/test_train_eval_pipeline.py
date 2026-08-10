from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


DIRECT_FILES = Path(__file__).resolve().parents[2]
FINALIZER = DIRECT_FILES / "scripts" / "finalize_chatts_best_checkpoint.py"
TRAIN_RUNNER = DIRECT_FILES / "scripts" / "full" / "run_chronos2_best_two_stage.sh"
EVAL_RUNNER = (
    DIRECT_FILES
    / "NetManAIOps-ChatTS"
    / "scripts"
    / "run_all_chatts_benchmarks.sh"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FinalizeCheckpointTest(unittest.TestCase):
    def test_finalizer_stamps_metadata_and_removes_only_checkpoint_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage2"
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
    def test_training_preflight_is_non_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "training"
            model = root / "base"
            chronos = root / "chronos2"
            output = root / "output"
            project.mkdir()
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
