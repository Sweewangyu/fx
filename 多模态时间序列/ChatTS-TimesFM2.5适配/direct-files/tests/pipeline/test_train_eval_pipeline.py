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
HOST_PIPELINE = CHATTS_REPO / "scripts" / "run_train_then_eval.sh"


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
  dataset_dir: /datasets/snapshot
  keep_stage1: true
  stage1:
    learning_rate: "3e-5"
    datasets: [align_256, ift]
  stage2:
    learning_rate: "8e-6"
    datasets: "sft,my_private_data"
evaluation:
  model_name: yaml-name
  benchmarks: tsrbench,timeseriesexam
  run_id: yaml-run
  protocol_hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  haystack_split: validation
  tiny_data_partition: search-dev
  tiny_partition_seed: 42
""".strip()
                + "\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["S1_LR"] = "9e-5"
            for name in (
                "SEED",
                "BASE_MODEL_PATH",
                "DATASET_DIR",
                "KEEP_STAGE1",
                "STAGE1_DATASETS",
                "S2_LR",
                "STAGE2_DATASETS",
                "MODEL_NAME",
                "BENCHMARKS",
                "RUN_ID",
                "EVAL_PROTOCOL_HASH",
                "HAYSTACK_SPLIT",
                "TINY_DATA_PARTITION",
                "TINY_PARTITION_SEED",
            ):
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
            self.assertEqual(assignments["DATASET_DIR"], "/datasets/snapshot")
            self.assertEqual(assignments["KEEP_STAGE1"], "1")
            self.assertEqual(assignments["STAGE1_DATASETS"], "align_256,ift")
            self.assertNotIn("S1_LR", assignments)
            self.assertEqual(assignments["S2_LR"], "8e-6")
            self.assertEqual(assignments["STAGE2_DATASETS"], "sft,my_private_data")
            self.assertEqual(assignments["MODEL_NAME"], "yaml-name")
            self.assertEqual(assignments["BENCHMARKS"], "tsrbench,timeseriesexam")
            self.assertEqual(assignments["RUN_ID"], "yaml-run")
            self.assertEqual(assignments["EVAL_PROTOCOL_HASH"], "a" * 64)
            self.assertEqual(assignments["HAYSTACK_SPLIT"], "validation")
            self.assertEqual(assignments["TINY_DATA_PARTITION"], "search-dev")
            self.assertEqual(assignments["TINY_PARTITION_SEED"], "42")

    def test_host_one_click_pipeline_runs_training_then_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared"
            shared.mkdir()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                """#!/usr/bin/env python3
import os
import subprocess
import sys

arguments = sys.argv[1:]
if arguments[:1] == ["inspect"]:
    print("true")
    raise SystemExit(0)
if arguments[:1] != ["exec"]:
    raise SystemExit("unsupported fake docker invocation: " + repr(arguments))
cursor = 1
environment = os.environ.copy()
while cursor < len(arguments) and arguments[cursor] == "-e":
    key, value = arguments[cursor + 1].split("=", 1)
    environment[key] = value
    cursor += 2
if cursor >= len(arguments):
    raise SystemExit("fake docker exec has no container")
cursor += 1  # The mock shares the host filesystem, so the container name is metadata.
command = arguments[cursor:]
if command[:2] == ["python", "-c"] and "torch.cuda.device_count" in command[2]:
    print("8")
    raise SystemExit(0)
raise SystemExit(subprocess.run(command, env=environment, check=False).returncode)
""",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)

            train_project = root / "training-project"
            eval_project = root / "evaluation-project"
            train_script = train_project / "scripts" / "train.sh"
            eval_script = eval_project / "scripts" / "eval.sh"
            train_script.parent.mkdir(parents=True)
            eval_script.parent.mkdir(parents=True)
            dataset = root / "dataset-snapshot"
            dataset.mkdir()
            base_model = root / "base-model"
            base_model.mkdir()
            chronos = root / "chronos2"
            chronos.mkdir()
            final_model = shared / "models" / "one-click-model"
            train_output = shared / "training"
            eval_output = shared / "evaluation" / "one-click"
            event_log = root / "events.log"

            train_script.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
[[ "$PIPELINE_MODE" == "full" ]]
[[ "$DATASET_DIR" == "$EXPECTED_DATASET_DIR" ]]
[[ "$KEEP_STAGE1" == "1" ]]
printf 'train|%s|%s|%s\n' "$PIPELINE_MODE" "$DATASET_DIR" "$KEEP_STAGE1" >> "$MOCK_EVENT_LOG"
mkdir -p "$FINAL_MODEL_PATH"
printf '%s\n' '{"architectures":["Qwen3TSForCausalLM"]}' > "$FINAL_MODEL_PATH/config.json"
printf '%s\n' '{"status":"complete","pipeline_mode":"full"}' > "$FINAL_MODEL_PATH/TRAINING_COMPLETE.json"
""",
                encoding="utf-8",
            )
            train_script.chmod(0o755)
            eval_script.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
[[ -f "$MODEL_PATH/config.json" ]]
[[ -f "$MODEL_PATH/TRAINING_COMPLETE.json" ]]
[[ "$REQUIRE_TRAINING_MARKER" == "1" ]]
printf 'eval|%s|%s|%s|%s|%s|%s\n' "$BENCHMARKS" "$RUN_ID" "$EVAL_PROTOCOL_HASH" "$HAYSTACK_SPLIT" "$TINY_DATA_PARTITION" "$TINY_PARTITION_SEED" >> "$MOCK_EVENT_LOG"
mkdir -p "$OUTPUT_ROOT"
printf 'suite\tstatus\nall\tPASS\n' > "$OUTPUT_ROOT/benchmark_status.tsv"
printf '# mock benchmark summary\n' > "$OUTPUT_ROOT/all_benchmarks_summary.md"
printf '%s\n' '{"status":"complete","suites":{"tsrbench":{"status":"pass"},"timeseriesexam":{"status":"pass"}}}' > "$OUTPUT_ROOT/metrics.json"
""",
                encoding="utf-8",
            )
            eval_script.chmod(0o755)

            for path in (
                root / "tsrbench",
                root / "tinybench",
                root / "haystack",
                root / "timeseriesexam",
            ):
                path.mkdir()
            exam_data = root / "timeseriesexam" / "dataset.json"
            write_json(exam_data, [])
            protocol_hash = "b" * 64
            config = root / "one-click.yaml"
            config.write_text(
                f"""
pipeline:
  seed: 42
  force_train: false
  force_eval: false
  preflight_only: false
  max_samples: 2
  offline: true
containers:
  training: mock-training
  evaluation: mock-evaluation
training:
  project_root: {train_project}
  script: {train_script}
  base_model_path: {base_model}
  output_root: {train_output}
  final_model_path: {final_model}
  chronos2_model_path: {chronos}
  dataset_dir: {dataset}
  keep_stage1: true
evaluation:
  project_root: {eval_project}
  script: {eval_script}
  model_name: one-click-model
  output_root: {eval_output}
  chronos2_model_path: {chronos}
  tsrbench_root: {root / 'tsrbench'}
  tinybench_dataset_root: {root / 'tinybench'}
  ts_haystack_root: {root / 'haystack'}
  timeseriesexam_root: {root / 'timeseriesexam'}
  timeseriesexam_data_file: {exam_data}
  benchmarks: tsrbench,timeseriesexam
  run_id: one-click-e2e
  protocol_hash: {protocol_hash}
  haystack_split: test
  tiny_data_partition: all
  tiny_partition_seed: 42
""".lstrip(),
                encoding="utf-8",
            )

            env = os.environ.copy()
            for name in (
                "TRAIN_CONTAINER",
                "EVAL_CONTAINER",
                "TRAIN_PROJECT_ROOT",
                "EVAL_PROJECT_ROOT",
                "TRAIN_SCRIPT",
                "EVAL_SCRIPT",
                "BASE_MODEL_PATH",
                "TRAIN_OUTPUT_ROOT",
                "FINAL_MODEL_PATH",
                "TRAIN_CHRONOS2_MODEL_PATH",
                "EVAL_CHRONOS2_MODEL_PATH",
                "DATASET_DIR",
                "KEEP_STAGE1",
                "PIPELINE_MODE",
                "MODEL_NAME",
                "EVAL_OUTPUT_ROOT",
                "BENCHMARKS",
                "RUN_ID",
                "EVAL_PROTOCOL_HASH",
                "HAYSTACK_SPLIT",
                "TINY_DATA_PARTITION",
                "TINY_PARTITION_SEED",
            ):
                env.pop(name, None)
            env.update(
                {
                    "PATH": str(fake_bin) + os.pathsep + env["PATH"],
                    "HOST_PYTHON_BIN": sys.executable,
                    "CONFIG_FILE": str(config),
                    "SHARED_ROOT": str(shared),
                    "EXPECTED_DATASET_DIR": str(dataset),
                    "MOCK_EVENT_LOG": str(event_log),
                }
            )
            result = subprocess.run(
                ["bash", str(HOST_PIPELINE)],
                env=env,
                check=True,
                timeout=20,
                capture_output=True,
                text=True,
            )

            self.assertIn("Pipeline completed successfully", result.stdout)
            events = event_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                events,
                [
                    f"train|full|{dataset}|1",
                    (
                        "eval|tsrbench,timeseriesexam|one-click-e2e|"
                        f"{protocol_hash}|test|all|42"
                    ),
                ],
            )
            self.assertTrue((final_model / "TRAINING_COMPLETE.json").is_file())
            self.assertTrue((eval_output / "benchmark_status.tsv").is_file())
            self.assertTrue((eval_output / "all_benchmarks_summary.md").is_file())
            self.assertTrue((eval_output / "metrics.json").is_file())

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
