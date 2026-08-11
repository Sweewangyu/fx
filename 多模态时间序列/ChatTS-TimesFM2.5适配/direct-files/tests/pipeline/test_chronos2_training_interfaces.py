from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "full" / "run_chronos2_best_two_stage.sh"
STAGE1_RUNNER = REPO_ROOT / "scripts" / "full" / "train_chronos2_best_stage1.sh"
STAGE2_RUNNER = REPO_ROOT / "scripts" / "full" / "train_chronos2_best_stage2.sh"
FINALIZER = REPO_ROOT / "scripts" / "finalize_chatts_best_checkpoint.py"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


class StageRunnerInterfaceTest(unittest.TestCase):
    def test_both_stage_runners_forward_dataset_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "training"
            model = root / "base"
            chronos = root / "chronos2"
            dataset = root / "snapshot"
            fake_bin = root / "bin"
            project.mkdir()
            chronos.mkdir()
            dataset.mkdir()
            fake_bin.mkdir()
            write_json(model / "config.json", {})

            capture = root / "deepspeed.args"
            make_executable(
                fake_bin / "deepspeed",
                """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\n' "$@" >> "$CAPTURE_PATH"
""",
            )
            fake_finalizer = root / "finalizer.py"
            fake_finalizer.write_text("raise SystemExit(0)\n", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "PROJECT_ROOT": str(project),
                    "MODEL_PATH": str(model),
                    "CHRONOS2_MODEL_PATH": str(chronos),
                    "DATASET_DIR": str(dataset),
                    "FINALIZER": str(fake_finalizer),
                    "PYTHON_BIN": sys.executable,
                    "CAPTURE_PATH": str(capture),
                    "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                    "STAGE1_PREPROCESSING_NUM_WORKERS": "1",
                    "STAGE2_PREPROCESSING_NUM_WORKERS": "1",
                }
            )

            stage1_out = root / "stage1"
            subprocess.run(
                ["bash", str(STAGE1_RUNNER), "1e-5", str(stage1_out)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            write_json(stage1_out / "config.json", {})
            write_json(stage1_out / "best_model_manifest.json", {})
            stage2_out = root / "stage2"
            subprocess.run(
                ["bash", str(STAGE2_RUNNER), "1e-5", str(stage1_out), str(stage2_out)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            args = capture.read_text(encoding="utf-8").splitlines()
            positions = [index for index, value in enumerate(args) if value == "--dataset_dir"]
            self.assertEqual(len(positions), 2)
            self.assertEqual({args[index + 1] for index in positions}, {str(dataset)})


class PipelineModeTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, str]:
        project = root / "training"
        model = root / "base"
        chronos = root / "chronos2"
        dataset = root / "snapshot"
        output = root / "output"
        project.mkdir()
        chronos.mkdir()
        dataset.mkdir()
        write_json(model / "config.json", {})

        stage1_runner = root / "fake_stage1.sh"
        make_executable(
            stage1_runner,
            """#!/usr/bin/env bash
set -Eeuo pipefail
lr="$1"
out="$2"
"$PYTHON_BIN" - "$out" "$lr" "$SEED" "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True)
(out / "config.json").write_text("{}", encoding="utf-8")
(out / "pytorch_model.bin").write_bytes(b"stage1-weights")
payload = {
    "stage": "stage1",
    "seed": int(sys.argv[3]),
    "learning_rate": sys.argv[2],
    "best_metric": 0.25,
    "selected_checkpoint": "checkpoint-10",
    "exported_model_dir": str(out),
    "input_model_dir": str(Path(sys.argv[4]).resolve()),
    "ts_encoder_type": "chronos2",
}
(out / "best_model_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
PY
""",
        )
        stage2_runner = root / "fake_stage2.sh"
        make_executable(
            stage2_runner,
            """#!/usr/bin/env bash
set -Eeuo pipefail
lr="$1"
stage1="$2"
out="$3"
"$PYTHON_BIN" - "$out" "$lr" "$SEED" "$stage1" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1]).resolve()
stage1 = Path(sys.argv[4]).resolve()
parent = json.loads((stage1 / "best_model_manifest.json").read_text(encoding="utf-8"))
out.mkdir(parents=True)
(out / "config.json").write_text("{}", encoding="utf-8")
(out / "pytorch_model.bin").write_bytes(b"stage2-weights")
payload = {
    "stage": "stage2",
    "seed": int(sys.argv[3]),
    "learning_rate": sys.argv[2],
    "best_metric": 0.125,
    "selected_checkpoint": "checkpoint-20",
    "exported_model_dir": str(out),
    "input_model_dir": str(stage1),
    "input_best_model": {
        "stage": "stage1",
        "exported_model_dir": str(stage1),
        "selected_checkpoint": parent["selected_checkpoint"],
        "best_metric": parent["best_metric"],
    },
    "ts_encoder_type": "chronos2",
}
(out / "best_model_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
PY
""",
        )

        env = os.environ.copy()
        env.update(
            {
                "PROJECT_ROOT": str(project),
                "MODEL_PATH": str(model),
                "CHRONOS2_MODEL_PATH": str(chronos),
                "DATASET_DIR": str(dataset),
                "OUTPUT_ROOT": str(output),
                "STAGE1_SCRIPT": str(stage1_runner),
                "STAGE2_SCRIPT": str(stage2_runner),
                "FINALIZER": str(FINALIZER),
                "AVAILABLE_GPUS_OVERRIDE": "8",
                "PYTHON_BIN": sys.executable,
            }
        )
        return env

    def test_default_full_mode_removes_stage1_and_records_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "shared-stage1"
            final = root / "output" / "final"
            env.update({"STAGE1_OUT": str(stage1), "FINAL_MODEL_PATH": str(final)})

            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertFalse(stage1.exists())
            marker = json.loads(
                (final / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["pipeline_mode"], "full")
            self.assertFalse(marker["stage1_model_retained"])
            self.assertEqual(
                Path(marker["resolved_configuration"]["DATASET_DIR"]),
                Path(env["DATASET_DIR"]).resolve(),
            )
            self.assertEqual(len(marker["resolved_configuration_sha256"]), 64)
            self.assertEqual(len(marker["commands_sha256"]), 64)
            self.assertEqual(len(marker["final_artifact"]["sha256"]), 64)
            self.assertTrue(
                all(
                    len(entry["sha256"]) == 64
                    for entry in marker["final_artifact"]["files"]
                )
            )
            run_manifest = root / "output" / "logs" / env.get(
                "RUN_NAME", "chronos2_seed42_s1lr_1e-5_s2lr_1e-5"
            ) / "training_run_manifest.json"
            self.assertTrue(run_manifest.is_file())

    def test_stage1_then_stage2_reuses_and_never_deletes_shared_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "shared-stage1"
            final = root / "output" / "trial-final"
            stage1_hash = "a" * 64
            data_hash = "d" * 64
            env.update(
                {
                    "PIPELINE_MODE": "stage1",
                    "STAGE1_OUT": str(stage1),
                    "TRIAL_ID": "shared-stage1",
                    "TRIAL_CONFIG_HASH": stage1_hash,
                    "DATASET_SNAPSHOT_HASH": data_hash,
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((stage1 / "STAGE1_COMPLETE.json").is_file())
            resumed = subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Stage1 already completed; reusing", resumed.stdout)

            stage2_env = env.copy()
            stage2_env.update(
                {
                    "PIPELINE_MODE": "stage2",
                    "STAGE2_FROM": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                    "TRIAL_ID": "proxy-001",
                    "TRIAL_CONFIG_HASH": "b" * 64,
                    "KEEP_STAGE1": "0",
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=stage2_env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(stage1.is_dir())
            marker = json.loads(
                (final / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["pipeline_mode"], "stage2")
            self.assertEqual(marker["trial_id"], "proxy-001")
            self.assertEqual(marker["trial_config_hash"], "b" * 64)
            self.assertEqual(marker["dataset_snapshot_hash"], data_hash)
            self.assertTrue(marker["stage1_model_retained"])
            self.assertIsNone(marker["commands"]["stage1"])
            self.assertEqual(
                Path(marker["training_lineage"]["stage2"]["input_model_path"]),
                stage1.resolve(),
            )
            self.assertEqual(len(marker["stage1_input_provenance"]["sha256"]), 64)

    def test_stage2_rejects_ancestor_output_without_deleting_shared_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            final = root / "output" / "container"
            stage1 = final / "shared-stage1"
            stage1.mkdir(parents=True)
            write_json(stage1 / "config.json", {})
            (stage1 / "pytorch_model.bin").write_bytes(b"shared-stage1-weights")
            write_json(
                stage1 / "best_model_manifest.json",
                {
                    "stage": "stage1",
                    "seed": 42,
                    "best_metric": 0.25,
                    "selected_checkpoint": "checkpoint-10",
                    "exported_model_dir": str(stage1.resolve()),
                    "ts_encoder_type": "chronos2",
                },
            )
            env.update(
                {
                    "PIPELINE_MODE": "stage2",
                    "STAGE2_FROM": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                    "FORCE_TRAIN": "1",
                }
            )

            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsafe overlapping model paths", result.stderr + result.stdout)
            self.assertTrue(stage1.is_dir())
            self.assertTrue((stage1 / "pytorch_model.bin").is_file())

    def test_stage2_cache_binds_stage1_source_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "legacy-shared-stage1"
            final = root / "output" / "trial-final"
            stage1.mkdir(parents=True)
            write_json(stage1 / "config.json", {})
            (stage1 / "pytorch_model.bin").write_bytes(b"shared-stage1-weights")
            manifest_path = stage1 / "best_model_manifest.json"
            manifest = {
                "stage": "stage1",
                "seed": 42,
                "best_metric": 0.25,
                "selected_checkpoint": "checkpoint-10",
                "exported_model_dir": str(stage1.resolve()),
                "ts_encoder_type": "chronos2",
            }
            write_json(manifest_path, manifest)
            env.update(
                {
                    "PIPELINE_MODE": "stage2",
                    "STAGE2_FROM": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest["selected_checkpoint"] = "checkpoint-999"
            write_json(manifest_path, manifest)
            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Stage1 input provenance digest mismatch",
                result.stderr + result.stdout,
            )
            marker = json.loads(
                (final / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["training_lineage"]["stage2"][
                    "input_stage1_selected_checkpoint"
                ],
                "checkpoint-10",
            )

    def test_completion_cache_rejects_tampered_final_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "stage1"
            final = root / "output" / "final"
            env.update(
                {
                    "STAGE1_OUT": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            weight = final / "pytorch_model.bin"
            original_digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            weight.write_bytes(b"tampered-stage2-weights")
            self.assertNotEqual(
                hashlib.sha256(weight.read_bytes()).hexdigest(), original_digest
            )
            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("artifact digest mismatch", (result.stderr + result.stdout).lower())
            self.assertEqual(weight.read_bytes(), b"tampered-stage2-weights")

    def test_legacy_default_marker_is_reused_and_run_manifest_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "stage1"
            final = root / "output" / "final"
            env.update(
                {
                    "STAGE1_OUT": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                }
            )
            write_json(final / "config.json", {})
            (final / "pytorch_model.bin").write_bytes(b"legacy-final-weights")
            write_json(final / "best_model_manifest.json", {"stage": "stage2"})
            write_json(
                final / "TRAINING_COMPLETE.json",
                {
                    "status": "complete",
                    "seed": 42,
                    "stage1_learning_rate": "1e-5",
                    "stage2_learning_rate": "1e-5",
                    "final_model_path": str(final.resolve()),
                },
            )

            result = subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Training already completed; reusing", result.stdout)
            run_manifest = (
                root
                / "output"
                / "logs"
                / "chronos2_seed42_s1lr_1e-5_s2lr_1e-5"
                / "training_run_manifest.json"
            )
            self.assertTrue(run_manifest.is_file())

    def test_force_train_refuses_unowned_custom_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "stage1"
            final = root / "output" / "custom-final"
            final.mkdir(parents=True)
            protected = final / "do-not-delete.txt"
            protected.write_text("owned by somebody else", encoding="utf-8")
            env.update(
                {
                    "STAGE1_OUT": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                    "FORCE_TRAIN": "1",
                }
            )

            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unowned final output", result.stderr + result.stdout)
            self.assertEqual(protected.read_text(encoding="utf-8"), "owned by somebody else")

    def test_force_train_replaces_owned_custom_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "custom-stage1"
            final = root / "output" / "custom-final"
            env.update(
                {
                    "STAGE1_OUT": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            first_marker = json.loads(
                (final / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
            )

            forced = env.copy()
            forced["FORCE_TRAIN"] = "1"
            result = subprocess.run(
                ["bash", str(RUNNER)],
                env=forced,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Removed previous final output", result.stdout)
            self.assertFalse(stage1.exists())
            second_marker = json.loads(
                (final / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                first_marker["completed_at_utc"], second_marker["completed_at_utc"]
            )

    def test_completion_cache_rejects_changed_dataset_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "stage1"
            final = root / "output" / "final"
            env.update(
                {
                    "STAGE1_OUT": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                    "TRIAL_CONFIG_HASH": "a" * 64,
                    "DATASET_SNAPSHOT_HASH": "d" * 64,
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            changed = env.copy()
            changed["DATASET_SNAPSHOT_HASH"] = "e" * 64
            result = subprocess.run(
                ["bash", str(RUNNER)],
                env=changed,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "completion marker mismatch for dataset_snapshot_hash",
                (result.stderr + result.stdout).lower(),
            )
            self.assertTrue(final.is_dir())

    def test_stage2_preflight_rejects_non_stage1_manifest_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "bad-stage1"
            stage1.mkdir()
            write_json(stage1 / "config.json", {})
            write_json(
                stage1 / "best_model_manifest.json",
                {
                    "stage": "stage2",
                    "seed": 42,
                    "exported_model_dir": str(stage1.resolve()),
                },
            )
            output = root / "output" / "must-not-exist"
            env.update(
                {
                    "PIPELINE_MODE": "stage2",
                    "STAGE2_FROM": str(stage1),
                    "FINAL_MODEL_PATH": str(output),
                    "PREFLIGHT_ONLY": "1",
                }
            )
            result = subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a Stage1 manifest", result.stderr + result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
