# ruff: noqa: PLW1510, PT009 -- this module intentionally uses unittest.TestCase.
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
DATASET_VERIFIER = REPO_ROOT / "scripts" / "verify_dataset_snapshot.py"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_studio_snapshot(
    root: Path,
    *,
    data_version: str | None = None,
    snapshot_schema: str = "chatts-dataset-snapshot-v2",
) -> str:
    stage_names = {
        "stage1": ["align_256", "ift"],
        "stage2": ["sft", "align_random", "finiverse_time_mqa", "finiverse_tsaqa"],
    }
    selection = {
        stage: {
            "sources": list(names),
            "qualities": [],
            "difficulties": [],
            "abilities": [],
        }
        for stage, names in stage_names.items()
    }
    input_identities = {
        name: {
            "raw_path": f"/source/{name}.jsonl",
            "annotation_path": f"/annotations/{name}.jsonl",
            "annotation_mode": "native",
            "raw_sha256": hashlib.sha256(f"raw:{name}".encode()).hexdigest(),
            "annotation_sha256": hashlib.sha256(f"annotation:{name}".encode()).hexdigest(),
        }
        for names in stage_names.values()
        for name in names
    }
    file_hashes: dict[str, str] = {}
    dataset_info: dict[str, object] = {}
    for stage, names in stage_names.items():
        for name in names:
            relative = f"{stage}/{name}.jsonl"
            content = json.dumps({"source": name}, separators=(",", ":")) + "\n"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            file_hashes[relative] = hashlib.sha256(content.encode()).hexdigest()
            dataset_info[name] = {"file_name": relative}

    counts = {"stage1": 2, "stage2": 4, "overlap": 0}
    if snapshot_schema == "chatts-dataset-snapshot-v2":
        snapshot_inputs = {
            name: {
                key: identity[key]
                for key in ("annotation_mode", "raw_sha256", "annotation_sha256")
            }
            for name, identity in input_identities.items()
        }
    else:
        snapshot_inputs = input_identities
    snapshot_hash = canonical_hash(
        {
            "schema_version": snapshot_schema,
            "selection": selection,
            "inputs": snapshot_inputs,
            "selected_outputs": dict(file_hashes),
            "counts": counts,
        }
    )
    selection_hash = canonical_hash(selection)
    for stage, names in stage_names.items():
        stage_manifest = {
            "schema_version": "chatts-dataset-stage-v1",
            "stage": stage,
            "selection_hash": selection_hash,
            "dataset_snapshot_hash": snapshot_hash,
            "rule": selection[stage],
            "dataset_names": names,
            "total_rows": len(names),
            "sources": {name: {"rows": 1} for name in names},
        }
        path = root / stage / "manifest.json"
        write_json(path, stage_manifest)
        file_hashes[f"{stage}/manifest.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_info_path = root / "dataset_info.json"
    write_json(dataset_info_path, dataset_info)
    file_hashes["dataset_info.json"] = hashlib.sha256(dataset_info_path.read_bytes()).hexdigest()
    training_env = root / "training.env"
    training_env.write_text("# fixture\n", encoding="utf-8")
    file_hashes["training.env"] = hashlib.sha256(training_env.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "chatts-dataset-studio-export-v1",
        "run_name": "test-fixture",
        "data_version": data_version,
        "created_at": "2026-08-11T00:00:00+00:00",
        "selection_hash": selection_hash,
        "dataset_snapshot_hash": snapshot_hash,
        "snapshot_hash_schema": snapshot_schema,
        "selection": selection,
        "preview": {"counts": counts},
        "dataset_names": stage_names,
        "input_identities": input_identities,
        "files": file_hashes,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(root / "manifest.json", manifest)
    return snapshot_hash


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
                    "STAGE1_MIX_STRATEGY": "concat",
                    "STAGE1_INTERLEAVE_PROBS": "",
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
            save_only_positions = [
                index for index, value in enumerate(args) if value == "--save_only_model"
            ]
            self.assertEqual(len(save_only_positions), 2)
            self.assertEqual(
                {args[index + 1] for index in save_only_positions}, {"False"}
            )
            best_model_positions = [
                index
                for index, value in enumerate(args)
                if value == "--load_best_model_at_end"
            ]
            self.assertEqual(len(best_model_positions), 2)
            self.assertEqual(
                {args[index + 1] for index in best_model_positions}, {"True"}
            )
            self.assertNotIn("--interleave_probs", args)


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
        write_studio_snapshot(dataset)
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
                "DATASET_VERIFIER": str(DATASET_VERIFIER),
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
            self.assertEqual(marker["data_version"], "")
            self.assertEqual(marker["resolved_configuration"]["DATA_VERSION"], "")
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
            logged = json.loads(run_manifest.read_text(encoding="utf-8"))
            self.assertEqual(logged["data_version"], "")
            self.assertEqual(logged["resolved_configuration"]["DATA_VERSION"], "")

    def test_stage1_then_stage2_reuses_and_never_deletes_shared_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "shared-stage1"
            final = root / "output" / "trial-final"
            stage1_hash = "a" * 64
            data_hash = write_studio_snapshot(
                Path(env["DATASET_DIR"]), data_version="datav3"
            )
            env.update(
                {
                    "PIPELINE_MODE": "stage1",
                    "STAGE1_OUT": str(stage1),
                    "TRIAL_ID": "shared-stage1",
                    "TRIAL_CONFIG_HASH": stage1_hash,
                    "DATASET_SNAPSHOT_HASH": data_hash,
                    "DATA_VERSION": "datav3",
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
            stage1_marker = json.loads(
                (stage1 / "STAGE1_COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stage1_marker["data_version"], "datav3")
            self.assertEqual(
                stage1_marker["resolved_configuration"]["DATA_VERSION"], "datav3"
            )
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
            self.assertEqual(marker["data_version"], "datav3")
            self.assertEqual(
                marker["resolved_configuration"]["DATA_VERSION"], "datav3"
            )
            self.assertTrue(marker["stage1_model_retained"])
            self.assertIsNone(marker["commands"]["stage1"])
            self.assertEqual(
                Path(marker["training_lineage"]["stage2"]["input_model_path"]),
                stage1.resolve(),
            )
            self.assertEqual(len(marker["stage1_input_provenance"]["sha256"]), 64)
            run_manifest = (
                root
                / "output"
                / "logs"
                / "chronos2_seed42_s1lr_1e-5_s2lr_1e-5_proxy-001"
                / "training_run_manifest.json"
            )
            logged = json.loads(run_manifest.read_text(encoding="utf-8"))
            self.assertEqual(logged["data_version"], "datav3")
            self.assertEqual(
                logged["resolved_configuration"]["DATA_VERSION"], "datav3"
            )

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

    def test_training_recipe_hash_reuses_completed_model_across_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            stage1 = root / "output" / "recipe-stage1"
            final = root / "output" / "recipe-final"
            recipe_hash = "c" * 64
            env.update(
                {
                    "STAGE1_OUT": str(stage1),
                    "FINAL_MODEL_PATH": str(final),
                    "TRAINING_RECIPE_HASH": recipe_hash,
                    "TRIAL_ID": "job-one",
                    "TRIAL_CONFIG_HASH": "a" * 64,
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            retried = env.copy()
            retried.update(
                {
                    "TRIAL_ID": "job-two",
                    "TRIAL_CONFIG_HASH": "b" * 64,
                }
            )
            result = subprocess.run(
                ["bash", str(RUNNER)],
                env=retried,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Training already completed; reusing", result.stdout)
            marker = json.loads(
                (final / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["training_recipe_hash"], recipe_hash)
            self.assertEqual(marker["trial_id"], "job-one")

            wrong_recipe = retried.copy()
            wrong_recipe["TRAINING_RECIPE_HASH"] = "d" * 64
            rejected = subprocess.run(
                ["bash", str(RUNNER)],
                env=wrong_recipe,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "training_recipe_hash",
                rejected.stderr + rejected.stdout,
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
                    "DATASET_SNAPSHOT_HASH": json.loads(
                        (Path(env["DATASET_DIR"]) / "manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )["dataset_snapshot_hash"],
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
                "dataset_snapshot_hash does not match",
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

    def test_preflight_rejects_noncanonical_data_version_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            output = root / "output" / "must-not-exist"
            env.update(
                {
                    "DATA_VERSION": "data-v3",
                    "FINAL_MODEL_PATH": str(output),
                    "PREFLIGHT_ONLY": "1",
                }
            )

            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("canonical datavN", result.stderr + result.stdout)
            self.assertFalse(output.exists())

    def test_preflight_verifies_v1_snapshot_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            snapshot_hash = write_studio_snapshot(
                Path(env["DATASET_DIR"]),
                data_version="datav3",
                snapshot_schema="chatts-dataset-snapshot-v1",
            )
            output = root / "output" / "must-not-exist"
            env.update(
                {
                    "DATASET_SNAPSHOT_HASH": snapshot_hash,
                    "DATA_VERSION": "datav3",
                    "FINAL_MODEL_PATH": str(output),
                    "PREFLIGHT_ONLY": "1",
                }
            )

            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dataset Studio snapshot verified", result.stdout)
            self.assertFalse(output.exists())

    def test_preflight_rejects_tampered_snapshot_file_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            output = root / "output" / "must-not-exist"
            (Path(env["DATASET_DIR"]) / "stage1" / "ift.jsonl").write_text(
                "tampered\n", encoding="utf-8"
            )
            env.update(
                {
                    "FINAL_MODEL_PATH": str(output),
                    "PREFLIGHT_ONLY": "1",
                }
            )

            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("snapshot file SHA256 mismatch", result.stderr + result.stdout)
            self.assertFalse(output.exists())

    def test_legacy_dataset_without_manifest_requires_no_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            legacy = root / "legacy-dataset"
            legacy.mkdir()
            output = root / "output" / "must-not-exist"
            env.update(
                {
                    "DATASET_DIR": str(legacy),
                    "FINAL_MODEL_PATH": str(output),
                    "PREFLIGHT_ONLY": "1",
                }
            )

            accepted = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("Legacy dataset directory", accepted.stdout)
            self.assertFalse(output.exists())

            identified = env.copy()
            identified["DATASET_SNAPSHOT_HASH"] = "d" * 64
            rejected = subprocess.run(
                ["bash", str(RUNNER)], env=identified, capture_output=True, text=True
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("manifest.json is missing", rejected.stderr + rejected.stdout)
            self.assertFalse(output.exists())

    def test_preflight_rejects_data_version_and_stage_key_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = self.make_fixture(root)
            snapshot_hash = write_studio_snapshot(
                Path(env["DATASET_DIR"]), data_version="datav3"
            )
            output = root / "output" / "must-not-exist"
            env.update(
                {
                    "DATASET_SNAPSHOT_HASH": snapshot_hash,
                    "DATA_VERSION": "datav4",
                    "FINAL_MODEL_PATH": str(output),
                    "PREFLIGHT_ONLY": "1",
                }
            )

            bad_version = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True
            )
            self.assertNotEqual(bad_version.returncode, 0)
            self.assertIn("DATA_VERSION does not match", bad_version.stderr + bad_version.stdout)
            self.assertFalse(output.exists())

            bad_keys_env = env.copy()
            bad_keys_env["DATA_VERSION"] = "datav3"
            bad_keys_env["STAGE2_DATASETS"] = "sft"
            bad_keys = subprocess.run(
                ["bash", str(RUNNER)], env=bad_keys_env, capture_output=True, text=True
            )
            self.assertNotEqual(bad_keys.returncode, 0)
            self.assertIn(
                "configured stage2 dataset keys do not match",
                bad_keys.stderr + bad_keys.stdout,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
