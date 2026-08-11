from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, dump_resolved
from .data import (
    DataCatalog,
    create_eval_dataset_views,
    create_eval_split_manifest,
    label_catalog,
    prepare_snapshot,
    validate_snapshot,
)
from .deepseek import (
    DeepSeekClient,
    DeepSeekError,
    proposal_validator,
    round_analysis_response_schema,
    round_analysis_validator,
)
from .hashing import command_fingerprint, hash_object, sha256_file
from .metrics import apply_gates, extract_badcases, load_metrics, sample_badcases
from .report import generate_report
from .runners import BlackBoxRunner
from .state import StateStore


class OrchestrationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_env(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _safe_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise OrchestrationError(f"Unsafe experiment id: {value!r}")
    return value


class Autoresearch:
    def __init__(
        self,
        config: Config,
        runner: BlackBoxRunner | None = None,
        deepseek_client: DeepSeekClient | None = None,
    ):
        self.config = config
        self.root = config.output_root
        for directory in (
            "configs",
            "logs",
            "models",
            "evaluations",
            "badcases",
            "figures",
            "commands",
            "analysis",
        ):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.state = StateStore(self.root / "state.sqlite3")
        self.catalog = DataCatalog(config)
        self.runner = runner or BlackBoxRunner(bool(config.get("runtime.dry_run", False)))
        self._deepseek = deepseek_client
        self._input_identity_cache: dict[tuple[str, str], str] = {}
        dump_resolved(config, self.root / "configs" / f"{config.fingerprint[:12]}.resolved.yaml")

    def close(self) -> None:
        if self._deepseek is not None:
            self._deepseek.close()

    @property
    def deepseek(self) -> DeepSeekClient:
        if not self.config.get("deepseek.enabled", True):
            raise OrchestrationError("DeepSeek is disabled in configuration")
        if self._deepseek is None:
            self._deepseek = DeepSeekClient(self.config.data["deepseek"], self.state)
        return self._deepseek

    def _export(self) -> None:
        self.state.export(self.root)

    def _input_identity(self, path_value: str | Path) -> str:
        path = Path(path_value).expanduser().resolve()
        stat_entries: list[tuple[Any, ...]] = []
        if path.is_dir():
            for item in sorted(path.rglob("*")):
                if item.is_symlink() and item.is_dir():
                    raise OrchestrationError(
                        f"Nested directory symlinks are not allowed in model inputs: {item}"
                    )
                if not item.is_file():
                    continue
                stat = item.stat()
                stat_entries.append(
                    (
                        item.relative_to(path).as_posix(),
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    )
                )
        signature = hash_object(stat_entries)
        key = (str(path), signature)
        if key not in self._input_identity_cache:
            self._input_identity_cache[key] = self._model_identity(path)
        return self._input_identity_cache[key]

    def _resolved_experiment_config(
        self,
        experiment_config: dict[str, Any],
        dataset_hash: str,
        protocol_hash: str,
        mode: str,
        stage1_path: Path,
    ) -> dict[str, Any]:
        clean = {key: value for key, value in experiment_config.items() if key != "_resolved"}
        return {
            **clean,
            "_resolved": {
                "training": copy.deepcopy(self.config.data["training"]),
                "data": copy.deepcopy(self.config.data["data"]),
                "evaluation": copy.deepcopy(self.config.data["evaluation"]),
                "gates": copy.deepcopy(self.config.data["gates"]),
                "search_policy": (
                    copy.deepcopy(self.config.data["search"])
                    if clean.get("role") != "baseline"
                    else None
                ),
                "deepseek_proposal_identity": {
                    "model": self.config.get("deepseek.model"),
                    "prompt_version": "chatts-search-v2",
                },
                "runtime": {
                    "seed": 42,
                    "gpu_ids": self.config.get("runtime.gpu_ids"),
                    "master_port": self.config.get("runtime.master_port"),
                },
                "train_script_sha256": sha256_file(self.config.require("paths.train_script")),
                "eval_script_sha256": sha256_file(self.config.require("paths.eval_script")),
                "interface_files": self._interface_file_hashes(),
                "autoresearch_files": self._autoresearch_file_hashes(),
                "base_model_identity": self._input_identity(
                    self.config.require("paths.base_model")
                ),
                "chronos2_identity": self._input_identity(
                    self.config.require("paths.chronos2_model")
                ),
                "stage1_input_identity": (
                    self._model_identity(stage1_path, require_weights=True)
                    if mode == "stage2"
                    else None
                ),
                "dataset_hash": dataset_hash,
                "protocol_hash": protocol_hash,
            },
        }

    def _interface_file_hashes(self) -> dict[str, str | None]:
        train_root = Path(str(self.config.require("paths.train_project"))).resolve()
        eval_root = Path(str(self.config.require("paths.eval_project"))).resolve()
        explicit = [
            train_root / "scripts/full/train_chronos2_best_stage1.sh",
            train_root / "scripts/full/train_chronos2_best_stage2.sh",
            train_root / "scripts/finalize_chatts_best_checkpoint.py",
            eval_root / "scripts/chatts_benchmark_artifacts.py",
            eval_root / "chatts/vllm/chatts_vllm.py",
            eval_root / "chatts/utils/llm_utils.py",
        ]
        discovered: list[Path] = []
        for pattern in ("scripts/run_chatts_*.sh", "scripts/evaluate_*.py"):
            discovered.extend(eval_root.glob(pattern))
        discovered.extend(eval_root.glob("chatts/utils/inference_*_vllm.py"))
        result: dict[str, str | None] = {}
        for path in sorted({*explicit, *discovered}):
            if path.is_relative_to(train_root):
                key = f"train:{path.relative_to(train_root).as_posix()}"
            else:
                key = f"eval:{path.relative_to(eval_root).as_posix()}"
            result[key] = sha256_file(path) if path.is_file() else None
        return result

    @staticmethod
    def _autoresearch_file_hashes() -> dict[str, str]:
        package_root = Path(__file__).resolve().parent
        return {
            path.name: sha256_file(path)
            for path in sorted(package_root.glob("*.py"))
            if path.is_file()
        }

    @staticmethod
    def _semantic_command_hash(commands: dict[str, dict[str, Any]]) -> str:
        fingerprints = {}
        for name, command in commands.items():
            env = dict(command["env"])
            for force_key in ("FORCE_TRAIN", "FORCE_EVAL"):
                if force_key in env:
                    env[force_key] = "0"
            fingerprints[name] = command_fingerprint(
                command["argv"], command["cwd"], env
            )
        return hash_object(fingerprints)

    def preflight(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        for key, kind in (
            ("paths.train_project", "dir"),
            ("paths.train_script", "file"),
            ("paths.eval_project", "dir"),
            ("paths.eval_script", "file"),
            ("paths.chronos2_model", "dir"),
            ("paths.base_model", "dir"),
        ):
            path = Path(str(self.config.require(key))).expanduser().resolve()
            passed = path.is_dir() if kind == "dir" else path.is_file()
            checks.append({"name": key, "path": str(path), "passed": passed})
        for key in ("paths.base_model", "paths.chronos2_model"):
            path = Path(str(self.config.require(key))).expanduser().resolve()
            try:
                identity = self._model_identity(path, require_weights=True)
            except OrchestrationError as exc:
                checks.append(
                    {
                        "name": f"model-artifacts:{key}",
                        "path": str(path),
                        "passed": False,
                        "detail": str(exc),
                    }
                )
            else:
                checks.append(
                    {
                        "name": f"model-artifacts:{key}",
                        "path": str(path),
                        "passed": True,
                        "identity": identity,
                    }
                )
        for key, kind in (
            ("paths.tsrbench_root", "dir"),
            ("paths.timeseriesexam_root", "dir"),
            ("paths.timeseriesexam_data_file", "file"),
            ("paths.ts_haystack_root", "dir"),
            ("paths.tinybench_root", "dir"),
        ):
            if not self.config.get(key):
                continue
            path = Path(str(self.config.get(key))).expanduser().resolve()
            passed = path.is_dir() if kind == "dir" else path.is_file()
            checks.append({"name": key, "path": str(path), "passed": passed})
        catalog = self.catalog
        checks.extend(
            {"name": f"datav2:{source.name}", "path": str(source.path), "passed": source.path.is_file()}
            for source in catalog.sources
        )
        for script_key in ("paths.train_script", "paths.eval_script"):
            script = Path(str(self.config.require(script_key))).resolve()
            if script.is_file():
                syntax = subprocess.run(
                    ["bash", "-n", str(script)], capture_output=True, text=True, check=False
                )
                checks.append(
                    {
                        "name": f"bash-n:{script_key}",
                        "path": str(script),
                        "passed": syntax.returncode == 0,
                        "detail": syntax.stderr.strip(),
                    }
                )
        gpu_ids = [item for item in str(self.config.get("runtime.gpu_ids")).split(",") if item]
        checks.append(
            {
                "name": "fixed-eight-gpu-mask",
                "path": str(self.config.get("runtime.gpu_ids")),
                "passed": len(gpu_ids) == 8 and len(set(gpu_ids)) == 8,
            }
        )
        global_batch = (
            len(gpu_ids)
            * int(self.config.get("training.per_device_batch_size"))
            * int(self.config.get("training.gradient_accumulation_steps"))
        )
        checks.append(
            {
                "name": "fixed-global-batch-512",
                "path": str(global_batch),
                "passed": global_batch == 512,
            }
        )
        split = create_eval_split_manifest(self.config)
        eval_views = create_eval_dataset_views(self.config)
        result = {
            "passed": all(item["passed"] for item in checks),
            "config_hash": self.config.fingerprint,
            "dataset_hash": catalog.fingerprint,
            "eval_split_hash": split["split_hash"],
            "eval_view_hash": eval_views["view_hash"],
            "deepseek_endpoint": self.config.get("deepseek.base_url"),
            "deepseek_network_check": "deferred until label/search",
            "checks": checks,
        }
        path = self.root / "preflight.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.state.metadata_put("preflight", result)
        self.state.metadata_put("eval_views", eval_views)
        self._export()
        if not result["passed"]:
            failed = [item["name"] for item in checks if not item["passed"]]
            raise OrchestrationError(f"Preflight failed: {', '.join(failed)}")
        return result

    def label(self) -> dict[str, int]:
        result = label_catalog(self.config, self.state, self.deepseek)
        self._export()
        return result

    def prepare_data(self) -> dict[str, Any]:
        result = prepare_snapshot(self.config, self.state)
        create_eval_split_manifest(self.config)
        eval_views = create_eval_dataset_views(self.config)
        self.state.metadata_put("eval_views", eval_views)
        self._export()
        return result

    def _baseline_data_policy(self) -> dict[str, Any]:
        """Return the data policy represented by ``baseline_snapshot``.

        A raw snapshot has no filtering, de-duplication, or re-sampling.  Data
        family trials start from that raw-equivalent policy so their one patch
        does not silently activate unrelated configured preprocessing.
        """
        policy = copy.deepcopy(self.config.data["data"])
        snapshot_name = str(self.config.get("data.snapshot_name"))
        baseline_snapshot = str(self.config.get("data.baseline_snapshot"))
        if baseline_snapshot in {"prepared", snapshot_name}:
            return policy
        policy.update(
            {
                "minimum_quality": 0.0,
                "missing_label_policy": "keep",
                "drop_exact_duplicates": False,
                "drop_cross_source_duplicates": False,
                "drop_near_duplicates": False,
                "source_weights": {},
                "difficulty_weights": {
                    "easy": 1.0,
                    "medium": 1.0,
                    "hard": 1.0,
                },
            }
        )
        return policy

    def _dataset_for(self, baseline: bool, patch: dict[str, Any] | None = None) -> tuple[Path, str]:
        snapshot_name = str(self.config.get("data.snapshot_name"))
        if patch and next(iter(patch), None) in {
            "source_weights",
            "minimum_quality",
            "difficulty_weights",
        }:
            patched = copy.deepcopy(self.config.data)
            patched["data"] = self._baseline_data_policy()
            key, value = next(iter(patch.items()))
            data_key = "minimum_quality" if key == "minimum_quality" else key
            if data_key in {"source_weights", "difficulty_weights"}:
                merged_weights = dict(patched["data"].get(data_key, {}))
                merged_weights.update(value)
                patched["data"][data_key] = merged_weights
            else:
                patched["data"][data_key] = value
            label_fingerprint = self.state.label_fingerprint(
                str(self.config.get("deepseek.prompt_version")),
                str(self.config.get("deepseek.model")),
            )
            derived_identity = hash_object(
                {"patch": patch, "label_fingerprint": label_fingerprint}
            )
            patched["data"]["snapshot_name"] = (
                f"{snapshot_name}-{derived_identity[:10]}"
            )
            derived = Config(patched, self.config.source_path)
            manifest = prepare_snapshot(derived, self.state)
            if key in {"minimum_quality", "difficulty_weights"} and manifest.get(
                "label_coverage"
            ) != 1.0:
                raise OrchestrationError(
                    f"{key} search requires complete quality/difficulty labels; "
                    f"observed coverage={manifest.get('label_coverage')!r}. Run label first."
                )
            return self.root / "datasets" / patched["data"]["snapshot_name"], manifest["snapshot_hash"]
        # A non-data trial must use exactly the baseline dataset; otherwise an
        # LR-only trial would silently change both LR and data preprocessing.
        baseline_snapshot = str(self.config.get("data.baseline_snapshot"))
        selected_snapshot = snapshot_name if baseline_snapshot in {"prepared", snapshot_name} else baseline_snapshot
        path = self.root / "datasets" / selected_snapshot
        if not (path / "manifest.json").is_file():
            raise OrchestrationError(
                f"Baseline dataset snapshot {selected_snapshot!r} is missing; run prepare-data first"
            )
        manifest = validate_snapshot(self.config, self.state, path)
        return path, manifest["snapshot_hash"]

    def _protocol_hash(self, split: str, benchmarks: str) -> str:
        split_manifest = create_eval_split_manifest(self.config)
        view_manifest_path = self.root / "eval_views" / "manifest.json"
        view_hash = None
        if view_manifest_path.is_file():
            view_hash = json.loads(view_manifest_path.read_text(encoding="utf-8")).get("view_hash")
        return hash_object(
            {
                "evaluation": self.config.data["evaluation"],
                "split": split,
                "benchmarks": sorted(item.strip() for item in benchmarks.split(",") if item.strip()),
                "split_hash": split_manifest["split_hash"],
                "eval_view_hash": view_hash,
                "eval_interface_files": self._interface_file_hashes(),
                "autoresearch_metrics_sha256": sha256_file(
                    Path(__file__).resolve().parent / "metrics.py"
                ),
                "seed": 42,
            }
        )

    def _training_env(
        self,
        experiment_id: str,
        experiment_config: dict[str, Any],
        dataset_dir: Path,
        dataset_hash: str,
        model_path: Path,
        stage1_path: Path,
        mode: str,
        force: bool,
        trial_config_hash: str,
    ) -> dict[str, str]:
        training = self.config.data["training"]
        patch = experiment_config.get("patch", {})
        learning_rate = float(patch.get("learning_rate", training["stage2_learning_rate"]))
        base_projector_ratio = float(training["stage2_timeseries_learning_rate"]) / float(
            training["stage2_learning_rate"]
        )
        projector_ratio = float(patch.get("projector_lr_ratio", base_projector_ratio))
        env: dict[str, Any] = {
            "PROJECT_ROOT": self.config.require("paths.train_project"),
            "MODEL_PATH": self.config.require("paths.base_model"),
            "CHRONOS2_MODEL_PATH": self.config.require("paths.chronos2_model"),
            "OUTPUT_ROOT": self.root / "models",
            "FINAL_MODEL_PATH": model_path,
            "STAGE1_OUT": stage1_path,
            "PIPELINE_MODE": mode,
            "KEEP_STAGE1": 1,
            "DATASET_DIR": dataset_dir,
            "TRIAL_ID": experiment_id,
            "TRIAL_CONFIG_HASH": trial_config_hash,
            "DATASET_SNAPSHOT_HASH": dataset_hash,
            "SEED": 42,
            "S1_LR": training["stage1_learning_rate"],
            "S2_LR": learning_rate,
            "STAGE1_TIMESERIES_SFT_LR": training["stage1_timeseries_learning_rate"],
            "STAGE2_TIMESERIES_SFT_LR": learning_rate * projector_ratio,
            "STAGE1_DATASETS": training["stage1_datasets"],
            "STAGE2_DATASETS": training["stage2_datasets"],
            "STAGE1_MIX_STRATEGY": training["stage1_mix_strategy"],
            "STAGE2_MIX_STRATEGY": training["stage2_mix_strategy"],
            "STAGE1_INTERLEAVE_PROBS": training.get("stage1_interleave_probs", ""),
            "STAGE2_INTERLEAVE_PROBS": training.get("stage2_interleave_probs", ""),
            "STAGE1_NUM_TRAIN_EPOCHS": training["stage1_epochs"],
            "STAGE2_NUM_TRAIN_EPOCHS": patch.get("epochs", training["stage2_epochs"]),
            "STAGE2_MAX_STEPS": experiment_config.get("max_steps", 0),
            "STAGE2_WARMUP_RATIO": patch.get("warmup_ratio", training["stage2_warmup_ratio"]),
            "STAGE2_LR_SCHEDULER_TYPE": patch.get("scheduler", training["stage2_scheduler"]),
            "STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE": training["per_device_batch_size"],
            "STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE": training["per_device_batch_size"],
            "STAGE1_GRADIENT_ACCUMULATION_STEPS": training["gradient_accumulation_steps"],
            "STAGE2_GRADIENT_ACCUMULATION_STEPS": training["gradient_accumulation_steps"],
            "STAGE1_CUTOFF_LEN": training["cutoff_len"],
            "STAGE2_CUTOFF_LEN": training["cutoff_len"],
            "STAGE1_VAL_SIZE": training["val_size"],
            "STAGE2_VAL_SIZE": training["val_size"],
            "DEEPSPEED_INCLUDE": "localhost:" + str(self.config.get("runtime.gpu_ids")),
            "MASTER_PORT": self.config.get("runtime.master_port"),
            "PYTHON_BIN": self.config.get("runtime.python_bin"),
            "FORCE_TRAIN": force,
        }
        if mode == "stage2":
            env["STAGE2_FROM"] = stage1_path
        return {key: _as_env(value) for key, value in env.items()}

    def _evaluation_env(
        self,
        experiment_id: str,
        model_path: Path,
        output_dir: Path,
        split: str,
        benchmarks: str,
        protocol_hash: str,
        force: bool,
    ) -> dict[str, str]:
        paths = self.config.data["paths"]
        env: dict[str, Any] = {
            "PROJECT_ROOT": paths["eval_project"],
            "MODEL_PATH": model_path,
            "MODEL_NAME": experiment_id,
            "CHRONOS2_MODEL_PATH": paths["chronos2_model"],
            "OUTPUT_ROOT": output_dir,
            "RUN_ID": experiment_id,
            "EVAL_PROTOCOL_HASH": protocol_hash,
            "EVAL_SPLIT": split,
            "BENCHMARKS": benchmarks,
            "SEED": 42,
            "EVAL_GPUS": self.config.get("runtime.gpu_ids"),
            "EVAL_NUM_GPUS": 8,
            "MAX_SAMPLES": (
                self.config.get("evaluation.final_max_samples", 0)
                if split == self.config.get("evaluation.final_split")
                else self.config.get(
                    "evaluation.search_max_samples",
                    self.config.get("evaluation.max_samples", 0),
                )
            ),
            "FORCE_EVAL": force,
            "PYTHON_BIN": self.config.get("runtime.python_bin"),
            "OFFLINE": 1,
            "HAYSTACK_SPLIT": (
                self.config.get("evaluation.haystack_search_split")
                if split == self.config.get("evaluation.search_split")
                else self.config.get("evaluation.haystack_final_split")
            ),
            "TINY_DATA_PARTITION": split,
            "TINY_PARTITION_SEED": self.config.get("evaluation.tiny_partition_seed"),
        }
        optional = {
            "TSRBENCH_ROOT": "tsrbench_root",
            "TSRBENCH_DATASET_ROOT": "tsrbench_root",
            "TIMESERIESEXAM_ROOT": "timeseriesexam_root",
            "TIMESERIESEXAM_DATA_FILE": "timeseriesexam_data_file",
            "TS_HAYSTACK_ROOT": "ts_haystack_root",
            "TINYBENCH_DATASET_ROOT": "tinybench_root",
        }
        for env_key, path_key in optional.items():
            if paths.get(path_key):
                env[env_key] = paths[path_key]
        views_root = self.root / "eval_views"
        view_manifest_path = views_root / "manifest.json"
        if view_manifest_path.is_file():
            view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
            if view_manifest.get("available", {}).get("tsrbench"):
                tsr_view = views_root / split / "tsrbench"
                if not tsr_view.is_dir():
                    raise OrchestrationError(f"Missing locked TSRBench view for split {split}")
                env["TSRBENCH_ROOT"] = tsr_view
                env["TSRBENCH_DATASET_ROOT"] = tsr_view
            if view_manifest.get("available", {}).get("timeseriesexam"):
                exam_view = views_root / split / "timeseriesexam" / "qa_dataset.json"
                if not exam_view.is_file():
                    raise OrchestrationError(f"Missing locked TimeSeriesExam view for split {split}")
                env["TIMESERIESEXAM_DATA_FILE"] = exam_view
        else:
            if paths.get("tsrbench_root") and Path(str(paths["tsrbench_root"])).is_dir():
                raise OrchestrationError("TSRBench views are missing; run prepare-data before evaluation")
            if paths.get("timeseriesexam_data_file") and Path(str(paths["timeseriesexam_data_file"])).is_file():
                raise OrchestrationError("TimeSeriesExam views are missing; run prepare-data before evaluation")
        return {key: _as_env(value) for key, value in env.items()}

    @staticmethod
    def _validation_loss(model_path: Path) -> float | None:
        for name in ("TRAINING_COMPLETE.json", "best_model_manifest.json"):
            path = model_path / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for key in ("stage2_best_eval_loss", "best_metric", "eval_loss"):
                if isinstance(payload.get(key), (int, float)):
                    return float(payload[key])
        return None

    def _assert_existing_matches(self, existing: dict[str, Any], expected: dict[str, Any]) -> None:
        for key in (
            "kind",
            "phase",
            "parent_id",
            "config_hash",
            "dataset_hash",
            "protocol_hash",
            "output_dir",
        ):
            if existing[key] != expected[key]:
                raise OrchestrationError(
                    f"Experiment id {existing['id']} exists with a different {key}"
                )
        if existing.get("config") != expected.get("config_json"):
            raise OrchestrationError(
                f"Experiment id {existing['id']} exists with different resolved config bytes"
            )

    def _execute_train_eval(
        self,
        experiment_id: str,
        kind: str,
        phase: str,
        experiment_config: dict[str, Any],
        baseline: bool,
        parent_id: str | None,
        stage1_path: Path,
        mode: str,
        benchmarks: str,
        split: str,
        baseline_metrics: dict[str, Any] | None,
        force: bool = False,
    ) -> dict[str, Any]:
        experiment_id = _safe_id(experiment_id)
        dataset_dir, dataset_hash = self._dataset_for(baseline, experiment_config.get("patch"))
        protocol_hash = self._protocol_hash(split, benchmarks)
        resolved_config = self._resolved_experiment_config(
            experiment_config, dataset_hash, protocol_hash, mode, stage1_path
        )
        config_hash = hash_object(resolved_config)
        model_path = self.root / "models" / experiment_id
        eval_dir = self.root / "evaluations" / experiment_id / split
        payload = {
            "id": experiment_id,
            "kind": kind,
            "phase": phase,
            "parent_id": parent_id,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "protocol_hash": protocol_hash,
            "config_json": resolved_config,
            "output_dir": str(eval_dir),
        }
        existing = self.state.create_experiment(payload)
        self._assert_existing_matches(existing, payload)
        train_env = self._training_env(
            experiment_id,
            experiment_config,
            dataset_dir,
            dataset_hash,
            model_path,
            stage1_path,
            mode,
            force or existing["status"] in {"failed", "running"},
            config_hash,
        )
        eval_env = self._evaluation_env(
            experiment_id,
            model_path,
            eval_dir,
            split,
            benchmarks,
            protocol_hash,
            force or existing["status"] in {"failed", "running"},
        )
        train_argv = ["bash", str(self.config.require("paths.train_script"))]
        eval_argv = ["bash", str(self.config.require("paths.eval_script"))]
        commands = {
            "train": {"argv": train_argv, "cwd": self.config.require("paths.train_project"), "env": train_env},
            "evaluate": {"argv": eval_argv, "cwd": self.config.require("paths.eval_project"), "env": eval_env},
        }
        combined_hash = self._semantic_command_hash(commands)
        (self.root / "commands" / f"{experiment_id}.json").write_text(
            json.dumps({"command_hash": combined_hash, **commands}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if existing["status"] == "completed":
            if existing.get("command_hash") != combined_hash:
                raise OrchestrationError(
                    f"Completed experiment {experiment_id} command hash no longer matches"
                )
            metrics = existing.get("metrics") or {}
            if metrics.get("model_identity") != self._model_identity(
                model_path, require_weights=True
            ):
                raise OrchestrationError(
                    f"Completed experiment {experiment_id} final model identity changed"
                )
            if metrics.get("stage1_identity") != self._model_identity(
                stage1_path, require_weights=True
            ):
                raise OrchestrationError(
                    f"Completed experiment {experiment_id} shared Stage1 identity changed"
                )
            if metrics.get("evaluation_identity") != self._directory_identity(
                eval_dir, require_metrics=True
            ):
                raise OrchestrationError(
                    f"Completed experiment {experiment_id} evaluation artifacts changed"
                )
            return existing
        if self.runner.dry_run:
            self._export()
            return self.state.get_experiment(experiment_id)
        self.state.mark_running(experiment_id, combined_hash, commands)
        try:
            train_result = self.runner.run(
                train_argv,
                self.config.require("paths.train_project"),
                train_env,
                self.root / "logs" / f"{experiment_id}.train.log",
            )
            eval_result = self.runner.run(
                eval_argv,
                self.config.require("paths.eval_project"),
                eval_env,
                self.root / "logs" / f"{experiment_id}.eval.{split}.log",
            )
            metrics = load_metrics(eval_dir)
            metrics = apply_gates(
                metrics,
                baseline_metrics,
                self.config.data["gates"],
                require_guards=phase in {"baseline", "full", "final-test"},
            )
            metrics["validation_loss"] = self._validation_loss(model_path)
            metrics["model_identity"] = self._model_identity(
                model_path, require_weights=True
            )
            metrics["stage1_identity"] = self._model_identity(
                stage1_path, require_weights=True
            )
            gpu_count = len(str(self.config.get("runtime.gpu_ids")).split(","))
            metrics["gpu_hours"] = (
                train_result.duration_seconds + eval_result.duration_seconds
            ) * gpu_count / 3600.0
            metrics["train_seconds"] = train_result.duration_seconds
            metrics["eval_seconds"] = eval_result.duration_seconds
            metrics["badcase_summary"] = extract_badcases(
                eval_dir,
                self.root / "badcases" / f"{experiment_id}.{split}.jsonl",
                {"tsrbench": self.root / "eval_views" / split / "tsrbench"},
            )
            metrics["evaluation_identity"] = self._directory_identity(
                eval_dir, require_metrics=True
            )
            self.state.mark_completed(experiment_id, metrics, str(model_path))
        except Exception as exc:
            self.state.mark_failed(experiment_id, str(exc))
            self._export()
            raise
        self._export()
        return self.state.get_experiment(experiment_id)

    def baseline(self) -> dict[str, Any]:
        config = {"role": "baseline", "encoder": "chronos2", "seed": 42, "max_steps": 0, "patch": {}}
        return self._execute_train_eval(
            "baseline",
            "baseline",
            "baseline",
            config,
            True,
            None,
            self.root / "models" / "shared-stage1",
            "full",
            self.config.get("evaluation.final_benchmarks"),
            self.config.get("evaluation.search_split"),
            None,
        )

    def _deterministic_proposals(self) -> list[dict[str, Any]]:
        search = self.config.data["search"]
        base_lr = float(self.config.get("training.stage2_learning_rate"))
        base_projector_ratio = float(
            self.config.get("training.stage2_timeseries_learning_rate")
        ) / float(self.config.get("training.stage2_learning_rate"))
        base_warmup = float(self.config.get("training.stage2_warmup_ratio"))
        base_scheduler = self.config.get("training.stage2_scheduler")
        by_family: dict[str, list[dict[str, Any]]] = {}

        def add(family: str, value: Any) -> None:
            by_family.setdefault(family, []).append(
                {
                    "family": family,
                    "patch": {family: value},
                    "rationale": "whitelist grid",
                }
            )

        for value in search["learning_rates"]:
            if float(value) != base_lr:
                add("learning_rate", float(value))
        for value in search["projector_lr_ratios"]:
            if float(value) != base_projector_ratio:
                add("projector_lr_ratio", float(value))
        for value in search["warmup_ratios"]:
            if float(value) != base_warmup:
                add("warmup_ratio", float(value))
        for value in search["schedulers"]:
            if value != base_scheduler:
                add("scheduler", value)
        for value in search["epochs"]:
            if int(value) != int(self.config.get("training.stage2_epochs")):
                add("epochs", int(value))
        baseline_policy = self._baseline_data_policy()
        source_names = sorted(source.name for source in self.catalog.sources)
        low, high = [float(value) for value in search["source_weight_range"]]
        if source_names:
            for value in (low, high):
                if value != 1.0:
                    add("source_weights", {source_names[0]: value})
        for value in search["minimum_qualities"]:
            if float(value) != float(baseline_policy["minimum_quality"]):
                add("minimum_quality", float(value))
        for difficulty in ("hard", "easy", "medium"):
            for value in (high, low):
                if value != float(baseline_policy["difficulty_weights"][difficulty]):
                    add("difficulty_weights", {difficulty: value})
                    break
        candidates: list[dict[str, Any]] = []
        family_order = (
            "learning_rate",
            "projector_lr_ratio",
            "source_weights",
            "minimum_quality",
            "difficulty_weights",
            "warmup_ratio",
            "scheduler",
            "epochs",
        )
        while any(by_family.values()):
            for family in family_order:
                if by_family.get(family):
                    candidates.append(by_family[family].pop(0))
        if not candidates:
            raise OrchestrationError("Search whitelist contains no changes from baseline")
        return candidates

    def _is_baseline_equivalent_patch(self, patch: dict[str, Any]) -> bool:
        if len(patch) != 1:
            return False
        key, value = next(iter(patch.items()))
        training = self.config.data["training"]
        data = self._baseline_data_policy()
        scalar_baselines = {
            "learning_rate": float(training["stage2_learning_rate"]),
            "projector_lr_ratio": float(training["stage2_timeseries_learning_rate"])
            / float(training["stage2_learning_rate"]),
            "warmup_ratio": float(training["stage2_warmup_ratio"]),
            "scheduler": training["stage2_scheduler"],
            "epochs": int(training["stage2_epochs"]),
            "minimum_quality": float(data["minimum_quality"]),
        }
        if key in scalar_baselines:
            return value == scalar_baselines[key]
        if key in {"source_weights", "difficulty_weights"} and isinstance(value, dict):
            merged = dict(data.get(key, {}))
            merged.update(value)
            return merged == data.get(key, {})
        return False

    def _deepseek_round_analysis(
        self,
        round_index: int,
        experiment: dict[str, Any],
        used_patches: list[dict[str, Any]],
        *,
        require_proposal: bool,
    ) -> dict[str, Any]:
        used_hashes = {hash_object(patch) for patch in used_patches}
        badcase_path = self.root / "badcases" / (
            f"{experiment['id']}.{self.config.get('evaluation.search_split')}.jsonl"
        )
        sampled = sample_badcases(badcase_path, 64)
        compact_cases = [
            {
                "badcase_id": item.get("badcase_id"),
                "suite": item.get("suite"),
                "task": item.get("task"),
                "source": item.get("source"),
                "difficulty": item.get("difficulty"),
                "prediction": str(item.get("prediction"))[:800],
                "gold": str(item.get("gold"))[:800],
                "question": str(item.get("question"))[:1200],
            }
            for item in sampled
        ]
        prompt = {
            "round_index": round_index,
            "source_experiment_id": experiment["id"],
            "observed_metrics": experiment.get("metrics"),
            "badcases": compact_cases,
            "allowed_search_space": self.config.data["search"],
            "allowed_sources": sorted(source.name for source in self.catalog.sources),
            "current_data_policy": {
                key: self._baseline_data_policy()[key]
                for key in (
                    "minimum_quality",
                    "source_weights",
                    "difficulty_weights",
                )
            },
            "already_used_patches": used_patches,
            "instruction": (
                "Group only the supplied badcase IDs by error type, state possible data causes, "
                "then recommend exactly one parameter family and one whitelisted Stage2 patch. "
                "When proposal_required is false, proposal must be null. Never return code or commands."
            ),
            "proposal_required": require_proposal,
        }
        analysis = self.deepseek.complete_json(
            purpose="badcase-round-analysis",
            system=(
                "You diagnose observed ChatTS time-series reasoning errors and propose one "
                "conservative Stage2 experiment. Return strict JSON with exactly error_groups, "
                "recommended_family, proposal. Each error group has error_type, "
                "likely_data_cause, badcase_ids. When proposal_required is true, proposal "
                "has family, patch, rationale; otherwise proposal is null. "
                "Use only supplied badcase IDs and the whitelist; never return shell, Python, "
                "scores, or prose outside JSON."
            ),
            user=json.dumps(prompt, ensure_ascii=False),
            validator=round_analysis_validator(
                self.config.data["search"],
                {source.name for source in self.catalog.sources},
                {
                    str(item["badcase_id"])
                    for item in compact_cases
                    if item.get("badcase_id") is not None
                },
                disallowed_patch_hashes=used_hashes,
                reject_patch=self._is_baseline_equivalent_patch,
                require_proposal=require_proposal,
            ),
            prompt_version="chatts-search-v2",
            response_schema=round_analysis_response_schema(
                self.config.data["search"],
                {source.name for source in self.catalog.sources},
                {
                    str(item["badcase_id"])
                    for item in compact_cases
                    if item.get("badcase_id") is not None
                },
                require_proposal=require_proposal,
            ),
        )
        payload = {
            "schema_version": "chatts-round-analysis-v1",
            "round_index": round_index,
            "source_experiment_id": experiment["id"],
            "sampled_badcases": len(compact_cases),
            "sampled_badcase_ids": [item["badcase_id"] for item in compact_cases],
            **analysis,
        }
        destination = self.root / "analysis" / f"round-{round_index:02d}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return analysis

    @staticmethod
    def _rank(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def key(item: dict[str, Any]) -> tuple[float, float, float]:
            metrics = item.get("metrics") or {}
            score = metrics.get("primary_score")
            gpu = metrics.get("gpu_hours")
            loss = metrics.get("validation_loss")
            return (
                -(float(score) if score is not None else -1.0),
                float(gpu) if gpu is not None else float("inf"),
                float(loss) if loss is not None else float("inf"),
            )

        return sorted(experiments, key=key)

    @staticmethod
    def _experiment_reference(experiment: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": experiment["id"],
            "status": experiment["status"],
            "phase": experiment["phase"],
            "parent_id": experiment.get("parent_id"),
            "config_hash": experiment["config_hash"],
            "dataset_hash": experiment["dataset_hash"],
            "protocol_hash": experiment["protocol_hash"],
            "command_hash": experiment.get("command_hash"),
            "metrics_hash": hash_object(experiment.get("metrics") or {}),
            "model_path": experiment.get("model_path"),
        }

    def _search_policy_hash(self) -> str:
        return hash_object(
            {
                "search": self.config.data["search"],
                "evaluation": self.config.data["evaluation"],
                "gates": self.config.data["gates"],
                "deepseek_model": self.config.get("deepseek.model"),
                "prompt_version": "chatts-search-v2",
                "autoresearch_files": self._autoresearch_file_hashes(),
            }
        )

    def _write_search_manifest(
        self,
        baseline: dict[str, Any],
        proxies: list[dict[str, Any]],
        ranked: list[dict[str, Any]],
        finalists: list[dict[str, Any]],
    ) -> dict[str, Any]:
        relevant_names = {f"round-{index:02d}.json" for index in range(1, len(proxies) + 1)}
        relevant_names.add("round-00.json")
        analysis_hashes = {
            path.name: sha256_file(path)
            for path in sorted((self.root / "analysis").glob("round-*.json"))
            if path.name in relevant_names
        }
        payload = {
            "schema_version": "chatts-search-complete-v1",
            "config_hash": self.config.fingerprint,
            "search_policy_hash": self._search_policy_hash(),
            "baseline": self._experiment_reference(baseline),
            "proxies": [self._experiment_reference(item) for item in proxies],
            "ranking": [item["id"] for item in ranked],
            "selected_proxy_ids": [item.get("parent_id") for item in finalists],
            "finalists": [self._experiment_reference(item) for item in finalists],
            "analysis_hashes": analysis_hashes,
            "completed_at": _utc_now(),
        }
        payload["search_hash"] = hash_object(payload)
        path = self.root / "SEARCH_COMPLETE.json"
        if path.is_file():
            existing = self._load_search_manifest(validate_experiments=False)
            ignored = {"completed_at", "search_hash"}
            old_comparable = {key: value for key, value in existing.items() if key not in ignored}
            new_comparable = {key: value for key, value in payload.items() if key not in ignored}
            if old_comparable != new_comparable:
                raise OrchestrationError(
                    "SEARCH_COMPLETE.json conflicts with the current ranking/finalists"
                )
            return existing
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        self.state.metadata_put("search_complete", payload)
        return payload

    def _load_search_manifest(self, *, validate_experiments: bool = True) -> dict[str, Any]:
        path = self.root / "SEARCH_COMPLETE.json"
        if not path.is_file():
            raise OrchestrationError(
                "Cannot freeze before all proxy trials and finalists complete and search creates "
                "SEARCH_COMPLETE.json"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OrchestrationError("SEARCH_COMPLETE.json is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != "chatts-search-complete-v1":
            raise OrchestrationError("Unsupported SEARCH_COMPLETE.json schema_version")
        stored_hash = payload.get("search_hash")
        unsigned = {key: value for key, value in payload.items() if key != "search_hash"}
        if not isinstance(stored_hash, str) or stored_hash != hash_object(unsigned):
            raise OrchestrationError("SEARCH_COMPLETE.json search_hash does not match its contents")
        if payload.get("config_hash") != self.config.fingerprint:
            raise OrchestrationError("Current configuration does not match SEARCH_COMPLETE.json")
        if payload.get("search_policy_hash") != self._search_policy_hash():
            raise OrchestrationError("Current search policy does not match SEARCH_COMPLETE.json")
        if len(payload.get("proxies", [])) != int(self.config.get("search.proxy_trials")):
            raise OrchestrationError("SEARCH_COMPLETE.json has the wrong proxy count")
        if len(payload.get("finalists", [])) != int(self.config.get("search.full_finalists")):
            raise OrchestrationError("SEARCH_COMPLETE.json has the wrong finalist count")
        analysis_hashes = payload.get("analysis_hashes")
        if not isinstance(analysis_hashes, dict):
            raise OrchestrationError("SEARCH_COMPLETE.json has invalid analysis hashes")
        if self.config.get("search.proposal_mode") == "deepseek":
            required_rounds = {
                f"round-{index:02d}.json"
                for index in range(0, int(self.config.get("search.proxy_trials")) + 1)
            }
            if not required_rounds.issubset(analysis_hashes):
                raise OrchestrationError("SEARCH_COMPLETE.json is missing DeepSeek round analyses")
        for name, expected_hash in analysis_hashes.items():
            if name != Path(name).name or not name.startswith("round-") or not name.endswith(".json"):
                raise OrchestrationError("SEARCH_COMPLETE.json has an unsafe analysis path")
            analysis_path = self.root / "analysis" / name
            if not analysis_path.is_file() or sha256_file(analysis_path) != expected_hash:
                raise OrchestrationError(f"Search analysis was deleted or changed: {name}")
        if not validate_experiments:
            return payload
        expected_proxy_ids = [
            f"proxy-{index:02d}"
            for index in range(1, int(self.config.get("search.proxy_trials")) + 1)
        ]
        baseline_reference = payload.get("baseline")
        if (
            not isinstance(baseline_reference, dict)
            or baseline_reference.get("id") != "baseline"
        ):
            raise OrchestrationError("SEARCH_COMPLETE.json has an invalid baseline reference")
        if [item.get("id") for item in payload["proxies"]] != expected_proxy_ids:
            raise OrchestrationError("SEARCH_COMPLETE.json has invalid proxy identities/order")
        all_references = [
            baseline_reference,
            *payload["proxies"],
            *payload["finalists"],
        ]
        reference_ids = [
            item.get("id") if isinstance(item, dict) else None
            for item in all_references
        ]
        if None in reference_ids or len(set(reference_ids)) != len(reference_ids):
            raise OrchestrationError("SEARCH_COMPLETE.json has duplicate experiment references")
        loaded_experiments: dict[str, dict[str, Any]] = {}
        for reference in all_references:
            if not isinstance(reference, dict) or not reference.get("id"):
                raise OrchestrationError("SEARCH_COMPLETE.json has an invalid experiment reference")
            try:
                experiment = self.state.get_experiment(reference["id"])
            except KeyError as exc:
                raise OrchestrationError(
                    f"Search experiment is missing from SQLite: {reference['id']}"
                ) from exc
            current = self._experiment_reference(experiment)
            if current != reference or experiment["status"] != "completed":
                raise OrchestrationError(
                    f"Search experiment no longer matches manifest: {reference['id']}"
                )
            loaded_experiments[experiment["id"]] = experiment
            metrics = experiment.get("metrics") or {}
            if metrics.get("model_identity") != self._model_identity(
                experiment["model_path"], require_weights=True
            ):
                raise OrchestrationError(
                    f"Search model identity has changed: {reference['id']}"
                )
            if metrics.get("evaluation_identity") != self._directory_identity(
                experiment["output_dir"], require_metrics=True
            ):
                raise OrchestrationError(
                    f"Search evaluation artifacts changed: {reference['id']}"
                )
            if metrics.get("stage1_identity") != self._model_identity(
                self.root / "models" / "shared-stage1", require_weights=True
            ):
                raise OrchestrationError(
                    f"Shared Stage1 identity changed after {reference['id']}"
                )
        selected = payload.get("selected_proxy_ids", [])
        if selected != [item.get("parent_id") for item in payload["finalists"]]:
            raise OrchestrationError("SEARCH_COMPLETE.json finalist lineage is inconsistent")
        if selected != payload.get("ranking", [])[: len(selected)]:
            raise OrchestrationError("SEARCH_COMPLETE.json finalists are not the ranked top candidates")
        actual_ranking = [
            item["id"]
            for item in self._rank(
                [loaded_experiments[experiment_id] for experiment_id in expected_proxy_ids]
            )
        ]
        if payload.get("ranking") != actual_ranking:
            raise OrchestrationError(
                "SEARCH_COMPLETE.json ranking does not match observed proxy metrics"
            )
        return payload

    def search(self) -> dict[str, Any]:
        if (self.root / "SEARCH_COMPLETE.json").is_file():
            manifest = self._load_search_manifest()
            return {
                "proxies": [item["id"] for item in manifest["proxies"]],
                "finalists": [item["id"] for item in manifest["finalists"]],
                "search_hash": manifest["search_hash"],
            }
        baseline = self.state.get_experiment("baseline")
        if baseline["status"] != "completed":
            raise OrchestrationError("A completed baseline is required before search")
        baseline_metrics = baseline.get("metrics") or {}
        if baseline_metrics.get("gate_pass") is not True:
            raise OrchestrationError("Baseline is missing required main/guard metrics")
        count = int(self.config.get("search.proxy_trials"))
        deterministic = self._deterministic_proposals()
        proxies: list[dict[str, Any]] = []
        used_hashes: set[str] = set()
        used_patches: list[dict[str, Any]] = []
        prior = baseline
        next_analysis: dict[str, Any] | None = None
        for index in range(1, count + 1):
            existing_id = f"proxy-{index:02d}"
            try:
                existing = self.state.get_experiment(existing_id)
            except KeyError:
                existing = None
            if existing is not None:
                try:
                    proposal = proposal_validator(
                        self.config.data["search"],
                        {source.name for source in self.catalog.sources},
                        disallowed_patch_hashes=used_hashes,
                        reject_patch=self._is_baseline_equivalent_patch,
                    )(existing["config"]["proposal"])
                except (DeepSeekError, KeyError, TypeError) as exc:
                    raise OrchestrationError(
                        f"Existing proposal for {existing_id} violates the current search policy"
                    ) from exc
            elif self.config.get("search.proposal_mode") == "deepseek":
                if next_analysis is None:
                    next_analysis = self._deepseek_round_analysis(
                        index - 1,
                        prior,
                        used_patches,
                        require_proposal=True,
                    )
                proposal = next_analysis["proposal"]
            else:
                try:
                    proposal = proposal_validator(
                        self.config.data["search"],
                        {source.name for source in self.catalog.sources},
                        disallowed_patch_hashes=used_hashes,
                        reject_patch=self._is_baseline_equivalent_patch,
                    )(
                        deterministic[(index - 1) % len(deterministic)]
                    )
                except DeepSeekError as exc:
                    raise OrchestrationError("Deterministic proposal violates search policy") from exc
            used_hashes.add(hash_object(proposal["patch"]))
            used_patches.append(copy.deepcopy(proposal["patch"]))
            experiment_config = {
                "role": "proxy",
                "encoder": "chronos2",
                "seed": 42,
                "max_steps": int(self.config.get("search.proxy_max_steps")),
                "proposal": proposal,
                "patch": proposal["patch"],
            }
            result = self._execute_train_eval(
                existing_id,
                "candidate",
                "proxy",
                experiment_config,
                False,
                "baseline",
                self.root / "models" / "shared-stage1",
                "stage2",
                self.config.get("evaluation.search_benchmarks"),
                self.config.get("evaluation.search_split"),
                baseline_metrics,
            )
            proxies.append(result)
            prior = result
            if (
                self.config.get("search.proposal_mode") == "deepseek"
                and result["status"] == "completed"
            ):
                next_analysis = self._deepseek_round_analysis(
                    index,
                    result,
                    used_patches,
                    require_proposal=index < count,
                )
        completed = [item for item in proxies if item["status"] == "completed"]
        finalist_count = int(self.config.get("search.full_finalists"))
        if len(completed) < finalist_count:
            raise OrchestrationError("Not enough completed proxy experiments for finalist selection")
        ranked = self._rank(completed)
        finalists = []
        for rank, proxy in enumerate(ranked[:finalist_count], 1):
            proxy_config = {
                key: value for key, value in proxy["config"].items() if key != "_resolved"
            }
            experiment_config = {
                **proxy_config,
                "role": "full-finalist",
                "max_steps": 0,
                "selected_from": proxy["id"],
            }
            finalist = self._execute_train_eval(
                f"full-{rank:02d}-{proxy['id']}",
                "candidate",
                "full",
                experiment_config,
                False,
                proxy["id"],
                self.root / "models" / "shared-stage1",
                "stage2",
                self.config.get("evaluation.final_benchmarks"),
                self.config.get("evaluation.search_split"),
                baseline_metrics,
            )
            finalists.append(finalist)
        manifest = self._write_search_manifest(baseline, proxies, ranked, finalists)
        self._export()
        return {
            "proxies": [item["id"] for item in proxies],
            "finalists": [item["id"] for item in finalists],
            "search_hash": manifest["search_hash"],
        }

    def resume(self) -> dict[str, Any]:
        if (self.root / "FROZEN.json").is_file():
            return {"status": "final-eval", **self.final_eval()}
        experiments = self.state.list_experiments()
        if not any(item["id"] == "baseline" and item["status"] == "completed" for item in experiments):
            baseline = self.baseline()
            if baseline["status"] != "completed":
                return {"status": "pending-dry-run", "next": "baseline"}
        return self.search()

    def _model_identity(
        self, model_path: str | Path, *, require_weights: bool = False
    ) -> str:
        model = Path(model_path).resolve()
        if not model.is_dir():
            if require_weights:
                raise OrchestrationError(f"Model directory is missing: {model}")
            return hash_object({"model_path": str(model), "files": {}})
        files = {}
        weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
        weight_count = 0
        for path in sorted(model.rglob("*")):
            if path.is_symlink() and path.is_dir():
                raise OrchestrationError(
                    f"Nested directory symlinks are not allowed in model artifacts: {path}"
                )
            if not path.is_file():
                continue
            relative = path.relative_to(model)
            if any(part in {".cache", "logs", "runs", "__pycache__"} for part in relative.parts):
                continue
            if path.suffix.lower() in {".tmp", ".log", ".lock", ".pyc"}:
                continue
            files[relative.as_posix()] = sha256_file(path)
            if path.suffix.lower() in weight_suffixes:
                weight_count += 1
        if require_weights and weight_count == 0:
            raise OrchestrationError(f"Model directory contains no weight files: {model}")
        return hash_object(
            {"model_path": str(model), "files": files, "weight_file_count": weight_count}
        )

    @staticmethod
    def _directory_identity(
        directory: str | Path, *, require_metrics: bool = False
    ) -> str:
        root = Path(directory).resolve()
        if not root.is_dir():
            raise OrchestrationError(f"Artifact directory is missing: {root}")
        if require_metrics and not (root / "metrics.json").is_file():
            raise OrchestrationError(f"Evaluation metrics.json is missing: {root}")
        files: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part.lower() in {"logs", "tmp", "temp", "__pycache__"} for part in relative.parts):
                continue
            if path.suffix.lower() in {".log", ".tmp", ".temp", ".partial", ".lock", ".pyc"}:
                continue
            files[relative.as_posix()] = sha256_file(path)
        if not files:
            raise OrchestrationError(f"Artifact directory is empty: {root}")
        return hash_object({"root": str(root), "files": files})

    def freeze(self) -> dict[str, Any]:
        baseline = self.state.get_experiment("baseline")
        if baseline["status"] != "completed":
            raise OrchestrationError("Cannot freeze without a completed baseline")
        if (baseline.get("metrics") or {}).get("gate_pass") is not True:
            raise OrchestrationError("Cannot freeze because baseline guard metrics are incomplete")
        search_manifest = self._load_search_manifest()
        completed_full = [
            self.state.get_experiment(reference["id"])
            for reference in search_manifest["finalists"]
        ]
        full = [
            item
            for item in completed_full
            if (item.get("metrics") or {}).get("gate_pass") is True
        ]
        eligible = [baseline, *full]
        champion = self._rank(eligible)[0]
        payload = {
            "schema_version": "chatts-autoresearch-freeze-v1",
            "config_hash": self.config.fingerprint,
            "dataset_hash": champion["dataset_hash"],
            "search_protocol_hash": champion["protocol_hash"],
            "search_manifest_hash": search_manifest["search_hash"],
            "baseline": {
                "experiment_id": baseline["id"],
                "model_path": baseline["model_path"],
                "dataset_hash": baseline["dataset_hash"],
                "model_identity": self._model_identity(
                    baseline["model_path"], require_weights=True
                ),
            },
            "champion": {
                "experiment_id": champion["id"],
                "model_path": champion["model_path"],
                "dataset_hash": champion["dataset_hash"],
                "model_identity": self._model_identity(
                    champion["model_path"], require_weights=True
                ),
                "primary_score": champion["metrics"].get("primary_score"),
                "gate_pass": champion["metrics"].get("gate_pass"),
            },
            "frozen_at": _utc_now(),
        }
        payload["freeze_hash"] = hash_object(payload)
        path = self.root / "FROZEN.json"
        if path.exists():
            existing = self._load_freeze()
            comparable = {key: value for key, value in existing.items() if key not in {"frozen_at", "freeze_hash"}}
            intended = {key: value for key, value in payload.items() if key not in {"frozen_at", "freeze_hash"}}
            if comparable != intended:
                raise OrchestrationError("FROZEN.json already exists with a different champion/protocol")
            return existing
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        self.state.metadata_put("freeze", payload)
        self._export()
        return payload

    def _load_freeze(self) -> dict[str, Any]:
        path = self.root / "FROZEN.json"
        if not path.is_file():
            raise OrchestrationError("Formal final evaluation is locked until freeze creates FROZEN.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OrchestrationError("FROZEN.json is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise OrchestrationError("FROZEN.json must contain one JSON object")
        if payload.get("schema_version") != "chatts-autoresearch-freeze-v1":
            raise OrchestrationError("Unsupported or missing FROZEN.json schema_version")
        stored_freeze_hash = payload.get("freeze_hash")
        unsigned = {key: value for key, value in payload.items() if key != "freeze_hash"}
        if not isinstance(stored_freeze_hash, str) or stored_freeze_hash != hash_object(unsigned):
            raise OrchestrationError("FROZEN.json freeze_hash does not match its contents")
        if payload.get("config_hash") != self.config.fingerprint:
            raise OrchestrationError("Current configuration does not match FROZEN.json")
        search_manifest = self._load_search_manifest()
        if payload.get("search_manifest_hash") != search_manifest.get("search_hash"):
            raise OrchestrationError("FROZEN.json does not match SEARCH_COMPLETE.json")
        for role in ("baseline", "champion"):
            if not isinstance(payload.get(role), dict):
                raise OrchestrationError(f"FROZEN.json is missing {role} metadata")
            required = {"experiment_id", "model_path", "dataset_hash", "model_identity"}
            if not required.issubset(payload[role]):
                raise OrchestrationError(f"FROZEN.json has incomplete {role} metadata")
            if self._model_identity(
                payload[role]["model_path"], require_weights=True
            ) != payload[role]["model_identity"]:
                raise OrchestrationError(f"Frozen {role} model identity has changed")
        return payload

    def _execute_eval_only(
        self, role: str, model: dict[str, Any], baseline_metrics: dict[str, Any] | None
    ) -> dict[str, Any]:
        experiment_id = _safe_id(f"final-{role}-{model['experiment_id']}")
        split = self.config.get("evaluation.final_split")
        benchmarks = self.config.get("evaluation.final_benchmarks")
        protocol_hash = self._protocol_hash(split, benchmarks)
        experiment_config = {"role": f"formal-{role}", "frozen_experiment": model["experiment_id"]}
        resolved_config = {
            **experiment_config,
            "_resolved": {
                "evaluation": copy.deepcopy(self.config.data["evaluation"]),
                "gates": copy.deepcopy(self.config.data["gates"]),
                "runtime": {"seed": 42, "gpu_ids": self.config.get("runtime.gpu_ids")},
                "eval_script_sha256": sha256_file(self.config.require("paths.eval_script")),
                "interface_files": self._interface_file_hashes(),
                "autoresearch_files": self._autoresearch_file_hashes(),
                "model_identity": self._model_identity(model["model_path"]),
                "dataset_hash": model["dataset_hash"],
                "protocol_hash": protocol_hash,
            },
        }
        config_hash = hash_object(resolved_config)
        eval_dir = self.root / "evaluations" / experiment_id / split
        payload = {
            "id": experiment_id,
            "kind": "final_eval",
            "phase": "final-test",
            "parent_id": model["experiment_id"],
            "config_hash": config_hash,
            "dataset_hash": model["dataset_hash"],
            "protocol_hash": protocol_hash,
            "config_json": resolved_config,
            "output_dir": str(eval_dir),
        }
        existing = self.state.create_experiment(payload)
        self._assert_existing_matches(existing, payload)
        env = self._evaluation_env(
            experiment_id,
            Path(model["model_path"]),
            eval_dir,
            split,
            benchmarks,
            protocol_hash,
            existing["status"] in {"failed", "running"},
        )
        argv = ["bash", str(self.config.require("paths.eval_script"))]
        command = {"evaluate": {"argv": argv, "cwd": self.config.require("paths.eval_project"), "env": env}}
        command_hash = self._semantic_command_hash(command)
        (self.root / "commands" / f"{experiment_id}.json").write_text(
            json.dumps({"command_hash": command_hash, **command}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if existing["status"] == "completed":
            if existing.get("command_hash") != command_hash:
                raise OrchestrationError(
                    f"Completed experiment {experiment_id} command hash no longer matches"
                )
            metrics = existing.get("metrics") or {}
            if metrics.get("evaluation_identity") != self._directory_identity(
                eval_dir, require_metrics=True
            ):
                raise OrchestrationError(
                    f"Completed experiment {experiment_id} evaluation artifacts changed"
                )
            return existing
        if self.runner.dry_run:
            return existing
        self.state.mark_running(experiment_id, command_hash, command)
        try:
            result = self.runner.run(
                argv,
                self.config.require("paths.eval_project"),
                env,
                self.root / "logs" / f"{experiment_id}.eval.{split}.log",
            )
            metrics = apply_gates(
                load_metrics(eval_dir),
                baseline_metrics,
                self.config.data["gates"],
                require_guards=True,
            )
            metrics["gpu_hours"] = result.duration_seconds * 8 / 3600.0
            metrics["eval_seconds"] = result.duration_seconds
            metrics["badcase_summary"] = extract_badcases(
                eval_dir,
                self.root / "badcases" / f"{experiment_id}.{split}.jsonl",
                {"tsrbench": self.root / "eval_views" / split / "tsrbench"},
            )
            metrics["evaluation_identity"] = self._directory_identity(
                eval_dir, require_metrics=True
            )
            self.state.mark_completed(experiment_id, metrics, model["model_path"])
        except Exception as exc:
            self.state.mark_failed(experiment_id, str(exc))
            self._export()
            raise
        self._export()
        return self.state.get_experiment(experiment_id)

    def final_eval(self) -> dict[str, Any]:
        frozen = self._load_freeze()
        baseline = self._execute_eval_only("baseline", frozen["baseline"], None)
        if frozen["champion"]["experiment_id"] == frozen["baseline"]["experiment_id"]:
            champion = baseline
        else:
            champion = self._execute_eval_only("champion", frozen["champion"], baseline["metrics"])
        return {"baseline": baseline["id"], "champion": champion["id"]}

    def report(self) -> Path:
        freeze_path = self.root / "FROZEN.json"
        freeze = self._load_freeze() if freeze_path.is_file() else None
        path = generate_report(self.state, self.root, freeze)
        self._export()
        return path
