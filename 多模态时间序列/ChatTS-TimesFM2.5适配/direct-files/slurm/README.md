# Dataset Studio Slurm launcher

`run_chatts_studio_pipeline.sbatch` is the trusted Slurm execution contract for
ChatTS Dataset Studio. It is separate from
`run_chronos2_all_data_one_stage.sbatch`, which remains a standalone, fixed
all-data experiment.

The Studio-compatible file contains this API marker:

```text
# CHATTS_STUDIO_SBATCH_API=1
```

It accepts exactly two positional arguments:

```bash
sbatch slurm/run_chatts_studio_pipeline.sbatch \
  /absolute/shared/path/to/<job-id>.resolved.yaml \
  <studio-job-id>
```

Dataset Studio normally supplies both arguments. The resolved YAML must be on a
filesystem visible from the login and compute nodes. Before Singularity starts,
the launcher verifies:

- the Studio job ID equals `pipeline.trial_id`;
- `pipeline.trial_config_hash` matches the complete frozen configuration;
- the dataset snapshot and training recipe hashes are valid SHA256 values;
- the output root ends in `recipe-<training_recipe_hash[:16]>` and the final
  model is strictly inside it;
- all Stage1 and Stage2 values are present and only whitelisted fields are used;
- the complete evaluation protocol is present, its protocol hash is valid, and no
  unknown evaluation field is silently ignored;
- the evaluation model exactly equals the resolved final training model, with
  `STAGE1_COMPLETE.json` for Stage1-only or `TRAINING_COMPLETE.json` for full;
- the selected mode is `full` or `stage1`.

For `stage1`, the standard
`scripts/full/run_chronos2_best_two_stage.sh` entrypoint receives
`PIPELINE_MODE=stage1` and `STAGE1_OUT=FINAL_MODEL_PATH`. The finalized Stage1
weights, `best_model_manifest.json`, and `STAGE1_COMPLETE.json` are therefore
retained directly in the recipe-specific final model directory. For `full`, the
standard two-stage behavior is unchanged. Both modes then run the selected
benchmarks in a second `srun` step inside the same sbatch allocation. A training
or marker/weight validation failure prevents that evaluation step from starting.
The evaluation output directory is scoped by
`protocol-<evaluation.protocol_hash[:16]>`, so different benchmark protocols do
not overwrite each other. The evaluation container's Hugging Face offline flags
strictly follow the frozen `pipeline.offline` value.

At compute-node startup the launcher takes a non-blocking shared-filesystem
`flock` on a sibling of the recipe output directory. A second job targeting the
same recipe fails clearly before its first `srun`; jobs for different recipe
hashes remain parallel. The lock is held by file descriptor 9 until the sbatch
process exits.

## Standalone evaluation allocation

`run_chatts_studio_evaluation.sbatch` is the separate evaluation-only contract.
It carries both the generic marker and
`# CHATTS_STUDIO_EVALUATION_SBATCH_API=1`, accepts the same frozen-config/job-ID
arguments, and never invokes a trainer. Dataset Studio may submit one job per
model in a batch; Slurm provides the cluster queue while each model/protocol
output also has a non-blocking shared-filesystem lock against duplicate writers.

The launcher validates the frozen YAML with the shared ChatTS
`scripts/load_studio_evaluation_config.py`, requires the selected external model
to contain `config.json` and non-empty weights during real preflight, and runs
`run_all_chatts_benchmarks.sh` with training-marker checks disabled. By default
it loads `ragas.sif` beside `CHATTS_SIF_IMAGE`; set the trusted
`integration.slurm_evaluation_sif_image` only when the image lives elsewhere.

```bash
sbatch slurm/run_chatts_studio_evaluation.sbatch \
  /absolute/shared/path/to/<evaluation-job-id>.yaml \
  <evaluation-job-id>
```

## Infrastructure settings

The browser does not control infrastructure paths. Cluster administrators set
them in Dataset Studio's trusted `server.yaml`; they are frozen into the resolved
YAML and covered by `trial_config_hash`. Submission-environment variables remain
fallbacks for hand-written/legacy jobs.

| Variable | Default |
| --- | --- |
| `CHATTS_TRAINING_DIR` | `${SLURM_SUBMIT_DIR:-$PWD}` (never the Slurm spool copy's `BASH_SOURCE`) |
| `CHATTS_HOST_ROOT` | `$HOME` |
| `CHATTS_SIF_IMAGE` | `$CHATTS_HOST_ROOT/chatts_v1.sif` |
| `CHATTS_EVALUATION_DIR` | resolved `integration.slurm_evaluation_root`, falling back to `integration.evaluation_root` |
| `CHATTS_EVAL_SIF_IMAGE` | resolved `integration.slurm_evaluation_sif_image`, otherwise `ragas.sif` beside `CHATTS_SIF_IMAGE` |
| `CHATTS_HOST_CHRONOS2_PATH` | `$CHATTS_HOST_ROOT/chronos2` |
| `CHATTS_CONTAINER_TRAINING_DIR` | `/workspace/ChatTS-Training` |
| `CHATTS_CONTAINER_CHRONOS2_PATH` | resolved YAML's Chronos-2 path |
| `CHATTS_SHARED_HOST_PATH` | `/share` |
| `CHATTS_SHARED_CONTAINER_PATH` | `/share` |
| `CHATTS_HOST_TSRBENCH_PATH` | host path corresponding to the resolved TSRBench container path |
| `CHATTS_HOST_TINYBENCH_PATH` | host path corresponding to the resolved tinyBenchmarks container path |
| `CHATTS_HOST_TS_HAYSTACK_PATH` | host path corresponding to the resolved TS-Haystack container path |
| `CHATTS_HOST_TIMESERIESEXAM_PATH` | host path corresponding to the resolved TimeSeriesExam container path |
| `CHATTS_JOB_TMP_ROOT` | `/tmp` |
| `CHATTS_HOST_PYTHON_BIN` | `python3` |
| `CHATTS_SRUN_BIN` | `srun` |
| `CHATTS_SINGULARITY_BIN` | `singularity` |
| `CHATTS_FLOCK_BIN` | `flock` |

Example for the current cluster layout:

```bash
export CHATTS_TRAINING_DIR=/data/hpc/home/yu.wang17/ChatTS-Training
export CHATTS_SIF_IMAGE=/data/hpc/home/yu.wang17/chatts_v1.sif
export CHATTS_HOST_CHRONOS2_PATH=/data/hpc/home/yu.wang17/chronos2
export CHATTS_EVALUATION_DIR=/data/hpc/home/yu.wang17/ChatTS
export CHATTS_HOST_TS_HAYSTACK_PATH=/data/hpc/home/yu.wang17/TS-Haystack
export CHATTS_HOST_TIMESERIESEXAM_PATH=/data/hpc/home/yu.wang17/TimeSeriesExam
export CHATTS_SHARED_HOST_PATH=/share
export CHATTS_SHARED_CONTAINER_PATH=/share

sbatch \
  --output=/share/airesearch/data/finiverse/traindata/chatts-studio-state/pipeline/logs/%j.out \
  --error=/share/airesearch/data/finiverse/traindata/chatts-studio-state/pipeline/logs/%j.err \
  slurm/run_chatts_studio_pipeline.sbatch \
  /share/airesearch/data/finiverse/traindata/chatts-studio-state/pipeline/configs/<job-id>.yaml \
  <job-id>
```

Command-line `--output` and `--error` override the fallback `#SBATCH` log paths,
allowing Dataset Studio to show the compute-node logs. The destination directory
must exist before submission and be shared with compute nodes.

The launcher uses array-based command construction and never evaluates YAML as
shell code. `load_studio_pipeline_config.py` emits only fixed environment names;
the Bash launcher validates that allowlist again before passing assignments to
the training and evaluation Singularity steps. Studio preflight submits one
real allocation with a fixed 10-minute wall-time cap so these checks also run on the compute node. Its frozen
`pipeline.preflight_only=true` makes both standard runners execute only their
non-mutating `--preflight` paths; no training, inference, model, or evaluation
artifact is produced.

## Checks

```bash
bash -n slurm/run_chatts_studio_pipeline.sbatch
python -m pytest -q tests/pipeline/test_studio_slurm_contract.py \
  tests/pipeline/test_standalone_evaluation_slurm.py
ruff check scripts/slurm/load_studio_pipeline_config.py \
  tests/pipeline/test_studio_slurm_contract.py
```
