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
- the selected mode is `full` or `stage1`.

For `stage1`, the standard
`scripts/full/run_chronos2_best_two_stage.sh` entrypoint receives
`PIPELINE_MODE=stage1` and `STAGE1_OUT=FINAL_MODEL_PATH`. The finalized Stage1
weights and `STAGE1_COMPLETE.json` are therefore retained directly in the
recipe-specific final model directory. For `full`, the standard two-stage
behavior is unchanged.

The Stage1 result is the validation-selected checkpoint, not the last training
step. The runner reloads and exports those weights, finalizes the inference
configuration, and writes both `best_model_manifest.json` and
`STAGE1_COMPLETE.json`. Repeating an identical recipe validates and reuses that
model; setting `pipeline.force_train: true` rebuilds only the current
recipe-specific Stage1 output.

## Infrastructure settings

The dashboard does not control infrastructure paths. Cluster administrators may
set the following environment variables at submission time, for example through
Slurm's trusted submission environment:

| Variable | Default |
| --- | --- |
| `CHATTS_TRAINING_DIR` | parent directory of this `slurm/` directory |
| `CHATTS_HOST_ROOT` | `$HOME` |
| `CHATTS_SIF_IMAGE` | `$CHATTS_HOST_ROOT/chatts_v1.sif` |
| `CHATTS_HOST_CHRONOS2_PATH` | `$CHATTS_HOST_ROOT/chronos2` |
| `CHATTS_CONTAINER_TRAINING_DIR` | `/workspace/ChatTS-Training` |
| `CHATTS_CONTAINER_CHRONOS2_PATH` | resolved YAML's Chronos-2 path |
| `CHATTS_SHARED_HOST_PATH` | `/share` |
| `CHATTS_SHARED_CONTAINER_PATH` | `/share` |
| `CHATTS_JOB_TMP_ROOT` | `/tmp` |
| `CHATTS_HOST_PYTHON_BIN` | `python3` |
| `CHATTS_SRUN_BIN` | `srun` |
| `CHATTS_SINGULARITY_BIN` | `singularity` |

Example for the current cluster layout:

```bash
export CHATTS_TRAINING_DIR=/data/hpc/home/yu.wang17/ChatTS-Training
export CHATTS_SIF_IMAGE=/data/hpc/home/yu.wang17/chatts_v1.sif
export CHATTS_HOST_CHRONOS2_PATH=/data/hpc/home/yu.wang17/chronos2
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
Singularity.

## Checks

```bash
bash -n slurm/run_chatts_studio_pipeline.sbatch
python -m pytest -q tests/pipeline/test_studio_slurm_contract.py
ruff check scripts/slurm/load_studio_pipeline_config.py \
  tests/pipeline/test_studio_slurm_contract.py
```
