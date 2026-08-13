# 全量数据单阶段训练

该实验从 Dataset Studio 当前激活快照选择数据集，并在同一个 Stage 中训练。默认选择：

- ChatTS Stage1：`chatts_align_256`、`chatts_ift`；
- ChatTS Stage2：`chatts_align_random`、`chatts_sft`；
- Stage2：`time_mqa`、`tsaqa`。

使用：

- Chronos-2 时间序列编码器；
- 完整 LLM + projector 全参数训练；
- `concat`，不传 `interleave_probs`；
- 单阶段、`1 epoch`；
- 8 张 H100，默认 global batch 为 `2 × 32 × 8 = 512`。

训练不会复制或改写数据快照。修改 sbatch 中的 `SELECT_DATASETS` 可以选择：

```bash
# ChatTS + Time-MQA + TSQA（默认）
SELECT_DATASETS=chatts,time-mqa,tsqa

# 只用 ChatTS
SELECT_DATASETS=chatts

# 只用 Time-MQA 和 TSQA
SELECT_DATASETS=time-mqa,tsqa

# 使用当前快照导出的所有 Stage1/Stage2 dataset key
SELECT_DATASETS=all
```

短名称会自动映射为当前 `datavN` 中的真实 LLaMAFactory dataset key。

提交前确认：

```bash
cd /data/hpc/home/yu.wang17/ChatTS-Training
mkdir -p log

# 确认镜像路径
ls -lh /data/hpc/home/yu.wang17/chatts_v1.sif

# 确认本地 Chronos-2 权重；若路径不同，修改 sbatch 中的 HOST_CHRONOS2_PATH
ls -lh /data/hpc/home/yu.wang17/chronos2

# 确认当前激活的数据版本
cat data/studio_versions/active.json

# 按实际路径修改 sbatch 文件中的 MODEL_PATH、CHRONOS2_MODEL_PATH、OUTPUT_DIR
sbatch slurm/run_chronos2_all_data_one_stage.sbatch
```

Slurm 脚本会在启动 Singularity 前检查当前激活快照必须同时存在：

```text
manifest.json
dataset_info.json
training.env
```

训练数据、基础模型和输出均在 `/share`，因此容器命令显式使用
`--bind /share:/share`。Chronos-2 另行从宿主机
`/data/hpc/home/yu.wang17/chronos2` 绑定到容器的 `/workspace/chronos2`。
它不会挂载其他代码仓库。

查看任务：

```bash
squeue -u yu.wang17
tail -f log/<job-id>-chatts-all1.out
```

实际训练入口是：

```text
scripts/full/run_chronos2_all_data_one_stage.sh
```
