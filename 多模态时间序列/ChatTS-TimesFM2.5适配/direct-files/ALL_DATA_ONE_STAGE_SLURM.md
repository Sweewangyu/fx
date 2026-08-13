# 全量数据单阶段训练

该实验把 Dataset Studio 当前激活快照中的 Stage1 和 Stage2 数据集 key 合并，使用：

- Chronos-2 时间序列编码器；
- 完整 LLM + projector 全参数训练；
- `concat`，不传 `interleave_probs`；
- 单阶段、`1 epoch`；
- 8 张 H100，默认 global batch 为 `2 × 32 × 8 = 512`。

训练不会复制或改写数据快照。Stage1/Stage2 如果包含同一 dataset key，只加入一次；不同 key
中的重复样本不会在训练时进行内容去重。

提交前确认：

```bash
cd /data/hpc/home/yu.wang17/ChatTS-Training
mkdir -p log

# 确认镜像路径
ls -lh /data/hpc/home/yu.wang17/chatts_v1.sif

# 确认当前激活的数据版本
cat data/studio_versions/active.json

# 按实际路径修改 sbatch 文件中的 MODEL_PATH、CHRONOS2_MODEL_PATH、OUTPUT_DIR
sbatch slurm/run_chronos2_all_data_one_stage.sbatch
```

查看任务：

```bash
squeue -u yu.wang17
tail -f log/<job-id>-chatts-all1.out
```

实际训练入口是：

```text
scripts/full/run_chronos2_all_data_one_stage.sh
```
