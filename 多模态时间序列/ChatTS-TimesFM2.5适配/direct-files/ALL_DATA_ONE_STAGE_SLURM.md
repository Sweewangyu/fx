# 六个全量数据集：单阶段训练

这个实验不读取 Dataset Studio 的 `datav3/stage1` 或 `datav3/stage2` 筛选结果，
而是直接读取以下六个完整 source 文件：

```text
/share/airesearch/data/finiverse/traindata/merged_labels/annotated/
├── chatts_align_256.jsonl
├── chatts_ift.jsonl
├── chatts_align_random.jsonl
├── chatts_sft.jsonl
├── time_mqa.jsonl
└── tsaqa.jsonl
```

`annotated` 文件是在完整训练样本上增加 quality/difficulty/taxonomy 审计字段；训练只读取
`input`、`output` 和 `timeseries`，不会根据标注字段过滤样本。

## 实验设置

- ChatTS + Time-MQA + TSQA 六个 source；
- 单阶段 `concat`；
- 不传 `interleave_probs`；
- `1 epoch`；
- `val_size=0`，不划出 5% validation；
- `eval_strategy=no`；
- `cutoff_len=10000`，避免旧配置中的 `2048` 让预处理器直接丢弃大量长样本；
- 保存完整一轮结束时的最终模型；
- 完整 LLM + projector 全参数训练，Chronos-2 backbone 冻结；
- ZeRO-3，8 GPU，global batch 为 `1 × 64 × 8 = 512`。

脚本不会复制大文件，只在任务自己的 `/tmp` 目录创建一个很小的
`dataset_info.json`，其中的绝对路径直接指向上述六个 JSONL。

训练启动前会逐文件统计行数并打印总数。默认要求至少 400,000 条：如果仍然只找到约
180,000 条，会在 DeepSpeed 启动前失败，不会消耗 GPU 进行错误实验。

## 提交

```bash
cd /data/hpc/home/yu.wang17/ChatTS-Training
git pull
mkdir -p log

ls -lh /data/hpc/home/yu.wang17/chatts_v1.sif
ls -lh /data/hpc/home/yu.wang17/chronos2
ls -lh /share/airesearch/data/finiverse/traindata/merged_labels/annotated/{chatts_align_256,chatts_ift,chatts_align_random,chatts_sft,time_mqa,tsaqa}.jsonl

sbatch slurm/run_chronos2_all_data_one_stage.sbatch
```

如果 Chronos-2 的宿主机路径不同：

```bash
sbatch \
  --export=ALL,HOST_CHRONOS2_PATH=/真实路径/chronos2 \
  slurm/run_chronos2_all_data_one_stage.sbatch
```

## 查看日志

```bash
squeue -u yu.wang17
tail -f log/<job-id>-chatts-all1.out
```

日志在 DeepSpeed 之前应显示类似：

```text
Raw total rows:   5xxxxx
   chatts_align_256          ...
   chatts_ift                ...
   chatts_align_random       ...
   chatts_sft                ...
   time_mqa                  ...
   tsaqa                     ...
Validation:       disabled; every valid row participates in training
```

Trainer 的 `Num examples` 应接近这里打印的 `Raw total rows`。如果两者仍有差距，查看日志中
的 `Dropped invalid example`、`Dropped lengthy example` 和 `[drop mismatch]`；那表示样本在
ChatTS tokenizer/时序占位符校验阶段无效，而不是再次发生 quality/difficulty 筛选。

## 容器和缓存

必要挂载：

```text
ChatTS-Training -> /workspace/ChatTS-Training
Chronos-2       -> /workspace/chronos2
/share          -> /share
```

Slurm 使用集群上已经验证过的：

```text
singularity run --cleanenv
LD_PRELOAD=/.singularity.d/libs/libcuda.so.1
HOME=/tmp/chatts_all1_<job-id>/home
TRITON_CACHE_DIR=/tmp/chatts_all1_<job-id>/triton
```

因此不会读取宿主机旧的 `~/.triton/cache/cuda_utils.so`。
