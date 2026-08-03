# NetManAIOps/ChatTS 四后端 vLLM 适配

这些文件用于修复外部时间序列 backbone checkpoint 在
NetManAIOps/ChatTS vLLM 评测中的权重加载错误，同时保持原始 ChatTS MLP-Patch
checkpoint 兼容：

- `chatts/vllm/chatts_vllm.py`：支持原始 MLP-Patch、TimesFM 2.5、Chronos-2 和 Zeus。
- `chatts/vllm/zeus_modeling.py`：Zeus 官方 checkpoint 兼容的 eager-attention 结构。
- `scripts/run_chatts_no_ragas_batch.sh`：逐 checkpoint 权重识别、评测和结果汇总脚本。

适配基线为 NetManAIOps/ChatTS `a16ca1a`。用户报错栈中的原文件行号与该基线一致。

文件 SHA-256：

- `chatts_vllm.py`：`de887ab3ea9ea8c8c5a5b84f85f0d239c9ad5c43c7937b87c98b79db501ada03`
- `zeus_modeling.py`：`d5850fb0d8d104f6d7d92580b74e48df5b0bf450536cdb662fff35fa66b5c271`

## 覆盖文件

```bash
cd /workspace/ChatTS/ChatTS-main
cp chatts/vllm/chatts_vllm.py chatts/vllm/chatts_vllm.py.bak
cp /path/to/NetManAIOps-ChatTS/chatts/vllm/chatts_vllm.py chatts/vllm/
cp /path/to/NetManAIOps-ChatTS/chatts/vllm/zeus_modeling.py chatts/vllm/
python -m py_compile chatts/vllm/chatts_vllm.py chatts/vllm/zeus_modeling.py
```

## 依赖

代码使用懒加载。评测原始 MLP-Patch 或 Zeus 时，不需要安装 TimesFM/Chronos；
只安装当前 checkpoint 对应的依赖：

```bash
pip install 'timesfm[torch]>=2.0.2'          # TimesFM 2.5
pip install 'chronos-forecasting==2.3.1'    # Chronos-2
```

Zeus 不需要 BasicTS 或 FlashAttention，但必须复制本目录的 `zeus_modeling.py`。

## 出现 “Using the native ChatTS MLP-Patch encoder”

如果 checkpoint 权重含 `ts_encoder.projector.*`，但启动日志显示：

```text
[ChatTS vLLM] Using the native ChatTS MLP-Patch encoder.
```

说明评测目录的 `config.json` 缺少外部编码器元数据，或仍是原始 ChatTS 配置。
更新本目录最新版 `chatts_vllm.py` 后，可先用环境变量明确指定本次评测架构：

```bash
# 三选一；原始 MLP 则使用 native
export CHATTS_TS_ENCODER_TYPE=timesfm2_5
# export CHATTS_TS_ENCODER_TYPE=chronos2
# export CHATTS_TS_ENCODER_TYPE=zeus

# 可选：评测机上的本地 backbone 路径；不设置就使用 config 或官方 Hugging Face ID
# export CHATTS_TIMESFM_MODEL_PATH=/models/timesfm-2.5-200m-pytorch
# export CHATTS_CHRONOS2_MODEL_PATH=/models/chronos-2
# export CHATTS_ZEUS_MODEL_PATH=/models/zeus
```

环境变量会同时修正模型端和 vLLM Processor 使用的 patch size：TimesFM/Zeus 为 32，
Chronos-2 为 16。它是进程级设置，因此一次评测进程只应用于一种 checkpoint 架构。

更推荐的永久修复是补齐 checkpoint 的 `config.json`。以 TimesFM 2.5 为例：

```json
{
  "ts_encoder_type": "timesfm2_5",
  "timesfm_model_name_or_path": "google/timesfm-2.5-200m-pytorch",
  "ts": {
    "patch_size": 32
  }
}
```

这里只展示需要确认的字段；不要用这段 JSON 覆盖整个配置文件。Chronos-2 对应
`chronos2`、`chronos2_model_name_or_path`、patch size 16；Zeus 对应 `zeus`、
`zeus_model_name_or_path`、patch size 32。

## `config.json` 契约

| `ts_encoder_type` | patch size | backbone 路径字段 |
|---|---:|---|
| 缺省、`native`、`mlp` | checkpoint 原值 | 无 |
| `timesfm2_5` | 32 | `timesfm_model_name_or_path` |
| `chronos2` | 16 | `chronos2_model_name_or_path` |
| `zeus` | 32 | `zeus_model_name_or_path` |

训练补丁保存 checkpoint 时会自动写入这些字段。若字段中保存的是训练机本地路径，
评测机不存在该路径，就把它改成对应 Hugging Face ID 或评测机本地模型目录。
若 `ts_encoder_type` 缺失，但只存在一个 backbone 路径字段，最新版推理代码也会自动推断。

## 批处理评测脚本怎么传参

[`scripts/run_chatts_no_ragas_batch.sh`](./scripts/run_chatts_no_ragas_batch.sh)
默认不传环境覆盖，会先读取每个 checkpoint 的配置，配置无元数据时再扫描
`ts_encoder` 权重名和 tensor shape：

```bash
bash scripts/run_chatts_no_ragas_batch.sh --infer-only
```

自动识别规则：

- `ts_encoder.mlp.*` 或 `ts_encoder.position_embedding.*` → 原始 ChatTS MLP。
- 1280 维 `ts_encoder.projector.*` → TimesFM 2.5。
- 768 维 projector + patch size 16 → Chronos-2。
- 768 维 projector + patch size 32 → Zeus。

Chronos-2 和 Zeus 的 projector 权重名与 shape 完全相同。如果连 patch size 也没保存，
仅根据权重在数学上无法区分，脚本会明确报错而不会猜。此时可仅对本次命令指定：

```bash
TS_ENCODER_TYPE=chronos2 \
bash scripts/run_chatts_no_ragas_batch.sh --infer-only
```

不需要在 shell 中先执行永久的 `export`。在最新批处理脚本中，这个值是
768 维 projector 的 Chronos-2/Zeus 歧义解决值，不再强制覆盖所有 checkpoint。
脚本会为每个模型先扫描权重：native MLP 和 1280 维 TimesFM 始终以各自权重为准，
然后把解析后的类型传给 Python 父进程与所有 vLLM spawn worker。也可以使用底层别名：

```bash
CHATTS_TS_ENCODER_TYPE=chronos2 \
bash scripts/run_chatts_no_ragas_batch.sh --infer-only
```

因此，`SEARCH_DIR` 可以混合 native MLP、TimesFM 和一种 768 维外部编码器。
如果同时混合 Chronos-2 和 Zeus，它们的权重 shape 相同，仍需要每个 checkpoint
保留 patch size/编码器元数据，或拆分为两次批处理。

## 离线加载 TimesFM 主干

ChatTS Stage 2 checkpoint 只保存训练过的 `ts_encoder.projector.*`。TimesFM 2.5
是冻结的 200M 外部主干，不会重复打包进每个网格实验 checkpoint。无网评测机需要
准备一份共享的 `model.safetensors`：

```bash
# 在能联网的机器上执行，再把整个目录复制到评测机
hf download google/timesfm-2.5-200m-pytorch \
  --local-dir /tmp/timesfm-2.5-200m-pytorch

# 在无网评测机上；最新批处理脚本会自动识别 TimesFM 类型
HF_HUB_OFFLINE=1 \
TIMESFM_MODEL_PATH=/workspace/timesf \
bash scripts/run_chatts_no_ragas_batch.sh --infer-only
```

`TIMESFM_MODEL_PATH` 可以指向包含 `model.safetensors` 的目录，也可直接指向该文件。
它会被脚本映射为 `CHATTS_TIMESFM_MODEL_PATH` 并传给所有 vLLM worker。

## 所有模型的汇总表

使用 `--all` 或 `--score-only` 跑完后，脚本会读取每个模型的 Dataset A/B
`result.json`，在终端打印总表，并默认生成：

```text
$SEARCH_DIR/logs/chatts_batch_summary.csv
$SEARCH_DIR/logs/chatts_batch_summary.md
```

CSV 是 UTF-8 with BOM，可直接用 Excel 打开。每行包含：Dataset A/B 的
categorical/numerical 分数、两个数据集的分项平均、四项指标的
`Macro Mean`、token 数和运行状态。表格默认按 `Macro Mean` 降序排名。

可以更改输出位置和文件名：

```bash
SUMMARY_DIR=/workspace/results \
SUMMARY_BASENAME=chronos2_grid \
bash scripts/run_chatts_no_ragas_batch.sh --score-only
```

`--infer-only` 不执行评分，因此不生成汇总表。如果某个模型本次运行失败，
脚本仍会汇总其他模型，并把该行标记为失败，避免把旧 `result.json` 误当成本次结果。
对已经评分完的现有结果，可以不重新推理或评分，只生成表格：

```bash
bash scripts/run_chatts_no_ragas_batch.sh --summary-only
```

## 正确启动标志

启动后应出现以下四种日志之一：

```text
[ChatTS vLLM] Using the native ChatTS MLP-Patch encoder.
[ChatTS vLLM] Using TimesFM 2.5 time-series encoder...
[ChatTS vLLM] Using Chronos-2 time-series encoder...
[ChatTS vLLM] Using Zeus time-series encoder...
```

外部 backbone checkpoint 还应正常加载 `ts_encoder.projector.*`，不再出现
`There is no module or parameter named 'ts_encoder.projector'`。

## 重要限制

- 四种后端都必须使用与训练时相同的 ChatTS Processor；不要在评测脚本外再次 z-score。
- TimesFM、Chronos-2、Zeus 主干均冻结并保持 FP32。
- 外部主干不在 ChatTS checkpoint 内；评测需能访问本地共享主干、模型缓存或 Hugging Face。
- tensor parallel 只切分 Qwen，外部时序主干在每个 vLLM worker 中各复制一份。
- 不要过滤或跳过 `ts_encoder.projector.*`；这样会得到能运行但无效的评测结果。
