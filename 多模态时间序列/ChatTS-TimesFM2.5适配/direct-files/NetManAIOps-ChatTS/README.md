# NetManAIOps/ChatTS 四后端 vLLM 适配

这两个文件用于修复外部时间序列 backbone checkpoint 在
NetManAIOps/ChatTS vLLM 评测中的权重加载错误，同时保持原始 ChatTS MLP-Patch
checkpoint 兼容：

- `chatts/vllm/chatts_vllm.py`：支持原始 MLP-Patch、TimesFM 2.5、Chronos-2 和 Zeus。
- `chatts/vllm/zeus_modeling.py`：Zeus 官方 checkpoint 兼容的 eager-attention 结构。
- `scripts/run_chatts_no_ragas_batch.sh`：用户批处理评测脚本的可选编码器覆盖版。

适配基线为 NetManAIOps/ChatTS `a16ca1a`。用户报错栈中的原文件行号与该基线一致。

文件 SHA-256：

- `chatts_vllm.py`：`c04cd7e854e96c3b8c9559a41c5de8be3c32b1a2b8009cc767e106edfbeadd94`
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
默认不传环境覆盖，由推理代码逐个读取 checkpoint 的 `config.json`：

```bash
bash scripts/run_chatts_no_ragas_batch.sh --infer-only
```

对于旧 checkpoint，如果权重已经是 TimesFM projector，但
`config.json` 没有 `ts_encoder_type` 和 backbone 路径字段，可以仅对这次命令指定：

```bash
TS_ENCODER_TYPE=timesfm2_5 \
bash scripts/run_chatts_no_ragas_batch.sh --infer-only
```

不需要在 shell 中先执行永久的 `export`。脚本会把它映射为
`CHATTS_TS_ENCODER_TYPE`，并传给 Python 父进程与所有 vLLM spawn worker。也可以直接用：

```bash
CHATTS_TS_ENCODER_TYPE=timesfm2_5 \
bash scripts/run_chatts_no_ragas_batch.sh --infer-only
```

如果 `SEARCH_DIR` 混合了多种编码器，不要设全局覆盖；应补齐每个
checkpoint 的 `config.json`。仅根据 projector 权重不能总是唯一识别：
Chronos-2 和 Zeus 的 projector 输入维度同为 768。

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
- 外部主干不在 ChatTS checkpoint 内；首次评测需能访问模型缓存或 Hugging Face。
- tensor parallel 只切分 Qwen，外部时序主干在每个 vLLM worker 中各复制一份。
- 不要过滤或跳过 `ts_encoder.projector.*`；这样会得到能运行但无效的评测结果。
