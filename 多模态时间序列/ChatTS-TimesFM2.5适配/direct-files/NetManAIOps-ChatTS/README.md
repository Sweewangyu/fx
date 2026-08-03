# NetManAIOps/ChatTS 四后端 vLLM 适配

这两个文件用于修复外部时间序列 backbone checkpoint 在
NetManAIOps/ChatTS vLLM 评测中的权重加载错误，同时保持原始 ChatTS MLP-Patch
checkpoint 兼容：

- `chatts/vllm/chatts_vllm.py`：支持原始 MLP-Patch、TimesFM 2.5、Chronos-2 和 Zeus。
- `chatts/vllm/zeus_modeling.py`：Zeus 官方 checkpoint 兼容的 eager-attention 结构。

适配基线为 NetManAIOps/ChatTS `a16ca1a`。用户报错栈中的原文件行号与该基线一致。

文件 SHA-256：

- `chatts_vllm.py`：`d573f0178f27752c350650568d9bbdf882470edcfb56e33e841bbe2cff7eb0f9`
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

## `config.json` 契约

| `ts_encoder_type` | patch size | backbone 路径字段 |
|---|---:|---|
| 缺省、`native`、`mlp` | checkpoint 原值 | 无 |
| `timesfm2_5` | 32 | `timesfm_model_name_or_path` |
| `chronos2` | 16 | `chronos2_model_name_or_path` |
| `zeus` | 32 | `zeus_model_name_or_path` |

训练补丁保存 checkpoint 时会自动写入这些字段。若字段中保存的是训练机本地路径，
评测机不存在该路径，就把它改成对应 Hugging Face ID 或评测机本地模型目录。

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
