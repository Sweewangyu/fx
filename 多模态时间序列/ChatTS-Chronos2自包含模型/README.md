# ChatTS：把 Chronos-2 + 两层投影器 + Qwen3 保存成一个完整权重

## 结论

你当前的 Chronos-2 训练适配器把 backbone 放在 `_Chronos2Handle` 这个非
`nn.Module` 对象中。这样训练时可以节省检查点空间，但 `state_dict()` 看不到
Chronos-2，所以现有检查点实际上只有：

```text
ts_encoder.projector.*     已训练的两层投影器（8 个 tensor）
model.*                    Qwen3
lm_head.*                  Qwen3 输出层
```

本方案把官方 Chronos-2 注册成正常子模块，最终目录变为：

```text
ts_encoder.backbone.*      完整 Chronos-2
ts_encoder.projector.*     你已经训练好的两层投影器
model.*                    你已经训练/保存的 Qwen3
lm_head.*                  Qwen3 输出层
```

下游只传这一个模型目录。`modeling_qwen3_ts.py` 不读取
`chronos2_model_name_or_path`，不调用 Chronos 的 `from_pretrained()`，也不会再联网下载权重。

## 为什么这样改

本目录以 Hugging Face 上的
[`bytedance-research/ChatTS-8B`](https://huggingface.co/bytedance-research/ChatTS-8B/tree/main)
官方非权重目录为基线，而不是重新实现 ChatTS：

- `configuration_qwen3_ts.py`：只增加四个显式配置字段和必要校验；
- `modeling_qwen3_ts.py`：保留官方 Qwen3、时序占位符展开和 generation 代码，只增加
  `Chronos2TimeSeriesEmbedding`，并让构造函数根据 `ts_encoder_type` 选择 encoder；
- `processing_qwen3_ts.py`：与官方文件逐字一致，没有修改；
- tokenizer、special tokens、chat template、generation config：从你的 1.7B 训练检查点复制。

官方 8B 的 `config.json` 不能直接用于你的 Qwen3-1.7B。导出脚本以你的检查点
`config.json` 为主体，只补 Chronos-2 字段，因此不会把 8B 的层数或 hidden size 写进
1.7B 模型。完整来源信息见 [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md)。

## 一次性生成完整模型

把本目录放到服务器，例如：

```bash
cd /workspace/ChatTS/ChatTS-main

pip install safetensors "chronos-forecasting>=2.3.1"

python /path/to/ChatTS-Chronos2自包含模型/scripts/export_self_contained_chronos2.py \
  --chatts-checkpoint /share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-PR-onestep/onestep_chronos2_lr3e-5 \
  --chronos2-checkpoint /workspace/chronos2 \
  --output-dir /share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-Chronos2-complete
```

三个参数分别是：

1. 已有的 `Qwen3 + 两层 projector` 完整训练检查点；
2. 训练时使用的本地 Chronos-2 原始权重目录；
3. 一个不存在或为空的新目录。脚本为防止误覆盖，拒绝写入非空目录。

由于 Chronos-2 在你的训练中被冻结，合并原始 Chronos-2 tensor 是严格正确的；不需要
重训，也不应该用随机初始化的 Chronos-2 替代。

## 脚本会自动做什么

- 读取两边的 `.bin` 或 `.safetensors` 及其 shard index；
- 原样保留 Qwen3 与 `ts_encoder.projector.*`；
- 给 Chronos-2 tensor 加上 `ts_encoder.backbone.` 前缀；
- 输出统一的 safetensors shards 和新的 `model.safetensors.index.json`；
- 复制你 1.7B 检查点中的 tokenizer、chat template、generation config 等文件；
- 用 [hf_files](hf_files) 中基于官方 ChatTS 修改的三个 Python 文件覆盖旧文件；
- 更新 `config.json`，删除运行时外部 Chronos 路径；
- 生成 `weight_audit.json`。

如果源检查点混入 native MLP、缺少 projector 的 8 个 tensor、Chronos hidden size 不是
768，或输入目录本身已经包含 backbone，脚本会直接报错，不会悄悄生成错误模型。

## 正确的 `config.json` 关键字段

导出后应包含：

```json
{
  "architectures": ["Qwen3TSForCausalLM"],
  "model_type": "qwen3ts",
  "ts_encoder_type": "chronos2",
  "chronos2_embedded": true,
  "chronos2_config": {
    "architectures": ["Chronos2Model"],
    "d_model": 768,
    "chronos_config": {
      "context_length": 8192,
      "input_patch_size": 16,
      "input_patch_stride": 16
    }
  },
  "projector_config": {
    "input_hidden_size": 768,
    "activation": "gelu",
    "num_linear_layers": 2
  },
  "ts": {
    "num_features": 2,
    "patch_size": 16,
    "max_sequence_length": 8192
  }
}
```

不应再有：

```json
"chronos2_model_name_or_path": "/workspace/chronos2"
```

来源名称可以保存在 `chronos2_backbone_name` 中用于审计，但它不参与加载。

## 验证完整性

验证脚本只读取 safetensors header，不会把 1.7B 模型载入内存：

```bash
python /path/to/ChatTS-Chronos2自包含模型/scripts/verify_self_contained_checkpoint.py \
  /share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-Chronos2-complete
```

成功时最后会显示：

```text
STATUS: OK — Qwen3 + two-layer projector + Chronos-2 are in one checkpoint.
```

## 用 Transformers 做一次加载测试

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

path = "/share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-Chronos2-complete"
config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype="auto",
    device_map="auto",
)

print(config.ts_encoder_type)
print(type(model.ts_encoder.backbone).__name__)
print(type(model.ts_encoder.projector).__name__)
PY
```

预期输出：

```text
chronos2
Chronos2Model
ExternalTimeSeriesProjector
```

这里仍需安装 `chronos-forecasting>=2.3.1`，因为它提供 Chronos-2 的 Python
类定义；但 Chronos-2 tensor 已经位于你的模型目录，加载过程中不需要第二个权重目录，也不需要网络。

## 本目录不包含什么

这里没有修改 vLLM。模型目录通过 `architectures`、`ts_encoder_type`、内嵌配置和明确的
state-dict 前缀把结构完整描述出来；部署组只需针对这一份稳定格式实现对应加载逻辑。

