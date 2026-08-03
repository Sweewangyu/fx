# ChatTS × TimesFM 2.5 / Chronos-2 / Zeus 适配实现

本目录提供一个可直接应用到
[xiezhe-24/ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training)
的完整补丁：保留 ChatTS 的数据格式、Processor、`<ts>...<ts/>` token 合并与两阶段训练流程，
将原生 MLP-Patch 时间序列编码器替换为冻结的时序基础模型和可训练的
TS-to-text projector。目前支持
[`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch)、
[`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) 和
[`GestaltCog/zeus`](https://huggingface.co/GestaltCog/zeus)。

## 架构

```text
ChatTS [value, valid_mask]
          │
          ├─ TimesFM 2.5: 32 点 patch → 20 层 backbone → 1280-d
          ├─ Chronos-2:   16 点 patch → encoder context tokens → 768-d
          └─ Zeus:        point-wise → U 型多尺度中心 32× → 768-d
          │
          ▼
Trainable LayerNorm → Linear → GELU → Linear → LayerNorm
          │
          ▼
Existing ChatTS token merge → Qwen LLM
```

三个编码器都继续返回 ChatTS 所需的 `(features, patch_cnt)`，无需修改 processor、数据格式、
`<ts>` token 合并或 LLM。TimesFM 2.5 与 Zeus 输出 `ceil(L / 32)` 个时序 token，
Chronos-2 输出 `ceil(L / 16)` 个。

## 目录内容

- [基础实现补丁](./0001-feat-add-frozen-TimesFM-2.5-encoder-for-ChatTS.patch)。
- [TS-Reasoner 训练配置修正补丁](./0002-fix-match-TS-Reasoner-two-stage-training-recipe.patch)。
- [ZEUS 与 Chronos-2 扩展补丁](./0003-feat-add-Zeus-and-Chronos-2-time-series-backbones.patch)。
- [单阶段混合训练补丁](./0004-feat-add-one-stage-mixed-TimesFM-training-recipe.patch)。
- [`direct-files/`](./direct-files/)：保持 ChatTS-Training 与 NetManAIOps/ChatTS
  相对路径的完整修改文件，可直接复制到服务器。
- 补丁基线：ChatTS-Training `bf30699`。
- 补丁实现提交：`9e24561`；训练配置修正提交：`f6abe7b`；多 backbone 提交：`c630197`；单阶段脚本提交：`4e0838e`。
- SHA-256：
  - `0001`：`60d3878a0f36e3e94b053894a0e64e10cbb4c9b6449b285869162e6c668a01b8`
  - `0002`：`1017284f4d1e94021eb10fa0089798f808628598c25dd0ed2a981b9a03e5425e`
  - `0003`：`a36216f285ed721df20050fb898eec23ab0b759983748650f2d005f24030d5c3`
  - `0004`：`46409109495226a7c6b90bea5a1c96437fe9124070925d783a63751296f831d1`

补丁包含：

1. TimesFM 2.5 编码器、官方 cumulative normalization 和变长 batch 处理。
2. `LayerNorm → Linear → GELU → Linear → LayerNorm` projector。
3. Stage 1 保存与 Stage 2 自动恢复 projector。
4. Full SFT 与 LoRA 的时间序列模块处理。
5. 两阶段训练脚本和单元测试。
6. Chronos-2 官方底层 `model.encode()` 接口，排除 REG/future token。
7. 与官方 167 个权重键/形状完全一致的 ZEUS eager-attention 实现，不依赖 BasicTS 或 FlashAttention。
8. `timesfm`、`chronos2` 和 `zeus` 可选依赖与 README 使用说明。
9. NetManAIOps/ChatTS 的 vLLM 推理适配：自动识别原始 MLP-Patch、TimesFM 2.5、
   Chronos-2 和 Zeus checkpoint，并加载 `ts_encoder.projector.*`。

## 直接复制修改文件

如果服务器上的 ChatTS-Training 基于 `bf30699`，下载本目录后可以直接覆盖：

```bash
rsync -av direct-files/ /path/to/ChatTS-Training/
```

直接文件包括：

- [`setup.py`](./direct-files/setup.py)
- [`dataset_info.ts_reasoner.json`](./direct-files/data/dataset_info.ts_reasoner.json)
- [`model_args.py`](./direct-files/src/llamafactory/hparams/model_args.py)
- [`finetuning_args.py`](./direct-files/src/llamafactory/hparams/finetuning_args.py)
- [`loader.py`](./direct-files/src/llamafactory/model/loader.py)
- [`timeseries.py`](./direct-files/src/llamafactory/model/model_utils/timeseries.py)
- [`timesfm2_5.py`](./direct-files/src/llamafactory/model/model_utils/timesfm2_5.py)
- [`chronos2.py`](./direct-files/src/llamafactory/model/model_utils/chronos2.py)
- [`zeus.py`](./direct-files/src/llamafactory/model/model_utils/zeus.py)
- [`zeus_modeling.py`](./direct-files/src/llamafactory/model/model_utils/zeus_modeling.py)
- [`timeseries_backbones.py`](./direct-files/src/llamafactory/model/model_utils/timeseries_backbones.py)
- [`test_timesfm2_5.py`](./direct-files/tests/model/test_timesfm2_5.py)
- [`test_external_ts_backbones.py`](./direct-files/tests/model/test_external_ts_backbones.py)
- [`train_timesfm2_5_stage1.sh`](./direct-files/scripts/full/train_timesfm2_5_stage1.sh)
- [`train_timesfm2_5_stage2.sh`](./direct-files/scripts/full/train_timesfm2_5_stage2.sh)
- [`train_timesfm2_5_one_stage.sh`](./direct-files/scripts/full/train_timesfm2_5_one_stage.sh)
- [`train_chronos2_stage1.sh`](./direct-files/scripts/full/train_chronos2_stage1.sh)
- [`train_chronos2_stage2.sh`](./direct-files/scripts/full/train_chronos2_stage2.sh)
- [`train_zeus_stage1.sh`](./direct-files/scripts/full/train_zeus_stage1.sh)
- [`train_zeus_stage2.sh`](./direct-files/scripts/full/train_zeus_stage2.sh)
- [vLLM 四后端版 `chatts_vllm.py`](./direct-files/NetManAIOps-ChatTS/chatts/vllm/chatts_vllm.py)
- [Zeus eager 模型结构 `zeus_modeling.py`](./direct-files/NetManAIOps-ChatTS/chatts/vllm/zeus_modeling.py)
- [Dataset A/B 批处理评测脚本](./direct-files/NetManAIOps-ChatTS/scripts/run_chatts_no_ragas_batch.sh)
- [NetManAIOps/ChatTS 覆盖与评测说明](./direct-files/NetManAIOps-ChatTS/README.md)

其中 `timesfm2_5.py`、`chronos2.py`、`zeus.py`、`zeus_modeling.py` 和
`timeseries_backbones.py` 是新增文件，其余 Python 文件包含基于 `bf30699` 的完整修改后内容。
如果服务器版本较新或已有本地改动，不建议直接覆盖，应使用下方 Git 补丁进行三方合并。

## NetManAIOps/ChatTS vLLM 评测

原始 NetManAIOps/ChatTS 的 `Qwen3TSForCausalLM` 总是构造 MLP-Patch 编码器，
因此外部 backbone checkpoint 中的 `ts_encoder.projector.*` 找不到对应模块，报错：

```text
ValueError: There is no module or parameter named 'ts_encoder.projector'
in Qwen3TSForCausalLM
```

本目录的推理版根据 checkpoint `config.json` 中的 `ts_encoder_type` 自动构造：

| `ts_encoder_type` | 推理编码器 | `ts.patch_size` | 额外安装 |
|---|---|---:|---|
| 缺省、`native`、`mlp` | 原始 ChatTS MLP-Patch | 沿用 checkpoint | 无 |
| `timesfm2_5` | `google/timesfm-2.5-200m-pytorch` | 32 | `timesfm[torch]>=2.0.2` |
| `chronos2` | `amazon/chronos-2` | 16 | `chronos-forecasting==2.3.1` |
| `zeus` | `GestaltCog/zeus` | 32 | 无；必须同时复制 `zeus_modeling.py` |

如果权重含 `ts_encoder.projector.*`，启动时却打印
`Using the native ChatTS MLP-Patch encoder`，说明评测目录没有保存外部编码器元数据。
最新版支持用环境变量立即指定：

```bash
# 按实际 checkpoint 三选一
export CHATTS_TS_ENCODER_TYPE=timesfm2_5
# export CHATTS_TS_ENCODER_TYPE=chronos2
# export CHATTS_TS_ENCODER_TYPE=zeus
```

也可以分别用 `CHATTS_TIMESFM_MODEL_PATH`、`CHATTS_CHRONOS2_MODEL_PATH`、
`CHATTS_ZEUS_MODEL_PATH` 指向评测机本地 backbone。永久方案仍是把正确的
`ts_encoder_type`、backbone 路径字段和 patch size 写回 checkpoint `config.json`。

对本目录附带的批处理脚本，默认留空就会先读取 checkpoint 配置，
再根据 `ts_encoder.mlp.*` / `ts_encoder.projector.*` 权重 shape 自动识别。
TimesFM 2.5 的 1280 维 projector 可唯一识别；Chronos-2 和 Zeus 同为 768 维，
需要保留 patch size 或提供单次 fallback，不需要先 `export`：

```bash
TS_ENCODER_TYPE=chronos2 \
bash direct-files/NetManAIOps-ChatTS/scripts/run_chatts_no_ragas_batch.sh --infer-only
```

最新脚本会逐个检查权重，所以这个 fallback 不会再把后续 native MLP 或
TimesFM checkpoint 错误构造成 Chronos-2。但如果同一 `SEARCH_DIR` 同时包含
Chronos-2 和 Zeus，两者仍需要各自的 patch size/元数据，或拆成两次运行。

TimesFM 冻结主干不会复制进每个 ChatTS checkpoint，只保存了训练的 projector。
无网评测时将一份 TimesFM `model.safetensors` 放在共享目录，然后：

```bash
HF_HUB_OFFLINE=1 \
TIMESFM_MODEL_PATH=/workspace/timesf \
bash direct-files/NetManAIOps-ChatTS/scripts/run_chatts_no_ragas_batch.sh --infer-only
```

批处理脚本用 `--all` 或 `--score-only` 完成后，还会把全部模型的
Dataset A/B categorical/numerical 指标、平均分、token 数和状态汇总到：

```text
$SEARCH_DIR/logs/chatts_batch_summary.csv
$SEARCH_DIR/logs/chatts_batch_summary.md
```

表格按四项总体指标的宏平均降序排名；可用 `SUMMARY_DIR` 和
`SUMMARY_BASENAME` 改变输出位置与文件名。已有 `result.json` 时可以用
`--summary-only` 直接生成表格，不重新推理或评分。

服务器上先备份，再覆盖两个文件：

```bash
cd /workspace/ChatTS/ChatTS-main
cp chatts/vllm/chatts_vllm.py chatts/vllm/chatts_vllm.py.bak
cp /path/to/direct-files/NetManAIOps-ChatTS/chatts/vllm/chatts_vllm.py chatts/vllm/
cp /path/to/direct-files/NetManAIOps-ChatTS/chatts/vllm/zeus_modeling.py chatts/vllm/
```

只安装当前 checkpoint 需要的依赖即可：

```bash
# TimesFM 2.5 checkpoint
pip install 'timesfm[torch]>=2.0.2'

# Chronos-2 checkpoint
pip install 'chronos-forecasting==2.3.1'
```

然后检查模型目录中的配置。`MODEL_PATH` 要指向 Stage 2 完整 checkpoint，不能只指向
projector adapter 目录：

```bash
MODEL_PATH=/path/to/stage2-checkpoint python - <<'PY'
import json
import os

path = os.environ["MODEL_PATH"]
with open(os.path.join(path, "config.json"), encoding="utf-8") as file:
    config = json.load(file)

encoder = config.get("ts_encoder_type", "native")
patch_size = config.get("ts", {}).get("patch_size")
expected = {"timesfm2_5": 32, "chronos2": 16, "zeus": 32}
print("ts_encoder_type =", encoder)
print("ts.patch_size =", patch_size)
if encoder in expected:
    assert patch_size == expected[encoder], (encoder, patch_size)
PY
```

保持原来的 ChatTS vLLM 启动/评测命令。多进程模式建议继续设置：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
python3 -m chatts.utils.inference_tsmllm_vllm
```

不要通过忽略 `ts_encoder.projector.*` 来绕过报错；那会让评测使用错误或随机初始化的
MLP 编码器。四种后端都保持训练时的输入契约和归一化路径；外部 backbone 在每个 vLLM
worker 中冻结并以 FP32 各保存一份，因此显存估算需额外计入该 backbone。

## 应用补丁

建议从对应基线开始：

```bash
git clone https://github.com/xiezhe-24/ChatTS-Training.git
cd ChatTS-Training
git checkout bf30699
git am "/path/to/0001-feat-add-frozen-TimesFM-2.5-encoder-for-ChatTS.patch"
git am "/path/to/0002-fix-match-TS-Reasoner-two-stage-training-recipe.patch"
git am "/path/to/0003-feat-add-Zeus-and-Chronos-2-time-series-backbones.patch"
```

如果你的 ChatTS-Training 已经包含后续提交，可以新建分支后尝试三方合并：

```bash
git switch -c timesfm2.5-adapter
git am -3 "/path/to/0001-feat-add-frozen-TimesFM-2.5-encoder-for-ChatTS.patch"
git am -3 "/path/to/0002-fix-match-TS-Reasoner-two-stage-training-recipe.patch"
git am -3 "/path/to/0003-feat-add-Zeus-and-Chronos-2-time-series-backbones.patch"
```

## 安装与训练

TimesFM 2.5 要求 Python 3.10 或更高版本。

```bash
pip install -e ".[timesfm,deepspeed]"
hf download google/timesfm-2.5-200m-pytorch model.safetensors config.json
```

Chronos-2 同样建议 Python 3.10 或更高版本，适配器固定使用已验证的
`chronos-forecasting==2.3.1`：

```bash
pip install -e ".[chronos2,deepspeed]"
```

ZEUS 使用仓库内置的纯 PyTorch eager-attention 兼容实现，不安装会强制降级
Transformers 的 `basicts==1.1.0`，也不需要额外安装 FlashAttention：

```bash
pip install -e ".[zeus,deepspeed]"
```

编辑脚本中的模型路径、输出路径和 GPU 数量，然后运行：

```bash
bash scripts/full/train_timesfm2_5_stage1.sh
bash scripts/full/train_timesfm2_5_stage2.sh

# 或将两阶段合成一次混合 SFT
bash scripts/full/train_timesfm2_5_one_stage.sh

# 或改用 Chronos-2
bash scripts/full/train_chronos2_stage1.sh
bash scripts/full/train_chronos2_stage2.sh

# 或改用 ZEUS
bash scripts/full/train_zeus_stage1.sh
bash scripts/full/train_zeus_stage2.sh
```

三个方案都不需要额外手工归一化：TimesFM 走官方 cumulative normalization；Chronos-2
在官方 encoder 内完成 instance normalization 与 `asinh`；ZEUS 适配器按有效点执行
instance normalization 与 `asinh`。重复 z-score 会改变预训练 backbone 看到的数据分布。

当前脚本已按 TS-Reasoner 论文 Table 4 对齐：

| 阶段 | 数据 | 可训练模块 | 冻结模块 | 全局 Batch | 学习率 | Epoch |
|---|---|---|---|---:|---:|---:|
| Stage 1 alignment | 120K captions | 完整 LLM + projector | 所选外部 backbone | 64 | `1e-5` | 1 |
| Stage 2 SFT | 30K instructions | 完整 LLM + projector | 所选外部 backbone | 32 | `2e-5` | 2 |

这里有一个容易误解但很重要的事实：TS-Reasoner 并不是“Stage 1 只训练 projector，
Stage 2 再训练 LLM”。论文 §3.2 明确写 LLM 在两个阶段都保持可训练，Table 4 也报告两个阶段
均有约 7.3B 可训练参数。TS-Reasoner 两阶段始终冻结的是 TimesFM；本扩展对
Chronos-2 与 ZEUS 采用相同冻结策略。

因此脚本中的 `--finetuning_type full` 是有意保留的：`full` 只会更新 ChatTS LLM 和
`ts_encoder.projector`；所选外部 backbone 在代码中通过 `requires_grad_(False)` 和
`torch.no_grad()` 双重冻结。

`--timeseries_sft_lr` 是可选参数。在三个外部 backbone 架构下，它都只控制两层
TS-to-text projector 的学习率，不是 TimesFM、Chronos-2 或 ZEUS 主干的学习率。
当前复现脚本不设置它，让 projector 与 LLM 共用论文给出的全局学习率。

论文和官方 shell 在 Stage 2 的 batch size 上存在一处不一致：论文 Table 4 报告 32，
而官方脚本按 8 卡 × 单卡 1 × 梯度累积 8 实际为 64。本目录脚本以论文表格为准，
Stage 2 使用梯度累积 4。

### 合并为一个训练阶段

`train_timesfm2_5_one_stage.sh` 使用：

```bash
--dataset "stage_1_120K,stage_2_30K" \
--mix_strategy "interleave_over" \
--interleave_probs "0.6667,0.3333"
```

这里的 `interleave_probs` 表示“下一条样本从各数据集抽取的概率”，顺序与
`--dataset` 一一对应；它不是 loss 权重，也不直接等于 epoch 数。
`interleave_over` 要等所有数据集都至少用完一次才停止，因此对 120K/30K 数据使用
`2/3,1/3` 时，实际暴露量约为 alignment 120K 一遍、instruction 30K 两遍。

如果你只想让两份数据各训练一遍，更简单的写法是：

```bash
--dataset "stage_1_120K,stage_2_30K" \
--mix_strategy "concat"
```

此时删掉 `--interleave_probs`，数据的自然比例就是 `120K:30K = 0.8:0.2`。

使用脚本前，请把
[`data/dataset_info.ts_reasoner.json`](./direct-files/data/dataset_info.ts_reasoner.json)
中的条目合并进服务器的 `data/dataset_info.json`。内容如下：

```json
{
  "stage_1_120K": {
    "file_name": "data/alignment/align_120K.jsonl",
    "columns": {"prompt": "input", "response": "output", "timeseries": "timeseries"}
  },
  "stage_2_30K": {
    "file_name": "data/finetuning/sft-30K.jsonl",
    "columns": {"prompt": "input", "response": "output", "timeseries": "timeseries"}
  }
}
```

## 重要行为

- TimesFM、Chronos-2、ZEUS 主干始终冻结，并保持 FP32；每个训练进程都持有一份外部主干。
- Stage 1 与 Stage 2 均训练完整 LLM 和两层 projector；这与 TS-Reasoner 原论文一致。
- 外部 backbone 权重不写入 ChatTS checkpoint，只保存 projector，避免重复保存 100M–200M 参数。
- 原 ChatTS MLP encoder 权重不能迁移，需要重新运行 Stage 1 对齐 projector。
- Stage 1 会写入具体 `ts_encoder_type`，Stage 2 使用 `auto` 自动恢复架构、模型路径与 projector。
- 单条序列上限：TimesFM 16,384 点，Chronos-2 8,192 点，ZEUS 4,096 点。
- 训练后的模型需通过打过补丁的 LLaMA-Factory loader 加载；普通的
  `AutoModelForCausalLM.from_pretrained(...)` 不会自动重建外部 backbone adapter。
- 三者可能带来更强的时序先验，但不应在完整对照实验前宣称
  ChatTS 问答指标必然提升。
- ZEUS 官方配置同时存在 `n_heads=[4,4,8,8,8,4,4]` 和论文式
  `num_heads=[6,12,12,12,6]`，而发布代码实际读取前者。本实现忠实匹配发布 artifact，
  不私自改变 attention heads；建议在论文实验中显式记录这一上游差异。

## 已完成验证

- Ruff 检查和格式检查通过。
- Python compileall 通过。
- 六个训练脚本的 `bash -n` 检查通过。
- 模拟 TimesFM backbone 的变长输入接口通过：长度 `[40, 65]` 得到 patch 数 `[2, 3]`。
- 冻结状态和“TimesFM 权重不进入 ChatTS state dict”通过。
- LoRA projector 隔离与完整保存逻辑通过。
- Stage 1 checkpoint → Stage 2 projector 自动恢复通过。
- 官方 `TimesFM_2p5_200M_torch.from_pretrained` API 已核对。
- 10 项 TimesFM/Chronos-2/ZEUS 单元测试全部通过。
- `amazon/chronos-2` 在 Transformers 4.51.3 下完成真实权重加载和前向：33 点得到
  3 个 context patch，encoder 输出形状为 `[1, 5, 768]`，排除 REG/future 后为 `[1, 3, 768]`。
- `GestaltCog/zeus` 官方 102,096,777 参数已真实加载；本地 eager 实现与官方
  safetensors 的 167 个键和 shape 完全一致，33 点前向得到 `[1, 2, 768]` 中心尺度特征。
- 两个新增 adapter 均验证为冻结状态，且外部 backbone 不会进入 ChatTS `state_dict`。

尚未在真实 8 卡 GPU 环境完成端到端训练，因此显存峰值、吞吐量和最终任务指标仍需实测。

## 设计依据

- [TS-Reasoner 论文](https://arxiv.org/abs/2510.03519)
- [TS-Reasoner-7B 模型页](https://huggingface.co/ParadiseYu/TS-Reasoner-7B)
- [TimesFM 官方仓库](https://github.com/google-research/timesfm)
- [TimesFM 2.5 PyTorch checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [Chronos-2 checkpoint](https://huggingface.co/amazon/chronos-2)
- [Chronos Forecasting 官方仓库](https://github.com/amazon-science/chronos-forecasting)
- [ZEUS checkpoint](https://huggingface.co/GestaltCog/zeus)
- [ZEUS 官方仓库](https://github.com/GestaltCogTeam/Zeus)
- [ZEUS 论文](https://arxiv.org/abs/2607.01918)
- [ChatTS-8B](https://huggingface.co/bytedance-research/ChatTS-8B)
