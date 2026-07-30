# ChatTS × TimesFM 2.5 适配实现

本目录提供一个可直接应用到
[xiezhe-24/ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training)
的完整补丁：保留 ChatTS 的数据格式、Processor、`<ts>...<ts/>` token 合并与两阶段训练流程，
将原生 MLP-Patch 时间序列编码器替换为冻结的
[`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
和可训练的 TS-to-text projector。

## 架构

```text
ChatTS [scaled_value, valid_mask]
                │
                ▼
     TimesFM 2.5 patching
        32 points / patch
                │
                ▼
 Frozen TimesFM 2.5 backbone
   20 layers · 1280 hidden · 200M
                │
                ▼
 Trainable 2-layer projector
          1280 → LLM hidden
                │
                ▼
  Existing ChatTS token merge
                │
                ▼
            Qwen LLM
```

编码器继续返回 ChatTS 所需的 `(features, patch_cnt)`。因此原始 MLP 的
`ceil(L / 8)` 个时间序列 token 会变为 `ceil(L / 32)`，通常约减少到四分之一。

## 目录内容

- [基础实现补丁](./0001-feat-add-frozen-TimesFM-2.5-encoder-for-ChatTS.patch)。
- [TS-Reasoner 训练配置修正补丁](./0002-fix-match-TS-Reasoner-two-stage-training-recipe.patch)。
- [`direct-files/`](./direct-files/)：保持 ChatTS-Training 相对路径的完整修改文件，可直接复制到服务器。
- 补丁基线：ChatTS-Training `bf30699`。
- 补丁实现提交：`9e24561`；训练配置修正提交：`f6abe7b`。
- SHA-256：
  - `0001`：`60d3878a0f36e3e94b053894a0e64e10cbb4c9b6449b285869162e6c668a01b8`
  - `0002`：`1017284f4d1e94021eb10fa0089798f808628598c25dd0ed2a981b9a03e5425e`

补丁包含：

1. TimesFM 2.5 编码器、官方 cumulative normalization 和变长 batch 处理。
2. `LayerNorm → Linear → GELU → Linear → LayerNorm` projector。
3. Stage 1 保存与 Stage 2 自动恢复 projector。
4. Full SFT 与 LoRA 的时间序列模块处理。
5. 两阶段训练脚本和单元测试。
6. `timesfm[torch]>=2.0.2` 可选依赖与 README 使用说明。

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
- [`test_timesfm2_5.py`](./direct-files/tests/model/test_timesfm2_5.py)
- [`train_timesfm2_5_stage1.sh`](./direct-files/scripts/full/train_timesfm2_5_stage1.sh)
- [`train_timesfm2_5_stage2.sh`](./direct-files/scripts/full/train_timesfm2_5_stage2.sh)

其中 `timesfm2_5.py` 是新增文件，其余 Python 文件包含基于 `bf30699` 的完整修改后内容。
如果服务器版本较新或已有本地改动，不建议直接覆盖，应使用下方 Git 补丁进行三方合并。

## 应用补丁

建议从对应基线开始：

```bash
git clone https://github.com/xiezhe-24/ChatTS-Training.git
cd ChatTS-Training
git checkout bf30699
git am "/path/to/0001-feat-add-frozen-TimesFM-2.5-encoder-for-ChatTS.patch"
git am "/path/to/0002-fix-match-TS-Reasoner-two-stage-training-recipe.patch"
```

如果你的 ChatTS-Training 已经包含后续提交，可以新建分支后尝试三方合并：

```bash
git switch -c timesfm2.5-adapter
git am -3 "/path/to/0001-feat-add-frozen-TimesFM-2.5-encoder-for-ChatTS.patch"
git am -3 "/path/to/0002-fix-match-TS-Reasoner-two-stage-training-recipe.patch"
```

## 安装与训练

TimesFM 2.5 要求 Python 3.10 或更高版本。

```bash
pip install -e ".[timesfm,deepspeed]"
hf download google/timesfm-2.5-200m-pytorch model.safetensors config.json
```

编辑脚本中的模型路径、输出路径和 GPU 数量，然后运行：

```bash
bash scripts/full/train_timesfm2_5_stage1.sh
bash scripts/full/train_timesfm2_5_stage2.sh
```

当前脚本已按 TS-Reasoner 论文 Table 4 对齐：

| 阶段 | 数据 | 可训练模块 | 冻结模块 | 全局 Batch | 学习率 | Epoch |
|---|---|---|---|---:|---:|---:|
| Stage 1 alignment | 120K captions | 完整 LLM + projector | TimesFM | 64 | `1e-5` | 1 |
| Stage 2 SFT | 30K instructions | 完整 LLM + projector | TimesFM | 32 | `2e-5` | 2 |

这里有一个容易误解但很重要的事实：TS-Reasoner 并不是“Stage 1 只训练 projector，
Stage 2 再训练 LLM”。论文 §3.2 明确写 LLM 在两个阶段都保持可训练，Table 4 也报告两个阶段
均有约 7.3B 可训练参数。两阶段始终冻结的是 TimesFM。

因此脚本中的 `--finetuning_type full` 是有意保留的：`full` 只会更新 ChatTS LLM 和
`ts_encoder.projector`；外部 TimesFM 在代码中通过 `requires_grad_(False)` 和
`torch.no_grad()` 双重冻结。

`--timeseries_sft_lr` 是可选参数。在 TimesFM 2.5 架构下，它只控制两层 TS-to-text
projector 的学习率，不是 TimesFM 20 层主干的学习率。当前复现脚本不设置它，让 projector
与 LLM 共用论文给出的全局学习率。

论文和官方 shell 在 Stage 2 的 batch size 上存在一处不一致：论文 Table 4 报告 32，
而官方脚本按 8 卡 × 单卡 1 × 梯度累积 8 实际为 64。本目录脚本以论文表格为准，
Stage 2 使用梯度累积 4。

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

- TimesFM 主干始终冻结，并保持 FP32；每个训练进程额外占用约 0.8 GB 权重显存。
- Stage 1 与 Stage 2 均训练完整 LLM 和两层 projector；这与 TS-Reasoner 原论文一致。
- TimesFM 权重不写入 ChatTS checkpoint，避免每个 checkpoint 重复保存约 200M 参数。
- 原 ChatTS MLP encoder 权重不能迁移，需要重新运行 Stage 1 对齐 projector。
- Stage 1 会写入 `ts_encoder_type=timesfm2_5`，Stage 2 使用 `auto` 自动恢复架构和模型路径。
- 当前单条时间序列最多接受 16,384 个点。
- 训练后的模型需通过打过补丁的 LLaMA-Factory loader 加载；普通的
  `AutoModelForCausalLM.from_pretrained(...)` 不会自动重建外部 TimesFM adapter。
- TimesFM 来自 forecasting 预训练，可能带来更强的时序先验，但不应在完整对照实验前宣称
  ChatTS 问答指标必然提升。

## 已完成验证

- Ruff 检查和格式检查通过。
- Python compileall 通过。
- 两个训练脚本的 `bash -n` 检查通过。
- 模拟 TimesFM backbone 的变长输入接口通过：长度 `[40, 65]` 得到 patch 数 `[2, 3]`。
- 冻结状态和“TimesFM 权重不进入 ChatTS state dict”通过。
- LoRA projector 隔离与完整保存逻辑通过。
- Stage 1 checkpoint → Stage 2 projector 自动恢复通过。
- 官方 `TimesFM_2p5_200M_torch.from_pretrained` API 已核对。

尚未在真实 8 卡 GPU 环境完成端到端训练，因此显存峰值、吞吐量和最终任务指标仍需实测。

## 设计依据

- [TS-Reasoner 论文](https://arxiv.org/abs/2510.03519)
- [TS-Reasoner-7B 模型页](https://huggingface.co/ParadiseYu/TS-Reasoner-7B)
- [TimesFM 官方仓库](https://github.com/google-research/timesfm)
- [TimesFM 2.5 PyTorch checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [ChatTS-8B](https://huggingface.co/bytedance-research/ChatTS-8B)
