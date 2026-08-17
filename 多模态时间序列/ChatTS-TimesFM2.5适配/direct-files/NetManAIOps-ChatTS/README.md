# NetManAIOps/ChatTS 四后端 vLLM 适配

这些文件用于修复外部时间序列 backbone checkpoint 在
NetManAIOps/ChatTS vLLM 评测中的权重加载错误，同时保持原始 ChatTS MLP-Patch
checkpoint 兼容：

- `chatts/vllm/chatts_vllm.py`：支持原始 MLP-Patch、TimesFM 2.5、Chronos-2 和 Zeus。
- `chatts/vllm/zeus_modeling.py`：Zeus 官方 checkpoint 兼容的 eager-attention 结构。
- `scripts/run_chatts_no_ragas_batch.sh`：逐 checkpoint 权重识别、评测和结果汇总脚本。
- `scripts/inspect_chatts_ts_encoder_checkpoints.py`：只读扫描权重并盘点真实编码器。
- `scripts/run_chatts_timeseriesexam.sh`：用原始数值时序评测 TimeSeriesExam。
- `scripts/run_all_chatts_benchmarks.sh`：四套 benchmark 串行执行，每套独占全部 8 张卡。
- `scripts/run_train_then_eval.sh`：宿主机一键完整训练评测，也支持只训练并保留 Stage1 权重。
- `configs/train_eval_chronos2.yaml`：两阶段训练、模型、数据和评测路径的集中配置。
- `scripts/load_train_eval_config.py`：将 YAML 安全展开为流水线环境变量。

适配基线为 NetManAIOps/ChatTS `a16ca1a`。用户报错栈中的原文件行号与该基线一致。

文件 SHA-256：

- `chatts_vllm.py`：`de887ab3ea9ea8c8c5a5b84f85f0d239c9ad5c43c7937b87c98b79db501ada03`
- `zeus_modeling.py`：`d5850fb0d8d104f6d7d92580b74e48df5b0bf450536cdb662fff35fa66b5c271`
- `inspect_chatts_ts_encoder_checkpoints.py`：`31f510c509cc9a57838ed4eaa101c4307ed4d7b111b904a8c07190beda596df0`

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

## 先盘点整个目录的真实权重

如果需要确认一批 checkpoint 到底保存了什么 encoder，运行：

```bash
python scripts/inspect_chatts_ts_encoder_checkpoints.py \
  /share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-PR-grid
```

脚本不修改 checkpoint，会扫描 `.safetensors` / `.bin` / `.pt` / `.pth`
的 tensor 名和 shape，并在终端输出：

```text
CHECKPOINT  DETECTED    STATUS  PROJ_DIM  NATIVE  PROJECTOR
model-a     timesfm2_5  OK      1280      0       8
```

默认还会生成：

```text
$SEARCH_DIR/logs/chatts_ts_encoder_inventory.csv
$SEARCH_DIR/logs/chatts_ts_encoder_inventory.md
```

CSV 的 `relevant_tensors` 列会保留具体权重键和 shape。对一个正常 TimesFM
checkpoint，应当看到 `detected_encoder=timesfm2_5`、`projector_dims=1280`、
`native_key_count=0` 和 `status=OK`。如果同时找到 native
`position_embedding/MLP` 和 external projector，会标记为
`mixed_native_external / ERROR`。

如果每个子目录下还有固定的权重子路径，使用：

```bash
python scripts/inspect_chatts_ts_encoder_checkpoints.py "$SEARCH_DIR" \
  --checkpoint-suffix stage2-checkpoint
```

脚本默认只使用 `torch.load(weights_only=True)`。老版 PyTorch/老格式不支持安全
加载时会明确报错；只有完全信任这批 checkpoint 时才可加
`--allow-unsafe-torch-load`。

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

## TSRBench：复用 ChatTS vLLM 推理后端

本目录额外提供四个新的可直接复制到服务器的文件，并复用此前已经提供的 checkpoint
检查器：

- `chatts/utils/llm_utils.py`：修复 `worker_vllm_ts` 忽略调用方
  `SamplingParams` 的问题，并支持 `CHATTS_VLLM_MAX_MODEL_LEN`；
- `chatts/utils/inference_tsrbench_vllm.py`：读取 TSRBench JSONL，复用现有
  `LLMClient(engine="vllm-ts")`，模型只加载一次即可依次跑完全部任务；
- `scripts/run_chatts_tsrbench.sh`：一键推理与评测；
- `scripts/evaluate_tsrbench.py`：纯本地多选题评测，不使用 RAGAS 或 judge API。
- `scripts/inspect_chatts_ts_encoder_checkpoints.py`：若 config 中没有
  `ts_encoder_type`，启动前直接根据 checkpoint 权重识别 MLP、TimesFM 2.5、
  Chronos-2 或 Zeus。

将四个文件按相同相对路径复制到 ChatTS 项目后运行。必须同时覆盖
`chatts/utils/llm_utils.py`，否则 vLLM-TS worker 仍会使用旧版硬编码的
`temperature=0.5, max_tokens=5000`：

```bash
PROJECT_ROOT=/workspace/ChatTS/ChatTS-main \
TSRBENCH_ROOT=/workspace/TSRBench \
DATASET_ROOT=/workspace/TSRBench/dataset \
MODEL_PATH=/path/to/your/chatts/checkpoint \
MODEL_NAME=my-chatts \
NUM_GPUS=8 \
NUM_GPUS_PER_PROCESS=2 \
bash scripts/run_chatts_tsrbench.sh
```

脚本会递归识别当前发布数据中的 12 个 JSONL 文件，同时兼容 TSRBench 旧代码里的
三个名称：`math_reasoning -> numerical_reasoning`、
`event_forecast -> event_prediction`、
`pattern_decision -> qualitative_decision`。中途退出后用同一条命令重跑即可从已有
`generated_answer.json` 继续；仅当 `FORCE_INFERENCE=1` 时才覆盖已有结果。

原始时间序列直接传入 vLLM 的 `multi_modal_data["timeseries"]`，不要在脚本外再次
归一化。ChatTS checkpoint 自带的 processor 会执行与 Dataset A/B 相同的 SP
归一化。MLP-Patch、TimesFM 2.5、Chronos-2、Zeus 都走同一个输入接口；encoder
类型和本地 backbone 路径的环境变量与 Dataset A/B 脚本完全相同。例如：

```bash
TS_ENCODER_TYPE=timesfm2_5 \
TIMESFM_MODEL_PATH=/workspace/timesf \
bash scripts/run_chatts_tsrbench.sh
```

通常无需指定 `TS_ENCODER_TYPE`：一键脚本会先用权重检查器自动识别。只有 768 维
projector 且 checkpoint 同时没有可用 patch size 时，Chronos-2 与 Zeus 无法仅靠
权重区分，此时才需要显式设置 `TS_ENCODER_TYPE=chronos2` 或 `zeus`。

只做快速冒烟测试：

```bash
DATASETS="perception numerical_reasoning" MAX_SAMPLES=20 \
bash scripts/run_chatts_tsrbench.sh
```

默认 `PROMPT_MODE=answer_only`：Qwen3 `enable_thinking=False`，只要求输出一个
选项字母；`max_model_len=12288`、`max_new_tokens=8`、`temperature=0.0`，并在
首个换行处停止，避免模型继续生成并截断解释；默认不做格式重试。若要复现
TSRBench 官方 ChatTS prompt 和显式参数，使用：

```bash
PROMPT_MODE=official FORCE_INFERENCE=1 \
DATASETS="perception numerical_reasoning" MAX_SAMPLES=20 \
bash scripts/run_chatts_tsrbench.sh
```

`official` 模式使用官方 `<think>...</think><answer>A</answer>` 指令、手写 ChatML
包装、`max_new_tokens=512`、输入上限 `8000`、`temperature=1.0`、最多十次生成，
以及官方的 `batch_size=1`；vLLM 的 `max_model_len` 设为 `12288`，为多模态
时序 tokens 预留额外空间。这里的
thinking 是 TSRBench 用户 prompt 明确要求的推理，不是额外开启 Qwen3 chat
template 的 thinking 开关。

推理和评测解析器均接受“第一行是单个选项字母、后面带解释”的历史输出。因此旧的
`generated_answer.json` 不需要重跑；更新 `scripts/evaluate_tsrbench.py` 后直接
重新执行评测即可恢复这类答案。

脚本会分别检查原始文本 token 数，以及 ChatTS processor 展开后的
“文本 + 时序 patch”总输入 token 数。默认给输出保留完整的
`MAX_NEW_TOKENS` 空间；超长样本会记录为 `INPUT_SKIPPED` 并继续评测，不会再让
整个 vLLM worker 退出、导致进度条永久阻塞。若日志仍显示
`effective_max_model_len=8192`，说明服务器保留了旧环境变量；请显式设置
`CHATTS_VLLM_MAX_MODEL_LEN=12288` 后重启脚本。

结果记录包含 `prompt_mode`，脚本不会把不同模式的断点结果混在一起；切换模式时仍
建议设置 `FORCE_INFERENCE=1`，便于得到一份干净的完整结果。

如果日志中出现下列组合：

```text
Qwen3TSForCausalLM has no vLLM implementation, falling back to Transformers
KeyError: <class 'vllm.model_executor.models.transformers.TransformersForCausalLM'>
```

说明 vLLM spawn worker 没有执行 ChatTS 的模型与多模态 processor 注册。
`inference_tsrbench_vllm.py` 已将 `import chatts.vllm.chatts_vllm` 固定在模块
顶层，不要再把它移进 `main()` 或其他函数。启动成功时每个 worker
都应使用 `Qwen3TSForCausalLM` 的 ChatTS vLLM 实现，不应再出现上述
fallback warning。

最终结果保存在 `${TSRBENCH_ROOT}/evaluation/results/embed/`，并额外生成
`tsrbench_summary_<model>.json` 和可直接用 Excel 打开的 CSV。主报告同时给出
`accuracy_strict=正确数/数据集总数` 与 `accuracy_parsed=正确数/成功解析数`，避免
因漏答或格式错误而虚高。

## TS-Haystack：四域长上下文评测

新增的 `inference_ts_haystack_vllm.py` 直接复用
`AI-X-Labs/TS-Haystack` 官方 dataset class，支持 Capture24、Sleep PSG
stages/arousals、LTAF ECG 与 UK-DALE。信号加载/重建、prompt、答案
类型解析、时间段 IoU 和 timestamp 容差都调用官方实现，不使用
judge model，也不会暗中下采样超长信号。

完整的数据布局、文件覆盖、四种 encoder 运行示例、长序列覆盖率与
汇总表定义见 [`TS_HAYSTACK.md`](./TS_HAYSTACK.md)。最小冒烟测试：

```bash
PROJECT_ROOT=/workspace/ChatTS/ChatTS-main \
TS_HAYSTACK_ROOT=/workspace/TS-Haystack \
MODEL_PATH=/workspace/checkpoints/my-chatts \
DATASETS="capture24 uk_dale" \
TASKS="existence localization" \
CONTEXT_LENGTHS="100 900" \
MAX_SAMPLES=20 \
bash scripts/run_chatts_ts_haystack.sh
```

结果会生成逐样本 JSON，以及包含 overall/per-dataset/per-task/per-context
的 JSON/CSV 汇总。`Strict` 把输入过长样本留在总分母中，`GenAcc`
只统计实际生成的样本，因此不会把 encoder 的上下文覆盖差异藏起来。

## TimeSeriesExam：基础时序理解考试

新增适配直接读取官方 TimeSeriesExam 的单序列 `ts` 或双序列 `ts1/ts2` 原始
数组，通过 ChatTS vLLM `timeseries` 模态推理，不需要转图片或逗号文本。
它支持 MLP-Patch、TimesFM 2.5、Chronos-2、Zeus，并默认对齐官方评测 shell：
固定 one-shot 示例、hint、最多 3 个 concepts 及示例、`temperature=0`、
`seed=42`、`max_new_tokens=1024`；额外 Qwen3 thinking 保持关闭。

完整的数据下载、文件覆盖、四后端命令、prompt 消融和指标定义见
[`TIMESERIESEXAM.md`](./TIMESERIESEXAM.md)。最小冒烟测试：

```bash
PROJECT_ROOT=/workspace/ChatTS/ChatTS-main \
TIMESERIESEXAM_ROOT=/workspace/TimeSeriesExam \
MODEL_PATH=/workspace/checkpoints/my-chatts \
NUM_GPUS=2 \
NUM_GPUS_PER_PROCESS=2 \
MAX_SAMPLES=20 \
OUTPUT_ROOT=/workspace/results/timeseriesexam-smoke \
bash scripts/run_chatts_timeseriesexam.sh
```

主表同时给出官方 `B) 选项正文` substring 规则的 `Official`、官方最后一行
`Strict`，以及稳健字母解析的 `LetterAcc`。这样既能复现论文协议，也能识别
“字母答对但输出格式没完全对齐”造成的误判。

## tinyBenchmarks 选择题：通用能力与灾难性遗忘筛查

这个版本不安装 `lm-eval`、`tinyBenchmarks` 或任何新环境，直接使用 ChatTS
已经安装的 `vllm==0.8.5`。数据只从本地读取，模型也只从本地加载；默认设置
`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，不会偷偷联网。

只跑五个选择题任务：

```text
tinyArc, tinyHellaswag, tinyMMLU, tinyTruthfulQA, tinyWinogrande
```

`tinyGSM8k` 不在本次评测中。评测器通过 vLLM 的 `prompt_logprobs`
计算选项条件似然：ARC、HellaSwag、MMLU 和 Winogrande 采用长度归一化
准确率，TruthfulQA 采用 MC2 正确答案概率质量。输入是纯文本，因此 TS
encoder 不参与时序特征 forward，也不存在“是否归一化时间序列”的问题；
但 ChatTS checkpoint 初始化时仍会按权重构建正确的 encoder 模块。

### 本地数据目录

`DATASET_ROOT` 是你已经下载的五个 tinyBenchmarks 数据集共同所在的上级目录。
脚本会递归查找官方评测 split：

- `tinyAI2_arc/ARC-Challenge/test-*.parquet`
- `tinyHellaswag/.../validation-*.parquet`
- `tinyMMLU/all/test-*.parquet`
- `tinyTruthfulQA/multiple_choice/validation-*.parquet`
- `tinyWinogrande/winogrande_xl/validation-*.parquet`

也支持 JSON/JSONL。若自动发现的目录布局不同，可逐项明确指定：

```bash
--task-file tinyArc=/workspace/datasets/tinyAI2_arc/ARC-Challenge/test-00000-of-00001.parquet
```

脚本先检查五个文件的 schema 和样本数，确认都是 100 条后才分配 GPU。
如果你的文件是 `datasets.save_to_disk` 格式，仅在当前环境本来就有
`datasets` 包时读取；缺包时只给出转换 JSONL 的提示，不会自动安装。

判断灾难性遗忘至少需要两个模型：开始 ChatTS 训练前的底座模型，
以及训练后 checkpoint。也可以在中间加入 Stage 1/Stage 2：

```bash
cd /workspace/ChatTS/ChatTS-main
CUDA_VISIBLE_DEVICES=0,1 \
NUM_GPUS=2 \
DATASET_ROOT=/workspace/datasets/tinyBenchmarks \
bash scripts/run_chatts_tinybenchmarks_mcq.sh \
  --model base=/workspace/models/qwen3-8b-before-chatts \
  --model stage1=/workspace/checkpoints/chatts-stage1 \
  --model chatts=/workspace/checkpoints/chatts-final \
  --baseline base
```

每个模型在独立 Python 进程中启动和退出，前一个模型的 CUDA/vLLM 状态不会
污染后一个模型。ChatTS checkpoint 的 encoder 类型由权重自动检测，支持：

```text
mlp-patch, timesfm2_5, chronos2, zeus
```

不必设置 `CHATTS_TS_ENCODER_TYPE`。外部 encoder 仍需给出本地 backbone：

```bash
CHATTS_TIMESFM_MODEL_PATH=/workspace/timesf
CHATTS_CHRONOS2_MODEL_PATH=/workspace/chronos-2
CHATTS_ZEUS_MODEL_PATH=/workspace/zeus
```

只设置当前 checkpoint 实际使用的那一个即可。显式设置
`CHATTS_TS_ENCODER_TYPE` 仍会作为高级覆盖项，但多 checkpoint 混合评测时
建议保持为空，让脚本逐个识别。

输出位于：

```text
/workspace/ChatTS/ChatTS-main/exp/tinybenchmarks_mcq/
  base/metrics.json
  base/samples_tinyArc.jsonl
  chatts/metrics.json
  tinybenchmarks_mcq_summary.csv
  tinybenchmarks_mcq_summary.json
  tinybenchmarks_mcq_summary.md
```

汇总表同时报告：

- 五个 tiny 锚点集的原始分数和实际样本数；
- 相对底座的每项和宏平均下降百分点；
- 通用能力保留率 `retention_percent`；
- 可配置的遗忘筛查标记。

为了不安装官方估计器及其校准资产，本脚本明确不声称 GPIRT/IRT++ 的
全量 benchmark 估计。这里的原始 100 条分数适合做训练前后 A/B 筛查。
每个任务都会保存 prompt/document/target 的 SHA-256；若本地数据不同，
汇总会标记 `protocol_match=mismatch`，不会把差异解释成遗忘。

默认只在“五项宏平均下降至少 5 个百分点，且至少三项下降”时
给出 `forgetting_warning`。这是低成本筛查，不是统计意义上的最终证明；
出现 warning 后应对对应全量 benchmark 复测。可以修改：

```bash
FORGETTING_THRESHOLD_PP=3.0 \
bash scripts/run_chatts_tinybenchmarks_mcq.sh --summary-only \
  --model base=/workspace/ChatTS/ChatTS-main/exp/tinybenchmarks_mcq/base \
  --model chatts=/workspace/ChatTS/ChatTS-main/exp/tinybenchmarks_mcq/chatts \
  --baseline base
```

先用 5 条样本做 vLLM smoke test：

```bash
MAX_SAMPLES=5 ALLOW_SIZE_MISMATCH=0 \
DATASET_ROOT=/workspace/datasets/tinyBenchmarks \
bash scripts/run_chatts_tinybenchmarks_mcq.sh \
  --model chatts=/workspace/checkpoints/chatts-final
```

smoke test 和正式评测应写到不同 `OUTPUT_ROOT`，或正式运行时加 `--force`，
避免把 5 条结果当作完整结果跳过。

对已经完成的结果只重建汇总表：

```bash
bash scripts/run_chatts_tinybenchmarks_mcq.sh --summary-only \
  --model base=/workspace/ChatTS/ChatTS-main/exp/tinybenchmarks_mcq/base \
  --model chatts=/workspace/ChatTS/ChatTS-main/exp/tinybenchmarks_mcq/chatts \
  --baseline base
```

## 一次跑完四套 benchmark

`scripts/run_all_chatts_benchmarks.sh` 会依次执行 TSRBench 全任务、
tinyBenchmarks 五个 MCQ 任务、TS-Haystack 全域和 TimeSeriesExam 全量数据。
每套任务运行期间独占 `0,1,2,3,4,5,6,7`。三个时序评测各启动 4 个双卡
vLLM worker；tinyBenchmarks 使用单个 8 卡 tensor-parallel 引擎。某套失败不会
阻止后续评测继续执行；全部结束后统一生成
`benchmark_status.tsv`、`run_manifest.json`、`metrics.json` 和
`all_benchmarks_summary.md`，只要
存在失败任务，总脚本就返回非零状态。

这个总控脚本固定使用 Chronos-2，不再自动判断或切换其他 encoder；四个子评测
都会显式收到 `CHATTS_TS_ENCODER_TYPE=chronos2`。默认模型是训练流水线产生的
8B Stage2 最佳权重：

```text
/share/airesearch/data/finiverse/output/ChatTS-msxf-8B-datav1/best_seed42
```

四套评测统一使用 `seed=42`；TSRBench 与 tinyBenchmarks 已补齐 vLLM engine 和
SamplingParams 的 seed，TS-Haystack 与 TimeSeriesExam 沿用各自已有的 seed 参数。
通常只需确认 TS-Haystack、TimeSeriesExam 和 Chronos-2 的本地路径：

```bash
cd /workspace/ChatTS/ChatTS-main

TS_HAYSTACK_ROOT=/workspace/TS-Haystack \
TIMESERIESEXAM_ROOT=/workspace/TimeSeriesExam \
CHRONOS2_MODEL_PATH=/workspace/chronos2 \
bash scripts/run_all_chatts_benchmarks.sh
```

先只检查模型、数据、runner 与 GPU 路径而不启动推理：

```bash
PREFLIGHT_ONLY=1 bash scripts/run_all_chatts_benchmarks.sh
```

默认断点续跑，不覆盖已经完成的结果。全部重算时使用：

```bash
FORCE_EVAL=1 bash scripts/run_all_chatts_benchmarks.sh
```

tinyBenchmarks 在这个总控流程中只评测本次流水线选择的最终模型，不加载训练前底座。
冒烟测试可设置 `MAX_SAMPLES=2`，全量评测保持默认 `MAX_SAMPLES=0`。

总控脚本会把影响结果的采样、提示词、离线模式和各 benchmark 参数纳入协议指纹，
并显式覆盖父进程中可能残留的同名环境变量。不同协议应写入不同的
`protocol-<hash前16位>` 输出目录，避免错误复用缓存或覆盖其他评测结果。

## 宿主机一键训练再评测

训练文件需要复制到 `chatts` 容器挂载的 ChatTS-Training 项目：

```bash
cp /path/to/direct-files/scripts/finalize_chatts_best_checkpoint.py \
  /workspace/ChatTS-Training/scripts/
cp /path/to/direct-files/scripts/full/train_chronos2_best_stage1.sh \
  /workspace/ChatTS-Training/scripts/full/
cp /path/to/direct-files/scripts/full/train_chronos2_best_stage2.sh \
  /workspace/ChatTS-Training/scripts/full/
cp /path/to/direct-files/scripts/full/run_chronos2_best_two_stage.sh \
  /workspace/ChatTS-Training/scripts/full/
cp /path/to/direct-files/scripts/verify_dataset_snapshot.py \
  /workspace/ChatTS-Training/scripts/
```

评测文件按原相对路径复制到 `ragas` 中的
`/workspace/ChatTS/ChatTS-main`，尤其要同时更新 `llm_utils.py`、四个 inference
Python 文件、四个子 runner 和 `run_all_chatts_benchmarks.sh`。

最后在宿主机运行本目录的入口，不需要手工进入任何容器：

```bash
cd /path/to/NetManAIOps-ChatTS
bash scripts/run_train_then_eval.sh
```

默认读取 `configs/train_eval_chronos2.yaml`。基础模型、输出目录、两阶段学习率、
`timeseries_sft_lr`、数据集、混合策略、epoch、batch size、梯度累积、
save/eval steps 以及四套评测数据路径均可在该 YAML 中修改。使用另一份配置：

```bash
CONFIG_FILE=/path/to/my_experiment.yaml bash scripts/run_train_then_eval.sh
```

如果只训练 Stage1、保存最优权重并直接评测，在 YAML 中设置：

```yaml
pipeline:
  pipeline_mode: stage1
training:
  stage1_model_path: /share/airesearch/data/finiverse/output/my-run/best_stage1_seed42
```

该模式以 `STAGE1_COMPLETE.json`、`best_model_manifest.json` 和非空权重为完成标志，
随后把这个 Stage1 最优目录作为 `MODEL_PATH` 运行同一套评测；不写
`pipeline_mode` 仍默认执行原来的 `full` 两阶段训练加评测。

Dataset Studio 启动的任务会另外写入 `DATA_VERSION`、
`DATASET_SNAPSHOT_HASH`、`TRIAL_ID` 和 `TRIAL_CONFIG_HASH`。训练容器会在写任何
输出前独立校验快照 manifest 与全部文件 SHA256，避免读到被修改或版本不匹配的数据。

命令行环境变量的优先级高于 YAML，例如临时做每套 2 条的冒烟评测：

```bash
MAX_SAMPLES=2 CONFIG_FILE=configs/train_eval_chronos2.yaml \
  bash scripts/run_train_then_eval.sh
```

它会先检查两个容器、8 张 GPU、代码和共享目录，再在 `chatts` 中依次完成
Stage1/Stage2。LLaMAFactory 会在结束时把验证集 `eval_loss` 最优 checkpoint
加载回内存并导出到阶段根目录；脚本验证该根目录后才删除 `checkpoint-*`。
Stage2 的 `--model_name_or_path` 明确指向 Stage1 最优导出目录；四套评测的
`MODEL_PATH` 又明确指向 Stage2 最优导出目录。Stage2 成功后会删除
Stage1 模型目录，最终只保留：

```text
/share/airesearch/data/finiverse/output/ChatTS-msxf-8B-datav1/best_seed42
```

常用控制参数：

```bash
PREFLIGHT_ONLY=1 bash scripts/run_train_then_eval.sh  # 只检查，不运行
MAX_SAMPLES=2 bash scripts/run_train_then_eval.sh     # 训练后做四套冒烟评测
FORCE_TRAIN=1 FORCE_EVAL=1 bash scripts/run_train_then_eval.sh
```

正常重跑具有幂等性：`TRAINING_COMPLETE.json` 与参数一致时复用最终模型；目录存在
但没有有效完成标记时会停止，不会静默覆盖。只有显式设置 `FORCE_TRAIN=1` 才会删除
该 seed 对应的 Stage1 临时目录和最终模型目录。

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
