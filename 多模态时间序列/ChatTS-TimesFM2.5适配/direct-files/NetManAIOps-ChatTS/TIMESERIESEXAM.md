# ChatTS 评测 TimeSeriesExam

本适配把 TimeSeriesExam 的原始数值数组直接送入 ChatTS vLLM
`timeseries` 模态，支持单序列与双序列题，兼容原始 MLP-Patch、TimesFM 2.5、
Chronos-2 和 Zeus 四种 encoder。不会把序列先画成图片，也不会把数值展开成
逗号文本。

## 对齐的官方协议

默认数据是官方 GitHub 仓库的最终 refinement round：

```text
output/round_3_folder/qa_dataset.json
```

默认复现官方 `evaluate_open_source.sh` / `evaluate_close_source.sh`：

- 题目、选项、`format_hint` 和固定 one-shot 问答示例保持官方内容；
- 加入 `question_hint`、最多 3 个 `relevant_concepts` 及其示例序列；
- `seed=42`、`temperature=0.0`、`max_new_tokens=1024`；
- Qwen3 chat template 的额外 thinking 默认关闭；题目本身仍按官方要求解释答案；
- 主指标使用官方 flexible 规则：回答中包含 `B) 正确选项正文`；
- 额外报告官方 strict 指标和稳健解析的选项字母准确率。

唯一必要的模态改动是：官方文本/图片时序位置改为 `<ts><ts/>`，对应的原始
`ts` 或 `ts1/ts2` 数组通过 vLLM 多模态字段传入。ChatTS Processor 会执行
训练时相同的 SP normalization，请勿在脚本外再次 z-score。

## 需要复制到服务器的文件

在已经覆盖四后端适配文件的 ChatTS 项目中，再按相同相对路径复制：

```text
chatts/utils/inference_timeseriesexam_vllm.py
scripts/evaluate_timeseriesexam.py
scripts/run_chatts_timeseriesexam.sh
```

检查语法：

```bash
cd /workspace/ChatTS/ChatTS-main
python -m py_compile \
  chatts/utils/inference_timeseriesexam_vllm.py \
  scripts/evaluate_timeseriesexam.py
bash -n scripts/run_chatts_timeseriesexam.sh
```

## 下载官方数据

推荐直接克隆官方仓库，因为默认 protocol 还会读取其中的
`evaluate/concepts.py`：

```bash
cd /workspace
git clone https://github.com/moment-timeseries-foundation-model/TimeSeriesExam.git
```

两个路径含义不同：

- `TIMESERIESEXAM_ROOT`：官方代码仓库根目录，用于读取 `concepts.py`，默认也从
  这里寻找 round-3 JSON。
- `DATA_FILE_PATH`：本次实际评测的数据文件。可指定官方 JSON、JSONL，或
  Hugging Face 下载的 Parquet。

Hugging Face 当前发布的 `AutonLab/TimeSeriesExam1` test split 与 GitHub
round-3 文件的行数并不相同。脚本不会悄悄混用：结果的 `protocol.source_file`
会记录实际文件路径，汇总分母以本次文件为准。

如果只使用 Hugging Face Parquet，仍建议保留官方仓库作为
`TIMESERIESEXAM_ROOT`；或者将 `ADD_CONCEPTS=0 ADD_EXAMPLES=0`，此时不读取
`concepts.py`。

## 最小冒烟测试

```bash
cd /workspace/ChatTS/ChatTS-main
CUDA_VISIBLE_DEVICES=0,1 \
PROJECT_ROOT=/workspace/ChatTS/ChatTS-main \
TIMESERIESEXAM_ROOT=/workspace/TimeSeriesExam \
MODEL_PATH=/workspace/checkpoints/my-chatts \
NUM_GPUS=2 \
NUM_GPUS_PER_PROCESS=2 \
MAX_SAMPLES=20 \
OUTPUT_ROOT=/workspace/results/timeseriesexam-smoke \
bash scripts/run_chatts_timeseriesexam.sh
```

正式跑全量时删除 `MAX_SAMPLES`，并换一个全新的 `OUTPUT_ROOT`，避免复用冒烟
测试的断点结果。

## 四种 encoder

脚本默认先扫描 ChatTS checkpoint 权重并自动识别类型。原始 MLP-Patch 不需要
外部 backbone：

```bash
MODEL_PATH=/workspace/checkpoints/chatts-mlp \
TIMESERIESEXAM_ROOT=/workspace/TimeSeriesExam \
bash scripts/run_chatts_timeseriesexam.sh
```

TimesFM 2.5：

```bash
MODEL_PATH=/workspace/checkpoints/chatts-timesfm \
TIMESERIESEXAM_ROOT=/workspace/TimeSeriesExam \
TIMESFM_MODEL_PATH=/workspace/timesf \
bash scripts/run_chatts_timeseriesexam.sh
```

Chronos-2：

```bash
MODEL_PATH=/workspace/checkpoints/chatts-chronos2 \
TIMESERIESEXAM_ROOT=/workspace/TimeSeriesExam \
CHRONOS2_MODEL_PATH=/workspace/chronos-2 \
bash scripts/run_chatts_timeseriesexam.sh
```

Zeus：

```bash
MODEL_PATH=/workspace/checkpoints/chatts-zeus \
TIMESERIESEXAM_ROOT=/workspace/TimeSeriesExam \
ZEUS_MODEL_PATH=/workspace/zeus \
bash scripts/run_chatts_timeseriesexam.sh
```

若一个旧 checkpoint 只有 768 维 projector 且未保存 patch size，单凭权重无法区分
Chronos-2 与 Zeus。只在这种情况下补充：

```bash
TS_ENCODER_TYPE=chronos2 bash scripts/run_chatts_timeseriesexam.sh
# 或 TS_ENCODER_TYPE=zeus
```

## Prompt 变体

默认是官方 shell 脚本采用的 `hint + concepts + examples`。要复现实验表中的
query-only 条件：

```bash
ADD_QUESTION_HINT=0 \
ADD_CONCEPTS=0 \
ADD_EXAMPLES=0 \
bash scripts/run_chatts_timeseriesexam.sh
```

其他消融可分别切换三个变量。不同设置会写入不同的结果目录，不会把断点混在一起。

不建议开启 `ENABLE_THINKING=1`：这会额外启用 Qwen3 的隐藏 thinking template，
并非 TimeSeriesExam 官方协议。官方 prompt 已经要求普通文本解释。

## 上下文长度与输出

默认：

```text
CHATTS_VLLM_MAX_MODEL_LEN=8192
MAX_NEW_TOKENS=1024
MAX_PROCESSED_INPUT_TOKENS=7168
```

脚本会在生成前按当前 encoder patch size 估算 ChatTS Processor 展开后的
“文本 + 时序 patch”总 token 数。超限样本记为 `skipped_input_length`，保留在严格
总分母中，不截断、不下采样。

输出目录示例：

```text
exp/timeseriesexam/
  my-chatts_query_hint_concepts_examples/
    generated_answer.json
    timeseriesexam_summary_my-chatts.json
    timeseriesexam_summary_my-chatts.csv
```

终端和汇总表包含：

- `Official`：官方 flexible accuracy，主复现指标；
- `Strict`：正确选项正文出现在回答最后一行；
- `LetterAcc`：稳健解析出的字母等于 gold；
- `Done`、`Parsed`：生成覆盖率与字母解析率；
- overall、category、subcategory、difficulty 四层统计。

只重建已有结果的汇总表：

```bash
SCORE_ONLY=1 \
MODEL_NAME=my-chatts \
OUTPUT_ROOT=/workspace/ChatTS/ChatTS-main/exp/timeseriesexam \
bash scripts/run_chatts_timeseriesexam.sh
```

## 常见问题

1. **为何没有手动归一化？** 训练和原 Dataset A/B 推理依赖 ChatTS Processor 的
   SP normalization；外部再做一次会改变输入语义。
2. **为何 checkpoint 仍要外部 TimesFM/Chronos/Zeus 目录？** Stage 2 通常只保存
   projector，冻结的主干没有重复打包进每个 ChatTS checkpoint。
3. **为何默认生成 1024 tokens，不只生成字母？** TimeSeriesExam 官方 one-shot
   prompt 明确要求解释，并用字母加选项正文评分；这里没有套用 TSRBench 的
   answer-only 模式。
4. **为什么 Official 与 LetterAcc 不同？** 模型可能只回答字母，或选项正文的格式
   与官方 substring 规则不一致。Official 用于论文协议复现，LetterAcc 用于诊断。
