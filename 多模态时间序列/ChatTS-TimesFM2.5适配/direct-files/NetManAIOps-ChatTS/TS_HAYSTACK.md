# ChatTS 评测 TS-Haystack

这个适配不重写 benchmark，而是直接导入
[`AI-X-Labs/TS-Haystack`](https://github.com/AI-X-Labs/TS-Haystack)
的官方 `QADataset`：

- 官方代码负责加载或重建原始信号；
- 官方 `pre_prompt` / channel label / `post_prompt` 保持不变；
- ChatTS 通过 vLLM `multi_modal_data["timeseries"]` 接收每个一维通道；
- 每条回答都由官方 `extract_answer()` 和 `evaluate_answer()` 解析/打分；
- 时间段 IoU、单时刻容差、boolean/integer/category 规则均使用官方实现，不使用 judge API。

支持的五个运行入口（Sleep 分两个 label class）：

| `DATASETS` 名称 | 官方 dataset class | 通道 / 频率 |
|---|---|---|
| `capture24` | `capture24_haystack_cot` | 3 / 100 Hz |
| `sleep_stages` | `sleep_psg_haystack` | 13 / 100 Hz |
| `sleep_arousals` | `sleep_psg_haystack` | 13 / 100 Hz |
| `ltaf` | `ltaf_haystack` | 2 / 128 Hz |
| `uk_dale` | `uk_dale_haystack` | 1 / 6 s 网格 |

## 1. 文件覆盖

把本目录下的文件按相同相对路径复制到 ChatTS：

```text
chatts/utils/inference_ts_haystack_vllm.py
chatts/utils/llm_utils.py
scripts/evaluate_ts_haystack.py
scripts/run_chatts_ts_haystack.sh
```

同时需要本适配已有的：

```text
chatts/vllm/chatts_vllm.py
chatts/vllm/zeus_modeling.py
scripts/inspect_chatts_ts_encoder_checkpoints.py
```

`llm_utils.py` 这次也必须覆盖。它除了保留之前的自定义
`SamplingParams` / `CHATTS_VLLM_MAX_MODEL_LEN` 修复，还把
`config/datagen_config.yaml` 改为相对 ChatTS 项目根目录解析。否则官方
TS-Haystack loader 切换工作目录后，vLLM spawn worker 会在启动阶段报
`config/datagen_config.yaml` 不存在。

## 2. 官方代码与数据

`TS_HAYSTACK_ROOT` 必须指向完整的官方仓库，不是某一个 parquet
文件夹：

```bash
git clone https://github.com/AI-X-Labs/TS-Haystack.git /workspace/TS-Haystack
```

推理脚本会直接把这个 checkout 加到 `sys.path`，不需要把
`ts-haystack` 安装成 Python package。但是官方 dataset loader 依赖必须装在
**实际运行 ChatTS/vLLM 的同一个 conda 环境**中：

```bash
# 先在你现有的 ChatTS Python 3.11 环境里执行；
# torch/numpy 应已存在。
python -c "import datasets, pyarrow, pandas, scipy, wfdb, matplotlib, torch"

# 只在上面报 ModuleNotFoundError 时安装缺失项。不要带 --upgrade，
# pip 默认的 only-if-needed 策略会保留已满足的 vLLM 环境依赖。
pip install datasets pyarrow pandas scipy wfdb matplotlib
```

官方当前 `pyproject.toml` 声明 Python `>=3.12`，而你现有 ChatTS 服务器
是 Python 3.11。因此不建议在这个环境直接 `pip install -e
/workspace/TS-Haystack`；本适配只导入 dataset 子模块，所用语法可在
Python 3.11 运行。若你需要运行官方训练栈，再用它的 `uv sync`
创建独立 Python 3.12 环境。

数据下载地址是官方
[Hugging Face collection](https://huggingface.co/collections/nz00shuuuu/ts-haystack)。
官方 README 给出的总入口是：

```bash
python scripts/data/download_from_hf.py
```

> 注意：我们检查的当前官方 `main` 中，README 引用了这个文件，
> 但公开 checkout 里并没有 `scripts/data/download_from_hf.py`。如果你的版本也
> 没有，请从 collection 的四个数据仓库下载原始文件，按各自
> dataset card 的目录放到下面路径。不要只下载 HF Dataset Viewer
> 自动转换的 `refs/convert/parquet`，它不包含动态重建信号所需的 sidecar。

运行时官方 loader 期待至少看到：

```text
/workspace/TS-Haystack/data/
├── capture24/ts_haystack/cot/{context}/{task}/{split}/data.parquet
├── sleep_psg/
│   ├── ts_haystack/{sleep_stages|arousals}/tasks/...
│   └── training_100hz/{subject}/{subject}.npy
├── ltafdb/
│   ├── ltaf_haystack/rhythms/tasks/...
│   └── training/{record}/{record}.npy
└── uk_dale/uk_dale_haystack/
    ├── tasks/{context}/{task}/{split}/data.parquet
    ├── signals/h{house}/m{meter}.npy
    ├── signals/h{house}/m{meter}.t.npy
    └── manifest.json
```

Capture24 的 `signals.npy` sidecar 可与 `data.parquet` 同目录；如果信号已内联
在 parquet 中，官方 loader 也能读。UK-DALE 的 `tasks/` 只存元数据，
`signals/` 与 manifest 是必需文件。

## 3. 先跑小样本

```bash
PROJECT_ROOT=/workspace/ChatTS/ChatTS-main \
TS_HAYSTACK_ROOT=/workspace/TS-Haystack \
MODEL_PATH=/workspace/checkpoints/my-chatts \
MODEL_NAME=my-chatts \
DATASETS="capture24 uk_dale" \
TASKS="existence localization" \
CONTEXT_LENGTHS="100 900" \
MAX_SAMPLES=20 \
NUM_GPUS=8 \
NUM_GPUS_PER_PROCESS=2 \
bash scripts/run_chatts_ts_haystack.sh
```

完整测试集：

```bash
PROJECT_ROOT=/workspace/ChatTS/ChatTS-main \
TS_HAYSTACK_ROOT=/workspace/TS-Haystack \
MODEL_PATH=/workspace/checkpoints/my-chatts \
MODEL_NAME=my-chatts \
DATASETS=all TASKS=all CONTEXT_LENGTHS=all SPLIT=test \
bash scripts/run_chatts_ts_haystack.sh
```

默认参数：

```text
temperature=0.0
max_new_tokens=500
Qwen3 enable_thinking=False
max_model_len=40960
batch_size=1
seed=42
```

这里没有把官方 prompt 改成 answer-only。官方 post-prompt 本身要求
step-by-step paragraph 并以 `Answer: <your answer>` 结尾；适配仅关闭
Qwen3 chat template 额外的 hidden thinking 模式。如果需要显式开启：

```bash
ENABLE_THINKING=1 bash scripts/run_chatts_ts_haystack.sh
```

## 4. MLP / TimesFM 2.5 / Chronos-2 / Zeus

一键脚本先用 checkpoint 权重检查器自动识别 encoder，四种后端使用
同一份 TS-Haystack 输入脚本。外部 backbone 仍然需要可访问的本地
路径：

```bash
# TimesFM 2.5
TIMESFM_MODEL_PATH=/workspace/timesf \
bash scripts/run_chatts_ts_haystack.sh

# Chronos-2（只有在 768-d projector 且 checkpoint 没元数据时才必须写 type）
TS_ENCODER_TYPE=chronos2 \
CHRONOS2_MODEL_PATH=/workspace/chronos-2 \
bash scripts/run_chatts_ts_haystack.sh

# Zeus
TS_ENCODER_TYPE=zeus \
ZEUS_MODEL_PATH=/workspace/zeus \
bash scripts/run_chatts_ts_haystack.sh

# 原始 ChatTS MLP-Patch 会自动识别为 native
```

不要在这个脚本里再做一次手工归一化。原始通道会交给 ChatTS
processor 执行 SP 编码；Sleep PSG 的每通道 z-score 是官方 loader
本身定义的数据处理，适配层不更改它。

## 5. 长序列与覆盖率

多模态 token 数大致为：

```text
通道数 × ceil(每通道采样点数 / encoder patch_size) + 文本 tokens
```

Sleep PSG 有 13 个 100 Hz 通道，长窗口很容易超出 Qwen/ChatTS
上下文。默认行为是“保留原信号并记录跳过”，不会暗中下采样、
截断或分块后冒充官方结果。输出中会看到：

```json
{"status": "skipped_input_length", "patch_token_lower_bound": 73125}
```

若 checkpoint 的真实上下文和 GPU 允许，可同时提高：

```bash
CHATTS_VLLM_MAX_MODEL_LEN=65536 \
MAX_PROCESSED_INPUT_TOKENS=65036 \
bash scripts/run_chatts_ts_haystack.sh
```

不能只改 `MAX_PROCESSED_INPUT_TOKENS`；它与 `MAX_NEW_TOKENS` 之和不能超过
`CHATTS_VLLM_MAX_MODEL_LEN`。

## 6. 结果

每个 domain 的逐样本记录：

```text
${OUTPUT_ROOT}/{dataset}_${MODEL_NAME}/generated_answer.json
```

汇总：

```text
${OUTPUT_ROOT}/ts_haystack_summary_${MODEL_NAME}.json
${OUTPUT_ROOT}/ts_haystack_summary_${MODEL_NAME}.csv
```

CSV 同时包含 overall、per-dataset、per-task 和 per-context 行，并保留
mean IoU 和 mean timestamp error。终端主表的：

- `Strict = correct / 选中的全部样本`，输入过长也留在分母中；
- `GenAcc = correct / 真正生成的样本`，用于区分模型正确率与上下文覆盖率。

只重建汇总表：

```bash
SCORE_ONLY=1 \
MODEL_NAME=my-chatts \
TS_HAYSTACK_ROOT=/workspace/TS-Haystack \
bash scripts/run_chatts_ts_haystack.sh
```
