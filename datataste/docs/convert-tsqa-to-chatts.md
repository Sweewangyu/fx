# 将 Time-MQA / TSAQA 转成 ChatTS 训练格式

转换器输出 ChatTS Training Dataset 的三字段 JSONL：

```json
{"input":"... <ts><ts/> ...","timeseries":[[1.0,2.0,3.0]],"output":"..."}
```

`input` 中的 `<ts><ts/>` 数量、顺序与 `timeseries` 中的序列严格一致。来源、任务、domain 和原始行号不会混入模型输入，而是单独写入 `*.audit.jsonl`。

## 1. 安装依赖

TSAQA 使用 Parquet，因此至少需要：

```bash
python3 -m pip install 'huggingface_hub>=0.27' 'pyarrow>=14,<22'
```

## 2. 下载训练集

```bash
# TSAQA：只下载 train.parquet，不下载 val/test
python3 scripts/convert_tsqa_to_chatts.py download \
  --dataset tsaqa \
  --output-dir data/raw/tsaqa \
  --revision main

# Time-MQA 是 gated dataset：先在网页同意协议，再登录
hf auth login
python3 scripts/convert_tsqa_to_chatts.py download \
  --dataset time-mqa \
  --output-dir data/raw/time_mqa \
  --revision main
```

正式训练应将 `--revision main` 改成下载时对应的 Hugging Face commit SHA，以保证复现。

默认不下载 Time-MQA 的 `Classification/classification.csv`。它含人体活动/传感器分类源，与 TSRBench 使用的部分源数据存在重合风险。只有明确完成样本级去污染后，才考虑加入 `--include-contaminated`。

## 3. 先检查实际字段

```bash
python3 scripts/convert_tsqa_to_chatts.py inspect \
  --input data/raw/time_mqa \
  --rows 3
```

Time-MQA 需要 Hugging Face 登录后才能取得 CSV。若后续仓库字段改变，先用该命令确认 `question/answer` 和数组的位置，不要静默转换。

## 4. 转换

```bash
python3 scripts/convert_tsqa_to_chatts.py convert \
  --dataset tsaqa \
  --input data/raw/tsaqa/train.parquet \
  --output data/chatts/tsaqa_train.jsonl

python3 scripts/convert_tsqa_to_chatts.py convert \
  --dataset time-mqa \
  --input data/raw/time_mqa \
  --output data/chatts/time_mqa_train.jsonl
```

每次转换产生三份文件：

- `*_train.jsonl`：可直接给 ChatTS-Training 的 SFT 数据。
- `*_train.audit.jsonl`：来源文件、行号、task、domain、序列长度及哈希。
- `*_train.manifest.json`：过滤数量、无效原因、任务分布和转换参数。

建议先做 100 条冒烟测试：

```bash
python3 scripts/convert_tsqa_to_chatts.py convert \
  --dataset tsaqa \
  --input data/raw/tsaqa/train.parquet \
  --output data/chatts/tsaqa_smoke.jsonl \
  --limit 100 \
  --fail-fast \
  --preview 3
```

## 5. 两个数据集的转换差异

TSAQA 的主序列来自 `input_ts`。Temporal、Puzzling、Data Transformation 等题还会把候选序列直接写在 `question` 中；转换器会把这些候选数组也抽出，并按题面出现顺序追加到 `timeseries`。`raw_ts` 是上游原始信号，和题面中经过变换的候选值不一定相同，因此转换器不会用它替换题面数组。

Time-MQA 官方 CSV 使用 `application_domain + task_type + QA_list` 字段，其中 `QA_list` 是包含 `question/answer` 的 JSON 片段；转换器会先拆出问答，再解析 question 中的数值数组并替换成 `<ts><ts/>`，答案保留为纯文本 `output`。对于 `X/NaN/None` 缺失值，默认生成两条等长序列：用 0 临时填充的值序列，以及 `1=观测、0=缺失` 的 mask；这样不会把 NaN 送进 ChatTS Processor。若不想使用 mask，可加 `--missing-policy drop` 丢弃这类样本。

## 6. 默认去污染行为

- 输入目录中的 `val/dev/test` 文件不会读取。
- Time-MQA 的整个 Classification 文件默认排除。
- TSAQA 中 `dataset` 字段含 `sunspot` 的样本默认排除。
- 完全相同的转换后样本会按 SHA256 去重。

`--include-nontrain` 和 `--include-contaminated` 可以关闭相应保护，但为了刷 TSRBench 榜，不建议使用。转换器只做已知源级过滤；正式训练前仍应将 audit 数据与 TSRBench valid/test 做序列指纹和问题模板的样本级查重。
