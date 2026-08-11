# DeepSeek 时间序列 QA 模板标注

这个目录只有一个运行脚本和一个 YAML 配置。它从 JSONL 提取问题/答案模板，每个唯一模板
只调用一次 DeepSeek，再把 `quality`、`difficulty`、`reason` 展开到原始样本。

## 运行

依赖只有 `httpx` 和 `PyYAML`：

```bash
python -m pip install httpx pyyaml
python label_tsqa.py --config label_config.yaml --dry-run
python label_tsqa.py --config label_config.yaml
```

数据路径、输出路径和 DeepSeek 参数全部写在 [`label_config.yaml`](label_config.yaml)。路径相对
于 YAML 文件所在目录解析。一个 YAML 可以在 `datasets` 下配置多个数据集。
当前配置已纳入 `datataste/data/versions/datav2` 的全部 16 个规范数据集，共 794,235 条；
没有重复加入 `raw/`、`chatts/`、datav0 和 datav1 中的同源副本。
本地全量扫描得到 229,834 个安全模板，整体减少到原逐条请求量的约 28.9%（3.456×
压缩），且 16 个数据集均为 0 坏行。逐数据集结果见
[`template_stats.json`](template_stats.json)。

服务器上先试跑单个数据集的 100 条：

```bash
python label_tsqa.py --config label_config.yaml \
  --dataset time_r1 --limit 100
```

`--dataset` 可以重复传入；不传时按 YAML 顺序运行全部数据集。`--limit` 只临时覆盖本次运行，
不会修改 YAML。

## 上传服务器后

上传包只包含脚本、配置、说明和本地模板统计，不重复打包约 8GB 的数据。服务器应保持：

```text
/workspace/SFT_CURATOR
/workspace/datataste
```

解压并先验证路径和服务：

```bash
mkdir -p /workspace/SFT_CURATOR
tar -xzf SFT_CURATOR-deepseek-template-labeler-20260810.tar.gz \
  -C /workspace/SFT_CURATOR
cd /workspace/SFT_CURATOR
python -m pip install httpx pyyaml

python label_tsqa.py --config label_config.yaml \
  --dry-run --dataset time_r1 --limit 100
```

正式后台运行：

```bash
mkdir -p labels
nohup env PYTHONUNBUFFERED=1 python label_tsqa.py \
  --config label_config.yaml > labels/label-all.log 2>&1 &
tail -f labels/label-all.log
```

连接中断不影响进程；任务中止后重新执行同一条命令即可 resume。

## 输入格式

每行一个 JSON 对象，支持两种字段名：

```json
{"sample_id":"可选","input":"问题 <ts><ts/>","timeseries":[[1.0,2.0]],"output":"答案"}
{"sample_id":"可选","question":"问题","context":"可选上下文","response":"答案"}
```

顶层 `timeseries` 不会发送给 DeepSeek。问题中的 `<ts><ts/>` 和问题/答案里的长数字序列
会转换成 `<ts>`；问题与代表答案中的实例标量会转换成 `<num>`，选择标签会转换成 `<label>`。
DeepSeek 实际看到的是：

```json
{
  "question_template": "Forecast the next <num> values from <ts>",
  "timeseries": "<ts>",
  "answer_structure": "tagged_time_series",
  "representative_answer_template": "<answer><ts></answer>"
}
```

模板键由“归一化问题模板 + 粗粒度答案结构”共同确定。答案的实例措辞不参与模板 ID，
但 `time_series / choice_label / reasoning_with_answer / text` 等不同结构不会被错误合并。
DeepSeek 会看到该组第一条答案的脱敏模板作为格式代表；因此这里评估的是模板级训练质量，
不是逐样本隐藏数值正确性。

## 输出与 Resume

假设配置的输出是 `labels/time_r1.quality-difficulty.jsonl`，脚本会产生：

- `time_r1.quality-difficulty.jsonl`：逐样本标签；
- `time_r1.quality-difficulty.templates.jsonl`：模板标签缓存，每个模板只请求一次；
- `time_r1.quality-difficulty.errors.jsonl`：坏行和失败请求。

逐样本输出示例：

```json
{"record_id":"s1","template_id":"...","quality":"good","difficulty":"hard","reason":"..."}
```

重复运行时会同时复用模板缓存和逐样本结果：成功模板不再请求 DeepSeek，已经写出的样本
不再重复展开；失败模板会重试。缓存同时校验 `model + prompt_version`，更换模型或 prompt
不会误用旧标签。`--dry-run` 不调用模型，只报告样本数、唯一模板数、预计 API 请求数和
模板压缩比例。

## 合并能力、质量和难度标签

全部质量标注跑完后，使用 [`merge_tsqa_annotations.py`](merge_tsqa_annotations.py) 和
[`merge_config.yaml`](merge_config.yaml) 将每条 QA 关联为：

```json
{
  "annotation_id": "time_r1:1",
  "ability_label": "TSF",
  "ability_bucket": "TSF",
  "ability_name": "time_series_forecasting",
  "ability_major": "prediction",
  "quality": "good",
  "difficulty": "hard",
  "quality_reason": "...",
  "ability_label_source": "final",
  "ability_join_method": "taxonomy_direct_hash"
}
```

先只生成不含 `timeseries` 的轻量 sidecar 和联合分布（推荐先做这一步）：

```bash
python merge_tsqa_annotations.py --config merge_config.yaml --dry-run
python merge_tsqa_annotations.py --config merge_config.yaml --labels-only
```

确认分布后，生成“原 QA + 标签”的逐来源完整 JSONL：

```bash
python merge_tsqa_annotations.py --config merge_config.yaml
```

完整 JSONL 会再占用约一份 datav2 的空间。任务按来源原子写入并支持 resume；中断后重复执行
同一命令即可。`--dataset time_r1` 可只合并一个来源，`--force` 可强制重建逐来源结果。

默认输出目录是 `merged_labels/`：

- `annotations/*.jsonl`：逐 QA 轻量标签 sidecar，不复制原始序列；
- `annotated/*.jsonl`：原始 `input/timeseries/output` 加顶层标签字段；
- `reports/DISTRIBUTION.md`：可直接阅读的总体摘要；
- `reports/ability_quality_difficulty.csv`：15 维能力 × 质量 × 难度完整立方体；
- `reports/source_ability_quality_difficulty.csv`：再按数据来源展开；
- `reports/coverage_by_source.csv`：逐来源覆盖率、标签来源和连接方式；
- `reports/distribution.json`：机器可读汇总；
- `taxonomy_labels.sqlite`：能力标注的可复用轻量索引。

连接不是简单按行号硬拼：原始 ChatTS 清洗来源按有效行顺序并校验规范化内容；未清洗来源
校验原始行 SHA-256；质量标签还会重新计算 `template_id/input_hash`。当前
`opentslm_ecg_qa_cot` 数据有 21,817 条，而旧能力结果只对应 11,543 条旧内容，因此该来源
会明确记录为 `audit_fallback`，使用当前逐行 audit 的 `primary_label`，不会误接旧标签。

能力标签采用 `final.primary_label` → `final.proposed_primary_label` →
`provisional.primary_label` 的优先级。明确 `out_of_scope` 的 QA 不会被强塞到某个能力维度，
其 `ability_label` 为 `null`，报表中归到 `UNMAPPED`；因此报表同时给出总 QA 数与“15 维有效
覆盖率”。为方便直接分组，每条记录还包含必有值的 `ability_bucket`：正常样本等于
`ability_label`，超出范围的样本为 `UNMAPPED`。

## 绘制每个数据集的质量和难度分布

合并完成后，[`plot_dataset_distributions.py`](plot_dataset_distributions.py) 直接读取已经聚合
好的 `source_ability_quality_difficulty.csv`，不重新扫描原始时间序列。安装绘图库并运行：

```bash
python -m pip install matplotlib

python plot_dataset_distributions.py \
  --input merged_labels/reports/source_ability_quality_difficulty.csv \
  --output-dir merged_labels/reports/dataset_plots
```

每个数据集单独生成两张 PNG：

```text
merged_labels/reports/dataset_plots/
├── time_r1/
│   ├── quality_distribution.png
│   └── difficulty_distribution.png
├── opentslm_tsqa/
│   ├── quality_distribution.png
│   └── difficulty_distribution.png
└── ...
```

每张图展示 5 个等级的样本数，并在柱顶标出数量和百分比。目录下同时生成：

- `dataset_distribution_summary.csv`：长表，每个数据集的每个等级一行；
- `dataset_distribution_wide.csv`：宽表，每个数据集一行，包含质量和难度各等级的数量与百分比；
- `manifest.json`：机器可读清单和完整分布。

只画一个或几个数据集时可重复传入 `--dataset`：

```bash
python plot_dataset_distributions.py \
  --dataset time_r1 \
  --dataset opentslm_tsqa
```
