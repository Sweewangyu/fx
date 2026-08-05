# 将 QA 训练集标注为 TSRBench 4 大类 / 15 小类

## 1. 标注对象与核心原则

标注对象是“回答问题所需的主要能力”，不是数据领域、答案中的关键词，也不是序列长相。每条样本保留四组字段：

- `primary_label`：唯一主标签，训练分桶只使用它。
- `secondary_labels`：确实需要多种能力时记录，不能代替主标签。
- `taxonomy_fit`：`exact`、`compatible`、`closest`、`mixed` 或 `out_of_scope`。
- `confidence`、`method`：用于质量过滤和审计。

不要强行把所有 QA 塞进 15 类。缺失值插补通常是 `out_of_scope`（`closest_label=TSF`）；没有时间序列语义的普通类别预测通常只是 `closest=PR`；一道题同时要求趋势、异常、原因与决策时标为 `mixed`，拆题后再用于单任务训练。

## 2. 官方 4×15 标签及判定问题

| 大类 | 标签 | 标注时问自己 |
|---|---|---|
| Perception | `PR` Pattern Recognition | 只需识别已观测序列的趋势、周期、平稳性、结构或核心统计特征吗？ |
| Perception | `NU` Noise Understanding | 核心要求是描述或量化随机噪声的尺度、幅度或形态吗？ |
| Perception | `AD` Anomaly Detection | 要寻找、定位或分类异常点/异常片段吗？ |
| Perception | `CA` Comparative Analysis | 要比较两个及以上序列的模式、分布、统计量、噪声或趋势吗？ |
| Reasoning | `ER` Etiological Reasoning | 要推断整段序列的生成来源或底层致因吗？ |
| Reasoning | `CD` Causal Discovery | 要判断多个序列间因果边的存在或方向吗？ |
| Reasoning | `AR` Abductive Reasoning | 要利用变化前后证据，找出解释局部变化的最可能隐事件吗？ |
| Reasoning | `TR` Temporal Relation Reasoning | 要定位事件并判断先后、重叠、持续或其他时间关系吗？ |
| Reasoning | `NR` Numerical Reasoning | 要结合上下文对序列数值做计算吗？ |
| Reasoning | `DR` Deductive Reasoning | 题目给定规则、公式或约束，要求推出必然结论吗？ |
| Reasoning | `IR` Inductive Reasoning | 要先从观察中归纳规则，再把规则用于新案例或未来事件吗？ |
| Prediction | `TSF` Time Series Forecasting | 输出是未来连续数值或一段未来数值序列吗？ |
| Prediction | `EP` Event Prediction | 输出是未来是否发生某个离散事件或事件类别吗？ |
| Decision-Making | `QualDM` Qualitative Decision-Making | 要根据序列和语境选择行动，但不需要定量模拟行动结果吗？ |
| Decision-Making | `QuantDM` Quantitative Decision-Making | 要定量模拟/比较不同操作的结果，再选择最优行动吗？ |

最容易混淆的边界：

- `PR` 描述已经发生的模式；`IR` 先归纳规则再外推；`TSF` 直接预测未来数值。
- `CA` 判断相关、相似或差异；`CD` 判断有方向的因果关系。
- `ER` 解释整段序列为何这样生成；`AR` 解释某个局部突变可能发生了什么。
- `NR` 计算一个数；`QuantDM` 计算多个行动后果并据此选行动。
- `EP` 预测将发生什么；`QualDM/QuantDM` 决定应该做什么。
- `DR` 的规则由题目预先给出；`IR` 的规则需要从样本中归纳。

## 3. 推荐标注管线

```text
数据源元信息 + 高精度规则
            ↓
按 source + 归一化问题模板聚类
            ↓
Qwen 首轮标低置信/冲突模板
            ↓
规则与 Qwen 一致则接收；不确定模板交给 DeepSeek 权威裁决
            ↓
人工覆盖 + 分层抽检
            ↓
生成标签索引，再物化成 15 个 ChatTS JSONL 训练桶
```

模板标签只能在同一数据源、同一任务定义内传播。标注者应同时看 `question`、`answer` 和数据集自带的 `task/question_type`；只有题意仍不清楚时才看原始序列。不要向标注者展示 TSRBench 测试题或测试答案，以免造成 benchmark contamination。

建议先人工精标每类至少 150 个模板，并让其中 10%–20% 由两位标注者独立完成。主标签 Cohen's kappa 达到 0.80、每类抽检精度达到 95% 后，才扩大自动传播范围。低频类别必须按类别抽样，不能只做总体随机抽检。

## 4. 本项目中的执行方式

源文件注册表是 `configs/tsr_annotation_sources.json`，管线脚本是 `scripts/annotate_tsr_taxonomy.py`。

### 第一步：规则初标和模板聚类

```bash
python scripts/annotate_tsr_taxonomy.py prepare \
  --registry configs/tsr_annotation_sources.json \
  --output-dir artifacts/tsr-taxonomy
```

主要输出：

- `provisional_labels.jsonl`：逐样本标签索引，不复制时序数组。
- `review_clusters.jsonl`：只含需复核的去重模板及代表样本。
- `invalid_source_rows.jsonl`：原始 JSONL 坏行审计；坏行不会进入训练桶。
- `annotation_state.sqlite`：模板簇和成员数，可用于统计与恢复。
- `prepare_manifest.json`：样本、标签、适配度和状态分布。

### 第二步：Qwen 首轮标注模糊模板

管线兼容任何 OpenAI-compatible `/chat/completions` 服务。Qwen 先处理全部待复核模板：

```bash
python scripts/annotate_tsr_taxonomy.py annotate-online \
  --input artifacts/tsr-taxonomy/review_clusters.jsonl \
  --output artifacts/tsr-taxonomy/votes-qwen36-all.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model QWEN_MODEL --allow-no-key --json-mode

python scripts/annotate_tsr_taxonomy.py resolve \
  --provisional artifacts/tsr-taxonomy/provisional_labels.jsonl \
  --clusters artifacts/tsr-taxonomy/review_clusters.jsonl \
  --votes artifacts/tsr-taxonomy/votes-qwen36-all.jsonl \
  --output artifacts/tsr-taxonomy/final_labels.qwen.jsonl
```

未解决模板会写到 `final_labels.qwen.human_review.jsonl`。

### 第三步：DeepSeek V4 Flash 权威裁决

DeepSeek 只读取 Qwen 未解决模板。使用思考模式时，`high` 是默认深度，`max` 用于更强推理：

```bash
python scripts/annotate_tsr_taxonomy.py annotate-online \
  --input artifacts/tsr-taxonomy/final_labels.qwen.human_review.jsonl \
  --output artifacts/tsr-taxonomy/vote-deepseek-v4-flash-authoritative.jsonl \
  --base-url http://localhost:30000/v1 \
  --model /models \
  --allow-no-key \
  --workers 8 \
  --max-tokens 2048 \
  --json-mode \
  --thinking-mode enabled \
  --reasoning-effort max
```

投票文件保存请求采用的思考模式、强度以及是否返回 `reasoning_content`，但不保存思考正文。只有最终 `content` 中的 JSON 参与解析。思考模式需要为最终 JSON 留出输出空间；如果出现 `model response has no final content`，应提高 `--max-tokens`，而不是把空内容当成标签。

### 第四步：解析普通票、权威票和人工覆盖

```bash
python scripts/annotate_tsr_taxonomy.py resolve \
  --provisional artifacts/tsr-taxonomy/provisional_labels.jsonl \
  --clusters artifacts/tsr-taxonomy/review_clusters.jsonl \
  --votes artifacts/tsr-taxonomy/votes-qwen36-all.jsonl \
  --authoritative-votes artifacts/tsr-taxonomy/vote-deepseek-v4-flash-authoritative.jsonl \
  --output artifacts/tsr-taxonomy/final_labels.jsonl
```

解析顺序是：人工覆盖 > 规则 `auto_accept` > DeepSeek 权威票 > 普通模型共识 > 单模型与规则一致 > `human_review`。权威票的置信度不会被改成 1.0；低置信结果仍可在物化时通过 `--min-confidence` 排除。

如果 DeepSeek 请求失败或没有合法权威票，该模板继续进入人工复核。

### 第五步：人工覆盖（可选）

人工覆盖文件采用 JSONL，每行至少包含：

```json
{"cluster_id":"...","primary_label":"AD","secondary_labels":["TR"],"taxonomy_fit":"exact","rationale":"要求定位异常片段"}
```

排除样本使用 `"primary_label": null, "taxonomy_fit": "out_of_scope"`。

如果希望在 Excel/Numbers 中标注，可导出 CSV：

```bash
python scripts/annotate_tsr_taxonomy.py export-human \
  --input artifacts/tsr-taxonomy/final_labels.human_review.jsonl \
  --output artifacts/tsr-taxonomy/human-labels.csv
```

填写后再次执行 `resolve` 并加上 `--human artifacts/tsr-taxonomy/human-labels.csv`。

### 第六步：物化训练桶

```bash
python scripts/annotate_tsr_taxonomy.py materialize \
  --registry configs/tsr_annotation_sources.json \
  --labels artifacts/tsr-taxonomy/final_labels.jsonl \
  --output-dir data/chatts/tsr15 \
  --splits train \
  --min-confidence 0.85 \
  --include-fit exact compatible
```

输出的 15 个文件仍严格保持 ChatTS 三字段格式。

## 5. 数据集级先验（不能替代逐题判断）

- ChatTS alignment：以 `PR/NU/AD/CA` 为主；多指标领域题可能覆盖推理与决策。
- ChatTS IFT：经常是一题多问，应优先标 `mixed` 或拆题。
- Time-MQA：forecasting→`TSF`，anomaly detection→`AD`；imputation 应排除；generic classification 只记 `closest=PR`。
- TSAQA：anomaly detection→`AD`，comparison→`CA`，temporal relationship→`TR`；classification 多数只与 `PR` 相邻；data transformation 通常是 `compatible=DR`，仍需抽检具体操作。

不要用这些先验覆盖题目中的明确操作。例如标题属于 forecasting，但问题实际询问异常点时，主标签仍应是 `AD`。

## 6. 当前全量首轮结果

本地 3 组数据源（ChatTS、Time-MQA、TSAQA）共扫描 550,104 行：549,971 条有效样本，133 条 ChatTS 原始 JSONL 坏行已隔离。规则直接接收 283,146 条；剩余 266,825 条被压缩为 93,301 个待复核模板。

当前的 `final_labels.jsonl` 是保守中间结果，不是完整的 15 类金标：低置信度的 `PR/NR/DR/ER` 等样本会保持 `human_review`，不会被误当作负样本。应先完成模型投票和人工仲裁，再执行 `materialize`。如果只是做管线冒烟测试，可以物化当前自动接收子集，但不应把它当成类别均衡的最终训练集。
