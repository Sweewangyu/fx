# DataTaste：TSRBench 训练数据处理工作区

这个目录集中保存 ChatTS、Time-MQA/TSQA、TSAQA 的下载数据、格式转换、TSRBench 4×15 标注、测试、技术文档和运行产物。

所有命令默认从本目录运行：

```bash
cd datataste
```

## 目录结构

```text
datataste/
├── README.md
├── configs/
│   └── tsr_annotation_sources.json
├── scripts/
│   ├── convert_tsqa_to_chatts.py
│   └── annotate_tsr_taxonomy.py
├── tests/
│   ├── test_convert_tsqa_to_chatts.py
│   └── test_tsr_taxonomy_annotation.py
├── docs/
│   ├── convert-tsqa-to-chatts.md
│   ├── annotate-tsr-taxonomy.md
│   └── tsrbench-4x15-annotation-pipeline-technical.md
├── data/
│   ├── raw/
│   │   ├── chatts_training/
│   │   ├── time_mqa_tsqa/
│   │   └── tsaqa/
│   └── chatts/
│       ├── time_mqa_train.jsonl
│       ├── time_mqa_train.audit.jsonl
│       ├── time_mqa_train.manifest.json
│       ├── tsaqa_train.jsonl
│       ├── tsaqa_train.audit.jsonl
│       └── tsaqa_train.manifest.json
└── artifacts/
    └── tsr-taxonomy/
        ├── provisional_labels.jsonl
        ├── review_clusters.jsonl
        ├── final_labels.jsonl
        ├── human-labels.csv
        ├── annotation_state.sqlite
        ├── prepare_manifest.json
        └── smoke-e2e/
```

## 数据处理流程

```text
下载原始数据
  → 转换成 ChatTS input/timeseries/output 三字段
  → 数据契约与坏行检查
  → TSRBench 元信息/规则初标
  → 模板聚类
  → Qwen 首轮复核
  → DeepSeek V4 Flash 权威裁决不确定模板
  → 人工仲裁
  → 最终标签解析
  → 物化 15 类 ChatTS 训练文件
```

## 常用命令

### 运行测试

```bash
python -m unittest \
  tests/test_tsr_taxonomy_annotation.py \
  tests/test_convert_tsqa_to_chatts.py
```

### 重新执行全量第一轮标注

建议使用新的输出目录保留旧版本：

```bash
python scripts/annotate_tsr_taxonomy.py prepare \
  --registry configs/tsr_annotation_sources.json \
  --output-dir artifacts/tsr-taxonomy-v2
```

### 使用 DeepSeek V4 Flash 思考模式裁决不确定模板

不要把密钥写入脚本或配置：

```bash
export DEEPSEEK_API_KEY='在终端中填写新密钥'

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

`--reasoning-effort` 支持 `high` 和 `max`；省略时使用服务端默认强度。思考正文不会写入投票文件，只记录是否返回了 `reasoning_content`。`content` 中的最终 JSON 才参与标签解析。

### 解析规则、模型和人工标签

```bash
python scripts/annotate_tsr_taxonomy.py resolve \
  --provisional artifacts/tsr-taxonomy/provisional_labels.jsonl \
  --clusters artifacts/tsr-taxonomy/review_clusters.jsonl \
  --votes artifacts/tsr-taxonomy/votes-qwen36-all.jsonl \
  --authoritative-votes artifacts/tsr-taxonomy/vote-deepseek-v4-flash-authoritative.jsonl \
  --human artifacts/tsr-taxonomy/human-labels.csv \
  --output artifacts/tsr-taxonomy/final_labels-v2.jsonl
```

解析优先级为：人工覆盖 > 规则 `auto_accept` > DeepSeek 权威票 > 普通模型共识 > 单模型与规则一致 > `human_review`。DeepSeek 权威票保留模型名、真实置信度、思考模式和思考强度，不伪装成人工标签。

### 物化 15 类训练集

```bash
python scripts/annotate_tsr_taxonomy.py materialize \
  --registry configs/tsr_annotation_sources.json \
  --labels artifacts/tsr-taxonomy/final_labels-v2.jsonl \
  --output-dir data/chatts/tsr15-v2-k8 \
  --splits train \
  --min-confidence 0.85 \
  --include-fit exact compatible \
  --max-per-template 8 \
  --template-cap-sources time_mqa tsaqa \
  --template-sample-seed 42
```

`--max-per-template 8` 表示每个“数据源 + 归一化问题模板”最多保留 8 条合格样本。程序按 `seed + sample_id` 的 SHA-256 排序取样，不会偏向源文件开头；同一 seed 重跑得到相同结果。这里建议只限制模板化较强的 Time-MQA 和 TSAQA，ChatTS alignment/SFT 不受影响。若希望所有训练源都执行 K 上限，省略 `--template-cap-sources`。

输出目录的 `manifest.json` 会记录候选数、保留数、过滤数、被截断模板簇数、最大模板簇，以及按数据源和 15 类拆分的过滤前后数量。首轮建议 `K=8`，再做 `K=4/8/16` 消融；Stage 1 alignment 不建议使用激进模板截断。

### 分数据集查看 15 类分布

将五个 ChatTS 子集合并为 ChatTS，并与 Time-MQA、TSAQA 分列统计。默认只统计 `train`，15 类占比以各数据集已经接受且具有合法主标签的样本为分母：

```bash
python scripts/annotate_tsr_taxonomy.py report-distribution \
  --labels artifacts/tsr-taxonomy/final_labels.jsonl \
  --splits train \
  --output-json artifacts/tsr-taxonomy/distribution-by-dataset.json \
  --output-csv artifacts/tsr-taxonomy/distribution-by-dataset.csv
```

终端会打印 ChatTS、Time-MQA、TSAQA 的 15 类数量和组内占比，并单独显示每个数据集的 `accepted`、`excluded`、`human_review`，避免把未解决样本混入能力分布。

## 当前状态

- 原始总行数：550,104；
- 有效样本：549,971；
- 隔离坏行：133；
- 模板簇：203,262；
- 规则自动接收：283,146；
- 待复核模板：93,301；
- DeepSeek 冒烟测试：通过；
- 全量双模型复核和人工仲裁：尚未执行；
- 正式 15 类训练集物化：应等待复核完成。

## 文档入口

- `docs/tsrbench-4x15-annotation-pipeline-technical.md`：按 A→O 数据流逐步解释完整标注管线；
- `docs/annotate-tsr-taxonomy.md`：简明标注操作手册；
- `docs/convert-tsqa-to-chatts.md`：Time-MQA/TSAQA 转 ChatTS 格式说明；
- `artifacts/tsr-taxonomy/smoke-e2e/SMOKE_TEST_REPORT.md`：DeepSeek API 端到端冒烟测试报告。

`data/` 和 `artifacts/` 体积较大，已在 `.gitignore` 中忽略。历史 audit/manifest 中的绝对路径已经更新到当前 `datataste` 位置；核心注册表使用相对路径，只要从本目录运行即可。
