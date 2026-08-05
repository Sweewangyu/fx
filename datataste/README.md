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
  → DeepSeek 或其他模型复核
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

### 使用 DeepSeek V4 Flash 复核模板

不要把密钥写入脚本或配置：

```bash
export DEEPSEEK_API_KEY='在终端中填写新密钥'

python scripts/annotate_tsr_taxonomy.py annotate-online \
  --input artifacts/tsr-taxonomy/review_clusters.jsonl \
  --output artifacts/tsr-taxonomy/vote-deepseek-v4-flash.jsonl \
  --base-url https://api.deepseek.com \
  --model deepseek-v4-flash \
  --api-key-env DEEPSEEK_API_KEY \
  --workers 8 \
  --json-mode \
  --disable-thinking
```

### 解析规则、模型和人工标签

```bash
python scripts/annotate_tsr_taxonomy.py resolve \
  --provisional artifacts/tsr-taxonomy/provisional_labels.jsonl \
  --clusters artifacts/tsr-taxonomy/review_clusters.jsonl \
  --votes artifacts/tsr-taxonomy/vote-model-a.jsonl \
          artifacts/tsr-taxonomy/vote-model-b.jsonl \
  --human artifacts/tsr-taxonomy/human-labels.csv \
  --output artifacts/tsr-taxonomy/final_labels-v2.jsonl
```

### 物化 15 类训练集

```bash
python scripts/annotate_tsr_taxonomy.py materialize \
  --registry configs/tsr_annotation_sources.json \
  --labels artifacts/tsr-taxonomy/final_labels-v2.jsonl \
  --output-dir data/chatts/tsr15-v2 \
  --splits train \
  --min-confidence 0.85 \
  --include-fit exact compatible
```

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
