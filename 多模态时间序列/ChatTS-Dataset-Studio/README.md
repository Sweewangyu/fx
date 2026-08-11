# ChatTS Dataset Studio

一个与 `SFT_CURATOR`、`datataste` 和训练仓库解耦的本地可视化数据筛选器。它读取
`datataste` 的数据注册表和 `SFT_CURATOR` 已合并的标签，按数据集、质量、难度、能力维度
组合规则，分别导出可直接交给 ChatTS 两阶段训练的 Stage1 / Stage2 数据。

真实 QA 不会送到浏览器：浏览器只接收聚合计数，JSONL 的扫描、筛选和写盘都在服务器端
流式完成。工具不调用模型、不联网，也不修改两个上游项目。

## 默认配方

页面首次扫描后会自动选中：

```text
chatts_align_256, chatts_align_random, chatts_ift,
chatts_sft, time_mqa, tsaqa
```

- Stage1：质量 `weak` 及以上，难度 `moderate` 及以下。
- Stage2：质量 `weak` 及以上，难度 `moderate` 及以上。
- `moderate` 会按需求同时进入两阶段；预览区会明确显示重叠数量。
- 能力维度默认不限制，可在页面中继续多选 15 维标签或 `UNMAPPED`。

## 先在本地只看格式

本地没有真实标签时，用 30 条明确标记为 demo 的合成记录启动页面：

```bash
cd /path/to/ChatTS-Dataset-Studio
PYTHON_BIN=/path/to/python3.10-or-newer bash scripts/run_demo.sh --open
```

演示数据只用于检查页面和导出格式，绝不能用于训练。默认预览结果应为：Stage1 12 条、
Stage2 18 条、两阶段重叠 6 条。

## 在内网服务器运行

要求 Python 3.10+。训练环境通常已经包含 `PyYAML`；也可以离线安装本项目：

```bash
cd /workspace/ChatTS-Dataset-Studio
python3 -m pip install -e .
cp configs/server.example.yaml configs/server.yaml
# 修改 server.yaml 中四个路径，然后启动：
chatts-dataset-studio serve -c configs/server.yaml
```

不安装包也可直接运行：

```bash
cd /workspace/ChatTS-Dataset-Studio
PYTHONPATH=src python3 -m chatts_dataset_studio serve -c configs/server.yaml
```

服务器建议只监听 `127.0.0.1`。从本机建立隧道后访问 `http://127.0.0.1:7865`：

```bash
ssh -L 7865:127.0.0.1:7865 user@server
```

不要把页面直接暴露到公网；它具有向配置的输出目录写文件的能力。

## 输入约定

`registry_path` 指向 datav2 的 `sources.json`，其中每个源至少有 `name` 与 `path`。
规范 QA JSONL 每行包含：

```json
{"input":"... <ts><ts/> ...","timeseries":[[0.1,0.2]],"output":"..."}
```

`annotations_root` 指向 `merge_tsqa_annotations.py` 的输出根目录。工具优先读取体积较小的
`annotations/<source>.jsonl`；若不存在则读取 `annotated/<source>.jsonl`。sidecar 与 QA 必须
严格一一同序，若 `annotation_source`、`source_index`、`line_number` 或 `annotation_id` 错位，
导出会立即失败并清理临时目录。

合法枚举：

```text
quality:   unusable < weak < acceptable < good < excellent
difficulty: very_easy < easy < moderate < hard < very_hard
```

## 导出产物

每个运行名只允许创建一次，避免误覆盖：

```text
<output_root>/<run_name>/
├── stage1/<source>.jsonl
├── stage2/<source>.jsonl
├── stage1_annotations/<source>.jsonl
├── stage2_annotations/<source>.jsonl
├── dataset_info.json
├── training.env
├── manifest.json
├── stage1/manifest.json
└── stage2/manifest.json
```

训练 JSONL 只保留 `input/timeseries/output` 三个字段。标签 sidecar、输入/输出 SHA256、筛选
规则、计数和数据快照哈希单独保存，便于审计与复现。

## 接入现有两阶段训练

导出完成后：

```bash
set -a
source /workspace/chatts-dataset-exports/<run_name>/training.env
set +a

cd /workspace/ChatTS-Training
bash scripts/full/run_chronos2_best_two_stage.sh
```

`training.env` 会设置 `DATASET_DIR`、两个阶段的数据集名、`concat` 混合策略以及
`DATASET_SNAPSHOT_HASH`。它不会覆盖学习率、batch size、epoch 等训练超参数。

## 命令行与测试

同一份 YAML 也可以完全不经过页面：

```bash
chatts-dataset-studio catalog -c configs/server.yaml
chatts-dataset-studio preview -c configs/server.yaml
chatts-dataset-studio export -c configs/server.yaml
pytest
```

`catalog` 扫描并校验标签，`preview` 只计算选择结果，`export` 才写文件。
