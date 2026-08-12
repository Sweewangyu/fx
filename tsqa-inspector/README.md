# TSQA Lens

用于逐条审查时间序列 QA 的本地网页。默认读取 DataTaste datav2 的全部来源，重点支持
OpenTSLM 数据质量检查，并原生兼容 TSRBench 官方评测集。

## 功能

- 兼容统一的 `input / timeseries / output` JSONL；
- 原生读取 TSRBench 的12个评测文件，包括普通 `question/choices/timeseries/answer` 样本和
  `abductive_reasoning` 的特殊比赛事件格式；
- 数据集切换、序号跳转、随机抽样和同模板随机抽样；
- 多通道时间序列折线图、通道开关、统计量和长序列降采样；
- 问题、答案原文与 Qwen 中文翻译并排检查，译文自动缓存；
- 展示 DataTaste audit 和 SFT_CURATOR 合并后的能力、质量、难度标签；
- 查看同一个 `taxonomy_cluster_id` 的成员、原始问法数和答案类别分布；
- 自动提示答案泄漏、视觉措辞错位和未验证推理。

## 本地运行

依赖 Node.js 22+、Python 3.9+ 和 PyYAML。首次运行先安装前端依赖：

```bash
cd /workspace/tsqa-inspector
npm install
python3 -m pip install -r requirements.txt
```

一条命令启动数据服务和网页：

```bash
./run_local.sh
```

浏览器打开 `http://服务器地址:3000`。数据接口默认监听 `8765`；服务器防火墙或 Docker
需要同时映射 `3000` 和 `8765`。

也可以分别启动：

```bash
python3 server.py --config inspector_config.yaml
npm run dev
```

## 配置

修改 `inspector_config.yaml`：

```yaml
data:
  root: ../datataste
  registry: ../datataste/data/versions/datav2/sources.json
  template_stats: ../SFT_CURATOR/template_stats.json
  annotations_dir: ../SFT_CURATOR/merged_labels/annotations
  tsrbench_root: ../TSRBench/dataset

qwen:
  base_url: http://10.112.164.1:30001/v1
  model: /share/global/pymaip/models/Qwen3.6-27B
```

路径相对于 YAML 文件解析。如果合并标签在其他位置，只需调整 `annotations_dir`。即使没有
合并标签，原始 QA、audit、模板和时间序列仍可浏览。

`tsrbench_root` 指向官方数据根目录，例如其中应包含：

```text
perception/perception.jsonl
reasoning/causal_reasoning.jsonl
prediction/time_series_forecasting.jsonl
decision/quantitative_decision.jsonl
```

服务会递归发现存在的任务文件；未下载 TSRBench 时不会报错，也不会影响训练集浏览。评测数据只作
查看，页面会明确标记为 `evaluation_only`，不会并入 DataTaste 训练来源。

如果 Qwen 需要密钥：

```bash
export DT_QWEN_API_KEY='...'
```

浏览器不会直接访问 Qwen，也不会接触密钥；翻译由 `server.py` 代理并缓存到
`.cache/translations.sqlite`。服务端调用会显式绕过系统 HTTP 代理，适合当前集群上的私网 Qwen
地址。

## 首次索引

第一次打开某个数据集时，服务会扫描该来源并创建 `.cache/<dataset>.index.sqlite`。索引只保存
文件偏移、模板ID、答案类别和风险标记，不复制完整时间序列。源文件或标签变化后，签名不匹配，
索引会自动重建。

## 服务器上传

需要上传整个 `tsqa-inspector/`，但不需要上传 `.cache/`、`.vinext/`、`.wrangler/`、`dist/`
或 `node_modules/`。DataTaste 和 SFT_CURATOR 默认与本目录同级；目录不同就修改 YAML。

## 快捷键

- `← / →`：上一条 / 下一条；
- `R`：随机抽样；
- `T`：打开同模板成员页。
