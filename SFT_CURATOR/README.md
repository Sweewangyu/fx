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
