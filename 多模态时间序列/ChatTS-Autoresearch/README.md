# ChatTS-Autoresearch

`ChatTS-Autoresearch` 是独立于 ChatTS 和 ChatTS-Training 的 Chronos-2 实验控制器。它不
import 两个项目的 Python 模块，只把训练与评测脚本作为黑盒子进程，通过 subprocess、
环境变量、退出码和 JSON/JSONL 产物交互。因此本目录可以单独复制到内网服务器，也不会
把自动搜索或 DeepSeek 逻辑耦合进原项目。

V1 固定 Chronos-2、seed 42 和当前 `run_chronos2_best_two_stage.sh` 配方：baseline 完整
训练一次并保留共享 Stage1；六个 `max_steps=300` 的 Stage2 proxy 从同一个 Stage1 启动；
排名前两个 proxy 再进行完整 Stage2 训练。每个 trial 只允许改变一个参数族。

## 安装与九个命令

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp configs/chronos2.example.yaml configs/chronos2.yaml

chatts-autoresearch preflight -c configs/chronos2.yaml
chatts-autoresearch label -c configs/chronos2.yaml
chatts-autoresearch prepare-data -c configs/chronos2.yaml
chatts-autoresearch baseline -c configs/chronos2.yaml
chatts-autoresearch search -c configs/chronos2.yaml
chatts-autoresearch resume -c configs/chronos2.yaml
chatts-autoresearch freeze -c configs/chronos2.yaml
chatts-autoresearch final-eval -c configs/chronos2.yaml
chatts-autoresearch report -c configs/chronos2.yaml
```

也可以不安装，使用 `PYTHONPATH=src python -m chatts_autoresearch <command> ...`。

各命令职责如下：

| 命令 | 作用 |
|---|---|
| `preflight` | 离线检查路径、脚本和本地模型，并创建或验证评测分区 |
| `label` | 流式补齐 quality、difficulty、taxonomy（包括 ECG）标签 |
| `prepare-data` | 生成不修改 datav2 原文件的数据快照和锁定评测视图 |
| `baseline` | 按当前配方完成 Stage1 + Stage2，并在 search-dev 上评测 |
| `search` | 运行六个 proxy、两个完整 finalist，并写入搜索完成清单 |
| `resume` | 从 SQLite 状态恢复，不重复已通过身份校验的实验 |
| `freeze` | 只从搜索完成清单中的 finalist 选冠军并写入冻结文件 |
| `final-eval` | 冻结后对 baseline 与 champion 运行完整正式评测 |
| `report` | 只用真实产物生成 Markdown 报告与 SVG 排名图 |

建议按表中顺序首次运行；中断后使用 `resume`。`final-eval` 在合法的 `FROZEN.json` 出现
前会拒绝执行。

## 数据快照与单变量搜索

`label` 流式读取 datav2，DeepSeek 只接收截断文本和时间序列统计摘要，不接收整条大数组。
标签缓存由样本哈希、prompt 版本和模型标识共同约束。`prepare-data` 生成质量过滤、难度重采样、
exact/near duplicate 与跨 source 泄漏标记，并输出 LLaMA-Factory `dataset_info.json`。

`data.baseline_snapshot` 是 baseline 的唯一数据基线。非数据参数实验（学习率、projector 学习率
比例、warmup、scheduler、epoch）严格复用同一份 baseline snapshot；只有
`source_weights`、`minimum_quality` 或 `difficulty_weights` 参数族会创建带独立哈希的派生
snapshot。这样一个 trial 不会在修改超参数时又悄悄更换训练数据。

## 锁定的 search-dev / final-test 协议

TSRBench 与 TimeSeriesExam 在 `prepare-data` 时物化为不可变视图：

- 只有当每条样本都有完整且同时包含 dev/test 的官方标记时，才采用官方划分；
- 否则使用 seed 42，按来源/类别/难度进行稳定哈希分层，精确划分 20% `search-dev` 与
  80% `final-test`；
- 生成的样本 ID、输入、计数和输出文件均写入清单并计算哈希；输入或视图被修改后不会复用；
- 搜索、proxy 排名和 full finalist 始终只使用 `search-dev`，`final-test` 在 `freeze` 前不会
  被控制器调用。

TS-Haystack 搜索固定使用官方 `validation`，正式评测使用官方 `test`。tinyBench 使用
`search-dev` / `final-test` 分区且固定 partition seed 42。`final-eval` 对 baseline 和冠军运行
`tsrbench,timeseriesexam,ts_haystack,tinybenchmarks` 全部四套 benchmark，并强制
`final_max_samples: 0`，即对锁定的正式分区做全量评测。

主排序分数是 TSRBench strict accuracy 与 TimeSeriesExam strict accuracy 的等权平均。
完整候选还必须满足 tinyBench 平均/单任务、TS-Haystack mean IoU 和 coverage 的保退化门槛；
同分时依次比较 GPU-hours 与 validation loss。

## DeepSeek 逐轮 badcase 分析

`search.proposal_mode: deepseek` 时，控制器在 baseline 和每轮 proxy 后从真实错误中分层抽取
最多 64 个 badcase，结果写入 `analysis/round-*.json`。请求和响应采用严格 JSON：

- 根对象只能包含 `error_groups`、`recommended_family`、`proposal`；
- 错误组只能引用本轮提供的 badcase ID，并填写错误类型和可能的数据原因；
- proposal 必须只修改一个白名单参数族、值必须在配置范围内，重复或等价于 baseline 的补丁会
  被拒绝；最后一轮 `proposal` 必须为 `null`；
- 非 JSON、额外字段、越界值、非法 ID、代码或 shell 命令均不会进入实验；响应会按配置重试，
  通过校验的结果才会缓存。

DeepSeek 只负责标签、错误归因和下一轮受限建议，不生成命令、不执行代码，也不产生或改写分数。
API key 只从 `deepseek.api_key_env` 指定的环境变量读取，不写入 resolved config、日志或 SQLite。
默认 `deepseek.response_format: json_schema` 会把带 `additionalProperties: false` 的严格 schema
交给服务端；若内网服务只实现 `json_object`，可显式降级，但本地 whitelist/schema 校验仍不会关闭。

搜索完整结束后写入 `SEARCH_COMPLETE.json`，其中锁定 baseline、六个 proxy、排名、两个
finalist、实验/配置/数据/协议/命令哈希以及逐轮分析哈希。`freeze` 会重新验证该清单、SQLite
记录、模型权重、共享 Stage1 和评测产物；旧实验或被修改的 checkpoint 不能混入冠军选择。

## 黑盒契约

训练脚本由 `paths.train_script` 指定。控制器传入 `PIPELINE_MODE`、`STAGE2_FROM`、
`KEEP_STAGE1`、`DATASET_DIR`、`TRIAL_ID`、配置/数据哈希和 Stage2 超参数。模型写入
`FINAL_MODEL_PATH`，共享 Stage1 写入 `STAGE1_OUT`。

评测脚本由 `paths.eval_script` 指定。控制器传入 `MODEL_PATH`、`OUTPUT_ROOT`、
`BENCHMARKS`、`RUN_ID`、`EVAL_PROTOCOL_HASH`、`EVAL_SPLIT`、`HAYSTACK_SPLIT`、
`TINY_DATA_PARTITION` 和 `TINY_PARTITION_SEED`。每个评测目录必须生成 `metrics.json`；
控制器保留原始预测并统一提取 badcase，不改变 ChatTS 的推理、答案解析或 benchmark 评分定义。

## 产物目录

所有状态和结果只写入 `runtime.output_root`：

```text
state.sqlite3
experiments.jsonl
leaderboard.csv
preflight.json
configs/
  *.resolved.yaml
  eval_splits.json
eval_views/
  manifest.json
  search-dev/{tsrbench,timeseriesexam}/...
  final-test/{tsrbench,timeseriesexam}/...
labels/quality_difficulty_taxonomy.jsonl
datasets/<snapshot-id>/{data/*.jsonl,dataset_info.json,manifest.json}
commands/*.json
logs/*.log
models/{shared-stage1,baseline,proxy-*,full-*}/...
evaluations/<experiment-id>/<split>/...
badcases/*.jsonl
analysis/round-*.json
SEARCH_COMPLETE.json
FROZEN.json
figures/leaderboard.svg
report.md
report_summary.json
```

SQLite 是恢复源，同时同步导出 JSONL/CSV。完整配置、源码接口、模型、数据、命令、协议和评测
目录均以内容身份绑定；缓存命中前会重新校验产物，避免不同 checkpoint 或协议错误复用。

## 内网离线复制

本项目运行时不搜索互联网，也不会自动下载 Chronos-2 或基础模型。复制到内网服务器时需要：

1. 复制整个 `ChatTS-Autoresearch` 目录（或 `dist/` 中的 wheel）以及 Python 依赖 wheelhouse；
2. 分别复制 ChatTS、ChatTS-Training、datav2、四套 benchmark、Chronos-2 和基础模型；
3. 将示例配置中的所有 `paths.*` 改为内网绝对路径；
4. 如果使用 DeepSeek，将 `base_url` 指向内网 OpenAI-compatible 服务，并设置配置声明的
   API key 环境变量；确定性模式可设置 `proposal_mode: deterministic`，不调用 DeepSeek 搜索建议。

离线安装示例：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --no-index --find-links /workspace/wheelhouse \
  dist/chatts_autoresearch-0.1.0-py3-none-any.whl
```

不要复制旧的 `runtime.output_root` 到另一套数据或代码后继续复用；若确需迁移实验，必须保持所有
路径内容与哈希一致。

## 测试

```bash
pytest
ruff check src tests
```

测试使用临时 mock trainer/evaluator，不启动 GPU 训练。报告只汇总实际产生并通过解析器校验的
指标，绝不会用 DeepSeek 填写推测性分数；当前 V1 固定单 seed 42，这一限制会写入报告。
