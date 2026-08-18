# ChatTS Dataset Studio

ChatTS 的本地数据版本、Stage1-only/两阶段训练和四套评测控制台。它不 import `ChatTS` 或
`ChatTS-Training`：数据通过不可变快照交付，训练评测通过固定脚本和 YAML 交互。

## 能做什么

- 扫描 `merged_labels/annotated/*.jsonl`，为**全部数据集**创建 source，不再写死 6 个。
- 为每个 source 分别设置 quality、difficulty、ability 和抽样百分比；Stage1 / Stage2
  规则彼此独立，页面只展示该数据集真实存在的标签维度。
- 发布 `datav3`、`datav4`……不可变数据版本，记录每版数据集组成、规则、行数和 SHA256。
- 在同级 `ChatTS-Training/data/studio_versions/` 注册版本，但不改全局
  `data/dataset_info.json`。
- 可选择“两阶段”或“只训练并保存 Stage1”；配置 LR、time-series encoder LR、epoch、batch、梯度累积、warmup、scheduler、
  max steps、保存/评测间隔等训练参数。
- Docker 宿主后端可配置四套 benchmark 并一键训练后评测；Slurm 后端通过可信
  `.sbatch` 在同一个 job allocation 内串行执行 Stage1-only/两阶段训练与四套评测。
- 模型、评测目录自动带 `-datavN`，任务配置和日志持久保存。

## 服务器目录

Dataset Studio 是轻量控制面，可以运行在 Docker 宿主机，也可以运行在能调用 `sbatch` 的
集群登录节点。Docker 模式通过 `docker exec` 调度 `chatts` 训练容器和 `ragas` 评测容器；
Slurm 模式把冻结配置交给 `ChatTS-Training/slurm/run_chatts_studio_pipeline.sbatch`，再由
Singularity 执行标准训练与评测脚本：

```text
<workspace>/
├── ChatTS-Dataset-Studio/
├── ChatTS-Training/
└── ChatTS/
```

如果只有前两个目录，可以完成 source 扫描、数据版本发布、训练注册和配置预览，但无法运行
现有四套评测。`ChatTS/scripts/run_train_then_eval.sh` 还要求训练容器 `chatts`、评测容器
`ragas` 正在运行，并能看到相同的 `/share/...` 数据、模型与输出目录。

不要在训练容器里启动控制面：容器里通常没有 Docker/Slurm 控制命令，也不应该为了该功能把
`/var/run/docker.sock` 挂入训练容器，因为这相当于授予宿主机级控制权限。把三个代码目录放在
宿主机的同一 workspace 即可；代码和数据仍可通过原有 volume 挂载给两个计算容器。

## 你的 traindata 路径

按当前服务器目录，核心配置应为：

```yaml
paths:
  registry_path: /share/airesearch/data/finiverse/traindata/sources.json
  annotations_root: /share/airesearch/data/finiverse/traindata/merged_labels
  data_root: /share/airesearch/data/finiverse/traindata
  output_root: /share/airesearch/data/finiverse/traindata/chatts-data-versions
  state_root: /share/airesearch/data/finiverse/traindata/chatts-studio-state
```

`tsr-taxonomy-datav2-v1/final_labels.jsonl` 是标签流程产物，不应被当成单个训练 source；实际
训练 source 来自 `merged_labels/annotated/` 下每个带 `input/timeseries/output` 的 JSONL。

## 安装与启动

```bash
cd /path/to/ChatTS-Dataset-Studio
python3 -m pip install -e .
cp configs/server.example.yaml configs/server.yaml
# 检查 server.yaml 中容器内路径、模型与 benchmark 路径后启动：
chatts-dataset-studio serve -c configs/server.yaml
```

不安装包也可以。推荐直接在宿主机/登录节点运行启动脚本；它会报告 Docker 与 Slurm
后端的可用性，真正的逐后端检查由页面 Preflight 完成：

```bash
cp configs/server.example.yaml configs/server.yaml
# 将 integration.training_root、evaluation_root、pipeline_script 改成宿主机绝对路径
bash scripts/start_host_control_plane.sh configs/server.yaml
```

等价的手动命令：

```bash
PYTHONPATH=src python3 -m chatts_dataset_studio serve -c configs/server.yaml
```

服务建议只监听 `127.0.0.1`，从电脑建立 SSH 隧道：

```bash
ssh -L 7865:127.0.0.1:7865 yu.wang17@<server>
```

浏览器打开 `http://127.0.0.1:7865`。

## 全量创建 source

`registry.auto_build` 默认就是 `true`（示例配置也显式写出），因此每次启动会扫描全部
`merged_labels/annotated/*.jsonl` 并原子刷新 `sources.json`。也可在启动前手动执行：

```bash
chatts-dataset-studio build-registry \
  --merged-labels-root /share/airesearch/data/finiverse/traindata/merged_labels \
  --data-root /share/airesearch/data/finiverse/traindata \
  --output /share/airesearch/data/finiverse/traindata/sources.json \
  --force
```

如需复用旧 registry 的 `family/split/training_role` 元数据，再加
`--metadata-registry /path/to/old/sources.json`。生成器只校验和登记文件，不改 QA 或标注。

YAML 的 `sources: ["*"]` 和页面的“全选可用 source”都会动态使用完整 catalog。新文件出现后
刷新 source 即可，不需要改 Python 名单。

页面会把逐数据集选择保存为兼容字段 `source_rules`。例如两个数据集可以使用完全不同的规则：

```yaml
stage1:
  sources: [dataset_a, dataset_b]
  qualities: [weak, acceptable, good, excellent]   # 旧配置 fallback，继续兼容
  difficulties: [very_easy, easy, moderate]
  abilities: []
  source_rules:
    dataset_a:
      qualities: [good, excellent]
      difficulties: [moderate, hard]
      abilities: [forecasting]
      sample_percent: 50                         # 筛选后稳定随机保留 50%
    dataset_b:
      qualities: [acceptable, good]
      difficulties: [easy, moderate]
      abilities: []                                # 空列表表示该数据集的全部能力
      sample_percent: 100
```

`source_rules` 中只能出现当前阶段已经选中的数据集，且只能选择该数据集实际存在的标签。
`sample_percent` 接受 1–100（最多 6 位小数），默认 100；抽样发生在质量、难度、能力筛选
之后。省略或显式设为 100 时不会写入 canonical selection，因此旧配置的 selection hash、
快照 hash 和导出 JSONL 字节保持不变。

预览会逐数据集检查三维联合结果；任何被选中但过滤后为 0 条的数据集都会被明确拒绝。
系统按 quality × difficulty × ability 细分层分配精确配额，再使用固定版本的样本哈希进行
单遍在线、可复现、无放回抽样；运行时每个细分层只保存固定数量的计数器，不保存行号或样本。
小数据集按百分比四舍五入后至少保留 1 条；同一输入快照与规则重复导出会得到相同的训练
JSONL。Stage1/Stage2 对同一 source、同一细分层先精确选择较大配额，再用第二个独立哈希在
其中精确选择较小配额，因此较小样本严格属于较大样本，预览中的 overlap 也是导出后的精确值。

`preview.counts` 与 `preview.by_source` 是抽样后的实际行数；过滤后的候选数分别记录在
`preview.filtered_counts` 及每个 source 的 `stage1_filtered/stage2_filtered`。预览还会汇总
Stage1/Stage2 所有 taxonomy 能力维度的条数和百分比：后端字段
`distributions.<stage>.ability_percentages` 的非空阶段合计约为 100%，页面补齐当前配方中
占比为 0 的维度，便于直接观察 15 个能力维度的最终配比。

Catalog 同时提供三个能力字段：`abilities` 保留数据中实际观测到的值以兼容旧客户端；
`ability_levels` 始终给出 TSRBench 4×15 的权威完整顺序；`ability_extras` 单独列出不在权威
15 类中的观测值。若实际标签与 `PR/NU/…/QuantDM` code 集合的匹配数更多，权威列表使用
15 个 codes，否则（包括完全没有匹配）使用 15 个完整英文 names；模式记录在
`ability_level_mode`。

## 数据版本与训练注册

首次发布默认为 `datav3`，之后自动递增。也接受 `data-v4` 输入，但落盘统一为 `datav4`。
同一内容重复发布会复用原版本；内容变化才会创建下一版。每版目录为：

```text
chatts-data-versions/datav3/
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

中央 `chatts-data-versions/ledger.json` 记录 parent、notes、规则、每阶段完整 source 组成、行数、
快照和文件哈希。发布或启动训练前会再次校验所有文件，修改旧版本会被拒绝。

“发布并注册”只在 Training 仓库新增：

```text
ChatTS-Training/data/studio_versions/
├── datav3.json
├── datav3.env
└── active.json
```

训练仍由快照自己的 `dataset_info.json` 注册：控制台把 `DATASET_DIR` 指到该 `datavN` 目录，
并从 manifest 注入 Stage1/Stage2 dataset keys。浏览器不能偷偷换训练脚本、项目根或任意输出
路径。

## Stage1-only、两阶段与评测

选择一个版本后：

1. 在“训练”页的数据版本下拉框选择任一已发布版本。默认跟随当前激活版，但这个选择只影响
   本次训练，不修改全局 active；未注册版本会在启动时自动注册。刷新版本列表会保留已选择的
   有效版本，若其被删除或变为不可用则安全回退到当前版或最新可训练版。
2. 选择“Chronos-2 两阶段”或“只训练并保存 Stage 1”，再调整实际执行阶段的参数；dataset
   列表只读，始终来自下拉框所选版本。
3. 所有训练方案和后端都可在“评测”页选择 `tsrbench`、`tinybenchmarks`、`ts_haystack`、
   `timeseriesexam`，并在训练完成后直接使用最终模型评测。Stage1-only 使用 Stage1 最终模型。
4. 先点 `Preflight`；它只检查冻结配置、训练与评测链路，不启动训练。通过后点击“训练 + 评测”。

例如 `datav3`、seed 42 会派生：

```text
TRAIN_OUTPUT_ROOT = .../ChatTS-msxf-8B-datav3/experiments/recipe-<16hex>
FINAL_MODEL_PATH  = .../experiments/recipe-<16hex>/best_seed42
MODEL_NAME        = chatts-msxf-8B-datav3-seed42-recipe-<16hex>
RUN_ID            = chronos2-datav3-seed42-recipe-<16hex>-full
```

Docker 底层固定调用：

```bash
CONFIG_FILE=<studio-generated.yaml> bash ../ChatTS/scripts/run_train_then_eval.sh
```

数据版本和 `DATASET_SNAPSHOT_HASH` 会同时进入训练与评测容器。两阶段模型以
`TRAINING_COMPLETE.json` 为准，Stage1-only 严格使用 `STAGE1_COMPLETE.json` +
`best_model_manifest.json` + 非空权重，不会伪造 `TRAINING_COMPLETE.json`。评测完成以
`metrics.json.status == "pass"` 为准。

### 单独评测与批量模型

“评测”页也可以不启动训练，直接输入一个或多个模型目录（每行一个，单批最多 64 个）。后端
先校验并去重全部绝对 POSIX 路径，再为每个模型冻结一份独立 YAML 和任务；任一路径非法时整批
拒绝，不会留下半批任务。同名目录会自动追加模型路径 SHA256 的前 12 位，输出固定为：

```text
<evaluation_output_base>/<模型目录名>-<路径hash12>/protocol-<协议hash16>/
```

Docker 单独评测使用 `ChatTS/scripts/run_eval_only.sh`，和训练流水线共用 Studio 的持久 FIFO；
Slurm 使用 `run_chatts_studio_evaluation.sbatch` 立即提交，页面中的 `PENDING` 就是集群调度队列。
单独评测不需要数据版本、ChatTS-Training 注册记录或训练完成 marker。兼容旧配置时会从
`evaluation_root/scripts/run_eval_only.sh` 自动找到 Docker 入口；也可显式配置：

TSRBench prompt 可选 `answer_only`、`official` 和 `json_reasoning`。选择
`json_reasoning` 时页面会使用 256 个生成 token、batch size 1 的推荐值，并把模式及实际资源参数
一起写入协议哈希；其严格输出格式为 `{"reason":"...","answer":"A"}`。训练后评测与单独评测、
Docker 与 Slurm 使用同一套设置。

```yaml
integration:
  evaluation_pipeline_script: /actual/workspace/ChatTS/scripts/run_eval_only.sh
  slurm_evaluation_sbatch: run_chatts_studio_evaluation.sbatch
  # 默认从训练 SIF 所在目录寻找 ragas.sif
  # slurm_evaluation_sif_image: /actual/images/ragas.sif
```

对应 API 为 `POST /api/evaluations` 和 `POST /api/evaluations/preflight`：

```json
{
  "model_paths": ["/share/models/model-a", "/share/models/model-b"],
  "execution": {"backend": "slurm"},
  "evaluation": {"benchmarks": ["tsrbench", "timeseriesexam"]}
}
```

## Slurm sbatch 后端

服务器配置示例：

```yaml
integration:
  execution_mode: slurm
  training_root: /workspace/ChatTS-Training
  slurm_root: /workspace/ChatTS-Training/slurm
  slurm_sbatch: run_chatts_studio_pipeline.sbatch
  slurm_evaluation_sbatch: run_chatts_studio_evaluation.sbatch
  # 以下是计算节点宿主机路径；评测项目默认复用 evaluation_root。
  slurm_evaluation_root: /actual/workspace/ChatTS
  # slurm_evaluation_sif_image: /actual/images/ragas.sif
  # slurm_ts_haystack_host_root: /actual/workspace/TS-Haystack
  # slurm_timeseriesexam_host_root: /actual/workspace/TimeSeriesExam
```

页面只能选择 `slurm_root` 内带 `# CHATTS_STUDIO_SBATCH_API=1` marker 的普通 `.sbatch`
文件；符号链接、目录越界和任意命令行参数都会被拒绝。每次提交固定使用不可变 resolved YAML、
Studio job id 和脚本 SHA256。通用脚本读取这份 YAML，所以数据版本、逐数据集快照、基础模型、
Stage1/Stage2 超参数、recipe hash 与输出路径都不会被忽略。旧的
`run_chronos2_all_data_one_stage.sbatch` 仍可独立运行，但没有 Studio marker，不会出现在页面中。

Slurm 后端先负责训练，训练成功后由流水线继续启动评测；Stage1-only 与两阶段都适用。任务页
同时展示 Studio 提交日志、Slurm job id、scheduler 状态、重定向后的 stdout/stderr 与后续评测
状态；控制面重启后会用已记录 job id 继续跟踪，不会重新提交第二份训练。
`paths.state_root`、冻结配置、数据、模型与输出路径必须位于计算节点可见的共享文件系统。

“训练”页的基础模型地址可直接填写，它是训练容器内可见的绝对路径，例如
`/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B`。`server.yaml` 中的
`integration.base_model_path` 只是页面默认值，不再锁定用户输入。

当模型目录名含 `8B`、`4B`、`1.7B` 等参数量时，页面和服务端会同步替换
`model_output_base` 与 `model_name_base` 中最后一个参数量标记。例如把基础模型从
`ChatTS-Qwen3-8B` 改成 `ChatTS-Qwen3-4B`，会将输出从
`ChatTS-msxf-8B-datav3` 自动改为 `ChatTS-msxf-4B-datav3`。

模型不会再仅按“数据版本 + seed”共用一个目录。Studio 会对数据快照、基础模型、Chronos-2、
seed、GPU 拓扑以及 Stage1/Stage2 的全部训练参数计算稳定的 `training_recipe_hash`，输出结构为：

```text
ChatTS-msxf-8B-datav3/
└── experiments/
    └── recipe-0123456789abcdef/
        ├── .stage1_seed42_s1lr_1e-5/
        ├── best_seed42/
        └── logs/
```

修改学习率、epoch、batch、混合策略等训练参数会生成新的 recipe 目录，不会覆盖旧权重；只改
评测参数、勾选强制评测或强制重训不会改变 recipe 身份。相同训练配方默认复用已完成模型，
“强制重训当前配方”只清理并重建当前 recipe 目录中的 Stage1/最终模型，不会触碰其他 recipe。

评测输出还会在模型名下追加 `protocol-<评测协议哈希前16位>/`。因此同一模型切换
benchmark、split、prompt、token budget 或 offline 策略时不会覆盖另一套协议的缓存和结果。

每次真正启动训练（Preflight 不计）前，Studio 都会先在
`<state_root>/pipeline/run-records/<job_id>/` 写入运行档案：

```text
pipeline_config.resolved.yaml       # 本次完整训练与评测参数
training_data.json                  # 数据版本、快照 SHA256、dataset keys 和 manifest
diff_from_previous.json             # 与上一次训练的机器可读差异
diff_from_previous.md               # 与上一次训练的易读差异表
comparison.json                     # 供下一次训练比较的规范化配置
run_record.json                     # 本次运行档案索引
```

这些文件在训练进程启动前落盘，并显示在任务详情的“产物”中。第一次训练会明确记录“无上次
训练”；第二次起会同时比较数据快照、基础模型、输出目录、Stage1/Stage2 参数以及评测协议。

Docker 与 Slurm 使用独立调度语义。Docker 共享固定训练/评测容器，因此使用持久化 FIFO，
同一时刻只启动一个本地流水线，后续 Docker 任务显示为“Docker 排队 #N”。Slurm 任务不进入
Studio FIFO：每次点击都会立即调用 `sbatch`，GPU 是否已满、何时从 `PENDING` 进入
`RUNNING` 完全由集群调度器决定，页面通过
`squeue/sacct` 展示真实状态和日志。两类任务都会在提交瞬间冻结 resolved YAML、数据快照 Hash
和运行档案；Studio 重启后也会分别恢复 Docker 队列及所有 Slurm tracker。不同 recipe 可并行；
同一 recipe 的 Slurm job 会在共享输出目录的 sibling lock 上做非阻塞互斥，后到任务明确失败，
避免并发重训或评测覆盖同一模型。

如果评测页显示“一键启动暂不可用”，页面会按当前“训练方案 + 执行后端”列出具体原因。
Docker 最常见的是没有传 `-c configs/server.yaml`，或 `pipeline_script` 不是宿主机上真实
存在的 `ChatTS/scripts/run_train_then_eval.sh`；Slurm 则通常是 `sbatch/squeue/sacct` 不可用、
`slurm_root` 不存在，或其中没有带 Studio marker 的通用脚本。部署路径建议写绝对路径：

```yaml
integration:
  training_root: /actual/workspace/ChatTS-Training
  evaluation_root: /actual/workspace/ChatTS
  pipeline_script: /actual/workspace/ChatTS/scripts/run_train_then_eval.sh
```

修改 YAML 后需要重启 Studio；先点“预检”，通过后再启动当前方案。Docker
预检只运行两个 runner 的 `--preflight`；Slurm 预检会提交一个固定 10 分钟上限的短暂 allocation，
在真实计算节点内校验冻结配置 loader、SIF、Chronos-2、benchmark bind 和两个
runner 的 `--preflight`，但不会启动训练、推理或写入模型/评测产物。

如果日志出现 `Docker CLI is unavailable`，请把控制面移到 Docker 宿主机，或在页面改用已配置
的 Slurm 后端；如果出现 `sbatch CLI is unavailable`，则需要在集群登录/控制节点启动 Studio。

如果首次运行因权限失败、重试时出现：

```text
Training registration conflicts with existing file: .../studio_versions/datav3.json
```

请先更新到包含幂等注册修复的版本，不要直接删除该文件，也不要使用 `chmod 777`。旧注册
把 `PROJECT_ROOT`、`OUTPUT_ROOT` 等部署路径写进了 profile；从容器控制面切换到宿主机控制面
后这些路径会变化，但它们不代表训练数据发生变化。新版按版本、数据快照 SHA256、manifest、
selection 和 Stage1/Stage2 composition 判断数据身份：身份一致时对已有 `.json/.env/active.json`
零写入复用，即使文件由另一个容器 UID 创建且当前用户只能读取也能重试；只有数据身份真的
变化才拒绝，并在错误中列出不同字段。

## 兼容 CLI

```bash
chatts-dataset-studio catalog -c configs/server.yaml
chatts-dataset-studio preview -c configs/server.yaml
# legacy export；推荐正式数据使用页面的 datavN 发布流程
chatts-dataset-studio export -c configs/server.yaml
```

## 测试

```bash
pytest
ruff check src tests
bash -n ../ChatTS/scripts/run_train_then_eval.sh
```

所有真实 QA 均在服务器端流式扫描、筛选和写盘；浏览器只接收聚合计数、版本组成、任务状态
和日志。该服务具有写数据版本和启动训练的能力，不要直接暴露到公网。
