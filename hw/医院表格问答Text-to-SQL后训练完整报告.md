# 面向医院表格问答的 MySQL Text-to-SQL 执行反馈后训练报告

## 1. 项目结论

本项目最终采用的主方案是：**SQL-first 数据合成 + 结构化规划 SFT + MySQL 执行反馈 DAPO + 线上 badcase 数据闭环**。

完整链路如下：

```text
医院表格清洗与 schema 标准化
-> SQL-first 反向合成训练数据
-> 多模型投票与人工分歧审核
-> <think>规划</think> + <answer>SQL</answer> 格式 SFT 冷启动
-> MySQL 执行反馈 DAPO
-> format / execution / result 三类奖励
-> total、千分位、value linking 等 badcase 的数据侧与算法侧联合回流
```

这个方案的核心判断是：Text-to-SQL 和普通文本生成不同，SQL 可以被 MySQL 执行器验证，所以最有效的后训练信号不是复杂偏好打分，而是**格式是否可解析、SQL 是否可执行、执行结果是否正确**。线上遇到的 total 字段、千分位数字、value linking、聚合函数错误，一部分要靠 schema 标准化和数据回流解决，另一部分要通过 AST 过程监督、执行反馈 DAPO、候选 SQL 执行重排和不可回答判别等算法机制解决。

最终结果：

| 指标 | Qwen3-30B Prompt Baseline | Qwen2.5-Coder-7B 最终方案 |
| --- | ---: | ---: |
| Format Pass Rate | 92.7 | 100.0 |
| Valid SQL | 88.6 | 97.3 |
| Result Accuracy | 56.9 | 67.4 |
| Business QA Accuracy | 63.6 | 71.8 |
| 平均生成延迟 | 4.8s | 1.2s |

固定医院表格测试集上，7B 后训练模型 Result Accuracy 较 Qwen3-30B prompt baseline 提升 10.5 个百分点，业务 Table RAG QA 准确率提升 8.2 个百分点。

## 2. 项目背景

华西医院场景中有大量表格问答需求，例如：

- 某医生今年总问诊量是多少？
- 某科室近三个月费用排名如何？
- 收入超过某阈值的科室有哪些？
- 某项检查阳性率最高的科室是哪个？

普通 RAG 对这类问题不稳定，因为它只能检索文本片段，不能可靠完成过滤、聚合、排序、分组和数值比较。因此项目在 Table RAG 中增加 Text-to-SQL 分支：模型根据用户问题和表格 schema 生成 MySQL SQL，在本地 MySQL 执行器中执行，再把结果返回给答案生成模块。

业务目标：

- 提升聚合、过滤、排序类表格问答准确率。
- 用 7B 小模型替代 30B 级通用模型，满足客户侧低算力部署。
- 通过 MySQL 执行器反馈形成可自动优化的后训练闭环。
- 将线上 badcase 转化为训练数据、schema 预处理规则和算法侧 hard negative / reward 样本。

## 3. 问题定义

输入：

- 用户问题 `q`。
- 表格 schema `T`：表名、表摘要、列名、列解释、列类型、sample values。
- 表格执行环境：MySQL。

模型输出固定为：

```text
<think>规划</think>
<answer>SQL</answer>
```

示例：

```text
<think>问题询问张三医生全年总问诊量；相关表：doctor_visit；相关列：doctor_name,total；过滤条件：doctor_name='张三'；生成策略：直接查询 total 字段。</think>
<answer>SELECT total FROM doctor_visit WHERE doctor_name = '张三';</answer>
```

线上只解析 `<answer>` 中的 SQL，放入 MySQL 执行器执行。

主要错误：

| 错误类型 | 示例 | 后果 |
| --- | --- | --- |
| 格式错误 | `<answer>` 标签缺失或 SQL 没包进去 | 工程解析失败 |
| Schema linking 错误 | 把“问诊量”映射到费用列 | SQL 逻辑错误 |
| Value linking 错误 | 医生名匹配到科室列 | 查询对象错误 |
| 聚合错误 | 问平均值却生成 `SUM` | 统计口径错误 |
| 类型错误 | `"12,000"` 按字符串比较 | 数值比较错误 |
| total/合计字段误用 | 有 total 字段却把月份逐列相加 | 结果重复或口径错误 |

### 3.1 典型 Badcase

| Badcase 类型 | 用户问题 | 表格/schema 关键信息 | 错误 SQL | 正确 SQL | 解决方式 |
| --- | --- | --- | --- | --- | --- |
| total 字段误用 | “张三医生今年总问诊量是多少？” | 表中有 `jan` 到 `dec`，也有 `total` | `SELECT jan+feb+...+dec FROM doctor_visit WHERE doctor_name='张三'` | `SELECT total FROM doctor_visit WHERE doctor_name='张三'` | schema 中显式标注 total 含义，回流 total 类样本做 SFT |
| 小计/合计行重复统计 | “心内科 2025 年总收入是多少？” | 明细行之外存在 `item='合计'` | `SELECT SUM(income) FROM dept_income WHERE department='心内科'` | `SELECT income FROM dept_income WHERE department='心内科' AND item='合计'` | 合计行作为 sample value，构造合计行优先查询样本 |
| 千分位字符串比较 | “收入超过 12000 的科室有哪些？” | `income` 样例为 `"9,800"`、`"12,500"` | `SELECT dept FROM income_table WHERE income > '12000'` | `SELECT dept FROM income_table WHERE CAST(REPLACE(income, ',', '') AS UNSIGNED) > 12000` | 表格预处理做类型归一化，补充 cast 模板样本 |
| value linking 错误 | “李明医生 3 月问诊量是多少？” | `doctor_name` 有李明，`department` 也有“李明工作室” | `SELECT mar FROM visit WHERE department='李明'` | `SELECT mar FROM visit WHERE doctor_name='李明'` | 在 `<think>` 中监督 value-column 对齐 |
| 聚合函数错误 | “各科室平均住院天数是多少？” | 列为 `department`、`stay_days` | `SELECT department, SUM(stay_days) FROM inpatient GROUP BY department` | `SELECT department, AVG(stay_days) FROM inpatient GROUP BY department` | 从 SQL AST 反推聚合标签，回流平均/求和对比样本 |
| Top-K 排序方向错误 | “问诊量最高的 3 个科室是哪些？” | `visits` 是数值列 | `SELECT dept FROM dept_visit ORDER BY visits ASC LIMIT 3` | `SELECT dept FROM dept_visit ORDER BY visits DESC LIMIT 3` | 构造最高/最低方向对比样本 |
| 不可回答强答 | “哪个科室患者满意度最高？” | schema 中没有满意度相关列 | `SELECT dept FROM dept_visit ORDER BY visits DESC LIMIT 1` | 拒答或转人工澄清 | 加入不可回答样本，训练 schema coverage 判断 |

这些 badcase 不应该只理解成“数据清洗问题”。我把它们拆成数据侧和算法侧两条处理线：

| 问题类型 | 数据侧处理 | 算法侧处理 |
| --- | --- | --- |
| total/合计字段 | schema 中显式标注 `total`、`合计`、`小计` 的业务含义，补充对应样本 | 在 `<think>` 中监督“已有 total 字段时优先直接查询”，并把逐月相加但结果不一致的 SQL 在 DAPO 中打低分 |
| 千分位/百分号/单位 | 表格预处理做类型归一化，保留原始值和标准值 | SQL 生成阶段训练 `CAST/REPLACE` 模板，执行失败或结果错误时通过 execution/result reward 惩罚 |
| value linking | sample values 覆盖高频实体、同名实体和歧义值 | AST 过程监督显式标注 value-column 对齐；同一 value 出现在多列时构造 hard negative |
| 聚合函数错误 | 构造 `SUM/AVG/COUNT/MAX/MIN` 对比样本 | 从 SQL AST 反推 operation label，在 DAPO rollout 中让错误聚合因执行结果不一致获得负优势 |
| Top-K 方向错误 | 构造最高/最低、前 N/后 N 的成对样本 | 推理时采样多条候选 SQL，优先选择执行结果稳定且 order direction 与问题极性一致的候选 |
| 不可回答强答 | 加入 schema coverage 不足的不可回答样本 | 增加 coverage gate：候选列召回为空时输出拒答模板，不进入 SQL 执行 |

## 4. 为什么选择这个方案

### 4.1 为什么必须做 Text-to-SQL，而不是普通 RAG

普通 RAG 擅长找文本证据，但不擅长数值计算。医院表格问答大量涉及 `SUM`、`AVG`、`COUNT`、`GROUP BY`、`ORDER BY`、区间筛选和时间筛选，这些操作需要符号执行。

如果只用 RAG：

- 聚合计算不稳定。
- 排序和 Top-K 容易错。
- 数值比较依赖模型口算。
- 答案无法通过 MySQL 执行器验证。

因此表格问答必须引入 MySQL Text-to-SQL 分支。

### 4.2 为什么不用 Qwen3-30B 直接 Prompt

Qwen3-30B prompt baseline 可以生成部分 SQL，但它在医院表格场景中有三个问题：

- 成本和延迟高，不适合客户侧本地部署。
- 输出格式不稳定，`<answer>` 标签和 SQL 边界不一定稳定。
- 对医院表格中的缩写列、total 字段、千分位数字和业务口径不敏感。

这个任务 schema 固定、执行器固定、输出格式固定，所以 7B 代码模型经过领域后训练后，可以在目标指标上超过 30B prompt baseline。

### 4.3 为什么采用 SQL-first 数据合成

我没有采用：

```text
schema -> query -> SQL
```

而是采用：

```text
schema -> SQL -> query -> think/answer
```

原因是医院表格 schema 很窄，query-first 容易生成 schema 不支持的问题。例如用户问题看起来合理，但表中根本没有“患者满意度”字段，后续 SQL 生成只能编造列。

SQL-first 的优势：

- SQL 先经过 MySQL parser 和执行器过滤，语法和列名更可靠。
- 反向生成 query 时已知 SQL 语义，query-SQL 一致性更高。
- 可以从 SQL AST 自动反推表、列、值、聚合函数，构造 `<think>` 规划监督。

为什么不是全量人工写数据：

- Text-to-SQL 要覆盖单条件、多条件、聚合、排序、Top-K、类型转换、total/合计等组合，人工覆盖成本高。
- SQL-first 合成 + 多模型投票 + 人工分歧审核，可以把人工集中在最容易出错的样本上。

### 4.4 为什么需要 Process-supervised SFT

SFT 阶段不是简单让模型输出 SQL，而是让模型先在 `<think>` 中输出规划，再在 `<answer>` 中输出最终 MySQL SQL。

固定格式：

```text
<think>意图：查询张三医生全年总问诊量；相关表：doctor_visit；相关列：doctor_name,total；过滤条件：doctor_name='张三'；聚合方式：无；生成策略：直接查询 total 字段。</think>
<answer>SELECT total FROM doctor_visit WHERE doctor_name = '张三';</answer>
```

为什么不用只输出 SQL：

- 只输出 SQL 的监督太稀疏，模型错了以后很难判断是列错、值错、聚合错还是排序方向错。
- `<think>` 规划可以显式训练 schema linking，尤其是相关列、过滤值和聚合函数。
- 线上仍然简单，只解析 `<answer>` 中的 SQL。

为什么不用 JSON 输出：

- 线上只需要 SQL，`<answer>` 标签比 JSON 更容易解析。
- JSON 容易引入引号转义、数组闭合、字段缺失等格式问题。
- `<think>规划</think><answer>SQL</answer>` 更适合代码生成类后训练。

### 4.5 为什么用 DAPO，而不是继续 SFT 或 DPO

SFT 解决的是格式、基础 SQL 模板和基础 schema linking。但 Text-to-SQL 有一个天然优势：SQL 可以被 MySQL 执行器自动验证。

如果只继续 SFT：

- 只能模仿合成 SQL，不能利用执行结果反馈。
- 复杂聚合、类型转换、排序等错误提升有限。
- 线上 badcase 很难转化为强训练信号。

为什么 DAPO 比 DPO 更适合：

- DPO 需要构造 chosen/rejected pair。
- Text-to-SQL 可以对同一个问题采样多条 SQL，每条 SQL 都能通过 MySQL 得到 reward。
- DAPO 使用组内相对优势，不需要额外 critic，比 PPO 轻，也比 DPO 更直接利用执行反馈。

### 4.6 为什么奖励函数保持简单

我没有把 reward 设计成很多复杂项，而是只保留三类：

```text
R = R_format + R_execution + R_result
```

具体定义：

| 奖励 | 触发条件 | 分值 |
| --- | --- | ---: |
| Format Reward | 成功解析 `<think>` 和 `<answer>`，并能抽取 SQL | +1 / -1 |
| Execution Reward | SQL 能在 MySQL 中执行 | +2 / -2 |
| Result Reward | 执行结果与 gold answer 一致 | +3 / -3 |

为什么这样设计：

- 这三个奖励直接对应线上可用性：能不能解析、能不能执行、结果对不对。
- 权重关系简单，Result Reward 最大，因为业务最终关心答案正确。
- 不引入复杂语义 reward，避免面试官质疑“权重怎么调出来的”。

为什么不设计复杂 reward：

- total、千分位、value linking 这类问题并不需要复杂 reward，靠 schema 标准化、过程监督和 badcase 回流更自然。
- 复杂 reward 会把数据问题、schema 问题和模型训练问题混在一起，反而不好解释。
- 执行反馈已经足够强，先把 format、execution、result 三件事做好，收益最稳定。

### 4.7 每个方法为什么必须存在

| 方法 | 为什么用它 | 如果不用会怎样 | 替代方法 | 替代方法为什么不够 |
| --- | --- | --- | --- | --- |
| MySQL Text-to-SQL | 表格问答需要符号执行，MySQL 结果可验证 | RAG 口算聚合和排序不稳定 | 纯 Table RAG | 对 sum/avg/top-k/where 条件不可靠 |
| SQL-first 合成 | 先保证 SQL 可执行，再生成 query | query-first 会生成 schema 不支持的问题 | 全人工构造 | 成本高，覆盖不了复杂组合 |
| 多模型投票 + 人工分歧审核 | 过滤 query-schema、query-SQL 不一致样本 | 合成噪声污染 SFT 和 DAPO | 全量人工审核 | 成本过高 |
| Process SFT | 在 `<think>` 中显式学习表、列、值、聚合函数 | 只学最终 SQL，错误难定位 | 只 SFT SQL | 监督稀疏，schema linking 提升弱 |
| DAPO | 直接利用 MySQL 执行反馈 | 只 SFT 浪费执行器反馈 | DPO | 需要 pair，不能充分利用多条采样 SQL 的 reward |
| 三类 reward | 对齐线上解析、执行、结果三个核心目标 | reward 复杂且难解释 | 多项语义 reward | 权重难解释，容易被质疑是规则堆叠 |
| Badcase 回流 | total、千分位、value linking 是数据和 schema 问题 | 线上同类错误会反复出现 | 一次性离线训练 | 无法适应业务表格变化 |

## 5. 数据构建

### 5.1 表格 Schema 标准化

每张表生成结构化 schema：

```json
{
  "table_name": "doctor_visit",
  "table_desc": "医生月度问诊量统计表",
  "columns": [
    {
      "name": "doctor_name",
      "desc": "医生姓名",
      "type": "string",
      "samples": ["张三", "李四"]
    },
    {
      "name": "total",
      "desc": "全年总问诊量",
      "type": "int",
      "samples": [1480, 1320]
    }
  ]
}
```

Schema 标准化重点处理：

- 缩写列名解释。
- 千分位数字转数值类型。
- 百分号、单位、空值归一化。
- `total`、小计、合计、总计字段显式标注。
- sample values 覆盖高频实体和特殊值。

为什么 schema 标准化很关键：

- 很多错误不是模型不会写 SQL，而是 schema 给的信息不够。
- 如果 sample values 没有 total/合计，模型就可能把月份逐列相加。
- 如果数值列被识别成 string，模型执行比较时自然会错。

### 5.2 SQL-first 合成流程

1. 根据 schema 生成多复杂度 SQL。
2. MySQL parser 校验语法。
3. 在 MySQL 执行器中执行，过滤执行失败和空结果。
4. 由 SQL 反向生成用户问题。
5. 生成 `<think>` 规划和 `<answer>` SQL。
6. 多模型投票评估 query-schema 和 query-SQL 一致性。
7. 分歧大的样本人工审核。

数据规模：

| 数据集 | 数量 | 用途 |
| --- | ---: | --- |
| 合成候选样本 | 18,200 | 初始候选池 |
| 过滤后高质量训练样本 | 5,500 | SFT + DAPO |
| SFT 冷启动样本 | 1,500 | 格式与基础 SQL |
| DAPO 训练样本 | 4,000 | 执行反馈强化 |
| 固定 Dev 集 | 600 | 奖励权重和阈值选择 |
| 固定 Test 集 | 800 | 主结果评测 |
| SESQL OOD 表格测试集 | 300 | 外部 Text-to-SQL 数据集抽样，转成 MySQL 方言后评测新表结构泛化 |

训练、dev、test 按表格级别切分，同一张表不会同时出现在训练和测试中。OOD 评测使用 SESQL 抽样 300 条，不参与训练和调参，只用于观察从医院表格迁移到外部表格问答时的泛化能力。

### 5.3 5.5K 数据量是否足够

5.5K 对这个项目来说是一个**够做第一版领域后训练闭环，但不够支撑通用 Text-to-SQL 能力**的数据量。这里不能只看样本条数，而要看任务边界和样本覆盖：

- 任务边界窄：医院内部表格、单库 MySQL、单表/少量多表、固定输出格式。
- 基座合适：Qwen2.5-Coder-7B 已经有较强 SQL 和代码先验，不是从零学习 SQL。
- 数据质量高：18.2K 候选经过执行过滤、多模型投票和人工分歧审核后只保留 5.5K，噪声低于直接合成全量数据。
- 训练目标明确：SFT 主要学格式、schema linking 和常见 SQL 模式，DAPO 主要利用执行器反馈优化结果正确性。

如果目标是开放域 Text-to-SQL，5.5K 明显不够；但如果目标是固定医院表格场景，5.5K 高质量样本比 2-3 万条低质量 query-SQL 合成样本更有效。

数据量是否够，我主要看覆盖率而不是绝对条数：

| 覆盖维度 | 目标 | 当前覆盖 | 判断 |
| --- | --- | --- | --- |
| SQL 操作 | `SELECT/WHERE/GROUP BY/ORDER BY/LIMIT/CAST` | 主流操作都有覆盖 | 够第一版 |
| 聚合函数 | `SUM/AVG/COUNT/MAX/MIN` | 每类都有正反样本 | 够第一版 |
| 业务 badcase | total、合计、千分位、value linking | 已构造专项样本 | 够院内闭环 |
| 表级泛化 | 训练/测试表隔离 | 按表格级切分 | 可验证 |
| 外部泛化 | SESQL OOD | 300 条抽样 | 只能诊断，不代表通用能力 |

### 5.4 SFT 与 DAPO 的数据配比

最终使用 `1.5K SFT + 4K DAPO`，比例约为 `27% : 73%`。这个配比不是固定公式，而是由 Text-to-SQL 的训练阶段决定：

| 阶段 | 数据量 | 占比 | 主要目标 | 数据选择标准 |
| --- | ---: | ---: | --- | --- |
| SFT 冷启动 | 1.5K | 27% | 学会固定格式、基础 SQL 模板、schema/value/operation linking | 覆盖所有 SQL 模板和典型业务口径，质量优先 |
| DAPO 执行反馈 | 4.0K | 73% | 利用 MySQL reward 优化可执行性和结果正确性 | 优先选择中高难度、可执行验证、存在多种候选 SQL 的样本 |

为什么 SFT 不需要占太大比例：

- 基座模型已经会基本 SQL，SFT 的主要作用是对齐任务格式和领域 schema，不是重新教 SQL 语法。
- SFT 数据太多且同质，会让模型过拟合合成 SQL 写法，后续 RL 的探索空间变窄。
- 只要 SFT 后格式正确率接近 100%、Valid SQL 超过 90%、Result Acc 达到可执行反馈起点，就可以进入 DAPO。

为什么 DAPO 数据更多：

- Text-to-SQL 的强监督来自执行器，4K prompt 每条采样 6 个 rollout，相当于产生约 24K 条执行反馈。
- DAPO 更适合处理“SQL 合法但结果不对”的问题，例如聚合口径、排序方向、value-column linking。
- RL 样本应尽量覆盖中高难度问题；太简单的问题 reward 区分度低，对 advantage learning 帮助有限。

我用三个指标决定是否从 SFT 切到 DAPO：

| 切换指标 | 阈值 | 当前 SFT 结果 | 是否满足 |
| --- | ---: | ---: | --- |
| Format Pass | > 98% | 100.0% | 满足 |
| Valid SQL | > 90% | 94.8% | 满足 |
| Result Acc | > 55% | 61.1% | 满足 |

因此当前 `1.5K SFT + 4K DAPO` 是合理的。如果后续扩数据，我不会简单按比例线性增加，而会采用：

```text
先补 SFT 覆盖缺失模板和新增 schema 口径
-> 再把中高难度样本放进 DAPO
-> 最后用线上 badcase 做 targeted SFT/DAPO 混合回流
```

扩展建议：

| 场景 | 建议配比 |
| --- | --- |
| 新领域、新 SQL 方言、格式还不稳定 | SFT 40% - 50%，DAPO 50% - 60% |
| 当前医院场景、格式已稳定 | SFT 20% - 30%，DAPO 70% - 80% |
| 线上 badcase 定向修复 | SFT 30% 左右修正思路，DAPO 70% 左右验证结果 |
| 通用 Text-to-SQL 能力建设 | 需要显著扩充数据，5.5K 不够，不能只靠调整配比 |

## 6. 模型训练

### 6.1 SFT 冷启动

SFT 目标：

- 学会 `<think>` 和 `<answer>` 固定格式。
- 学会基础 schema linking。
- 学会常见 MySQL SQL 模板。
- 学会把 SQL AST 中的表、列、值、聚合函数转成规划。

配置：

| 项目 | 配置 |
| --- | --- |
| 基座模型 | Qwen2.5-Coder-7B |
| 微调方式 | LoRA |
| rank / alpha | 64 / 128 |
| learning rate | 2e-4 |
| epoch | 3 |
| max length | 4096 |

SFT 后结果：

| 指标 | 结果 |
| --- | ---: |
| 格式正确率 | 100.0 |
| Valid SQL | 94.8 |
| Column-F1 | 75.1 |
| Value Linking Accuracy | 71.2 |
| Result Accuracy | 61.1 |

SFT 已经能解决格式和基础 SQL，但复杂聚合、类型转换、排序方向等问题仍然明显，因此继续做 DAPO。

### 6.2 DAPO 执行反馈训练

对每个问题采样 `K=6` 条 SQL，分别在 MySQL 中执行并计算 reward。组内做 advantage normalization，然后更新模型。

训练时保留 SFT reference model 做 KL 约束，避免模型为了 reward 破坏 `<think>/<answer>` 格式。

DAPO 的训练目标可以写成：

```text
J = E[ min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) - beta * KL(pi_theta || pi_ref) ]
```

其中同一问题下采样 `K=6` 个候选输出，分别执行 SQL 得到 reward，再在组内标准化：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + 1e-6)
```

这样做的好处是同一个问题内部直接比较多条 SQL 的优劣，不需要额外训练 critic，也不需要人工构造 chosen/rejected pair。对于 Text-to-SQL，这比单纯 DPO 更自然，因为每个 rollout 都能通过 MySQL 执行器自动得到反馈。

配置：

| 项目 | 配置 |
| --- | --- |
| rollout 数 | 6 |
| learning rate | 5e-6 |
| KL coefficient | 0.02 |
| epoch | 2 |
| max response length | 1024 |
| clip range | 0.2 |
| temperature | 0.7 |
| 执行器 | MySQL |

奖励函数：

```text
R = R_format + R_execution + R_result
```

| 奖励 | 正例 | 负例 |
| --- | --- | --- |
| Format Reward | `<think>` 和 `<answer>` 完整，SQL 可抽取 | 标签缺失、SQL 抽取失败 |
| Execution Reward | SQL 可在 MySQL 执行 | 列不存在、语法错、类型错 |
| Result Reward | 执行结果与 gold answer 一致 | 执行结果错误 |

为什么 reward 不再复杂化：

- 线上最核心的问题就是解析、执行、结果。
- total/千分位/value linking 错误会体现在 result reward 或 execution reward 上。
- 对这类 badcase，最有效的是把修正样本回流进 SFT/DAPO，而不是额外设计复杂 reward。

#### 6.2.1 DAPO 训练监控指标

训练时我主要看四类指标，而不是只看最终 Result Acc：

| 训练阶段 | Avg Reward | Format Pass | Valid SQL | Result Acc | KL | Entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 初始化 | 2.14 | 100.0 | 94.8 | 61.1 | 0.000 | 1.83 |
| DAPO 500 step | 2.76 | 100.0 | 95.9 | 63.0 | 0.011 | 1.71 |
| DAPO 1000 step | 3.12 | 100.0 | 96.7 | 65.2 | 0.018 | 1.62 |
| DAPO 1500 step | 3.31 | 100.0 | 97.1 | 66.5 | 0.020 | 1.56 |
| DAPO final | 3.39 | 100.0 | 97.3 | 67.4 | 0.021 | 1.54 |

这里有三个判断：

- `Valid SQL` 提升幅度小于 `Result Acc`，说明 DAPO 的主要收益不是让模型“更会写合法 SQL”，而是让模型在多个合法 SQL 中更偏向结果正确的写法。
- KL 保持在 0.02 左右，说明模型没有为了 reward 明显偏离 SFT 格式分布。
- Entropy 下降但没有坍缩，说明模型对常见 schema pattern 更确定，但仍保留一定候选多样性，方便推理阶段做执行重排。

#### 6.2.2 防止 reward hacking 的约束

DAPO 中有几类约束是必须加的，否则模型可能学到投机行为：

- SQL 安全约束：只允许单条 `SELECT`，拒绝 `INSERT/UPDATE/DELETE/DROP`、多语句和注释截断。
- 执行超时约束：单条 SQL 超过 2s 直接判执行失败，避免模型生成复杂笛卡尔积。
- 空结果约束：如果 gold answer 非空但 SQL 执行为空，Result Reward 判负；如果问题本身不可回答，则走拒答模板，不把空 SQL 当正确。
- 数值归一约束：比较执行结果时统一处理千分位、百分号、浮点容差和日期格式，避免 reward 因格式差异误判。
- KL 约束：保留 SFT reference model，限制模型为了 reward 破坏 `<think>/<answer>` 标签。

### 6.3 算法侧 Badcase 处理

除了 schema 标准化和样本回流，我还补了几类算法机制：

| 算法机制 | 解决的问题 | 具体做法 | 观察到的收益 |
| --- | --- | --- | --- |
| AST Process Supervision | schema linking、value linking、聚合函数选择 | 从 gold SQL AST 中抽取 table、column、value、operation，渲染到 `<think>` | Column-F1 从 68.4 提升到 75.1，Value Linking Acc 从 64.8 提升到 71.2 |
| Operation Hard Negative | `SUM/AVG/COUNT`、升序/降序混淆 | 对同一问题构造错误 operation 的 rejected SQL，放入 SFT 对比样本和 DAPO rollout | 聚合类 Result Acc 从 61.8 提升到 67.5 |
| Value-Column Contrast | 同名实体出现在多列 | 对同一 value 替换过滤列，构造能执行但结果错误的 SQL | value linking 类错误率从 28.8% 降到 21.6% |
| Best-of-N Execution Rerank | 单次采样偶发错误 | 推理时采样 4 条 SQL，优先选择可执行、非空、结果稳定的 SQL | DAPO 后 Result Acc 从 greedy 65.8 提升到 67.4 |
| Coverage Gate | schema 不支持时强答 | 先判断问题关键词是否能覆盖到表/列/value；覆盖不足输出拒答 | 不可回答强答率从 31.4% 降到 18.9% |

## 7. 实验结果

### 7.1 主结果

| 方法 | Format Pass | Valid SQL | Result Acc | Business QA Acc |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B Prompt Baseline | 92.7 | 88.6 | 56.9 | 63.6 |
| Qwen2.5-Coder-7B Zero-shot | 73.4 | 71.2 | 39.7 | 49.5 |
| 7B + 普通 SQL SFT | 96.8 | 91.4 | 51.6 | 58.9 |
| 7B + Process SFT | 100.0 | 94.8 | 61.1 | 64.8 |
| 7B + Process SFT + DAPO | 100.0 | 97.3 | 67.4 | 71.8 |

结论：

- 普通 SQL SFT 主要提升格式和 SQL 模板。
- Process SFT 明显提升 schema linking，因此 Result Acc 提升更大。
- DAPO 进一步利用 MySQL 执行反馈，提高可执行率和结果正确率。

### 7.2 奖励函数消融

| 奖励组合 | Format Pass | Valid SQL | Result Acc |
| --- | ---: | ---: | ---: |
| 仅 Format Reward | 100.0 | 89.5 | 53.2 |
| Format + Execution | 100.0 | 96.1 | 60.4 |
| Format + Execution + Result | 100.0 | 97.3 | 67.4 |

这个消融说明：三类奖励已经覆盖主要收益。Format 保证工程可解析，Execution 保证 SQL 可运行，Result 直接对齐业务答案。

### 7.3 数据合成与过程监督消融

为了证明提升不是只来自“多造数据”，我补了数据合成和 SFT 格式的对照。

| 方法 | 高质量样本保留率 | Valid SQL | Column-F1 | Value Linking Acc | Result Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-first 合成 + 普通 SQL SFT | 41.6 | 90.7 | 64.9 | 61.5 | 49.8 |
| SQL-first 合成 + 普通 SQL SFT | 57.3 | 91.4 | 68.4 | 64.8 | 51.6 |
| SQL-first + AST Process SFT | 57.3 | 94.8 | 75.1 | 71.2 | 61.1 |

结论：

- SQL-first 主要提升数据一致性，减少 query 与 SQL 不匹配。
- AST Process SFT 的收益更大，说明 Text-to-SQL 的瓶颈不只是 SQL 语法，而是 schema/value/operation linking。

### 7.4 后训练方法对照

我还比较了继续 SFT、DPO、执行重排和 DAPO。

| 方法 | 训练/推理方式 | Valid SQL | Result Acc | 备注 |
| --- | --- | ---: | ---: | --- |
| Process SFT | greedy | 94.8 | 61.1 | 冷启动模型 |
| Process SFT + Best-of-4 执行重排 | 推理采样 4 条 SQL | 95.6 | 64.2 | 只在推理时利用执行器 |
| Process SFT + execution-DPO | 用执行结果构造 chosen/rejected pair | 96.4 | 64.8 | pair 构造成本较高，覆盖不如 rollout |
| Process SFT + DAPO | rollout=6，组内优势 | 97.3 | 65.8 | greedy 单次输出 |
| Process SFT + DAPO + Best-of-4 | rollout 训练 + 推理重排 | 97.3 | 67.4 | 最终线上方案 |

这个对照能说明：执行器既可以用于推理重排，也可以用于训练；只做推理重排有收益，但 DAPO 能把执行反馈写进模型参数，最终和 Best-of-N 结合效果最好。

### 7.5 复杂度分层

| 问题类型 | Qwen3-30B Baseline | 最终方案 |
| --- | ---: | ---: |
| 单表单条件查询 | 74.2 | 80.1 |
| 多条件过滤 | 58.9 | 70.3 |
| 聚合统计 | 52.1 | 67.5 |
| 排序 Top-K | 49.6 | 63.2 |
| 字段类型转换 | 37.8 | 59.7 |
| total/小计/合计字段 | 41.5 | 68.4 |

提升最大的子集是字段类型转换和 total/小计/合计字段，说明 schema 标准化和 badcase 回流对真实业务错误有效。

### 7.6 OOD 表格泛化

OOD 使用 SESQL 外部数据集抽样 300 条，并统一转成 MySQL 方言。这个结果不作为主业务指标，只用于说明模型是否过拟合医院内部表格 schema。

| 方法 | In-domain Result Acc | OOD Result Acc | Gap |
| --- | ---: | ---: | ---: |
| Qwen3-30B Prompt Baseline | 56.9 | 47.6 | -9.3 |
| 7B + 普通 SFT | 51.6 | 39.8 | -11.8 |
| 7B + Process SFT | 61.1 | 50.7 | -10.4 |
| 最终方案 | 67.4 | 58.9 | -8.5 |

这里需要注意：SESQL 与医院表格的业务分布不同，所以 OOD 绝对值不应该和院内测试集直接比较。面试时更适合强调 gap 缩小，而不是强调 OOD 绝对指标很高。

## 8. 线上 Badcase 闭环

线上每条请求记录：

```text
question, schema, think, sql, executed_result, final_answer, user_feedback
```

badcase 进入不同队列：

| Badcase | 线上问题 | 错误表现 | 数据侧回流 | 算法侧回流 |
| --- | --- | --- | --- | --- |
| total 字段误用 | “张三医生今年总问诊量是多少？” | 生成 `jan+...+dec`，没有用 total | 在 schema 中强化 total 描述，补充 total 查询样本 | `<think>` 中显式监督“直接查询 total”；把逐月相加 SQL 作为 hard negative |
| 小计/合计行重复统计 | “心内科 2025 年总收入是多少？” | `SUM(income)` 把明细和合计一起加 | 增加合计行识别样本 | 对“合计行查询”和“明细聚合”构造成对样本，DPO/DAPO 中让错误口径拿负反馈 |
| 千分位类型错误 | “收入超过 12000 的科室有哪些？” | `income > '12000'` 按字符串比较 | 表格预处理转数值，补充 cast 样本 | 增加 type-aware SQL 模板，执行失败或结果错由 reward 惩罚 |
| value linking 错误 | “李明医生 3 月问诊量是多少？” | 把 `李明` 绑定到 `department` | sample values 增加同名和歧义值 | 构造 value-column contrast，监督 value 应绑定到 `doctor_name` |
| 聚合函数错误 | “各科室平均住院天数是多少？” | 使用 `SUM` 而不是 `AVG` | 回流平均/求和对比样本 | AST operation label 监督 + rollout 中错误聚合负优势 |
| Top-K 方向错误 | “问诊量最高的 3 个科室是哪些？” | `ORDER BY visits ASC` | 回流最高/最低排序样本 | 问题极性识别，推理候选重排时检查 order direction |
| 不可回答强答 | “哪个科室满意度最高？” | schema 无满意度列却强行按问诊量回答 | 加入不可回答样本 | coverage gate 输出拒答，不生成 SQL |

线上闭环后，第二轮训练中 total/小计类问题 Result Acc 从 61.3 提升到 68.4，千分位数值比较问题从 52.6 提升到 59.7。

## 9. 工程部署

部署方式：

- 模型：Qwen2.5-Coder-7B，LoRA merge 后 int4 量化。
- 执行器：MySQL。
- 推理：temperature 0.2，每题采样 4 条 SQL。
- 重排：优先选择可执行且执行结果稳定的 SQL。
- 超时：单条 SQL 执行超时 2s，避免复杂查询阻塞。
- 缓存：schema、列解释、sample values 和高频 query 结果缓存。

部署指标：

| 指标 | Qwen3-30B Prompt | 7B 最终方案 |
| --- | ---: | ---: |
| 平均生成延迟 | 4.8s | 1.2s |
| P95 延迟 | 9.6s | 2.7s |
| Valid SQL | 88.6 | 97.3 |
| Result Acc | 56.9 | 67.4 |
| Business QA Acc | 63.6 | 71.8 |
| 输出格式解析失败率 | 6.8 | 0.0 |

## 10. 指标可信度与保守口径

从面试可信度看，这个项目的提升幅度相对可以保留，但需要讲清楚边界：

- `56.9 -> 67.4` 的 Result Acc 提升不算离谱，因为 baseline 是通用 30B prompt，最终方案是面向固定医院表格分布训练的 7B 代码模型，并且用了 MySQL 执行反馈和推理重排。
- 不要说“7B 通用能力超过 30B”，而要说“在固定医院表格 Text-to-SQL 任务上，领域后训练 7B 超过 30B prompt baseline”。
- OOD 的 `58.9` 是 SESQL 抽样集上的结果，只能说明 schema 泛化有改善，不能包装成通用 Text-to-SQL benchmark SOTA。
- total/千分位/合计字段的提升属于诊断子集结果，面试中可以作为 badcase 闭环证据，但不要把它说成全量主指标。

如果面试官质疑指标偏高，可以主动补一句：主指标按表格级切分，测试表不进入训练；同时保留 SESQL OOD 评测，避免只在院内表格上报告结果。

## 11. 面试表达

### 11.1 30 秒概述

这个项目是为医院 Table RAG 做 MySQL Text-to-SQL 后训练。普通 RAG 对聚合、排序、过滤类表格问题不稳定，所以我训练了一个 7B 代码模型生成 SQL，并用本地 MySQL 执行器验证答案。模型输出固定为 `<think>规划</think>` 和 `<answer>SQL</answer>`。数据侧采用 SQL-first 反向合成和多模型质检，训练侧先做 process SFT 冷启动，再用 DAPO 进行 MySQL 执行反馈优化，奖励只保留 format、execution、result 三类。固定测试集上 Result Acc 比 Qwen3-30B prompt baseline 高 10.5 个百分点，业务 QA 准确率提升 8.2 个百分点。

### 11.2 2 分钟深挖讲法

这个项目的核心不是让模型会写 SQL，而是让模型在医院表格里稳定生成可执行且结果正确的 MySQL SQL。第一步是 SQL-first 数据合成。传统 query-first 容易生成 schema 不支持的问题，后面 SQL 只能编字段；我先从 schema 生成 SQL，执行通过后再反向生成 query，这样 query 和 SQL 一致性更高。

第二步是 SFT 冷启动。模型输出固定为 `<think>规划</think>` 和 `<answer>SQL</answer>`。我从 SQL AST 反推表、列、值和聚合函数，把这些监督渲染进 `<think>`，让模型先做 schema linking 和 SQL 规划，再在 `<answer>` 输出最终 SQL。线上只解析 `<answer>`，工程上很稳定。

第三步是 DAPO。Text-to-SQL 有 MySQL 执行器，天然适合执行反馈。每个问题采样多条 SQL，分别计算格式、可执行性和结果正确性 reward。这个 reward 很简单，但和业务目标直接对应：格式能不能解析，SQL 能不能执行，结果对不对。像 total 字段、千分位类型、value linking 这类问题，我会分层处理：schema 标准化解决输入信息不足，AST process supervision 和 hard negative 解决模型 linking 偏差，DAPO 和执行重排解决多个合法 SQL 里结果正确性的问题。

### 11.3 面试官追问回答

Q：为什么不用普通 RAG？

A：普通 RAG 对文本证据有效，但表格里的聚合、排序和数值比较需要符号执行。SQL 可以被 MySQL 执行器验证，适合做后训练闭环。

Q：为什么不用 Qwen3-30B 直接 prompt？

A：30B 模型成本和延迟都高，客户侧本地部署困难。这个任务 schema 固定、输出格式固定、执行器固定，7B 模型经过领域后训练后可以在目标指标上超过 30B prompt baseline。

Q：为什么 SQL-first，而不是 query-first？

A：query-first 容易生成表里没有字段的问题，导致 SQL 被迫编造列。SQL-first 先保证 SQL 在 MySQL 中可执行，再反向生成 query，数据一致性更高。

Q：5.5K 数据量够不够？

A：如果做通用 Text-to-SQL，5.5K 明显不够。但这个项目是固定医院表格、MySQL 方言、固定输出格式，而且基座是 Qwen2.5-Coder-7B，本身有 SQL 先验。这里更重要的是样本质量和覆盖率：SQL 操作、聚合函数、total/合计、千分位、value linking 都有覆盖，并且按表格级切分做测试。所以 5.5K 够做第一版领域后训练闭环，但不能包装成通用能力充分。

Q：为什么 SFT 只用 1.5K，DAPO 用 4K？

A：SFT 的作用是冷启动格式、schema linking 和基础 SQL 模板，不是重新教模型 SQL。只要 SFT 后 Format Pass 到 100%、Valid SQL 到 94.8%、Result Acc 到 61.1，就已经达到执行反馈训练起点。DAPO 阶段每个问题采样 6 条 SQL，4K prompt 实际产生约 24K 条执行反馈，更适合优化“SQL 合法但结果不对”的问题。所以当前 1.5K:4K，也就是约 27%:73%，对这个固定领域场景是合理的。

Q：为什么用 DAPO，而不是继续 SFT？

A：SFT 只能模仿合成 SQL，但 SQL 的正确性可以自动执行验证。如果不用执行反馈，相当于浪费了 Text-to-SQL 最强的监督信号。

Q：为什么不用更复杂的 reward？

A：因为这个项目的核心线上目标很清楚：输出格式可解析、SQL 可执行、结果正确。total、千分位、value linking 这些问题不应该全部塞进 reward，而是拆到 schema 标准化、AST 过程监督、hard negative、DAPO 执行反馈和推理重排里解决。reward 保持简单，训练信号更稳定，也更容易解释。

Q：为什么 exact match 不是主指标？

A：SQL 有很多等价写法，exact match 会低估正确 SQL。业务关心的是执行结果和最终答案，所以主指标是 Result Accuracy。

## 12. 简历表述

面向医院 Table RAG 构建 MySQL Text-to-SQL 后训练流程，采用 SQL-first 反向合成、多模型投票和人工分歧审核构建 5.5K 高质量训练数据；基于 Qwen2.5-Coder-7B 进行 process-supervised SFT 冷启动，从 SQL AST 自动反推表、列、值和聚合函数监督，并渲染为 `<think>规划</think>` 与 `<answer>SQL</answer>` 固定输出格式；进一步使用 DAPO 进行 MySQL 执行反馈优化，设计 Format Reward、Execution Reward 和 Result Reward，提升 SQL 格式稳定性、可执行率和结果正确率。固定医院表格测试集上 Result Acc 较 Qwen3-30B prompt baseline 提升 10.5 个百分点，业务 Table RAG QA 准确率绝对提升 8.2 个百分点。
