# 做LLM强化学习，选在线RL还是离线RL？

LLM 强化学习训练面临一个核心架构选择：是使用在线 RL（Online RL）——在训练中实时采样当前策略的 rollout，还是使用离线 RL（Offline RL）——利用预先收集好的固定数据集进行优化？

这两种方法在数据效率、计算成本、训练稳定性和最终性能之间存在根本性的权衡，理解这一权衡是设计 LLM 训练系统的核心决策。

DPO（Direct Preference Optimization）是最成功的离线 RL 代表，PPO/GRPO 是在线 RL 的主流实现，而最新趋势是将两者结合，形成迭代式在线-离线混合方法（如 Online DPO、Iterative RLHF）。

## 01 在线 RL 与离线 RL 的本质区别

（1）在线 RL（PPO/GRPO）

在每个训练步骤，使用当前策略生成新的 rollout，然后立即用这些新数据进行参数更新：

数据流：当前策略 π_θ→ 生成 rollout → 计算奖励 → 更新参数 → 新策略π_θ' → 生成新 rollout → …

核心特性：

训练数据始终来自当前策略，无分布偏移问题

每次策略更新都能探索当前策略能力边界附近的新区域

计算开销高：每步需要完整的推理前向传播（通常占总计算的 60-70%）

需要实时奖励模型或验证器（无法使用批量预标注数据）

（2）离线 RL（DPO/RAFT）

使用预先收集好的固定数据集（通常是偏好数据对）进行优化，无需实时采样：

数据流：固定偏好数据集 D ={(x,y_w,y_l)} → 计算 DPO/SFT 损失 → 更新参数

核心特性：

数据复用效率高：一份数据集可以多轮训练

无需实时推理，GPU 利用率高（无生成开销）

存在分布偏移：数据由初始策略生成，随训练进行数据越来越不代表当前策略分布

无法探索训练集之外的新解法（被数据集限制）

## 02 关键性能权衡

（1）数学推理任务

在线 RL（GRPO）的优势明显：

每轮训练生成新的推理路径，不受初始数据集中推理风格的束缚

可以自主发现数据集中不存在的推理策略（如不寻常但有效的解题路径）

DeepSeek-R1 实验显示，GRPO 在 MATH 数据集上比相同数据量的 DPO 高约 5-10%

离线 DPO 的局限：

DPO 数据集中的正确推理路径由某个特定模型生成，当前训练模型可能”无法理解”该路径（超出当前能力范围），导致监督信号无效

缺少对“接近正确但有缺陷”的推理路径的细粒度对比，负样本质量有限

（2）通用对话任务

离线 DPO 表现良好：

对话质量的偏好较稳定（礼貌性、格式、完整性），不随策略能力强烈变化

高质量人工偏好数据集（如 HH-RLHF、UltraFeedback）包含了大量多样化场景

计算成本远低于在线 RL，适合资源有限场景

在线 RL 的优势不明显：

通用对话任务的”最优回答”不那么唯一（创意性更强），在线探索的收益较小

PPO 在对话任务中的 Critic 训练难度高（难以精确估计对话价值）

## 03 迭代式在线-离线混合方法

（1）Online DPO（ODPO）

结合了在线 RL 的数据新鲜度和 DPO 的简单性：

流程：

从当前策略 π_θ 对每个 prompt 采样 2 个候选回答

用奖励模型/LLM-as-Judge 判断哪个更好，生成新的偏好对

用新偏好对做一步 DPO 更新

循环

Online DPO 的数据始终来自当前策略，避免了离线 DPO 的分布偏移问题，同时保留了 DPO 的简单性（无需 Critic）。

实验表明，Online DPO 比离线 DPO 在 MT-Bench 上高约 5-8%，且计算成本仅为 PPO 的 30-40%。

（2）Iterative RLHF（迭代式强化学习）

3 阶段迭代循环：

阶段 1（离线 SFT）：使用高质量示例数据进行监督微调，建立基础能力

阶段 2（在线 RL）：使用 PPO/GRPO 强化关键能力（数学推理、安全性），探索当前数据分布边界

阶段 3（离线 DPO/RLHF）：收集在线 RL 产生的新数据，经人工/AI 标注后更新偏好模型，为下一轮迭代准备

LLaMA-2、LLaMA-3 的训练均采用多轮 Iterative RLHF，每轮 5000-20000 条新偏好标注，已进行 3-5 轮迭代。

（3）Rejection Sampling Fine-tuning（RSFT/RAFT）

离线 RL 的一种特殊形式，利用在线采样但以 SFT 方式训练：

从当前策略对每个 prompt 采样 K（如 64）个候选回答

用奖励模型选择得分最高的候选作为”最优回答”

对这些”最优回答”做 SFT（而非 RL）

循环

优点：

无需 Critic，无 PPO 的训练不稳定性

SFT 训练稳定且高效

每轮使用当前策略采样，隐式解决了分布偏移

缺点：

无法直接利用“负样本”信息（与 DPO 相比）

每次需要采样 K 个候选，推理成本高

## 04 选择方法的决策树

任务有可验证奖励（数学、代码）？├── 是 → 在线RL（GRPO/PPO）效果最好，计算预算允许则首选└── 否（主观任务）├── 有高质量偏好数据？│ ├── 是 → 离线DPO（简单高效）│ └── 否 → Online DPO（自动生成在线偏好数据）└── 计算预算充足？├── 是 → Iterative RLHF（最佳效果）└── 否 → RSFT（RAFT，平衡性价比）

## 05 工程实践建议

资源有限的团队优先离线 DPO：实现简单，无需推理基础设施，一块 A100 可以训练 7B 模型。

高性能推理任务（数学/代码）必须使用在线 RL：离线数据的分布局限会严重限制推理能力上限。

Iterative RLHF 是工业级对齐的标准：Meta、Google、Anthropic 均采用多轮迭代，每轮更新数据和奖励模型。

Online DPO 是在线 RL 和离线 DPO 之间的最优折衷：比 DPO 效果好 5-8%，比 PPO 计算成本低 50-70%。

参考文献：

Rafailov et al., Direct Preference Optimization: Your Language Model is Secretly a Reward Model, NeurIPS 2023

Dong et al., RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment, TMLR 2023

Guo et al., Direct Language Model Alignment from Online AI Feedback, arXiv 2024

Touvron et al., LLaMA 2: Open Foundation and Fine-Tuned Chat Models, 2023

作者：硅基趣玩喵

来源：https://zhuanlan.zhihu.com/p/2016080986690041206
