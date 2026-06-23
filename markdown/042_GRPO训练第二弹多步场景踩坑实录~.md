# GRPO训练第二弹：多步场景踩坑实录~

上一篇我们把工具选择准确率从 63% 做到了 97%。模型上线后，我们以为可以收工了。直到拆开线上数据一看——多步场景的准确率只有 63%。又回到了原点。

## 01 97% 的假象

模型上线跑了两周，整体指标很漂亮：tool 准确率稳定在 95%+，但分场景统计时发现：多步组合场景下（例如"如果余额够买 X 股，就帮我用下单"）的准确率只有**63%**。

## 02 多步场景到底难在哪

拉了几十条多步场景的错误 case，发现一个规律：模型的 reasoning 写得头头是道，但最终选的 tool 经常是错的。

典型错误：

用户:"先查一下余额，如果够的话帮我转 10 万到理财账户"模型输出:{"thought":"用户想先确认余额是否充足，然后转账。需要先查余额。","action": {"tool":"transfer_funds","params": {"amount":100000}},"plan": ["check balance","transfer if sufficient"]}

reasoning 完全正确——"需要先查余额"。但 action 直接跳到了 transfer_funds。thought 和 action 脱节了。

单步场景下不会出现这个问题。"查余额" 只对应 check_balance，"买基金" 只对应 place_order，意图和工具是一对一映射。

但多步场景里，一句话同时涉及 check_balance 和 transfer_funds，模型需要理解执行顺序，选出当前步骤应该执行的工具。这对推理能力的要求完全不同。

问题出在两个层面：一是输出格式不够用，二是奖励函数的精度跟不上。

## 03 奖励函数升级：从规则匹配到 Ground Truth 比对

上一版的奖励函数是基于规则的：检查格式是否完整、工具是否在合法列表里、有没有 reasoning。这在单步场景下够用——格式正确的输出，tool 大概率也是对的。

但多步场景下，check_balance 和 transfer_funds 都是合法工具，格式都完整，规则匹配给出的分数几乎一样。奖励函数区分不了"对的第一步"和"对的第二步"。

升级方向很明确：直接比对 Ground Truth。不再用规则猜测，而是拿模型输出和正确答案做精确匹配。

classMultistepGTReward:def__call__(self, prompts, completions, **kwargs):solutions = kwargs.get("solution", [])rewards = []forcompletion, solutioninzip(completions, solutions):parsed = parse_json(completion)expected_tool = parse_json(solution).get("action", {}).get("tool")actual_tool = parsed.get("action", {}).get("tool")# Tool correctness — 占最大权重ifexpected_toolandactual_tool == expected_tool:t_score =0.5# 选对了elifactual_toolinVALID_TOOLS:t_score =0.1# 选错了但合法，保留格式激励else:t_score =0.0# Params bonusp_score =0.2ifhas_valid_params(parsed)else0.0# Reasoning + Planr_score = reasoning_score(parsed)# 最高 0.3rewards.append(t_score + p_score + r_score)returnrewards

分数设计的几个考量：

Tool correctness 占 0.5，是最大的单项分。选对和选错的分差是 0.4，足够产生明确的学习信号

选错但合法给 0.1，不是 0。完全不给分会导致模型连格式都不愿意输出——它会发现"什么都不输出"比"输出错误工具"的期望 reward 更高

Reasoning 和 Plan 保留为辅助信号，鼓励模型思考，但权重明确低于正确性

原则就一条：你最关心什么，就把最大的分给什么。

## 04 Multistep 格式强化

奖励函数升级解决了"怎么评判"的问题，但多步场景还需要解决"怎么表达"的问题。原来的输出格式太紧凑了，模型没有足够的空间做深入推理。

几个调整：

加长思考空间：max_completion_length 从 200 提到 350。原来 200 token 对简单场景够用，但多步推理需要模型把"先做 A，因为 B，然后 C"这个链条写清楚。

增加探索多样性：num_generations 从 4 提到 8。多步场景的正确答案不那么显然，需要更多采样才能覆盖到正确的推理路径。

奖励推理深度：thought 里包含逻辑关键词（"first"、"then"、"because"、"before"）额外加分。

不是为了凑字数，而是鼓励模型显式地把推理链写出来——写出来的推理比"心算"更可靠。

LOGIC_KEYWORDS = ["first","then","before","because","therefore","need to","in order to","since"]# thought 中包含逻辑词 → +0.05ifany(kwinthought.lower()forkwinLOGIC_KEYWORDS):r_score +=0.05

Plan 字段从装饰变成核心：之前 plan 只是可选的加分项，现在要求 plan 非空才给分。多步场景的 plan 不是锦上添花，是模型理解任务结构的证据。

## 05 数据准备

多步训练数据需要特殊处理。每个多步场景会展开成多条训练样本，每条对应执行链中的一步：

场景:"查余额，够的话转账 10 万"样本1(第一步):input:"查余额，够的话转账 10 万"output: {"thought":"先查余额...","action": {"tool":"check_balance"},"plan": ["check","transfer"]}样本2(第二步):input:"查余额，够的话转账 10 万\n\nPrevious steps:\nStep 1: check_balance → 余额 50 万"output: {"thought":"余额充足，执行转账...","action": {"tool":"transfer_funds"},"plan": ["transfer"]}

第二步的 input 包含了前一步的执行结果，模型需要基于上下文决定下一步操作。这比单步场景难得多——模型不仅要理解用户意图，还要理解当前执行状态。

最终数据集：454 条基线单步数据 + 823 条多步展开数据。

## 06 NGRPO 依然不可或缺

跑完训练看 NGRPO 统计：Zero std: 3243/5108 (63.5%)。

超过六成的 group 内所有生成结果的 reward 完全相同。如果没有 NGRPO 的虚拟满分样本注入，这些 group 的梯度就是零，模型什么都学不到。

有意思的是，上一轮简单场景训练时 zero-std 比例是 67%，现在多步场景反而降到了 63.5%。

可能是因为多步场景更难，模型的生成结果更分散，reward 方差自然更大一些。

不管怎样，NGRPO 不是一次性的 trick，而是 GRPO 训练的基础设施。只要你的任务存在大量"模型已经会了"的样本，就需要它。

## 07 结果

多步场景准确率：63% → 97%。

整体准确率维持 97%，单步场景没有退化。

剩余 3 个错误全是同一类 case：

"Ineedtotake out75rupeesfromACC888" → 模型选 withdraw，标注是 check_balance"Withdraw50rupeesfromACC002" → 模型选 withdraw，标注是 check_balance"Made profit in day trading. CanIwithdraw it tonight?" → 模型选 withdraw，标注是 check_balance

用户明确说了"取钱"、"withdraw"，模型选 withdraw 完全合理。标注认为应该先 check_balance 确认余额，也有道理。这是标注层面的歧义，不是模型的问题。

97% 大概就是这个数据集的天花板了。

## 08 经验总结

奖励函数的精度要跟着任务难度升级。简单任务用规则匹配够了，复杂任务需要 Ground Truth 直接比对。不是一开始的方案有问题，而是场景变了，方案要跟着变。

分数分配反映优先级。Tool correctness 0.5，Params 0.2，Reasoning 0.3——你最关心什么，就把最大的分给什么。

给模型足够的思考空间。completion length、generation 数量都要匹配任务复杂度。多步推理需要更长的输出和更多的探索。

知道什么时候该停下来。剩余错误是标注歧义时，继续优化模型没有意义。

作者：LLMCat，亚马逊 数据科学家

来源：https://zhuanlan.zhihu.com/p/2012306320288658180
