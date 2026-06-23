# 面了个算法大佬：GRPO和DAPO差在哪？

最近面了一个学术界做科研出身的 LLM 算法候选人，大佬能力很强，他在面试中回答的内容很值得我学习，特此记录。

我问了一个开放问题：网上很多人说现在的 GRPO、DAPO 这些方法本质上是 recipe engineering 而不是算法突破。你怎么看？

他的回答和我预想的有些差异，很有趣，并不是因为他说得不对，而是他看这些方法的角度跟我很不一样。

他是从算法机制出发的，我是从训练实际出问题时怎么修出发的。聊下来我发现这两个角度拼在一起，才是对这些方法比较完整的理解。

下面把我觉得有价值的部分用 AI 整理出来。

## 01 回答

他说了一个我觉得很准确的判断：

我觉得不能按算法名字去看这些东西。GRPO、DAPO 本质上是一堆针对 RL 训练过程中具体问题的组件，拼在一起就是一个方法。

我拿 DAPO 举例，拆开来看的话，其实每个 trick 都是在解决真实的问题：

Token-level loss：做 long-CoT 的时候 response 长度差异很大，如果按 response 级别平均 loss，一个 2000 token 的回答和一个 200 token 的回答对梯度的贡献差了一个数量级，这肯定不合理。

Token-level loss 就是把信号打散到每个 token 上，让长短样本的贡献更均衡。

Clip-Higher：PPO/GRPO 里 ratio 被 clip 在一个对称区间内，比如 0.8 到 1.2。DAPO 把它改成非对称的，下界还是 0.8，但上界放宽到 1.3。

下界不动是因为你不希望某些 token 概率被压得太低，那样采样空间会坍缩；上界放宽是为了让那些偶尔被采到的好 token 能更快地提升概率，不然它们被 clip 卡住学不动。

Dynamic Sampling：同一个 prompt 采样一组 response，如果全对或全错，group 内没有区分度，advantage 算出来全是零，没有有效梯度。

Dynamic Sampling 就是把这些无效 prompt 过滤掉，继续采样直到 batch 填满有信号的样本。

DAPO 的 ablation 里这个组件单项收益是最大的，比 Clip-Higher 和 Token-level Loss 大不少。

直觉上也好理解：如果一个 prompt 太简单或太难，采多少条都是一样的结果，这些计算本质上是浪费的。

Overlong penalty：这个其实是两层。第一层比较粗暴，超长被截断的样本直接不算 loss，mask 掉；第二层柔和一些，快到长度上限的时候逐渐加惩罚。实际效果来看，第一层（直接 mask）对稳定训练帮助更大。

还有一个比较大胆的选择：DAPO 完全去掉了 KL penalty，不维护 reference model。

因为做 long-CoT 推理的时候，模型需要大幅偏离初始分布才能涌现新能力，KL 约束在这个场景下反而是拖后腿的。

这个分类方式本身我完全同意：要按 failure mode 理解，不按论文标题理解。

## 02 我的视角

工业界做训练，分析问题的出发点和学术界不太一样，重点是基于问题找解法，而不是基于方法找问题。

我不会先想怎么基于这个组件的数学性质找问题，而是先看训练跑起来之后分析哪里有问题，再对症下药找解法：

wandb 上看 log，例如 entropy 前几百步就塌了，说明探索出了问题，可能会考虑试试用 Clip-Higher 或改 temperature；

batch 里很多 prompt 的 reward 全是 0/1，应该是有效信号太少，可以试试 Dynamic Sampling ，NGROP或者换数据配比；

accuracy 在涨但平均输出长度从 600 飙到 2000，代表有 length bias，应该看 overlong penalty 或者 reward 设计本身；

KL 在某个阶段突然起飞，可以推测模型被 reward 推偏了，要看是 clip 不够紧、还是 rollout 太多步没更新 reference，或者是某类 prompt 的 reward 信号本身有问题。

他是从组件出发找 failure mode，我是从 failure mode 出发找组件。方向相反，但最终指向同一个结论：这些 trick 不是独立的算法突破，是诊断训练问题后的对症处理。

当然，从我个人的经验里还有一层可以展开的：同一个 failure mode 的处理方式不止一种，选哪个要看你的约束条件。

比如有效 batch 信号太少，你可以用 Dynamic Sampling，也可以调数据配比让中等难度的 prompt 占比更高，也可以增大 group 的采样数。

选哪个取决于当时的 GPU 预算、数据灵活度、训练流水线的复杂度容忍度。组件本身只是解法之一，不是唯一解。

## 03 追问：你怎么判断一个 trick 是真的有用？

这里我们聊得比较深。

候选人的方法论总结

他给了一个三步判断链，我觉得思路很清晰：

第一步，看直接证据。比如 Clip-Higher：统计 correct response 中 token ratio 触达 upper clip 的比例。

如果加之前有很多被卡住，加之后放开了，涨分跟机制对得上。如果加之前就没怎么触达，涨分可能来自别的地方。

第二步，排除混淆。分数涨了但 response length 翻倍，可能是 token budget 换来的。

pass@1 涨了但 pass@k 掉了，说明搜索空间变窄了。in-domain 涨了但 OOD 掉了，benchmark recipe。

第三步，实验设计。incremental ablation，每次只加一个组件，固定 base model、数据、rollout budget、reward、max length、decoding 参数，跑多个 seed。

不只看 final score，看训练过程中 entropy、KL、length、reward variance 的变化曲线。

## 04 我追问了一个问题：那你实际做下来，哪些情况你觉得涨分是假的？

他列了几种：

分数涨了，输出长度却变长甚至翻倍；

pass@1 涨了，但 pass@k 掉了，entropy 暴跌；

KL 失控，模型被 reward 推得太远；

只在特定 benchmark 涨，换格式换难度就掉；

论文同时改了七八个东西，ablation 不干净。

## 05 我的理解

他这套方法论在逻辑上没问题。但我会多关注几个他没提到的部分：

首先是组件之间的交互效应。他讲的是 incremental ablation，每次加一个。但实际上组件之间不一定是独立的。

Dynamic Sampling 改变了 batch 的难度分布，这会影响 Clip-Higher 的作用（因为中等难度的 prompt 里 correct token 触达 upper clip 的比例跟极难 prompt 里完全不同）。

我见过 Dynamic Sampling 单独加涨 3 个点，Clip-Higher 单独加涨 1.5 个点，两个一起加反而只涨 2 个点的情况。

还有一个相关的坑：GRPO 一个 group 里如果多条正确 response 几乎一样（数学场景里正确答案格式高度一致），advantage 估计的方差会被严重低估。

实际做的时候需要对 group 内做 near-duplicate filtering，只保留足够 diverse 的样本来算 baseline，否则梯度信号会退化。或者可以用NGRPO，加一个虚拟满分样本。

GRPO 本身的数学偏差。这个跟 DAPO 的 token-level loss 相关但角度不同。

Dr.GRPO 里面也写过 GRPO 原始实现里除以 response 长度 1/|o_i| 会引入一个隐蔽的 bias：正 advantage 时偏好短回答（因为短 response 的梯度没被长度稀释），负 advantage 时对长错误回答惩罚不足。

另外除以 group 内 reward 的 std 也会导致太简单/太难的 prompt 获得过大的有效学习率。

修复方法反而很简单去掉这两个归一化。这说明有时候问题不是该加什么组件，而是原始实现里有什么隐含 bug。

工程复杂度 vs 收益的取舍。这是学术论文不会讨论的。Dynamic Sampling 意味着 batch 里有效 prompt 数量不确定，训练吞吐不稳定，要写额外的调度逻辑。

如果它只带来 1 个点的提升，在生产环境里这个工程成本可能不值得。学术界不需要考虑这个，但工业界每个 trick 都有维护成本。

同一个 trick 在训练不同阶段的表现差异。Clip-Higher 在训练早期帮助很大：正确轨迹稀疏，模型偶尔采到好路径但 clip 卡着学不动。

但训练后期模型已经收敛得差不多了，继续放宽 upper clip 反而可能加速 KL 飘移和模板化。

我的做法是前期开、后期收，或者根据 correct token 的 clip 触达率动态调整。

如果做多轮迭代训练（iterative RLHF），这一点更明显：经过几轮之后剩余的被 clip 卡住的高 advantage 样本大概率是 reward hack 而不是真正的好 response，继续放大它们的梯度等于加速 over-optimization。

他讲的是静态地分析一个组件该不该加，这是常见的科研思维，但实际训练中这些组件往往需要随阶段调整。

Entropy 看起来正常不代表模型没出问题。Token-level entropy 数值还健康（比如 >2.0），但模型可能已经 collapse 到几个固定的长模板上，因为这些模板本身足够长且内部词汇有一定多样性，平均 token entropy 看起来还过得去。

更可靠的诊断方法是看 group 内 response 之间的 pairwise BLEU 或 embedding similarity：同一 prompt 下多个 rollout 的 pairwise BLEU-4 超过 0.3（正常应该 <0.15），模型大概率已经实质性 collapse 了。

我一般会组合监控三个指标：token entropy + unique response rate（去重后比例，正常 >80%）+ length CV（std/mean，正常 >0.3），任意两个同时恶化就要停下来检查。

## 06 追问：这些方法能搬到哪些场景里？

这个问题他答得也很好：

不能直接搬。DAPO 的 recipe 是为 long-CoT 数学推理设计的。Agent 场景里目标完全不同：输出要短、延迟要低、工具调用要准。response length 变长在这个场景里是负面的。

做 GRPO 时我会把 reward 拆成路由准确率、工具调用合法性、任务完成率、响应长度、延迟。

关注的是 tool-call valid rate 和 latency-aware reward，不是长推理能力。

DAPO 的价值不在于 recipe 能照搬，而在于它展示了一种思路：针对具体 failure mode 做组件化修正。思路可以迁移，recipe 不行。

我个人从实际项目的理解：Agent 场景里 Dynamic Sampling 的行为跟数学场景很不同。数学题的 reward 是 binary 的（对或错），group 内全对全错很好判断。

但 Agent 场景的 reward 往往是多维度加权的，一个 response 可能工具调用对了但格式不好，另一个格式好但调用错了：group 内的 reward variance 一直存在，Dynamic Sampling 的过滤逻辑需要重新设计。直接搬 DAPO 的实现会发现它几乎不过滤任何东西，等于没加。

## 07 最后

聊完之后我的感受是：他对算法机制的理解很扎实，failure mode 的分类、归因的方法论都是对的。

我们的差异主要在于，他更关心这个组件在数学上解决什么问题，我更关心加了这个东西之后训练pipeline会变成什么样。

这两个视角不是谁对谁错：

理解机制，才能在新场景下判断一个组件该不该用：而不是靠上次这么搞调有效的经验。

从实际完成项目的角度出发，而不是单纯考虑提点，则会注意到组件之间的交互、部分论文中刻意忽略的scale 差异、阶段性变化这些东西。

不过我这个按 failure mode 理解的框架本身也有局限：它假设 failure mode 是可分解的、正交的。

但实际中最棘手的情况是 failure mode 之间的耦合：修复 entropy collapse 的手段（如放宽 clip）可能加剧 reward hacking；修复 reward hacking 的手段（如收紧 KL）可能触发 mode collapse。

当系统有三个以上相互耦合的 failure mode 时，按单一 failure mode 诊断容易陷入打地鼠无限循环，所以要尽量多读 paper 来提升自己的判断力。

另外一个值得留意的趋势：当 policy 已经比较强的时候，GRPO 的 group 内对比开始退化：同一 prompt 下的多个 response 质量非常接近，group 内方差趋近于零，梯度信号退化为噪声。

这时候最朴素的方法：生成 N 个，挑最好的，做 SFT（即 rejection sampling 迭代），反而可能比继续做 PO 更稳定。这不是说 PO 无用，而是它可能在训练的不同阶段有不同的性价比。

如果只记一句话：PO 方法按 failure mode 选组件，但选完之后还要看组件之间怎么配合、在当前 scale 和训练阶段是否还成立。单个组件的机制分析是起点，不是终点。

作者：亚马逊 AWS 数据科学家，刘明

来源：https://zhuanlan.zhihu.com/p/2050939460137775428
