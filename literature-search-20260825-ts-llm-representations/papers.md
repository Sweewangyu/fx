# 时序大模型表示与路由：筛选论文清单

> 检索日期：2026-08-25  
> 用途：支撑《时序大模型技术路线与下一步计划》内部汇报  
> 评分：相关性 / 证据强度 / 可复现性，满分 5 分；质量标签 A（核心证据）、B（重要补充）、Risk（负结果或创新碰撞）。

## 最终筛选（15 篇）

| # | 论文 | 年份/来源 | 类型 | 核心结论与本项目关系 | 评分 | 标签 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | [Large Language Models for Time Series: A Survey](https://arxiv.org/abs/2402.01801) | IJCAI 2024 | Survey | 给出 alignment、tokenization、prompting 等总览，用于建立分类口径。 | 4/4/4 | B |
| 2 | [Time-LLM](https://arxiv.org/abs/2310.01728) | ICLR 2024 | Pure method | patch + reprogramming + frozen LLM，是独立时序前端路线的经典基线。 | 5/5/4 | A |
| 3 | [TEST](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a4352e2c9d93582898a2a20e1f514e8f-Abstract-Conference.html) | ICLR 2024 | Pure method | 独立 TS tokenizer/encoder，并用多层对比学习对齐文本原型。 | 5/5/4 | A |
| 4 | [ChatTS](https://arxiv.org/abs/2412.03104) | PVLDB 2025 | Method + benchmark | 以 patch/MLP 编码时序并插入文本位置，直接覆盖时序理解与推理。 | 5/4/4 | A |
| 5 | [AutoTimes](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dcf88cbc8d01ce7309b83d0ebaeb9d29-Abstract-Conference.html) | NeurIPS 2024 | Pure method | 将时序片段投影到 token 空间并自回归生成，支持可变预测长度。 | 4/5/4 | A |
| 6 | [Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820) | NeurIPS 2023 | Pure method | 数值字符序列直接进 LLM；证明 tokenizer、精度和分隔符决定纯文本路径表现。 | 5/5/4 | A |
| 7 | [PromptCast](https://arxiv.org/abs/2210.08964) | TKDE 2023 | Method + benchmark | 把历史值与预测问题改写为 sentence-to-sentence，建立 prompt 路线。 | 4/4/3 | B |
| 8 | [LSTPrompt](https://aclanthology.org/2024.findings-acl.466/) | Findings of ACL 2024 | Pure method | 用长短期提示分解注入时序机制，是纯文本 prompt 的增强基线。 | 4/4/3 | B |
| 9 | [Are Language Models Actually Useful for Time Series Forecasting?](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) | NeurIPS 2024 | Empirical audit | 移除或替换 LLM 往往不降低 forecasting 表现，要求单独证明语言能力贡献。 | 5/5/5 | Risk |
| 10 | [Revisiting LLMs as Zero-Shot Time-Series Forecasters](https://aclanthology.org/2025.acl-short.71/) | ACL 2025 | Empirical audit | 纯 prompt 零样本预测对小噪声敏感，提供稳健性压力测试协议。 | 5/5/5 | Risk |
| 11 | [ChatTime](https://ojs.aaai.org/index.php/AAAI/article/view/33384) | AAAI 2025 | Method + benchmark | 10K 量化 bins 加入词表，约一值一 token，是数字扩词表的直接基线。 | 5/5/4 | A |
| 12 | [Continuity and Ordinality Matter](https://arxiv.org/abs/2605.28866) | arXiv 2026 | Pure method | 用有序流形初始化及序数/单调正则修复新数值 token 的几何结构。 | 5/3/4 | A / Risk |
| 13 | [TimeOmni-1](https://arxiv.org/abs/2509.24803) | ICLR 2026 | Method + benchmark | 同基座 text/encoder 对比显示质量、有效回答率、显存和延迟存在交叉。 | 5/5/4 | A |
| 14 | [OpenTSLM](https://arxiv.org/abs/2510.02410) | arXiv 2026 | Pure method | SoftPrompt 与 Flamingo 在短/长序列上的内存和性能交叉，支持规模感知表示。 | 5/3/4 | A / Risk |
| 15 | [TSRouter](https://arxiv.org/abs/2607.08940) | COLM 2026 | Pure method | 已做时序文本/图像/混合模态与模型的质量-成本路由，是命名和创新的直接碰撞。 | 5/4/4 | Risk |

## 关键研究机会

1. **不是简单长度阈值，而是表示率选择。** raw text、数值专用 token、encoder latent 分别对应高、中、低输入率；核心问题是质量、信息损失与系统成本的 Pareto 权衡。
2. **路由前先证明互补。** 必须报告逐样本 Oracle 相比最佳单专家的增益；没有 Oracle gap 就没有路由价值。
3. **用真实 token 数替代 raw length。** 文本成本由 tokenizer、精度、符号和分隔符共同决定，同样的时间步数可能产生完全不同的上下文成本。
4. **创新边界。** 与 TSRouter 的区别应落在共享 LLM 内部的双/三表示选择、encoder 压缩失真、预算约束及 Oracle regret，而不是“动态选择”本身。
5. **扩词表的真正难点是几何。** ChatTime 解决 token 数，COM 进一步解决数值 token 的连续性与序数性；两者应作为连续基线而非两个孤立方向。

## 补充但未计入最终 15 篇

- [Chronos](https://mlanthology.org/tmlr/2024/ansari2024tmlr-chronos/)：量化词表的时序基础模型参照，但不以保留文本 LLM 能力为目标。
- [SciTS](https://arxiv.org/abs/2510.03255)：科学时序长度覆盖广，可用于极长序列/OOD 评测设计。
- [CALF](https://ojs.aaai.org/index.php/AAAI/article/view/34082)：输入、中间层、输出三级跨模态对齐，提示 projector-only 可能不足。

