# 文献检索：ChatTS Time Encoder 架构更新

日期：2026-07-28；目的：为 ChatTS 的逐 patch MLP 寻找原生时间序列、尽量简单、能够保持 token 接口的替代架构。
来源策略：优先正式会议/期刊、官方 proceedings、PMLR、OpenReview、PVLDB、AAAI 与 arXiv 原始记录；排除政策不纳入来源及仅有搜索摘要的记录。

## 摘要

- 最适合架构替换：ModernTCN-lite、InceptionTime-lite、P-sLSTM、Bi-Mamba+。
- 必须对照：PatchTST-style。
- 最接近 ChatTS 的后续 TS-QA 工作：ITFormer，但它主要更新跨模态融合。
- 2026 新信号：一项 encoder+冻结 LLM 对比中，Inception 是唯一持续获得正向增益的编码器家族；Chronicle 与 HORAI 则代表更重的联合预训练方向。
- 最大风险：现有工作没有在同一 ChatTS QA/推理协议下直接证明任何候选优于 MLP。

## 论文表

评分均为 1-5，评价论文证据质量，不是 ChatTS 上的性能预测。

| # | 论文 | 年份/来源 | 类型 | Insight | 完整性 | 数字证据 | 总体 | 与本任务的关系 |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | [ChatTS](https://www.vldb.org/pvldb/vol18/p2385-xie.pdf) | PVLDB 2025 | method + benchmark | 4 | 4 | 4 | A/Risk | 目标基线；明确使用 fixed patch + 5-layer MLP |
| 2 | [ModernTCN](https://proceedings.iclr.cc/paper_files/paper/2024/hash/86b1437c1e4c3b3c4debff98234a67e7-Abstract-Conference.html) | ICLR 2024 Spotlight | pure method | 4 | 4 | 4 | A | 大核时序卷积；最适合低复杂度 token-preserving 适配 |
| 3 | [InceptionTime](https://arxiv.org/abs/1909.04939) | DMKD 2020 | pure method | 4 | 5 | 5 | A | 多尺度 1D 卷积；可移除 GAP 后保留 N 个 token |
| 4 | [An Exploratory Study to Repurpose LLMs to a Unified Architecture for TSC](https://arxiv.org/abs/2601.09971) | arXiv 2026 | other: empirical study | 4 | 3 | 3 | B/Risk | 直接比较 encoder+LLM；支持 Inception，但仅分类且为预印本 |
| 5 | [P-sLSTM](https://ojs.aaai.org/index.php/AAAI/article/view/33303) | AAAI 2025 | pure method | 4 | 4 | 4 | A | fixed patch + channel independent + sLSTM |
| 6 | [Bi-Mamba+](https://arxiv.org/abs/2404.15772) | arXiv v3 | pure method | 4 | 3 | 4 | Risk | 双向 patch 级状态扫描；接口合适但发表/依赖风险高 |
| 7 | [PatchTST](https://arxiv.org/abs/2211.14730) | ICLR 2023 | pure method | 4 | 5 | 5 | A | 与 ChatTS patch 接口最贴的 Transformer 强对照 |
| 8 | [TimeMixer++](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html) | ICLR 2025 | pure method | 5 | 4 | 5 | A | 原生通用时序架构，但多尺度时频系统过重 |
| 9 | [ITFormer](https://proceedings.mlr.press/v267/wang25av.html) | ICML 2025 | method + benchmark | 5 | 5 | 4 | A/Risk | 最接近的 TS-QA；改变 bridge 而非只换 encoder |
| 10 | [OpenTSLM](https://arxiv.org/abs/2510.02410) | arXiv 2025 | method + benchmark | 5 | 4 | 3 | B/Risk | 医疗文本+时序推理；属于更重的多模态系统路线 |
| 11 | [TsLLM](https://arxiv.org/abs/2510.01111) | arXiv 2025 | pure method | 4 | 4 | 3 | B | 通用时序理解/预测；训练与接口变化较大 |
| 12 | [Chronicle](https://arxiv.org/abs/2605.20268) | arXiv 2026 | pure method | 5 | 4 | 4 | Risk | 文本与时序从头联合预训练；不是 drop-in encoder |
| 13 | [HORAI](https://arxiv.org/abs/2602.05646) | arXiv 2026 | method + benchmark | 5 | 4 | 4 | Risk | 大规模多模态预训练与频率增强融合；远超本轮范围 |
| 14 | [LightGTS](https://proceedings.mlr.press/v267/wang25ch.html) | ICML 2025 | pure method | 4 | 4 | 4 | B | 轻量时序模型，但 tokenizer/decoder 改动较大 |
| 15 | [TimeBase](https://proceedings.mlr.press/v267/huang25az.html) | ICML 2025 | pure method | 4 | 4 | 4 | B | 极简预测模型；输出范式不适合逐 patch token |
| 16 | [iTransformer](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html) | ICLR 2024 Spotlight | pure method | 4 | 5 | 5 | A | token 轴是变量，不是 ChatTS 所需的 temporal patch 轴 |
| 17 | [Pathformer](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html) | ICLR 2024 | pure method | 4 | 4 | 4 | A | 自适应多尺度路径；机制和超参数过多 |
| 18 | [MOMENT](https://proceedings.mlr.press/v235/goswami24a.html) | ICML 2024 | method + benchmark | 4 | 5 | 5 | A | 预训练时序基础模型路线，不是简单替换 |
| 19 | [Moirai](https://proceedings.mlr.press/v235/woo24a.html) | ICML 2024 | pure method | 4 | 5 | 5 | A | 通用预测基础模型；tokenizer 和训练定义均变化 |
| 20 | [TimesNet](https://openreview.net/forum?id=ju_Uqw384Oq) | ICLR 2023 | pure method | 5 | 5 | 5 | A | 时间二维化的强通用模型，但不是轻量 drop-in |
| 21 | [DualMamba](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-41293-5) | Frontiers of Computer Science 2026 | pure method | 4 | 3 | 4 | B | patch + dual Mamba；实现和 token 适配仍需较大改动 |
| 22 | [Time2Lang](https://proceedings.mlr.press/v287/pillai25a.html) | CHIL 2025 | pure method | 4 | 4 | 3 | B | TFM→LLM 桥接证据；健康分类场景且依赖预训练 TFM |

## 最接近工作的聚类

### 聚类一：保持 patch token 的时间 mixer

- 代表论文：ModernTCN、InceptionTime、P-sLSTM、Bi-Mamba+、PatchTST。
- 已覆盖：卷积、递归、状态空间与 attention 四类跨时间建模方式。
- 未覆盖：在相同 ChatTS QA/推理协议中的受控比较。
- 可区分方向：固定 token 接口，只比较 pre-LLM temporal mixing。

### 聚类二：时间序列与 LLM 的桥接

- 代表论文：ChatTS、ITFormer、Time2Lang、OpenTSLM、TsLLM。
- 已覆盖：patch token 插入、问题条件化融合、TFM→LLM 映射。
- 未覆盖：控制 bridge 不变时，encoder 家族的独立贡献。
- 可区分方向：不改融合，只换 encoder，做严格归因。

### 聚类三：联合多模态基础模型

- 代表论文：Chronicle、HORAI。
- 已覆盖：共享 backbone 或大规模多模态预训练。
- 未覆盖：小改动、低算力、兼容现有 ChatTS 的升级路径。
- 可区分方向：把本工作定位为低成本 architecture-only update，而不是基础模型重训。

## 机会图

| 聚类 | 状态 | 开放缺口 | 可行方向 | 所需证据 | 风险 |
| --- | --- | --- | --- | --- | --- |
| Patch temporal mixer | crowded but open | 缺少 ChatTS QA 受控比较 | 统一 N token 接口比较五类 mixer | QA、趋势、周期、异常、跨段推理 | 只换 backbone 的创新性有限 |
| Encoder + LLM | mechanism gap | encoder 选择影响缺少系统解释 | 分析 pre-LLM mixing 对任务类型的影响 | 按题型分组和注意力/感受野分析 | LLM 容量可能掩盖 encoder 差异 |
| Joint multimodal FM | deployment/system gap | 训练成本高、难复现 | 强调低成本兼容性 | 参数、吞吐、显存、训练预算 | 性能上限可能不及大规模预训练 |

## 引用与定位注意事项

- 不能把 ModernTCN-lite、InceptionTime-lite 或 PatchTST-style 写成原论文完整结构。
- 不能用 UCR 分类上的 Inception 结果宣称 ChatTS QA 必然提升。
- Bi-Mamba+ 的正式发表状态必须继续标注为未确认/预印本，除非后续找到正式记录。
- ITFormer 是最可能被审稿人要求补充的最近邻 TS-QA 工作。
- 需要把“架构适配性排序”和“真实 ChatTS 结果”严格分开。
