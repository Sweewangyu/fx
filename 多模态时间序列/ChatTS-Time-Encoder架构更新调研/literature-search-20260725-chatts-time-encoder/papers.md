# 候选论文清单

调研日期：2026-07-25
任务：为 ChatTS 的固定 patch MLP 寻找原生时间序列、尽量简单、可保持 token 接口的替代架构。

评分含义：相关性 / 新颖性 / drop-in 适配均为 1–5；它们是面向本项目约束的工程筛选分，不代表论文质量或真实性能排名。

| # | 论文 | 状态 | 相关 | 新颖 | 适配 | 决策 |
|---:|---|---|---:|---:|---:|---|
| 1 | [ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning](https://www.vldb.org/pvldb/vol18/p2385-xie.pdf) | PVLDB 18(8) · formal | 5 | 5 | 5 | include-baseline |
| 2 | [ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://proceedings.iclr.cc/paper_files/paper/2024/hash/86b1437c1e4c3b3c4debff98234a67e7-Abstract-Conference.html) | ICLR Spotlight · formal | 5 | 4 | 5 | shortlist |
| 3 | [Unlocking the Power of LSTM for Long Term Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/33303) | AAAI · formal | 5 | 4 | 4 | shortlist |
| 4 | [Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting](https://arxiv.org/abs/2404.15772) | arXiv v3 · preprint | 5 | 4 | 4 | shortlist-risk |
| 5 | [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://iclr.cc/virtual/2023/poster/10876) | ICLR · formal | 5 | 3 | 5 | include-control |
| 6 | [TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html) | ICLR · formal | 4 | 5 | 2 | boundary |
| 7 | [ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset](https://proceedings.mlr.press/v267/wang25av.html) | ICML · formal | 5 | 5 | 2 | boundary-nearest |
| 8 | [OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data](https://arxiv.org/abs/2510.02410) | arXiv v3 · preprint | 5 | 5 | 2 | related |
| 9 | [TsLLM: Augmenting LLMs for General Time Series Understanding and Prediction](https://arxiv.org/abs/2510.01111) | arXiv v2 · preprint | 4 | 4 | 2 | related |
| 10 | [DualMamba: a patch-based model with dual mamba for long-term time series forecasting](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-41293-5) | Frontiers of Computer Science 20(2) · formal | 4 | 4 | 3 | watch |
| 11 | [LightGTS: A Lightweight General Time Series Forecasting Model](https://proceedings.mlr.press/v267/wang25ch.html) | ICML · formal | 3 | 4 | 2 | exclude-current |
| 12 | [TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting](https://proceedings.mlr.press/v267/huang25az.html) | ICML · formal | 3 | 4 | 2 | exclude-current |
| 13 | [S-Mamba: Is Mamba Effective for Time Series Forecasting?](https://doi.org/10.1016/j.neucom.2024.129178) | Neurocomputing 619 · formal | 3 | 3 | 1 | exclude-axis-mismatch |
| 14 | [ms-Mamba: Multi-Scale Mamba for Time-Series Forecasting](https://arxiv.org/abs/2504.07654) | Neurocomputing 680 · formal | 3 | 4 | 1 | exclude-axis-mismatch |
| 15 | [Affirm: Interactive Mamba with Adaptive Fourier Filters for Long-term Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/35463) | AAAI · formal | 4 | 5 | 1 | exclude-complex |
| 16 | [Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html) | ICLR · formal | 3 | 4 | 2 | exclude-complex |
| 17 | [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html) | ICLR Spotlight · formal | 3 | 4 | 1 | exclude-axis-mismatch |
| 18 | [MOMENT: A Family of Open Time-series Foundation Models](https://proceedings.mlr.press/v235/goswami24a.html) | ICML · formal | 4 | 4 | 2 | related |
| 19 | [Moirai: Unified Training of Universal Time Series Forecasting Transformers](https://proceedings.mlr.press/v235/woo24a.html) | ICML · formal | 3 | 4 | 1 | related |
| 20 | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/forum?id=ju_Uqw384Oq) | ICLR · formal | 3 | 4 | 2 | related |

## 纳入主报告的架构

1. ChatTS MLP-Patch：目标基线。
2. ModernTCN-lite：主推荐，删除多 stage，只保留单尺度大核卷积 block。
3. P-sLSTM：2025 原生 patch 时序递归方案。
4. Bi-Mamba+ Patch Encoder：沿 patch 轴双向扫描，列为高潜力/高依赖风险方案。
5. PatchTST-style Encoder：强对照。
6. TimeMixer++：能力上限与复杂度边界。
7. ITFormer：最近邻 TS-QA 融合架构边界。

## 关键排除

- S-Mamba、ms-Mamba、iTransformer：主 token/scan 轴是变量，不是 ChatTS 所需的时间 patch 轴。
- LightGTS：主要创新在周期 tokenizer 与 decoder；会改变 token 数。
- Affirm、Pathformer、TimeMixer++：机制强，但已不是简单替换。
- MOMENT、Moirai：预训练 foundation model 路线，改变训练与部署问题定义。
