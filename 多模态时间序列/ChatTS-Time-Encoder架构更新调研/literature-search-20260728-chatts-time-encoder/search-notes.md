# 检索与筛选记录

## 范围

- 目的：只更新 ChatTS time encoder 架构。
- 硬约束：固定 patch size、每样本 patch 数、每 patch 一个输出 token、原文本占位符替换路径。
- 偏好：原生时间序列证据；不能把视觉架构误写成时序论文。
- 截止：2026-07-28（Asia/Shanghai）。
- 模式：standard literature search。

## 使用的公开检索式

1. `"time series patch encoder" ICLR ICML AAAI`
2. `"time series encoder" LLM 2026 architecture comparison`
3. `"time series Mamba" patch bidirectional official paper`
4. `"time series LSTM" patch encoder AAAI`
5. `"time series multimodal QA" encoder ICML`
6. `"ChatTS" five layer MLP patch encoder`
7. `2026 time series multimodal foundation model`

## 核验来源

- PVLDB、ICLR proceedings、PMLR、AAAI/OJS、OpenReview。
- arXiv 原始记录用于预印本和论文 PDF。
- 官方 GitHub 仅用于确认实现接口与依赖，不作为论文结论的替代来源。
- 没有在最终表中使用仅能看到搜索摘要、无法稳定核验的记录。
- 按来源政策排除了不纳入来源及低信号页面。

## 直接证据

- ChatTS §3.4.1 明确写 fixed-size patches 与每 patch 的 5-layer MLP。
- ChatTS 公开实现计算 `patch_cnt = ceil(valid_length / patch_size)`，补齐最后一个 patch，再用共享 MLP 独立编码。
- ModernTCN Figure 2 明确区分 temporal DWConv 与 ConvFFN feature mixing；报告中的 lite 版本删除多 stage/downsampling。
- InceptionTime Figure 1/2 明确使用多尺度并行 1D 卷积、MaxPool 分支和 residual。
- 2026 encoder+LLM 预印本的 Table 1 报告：在其 UCR 分类设置中，Inception 是唯一在两种口径下接入 LLM 后均提高的家族。此结果没有外推为 ChatTS QA 结论。
- P-sLSTM Figure 1 明确包含 patching、projection、sLSTM backbone 与 channel independence。
- Bi-Mamba+ Figure 1/2 明确包含 channel-independent temporal patch path 与正反向 Mamba+。
- PatchTST Figure 1 明确保留 patch token 经 Transformer Encoder 的中间表示。
- TimeMixer++ 与 ITFormer 均为正式论文，但分别会改变多尺度 token 对齐和跨模态融合接口。

## 筛选问题

主要问题：

> 架构是否在进入 LLM 前沿 temporal patch 轴交换信息，并且仍输出 N 个 token？

- 是，且仅依赖标准算子：主 shortlist。
- 是，但需要自定义 CUDA 或正式发表状态不足：条件 shortlist。
- 主要沿变量轴混合：axis mismatch。
- 需要新 tokenizer、decoder、多尺度对齐或预训练：边界/排除。
- 主要改变跨模态 bridge：相关工作，不放入 architecture-only 主实验。

## 未知项与风险

- 没有完全相同 ChatTS QA 协议的公开 encoder 对照。
- Bi-Mamba+ 的正式会刊状态未在本轮稳定来源中确认。
- 2026 encoder+LLM 对比为短预印本，任务是 UCR 分类，不是生成式 QA。
- Chronicle、HORAI 等 2026 工作代表不同研究定义，不能作为简单替换的直接性能依据。
- 所有推荐都必须通过本项目自己的受控实验验证。

## 写作交接

- 主张：ChatTS 缺少 pre-LLM cross-patch temporal mixing。
- 最近邻：ITFormer。
- 最稳主线：ModernTCN-lite。
- 2026 支持线索：InceptionTime-lite。
- 强对照：PatchTST-style。
- 写作禁区：不能把适配版本冒充原论文完整模型，不能把跨数据集结果写成 ChatTS 性能。
