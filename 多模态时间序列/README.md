# 多模态时间序列

本目录收录时间序列与大语言模型、多模态问答及跨模态编码相关的调研与设计材料。

## ChatTS Time Encoder 架构更新调研

调研目标：在不修改 ChatTS tokenizer、patch 数量、LLM token merge 和训练任务的前提下，寻找能够替换原始 MLP-Patch 的原生时间序列编码架构。

- [GitHub 可直接阅读的 Markdown 报告](./ChatTS-Time-Encoder架构更新调研/report.md)
- [完整 HTML 报告](./ChatTS-Time-Encoder架构更新调研/report.html)
- [架构图与证据包说明](./ChatTS-Time-Encoder架构更新调研/README.md)
- [20 篇候选论文筛选表](./ChatTS-Time-Encoder架构更新调研/literature-search-20260725-chatts-time-encoder/papers.md)

主要结论：

1. **ModernTCN-lite**：最务实的主方案，使用单尺度大核时序卷积，接口不变、工程依赖少。
2. **P-sLSTM**：2025 年原生时序递归架构，适合强调架构更新，但实现复杂度略高。
3. **Bi-Mamba+ Patch Encoder**：接口适配度高、长依赖建模能力强，但存在预印本和 CUDA 依赖风险。
4. **PatchTST-style Encoder**：应保留为 Transformer 强对照，而不是主要创新。

调研及资料核验截止日期：2026-07-25。
