# 多模态时间序列

本目录收录时间序列与大语言模型、多模态问答及跨模态编码相关的调研与设计材料。

## ChatTS × TimesFM 2.5 适配实现

基于 TS-Reasoner 的“冻结时间序列基础模型 + 可训练 projector + 两阶段训练”思路，
将 ChatTS 原生 MLP-Patch 编码器替换为 `google/timesfm-2.5-200m-pytorch`。

- [适配说明、完整 Git 补丁与训练方法](./ChatTS-TimesFM2.5适配/README.md)

## ChatTS Time Encoder 架构更新调研

目标：在不修改 ChatTS tokenizer、patch 数量、LLM token merge 和训练任务的前提下，寻找能够替换原始 MLP-Patch 的原生时间序列编码架构。

- [原论文架构图版完整报告](./ChatTS-Time-Encoder架构更新调研/report.md)
- [浏览器版 HTML 报告](./ChatTS-Time-Encoder架构更新调研/report.html)
- [原图图号、页码与来源清单](./ChatTS-Time-Encoder架构更新调研/SOURCE_MANIFEST.md)
- [22 篇候选论文筛选与评分](./ChatTS-Time-Encoder架构更新调研/literature-search-20260728-chatts-time-encoder/papers.md)

主要结论：

1. **ModernTCN-lite**：最稳妥的主方案。
2. **InceptionTime-lite**：多尺度卷积，获得 2026 encoder+LLM 直接经验信号支持。
3. **P-sLSTM**：正式发表、时间状态清晰的现代递归方案。
4. **Bi-Mamba+ Patch**：高潜力，但有预印本和 CUDA 依赖风险。
5. **PatchTST-style**：应保留为 Transformer 强对照。

资料核验截止日期：2026-07-28。
