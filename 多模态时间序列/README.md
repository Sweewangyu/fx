# 多模态时间序列

本目录收录时间序列与大语言模型、多模态问答及跨模态编码相关的调研与设计材料。

## ChatTS Chronos-2 自包含权重

将已有的 `Qwen3 + 已训练两层 projector` 检查点与本地 Chronos-2 合并为一个标准
Hugging Face 模型目录。最终 state dict 同时包含 `ts_encoder.backbone.*`、
`ts_encoder.projector.*` 和 Qwen3 权重，运行时不再需要第二个 Chronos 权重路径。
模型代码以 `bytedance-research/ChatTS-8B` 官方非权重目录为基线，只做最小修改。

- [合并脚本、修改后的 Python 文件与服务器命令](./ChatTS-Chronos2自包含模型/README.md)

## ChatTS × TimesFM 2.5 / Chronos-2 / ZEUS 适配实现

基于 TS-Reasoner 的“冻结时间序列基础模型 + 可训练 projector + 两阶段训练”思路，
将 ChatTS 原生 MLP-Patch 编码器替换为 `google/timesfm-2.5-200m-pytorch`、
`amazon/chronos-2` 或 `GestaltCog/zeus`。目录包含连续 Git 补丁、可直接覆盖到服务器的
完整 `.py` 文件和每种 backbone 的 Stage 1 / Stage 2 脚本。
同时提供 NetManAIOps/ChatTS 的四后端 vLLM 评测文件，兼容原始 MLP-Patch、
TimesFM 2.5、Chronos-2 与 Zeus checkpoint。

- [三种 backbone 的适配说明、完整 Git 补丁、直接文件与训练方法](./ChatTS-TimesFM2.5适配/README.md)

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
