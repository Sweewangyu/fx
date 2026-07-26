# ChatTS Time Encoder 架构更新调研

本资料包讨论如何在保持 ChatTS 现有 patch/token 接口不变的前提下，把独立的 MLP-Patch 投影更新为真正具有跨 patch 时间建模能力的编码器。

## 阅读入口

- [Markdown 完整报告](./report.md)：适合直接在 GitHub 阅读。
- [HTML 可视化报告](./report.html)：包含侧边导航与完整排版，下载后可直接打开。
- [论文筛选表](./literature-search-20260725-chatts-time-encoder/papers.md)
- [机器可读 CSV](./literature-search-20260725-chatts-time-encoder/papers.csv)
- [检索与证据记录](./literature-search-20260725-chatts-time-encoder/search-notes.md)

## 架构图

1. [ChatTS 原始 MLP-Patch](./figures/01-chatts-mlp-patch.svg)
2. [ModernTCN-lite](./figures/02-moderntcn-lite.svg)
3. [P-sLSTM](./figures/03-p-slstm.svg)
4. [Bi-Mamba+ Patch Encoder](./figures/04-bi-mamba-plus-patch.svg)
5. [PatchTST-style Encoder](./figures/05-patchtst-transformer.svg)
6. [TimeMixer++ 能力边界](./figures/06-timemixer-plus-plus.svg)
7. [ITFormer 跨模态桥接参考](./figures/07-itformer-bridge.svg)

所有 SVG 均为根据论文与公开实现重新绘制的架构解释图，并标注了论文来源和 ChatTS-specific adaptation。
