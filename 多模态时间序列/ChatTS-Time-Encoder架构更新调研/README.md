# ChatTS Time Encoder 架构更新调研

本资料包讨论如何在保持 ChatTS patch/token 接口不变的前提下，把独立的 MLP-Patch 投影更新为真正具有跨 patch 时间建模能力的编码器。

## 阅读入口

- [Markdown 完整报告](./report.md)：包含原论文架构图和逐图解释。
- [HTML 完整报告](./report.html)：适合浏览器阅读。
- [原论文图来源清单](./SOURCE_MANIFEST.md)：图号、PDF 页码、发表状态与链接。
- [22 篇候选论文与质量评分](./literature-search-20260728-chatts-time-encoder/papers.md)
- [机器可读 CSV](./literature-search-20260728-chatts-time-encoder/papers.csv)
- [检索与证据记录](./literature-search-20260728-chatts-time-encoder/search-notes.md)

## 主要结论

1. **ModernTCN-lite**：最务实的主方案，标准 Conv1d、token 数不变、工程风险最低。
2. **InceptionTime-lite**：多尺度卷积方案，并有 2026 encoder+LLM 对比论文提供直接经验信号。
3. **P-sLSTM**：2025 年正式发表的 patch 级递归方案，时序信息流最容易解释。
4. **Bi-Mamba+ Patch**：高潜力但存在预印本、自定义 CUDA 与短序列收益风险。
5. **PatchTST-style**：必须保留的全局 attention 强对照。

## 原论文图

1. [ChatTS Figure 6](./figures-original/01-chatts-figure6.png)
2. [ModernTCN Figure 2](./figures-original/02-moderntcn-figure2.png)
3. [InceptionTime Figure 1](./figures-original/03-inceptiontime-figure1.png)
4. [InceptionTime Figure 2](./figures-original/03b-inceptiontime-figure2.png)
5. [2026 Encoder+LLM Study Table 1](./figures-original/03c-llm-encoder-study-2026-table1.png)
6. [P-sLSTM Figure 1](./figures-original/04-pslstm-figure1.png)
7. [Bi-Mamba+ Figure 1](./figures-original/05-bimamba-figure1.png)
8. [Bi-Mamba+ Figure 2](./figures-original/05b-bimamba-figure2.png)
9. [PatchTST Figure 1](./figures-original/06-patchtst-figure1.png)
10. [TimeMixer++ Figure 2](./figures-original/07-timemixerpp-figure2.png)
11. [ITFormer Figure 2](./figures-original/08-itformer-figure2.png)

所有 PNG 均从原始论文 PDF 裁切，未重绘或修改图内内容。报告中的 `lite` / `style` 名称表示面向 ChatTS 接口的最小适配，不是原论文官方模型名称。
