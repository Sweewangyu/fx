# 检索记录

## 研究问题

1. 哪些工作让时间序列独立经过 encoder/projector，再与文本进入 LLM？
2. 哪些工作把数值序列直接序列化到 prompt？
3. 把量化数字扩展到词表后，如何处理 token 数、量化与 embedding 几何？
4. 是否已有按长度、模态、任务或预算路由时序表示/模型的工作？

## 检索式与入口

- `time series LLM separate encoder multimodal`
- `time series large language model numeric sequence prompt tokenizer`
- `zero-shot time series forecasting digit tokenization`
- `time series vocabulary expansion numerical token`
- `arxiv 2412.11376`
- `arxiv 2605.28866`
- `time series router long sequence encoder short sequence text`
- `TSRouter time series reasoning`
- `time series LLM text based versus embedding encoder`
- 论文引用链：Time-LLM、LLMTime、ChatTime、COM、TimeOmni-1、OpenTSLM、TSRouter 的参考文献与相关工作。

## 来源策略

优先使用论文原文和正式出版页面：arXiv、NeurIPS Proceedings、ICLR Proceedings/OpenReview、ACL Anthology、AAAI、ACM Digital Library、TMLR。关键架构和表格结论均回到 PDF 核验；正文 4 张图直接截自对应论文 PDF。

## 纳入标准

- 与独立时序编码、数值文本 prompt、数值词表或动态路由至少一项直接相关；
- 方法结构或实验结论能支持路线选择、风险判断或最小实验设计；
- 有可访问的论文原文或正式出版记录；
- 负结果和直接创新碰撞优先保留，不只筛选支持性论文。

## 排除与降权

- 排除 MDPI 结果及二次转载、博客、ResearchGate 镜像等非首选来源；
- 只讨论通用 Transformer、但不涉及语言模型/文本能力的纯时序模型未列为核心候选；
- 只做模型选择、但没有时序场景或表示选择问题的通用 router 仅作背景，不进入最终表；
- 2026 年仅有 arXiv 状态的 COM、OpenTSLM 降低 evidence strength，并在汇报中明确为预印本。

## 关键核验点

- ChatTime：`[-1,1]`、10K bins、专用 TS tokens、扩展 embedding/LM head；论文示例中 4 个数值分别需 GPT 34、LLaMA 22、ChatTime 7 tokens。
- COM：硬约束初始化 + ordinality/monotonicity 软约束；TSQA 表中 PCA-Main 91.23、Default 50.04、打乱 PCA-Main 50.71。仅作为论文特定设置下结果。
- TimeOmni-1 Appendix F.3：text 与 ChatTS-style encoder 在不同任务上互有胜负；text 路径更快、encoder 峰值显存更低；长度与赢家并非单调关系。
- TSRouter：题名已占用，且已联合路由时序模态与模型；本项目必须在共享 backbone 的表示率/失真/预算路由上形成差异。

## 未决问题

- COM 和 OpenTSLM 的结论尚需正式同行评审或更多独立复现。
- 现有公开结果不足以证明“短序列必然适合文本、长序列必然适合 encoder”；需同任务、配对长度控制实验。
- 端到端延迟高度依赖 encoder kernel、批处理、服务框架和硬件，不能只由输入 token 数推断。
- 最终 router 是否值得训练取决于本项目数据上的 Oracle gap，无法由文献替代。

## 交付物

- 主汇报：`output/时序大模型研究汇报.md`
- 图片：`output/时序大模型汇报.assets/`
- 筛选表：`papers.md`、`papers.csv`

