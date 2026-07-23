# ChatTS 数据合成方法与 Pipeline 详细报告

**论文：** ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning

**会议：** VLDB'25  ·  **arXiv：** 2412.03104 (v3, 2025-04-16)  ·  **作者：** Zhe Xie, Zeyan Li, Xiao He 等（清华 NetMan / 字节跳动）

**代码：** github.com/NetManAIOps/ChatTS  ·  **模型：** ChatTS-14B / 8B (基座 Qwen2.5)

> **关键词：** 纯合成数据训练 · 属性驱动生成 · Time Series Evol-Instruct · 多元时间序列 MLLM · MIT License

## ·目录

  1. 核心结论一句话
  2. 为什么要合成数据
  3. 数据合成总体 Pipeline
  4. 阶段一：属性驱动的时间序列合成
  5. 属性 → 文本描述（对齐语料）
  6. 阶段二：QA 合成与 TSEvol
  7. 数据集规模统计
  8. GitHub 仓库结构映射
  9. 一键复现命令
  10. 要点评析

## 1核心结论一句话

ChatTS 的训练数据 **100% 由程序合成** ，不依赖任何真实标注。它的关键创新在于把"生成时间序列"和"生成标签"这两个通常割裂的环节 **合二为一** ：

**先定属性，再生成波形。** 系统先用规则 + LLM 随机采样出一组"属性池"（趋势/周期/噪声/局部变化及其精确数值），再用规则化函数按属性池**确定性地** 合成出对应波形。因此每条时间序列都**自带一份 100% 准确、可精确到数值的文本标签** —— 从根本上绕开了"给真实时间序列做人工标注既贵又不准"的瓶颈。 

在此之上，再用 **Time Series Evol-Instruct (TSEvol)** 把简单标签演化成多样、复杂的问答对，用于提升推理能力。

## 2为什么要合成数据

时间序列多模态大模型（TS-MLLM）落地的最大障碍是**缺少"时间序列↔文本"对齐的高质量数据集** 。真实数据存在三大问题：

  * **标注昂贵** ：让人描述一条曲线的趋势、周期、异常点，成本高、主观性强。
  * **标注不精确** ：人无法说出"第 137 个点有一个幅度 4.2 的向上尖峰"这类数值级细节。
  * **模式覆盖不全** ：真实数据的形态分布长尾且不可控，难以系统覆盖各种模式组合。

ChatTS 的解法是**反过来做** ：不去标注既有数据，而是先决定"要什么属性"，再生成"符合属性的数据"，标签天然精确、模式可控、数量无限。

## 3数据合成总体 Pipeline

整个数据工厂分为两大阶段、五条产线，最终汇成对齐（Alignment）+ 微调（SFT）两套训练集：

**阶段 A · 属性驱动的时间序列合成** （`chatts/ts_generator/`） 

Metric 采样 → GPT 选属性子集 → 属性采样器赋数值 → 规则生成器造波形 → 输出「波形 + 属性池 + 精确文本标签」

**阶段 B · 问答对合成** （`chatts/align/` \+ `chatts/sft/`） 

① 模板 QA ② LLM 种子 QA ③ TSEvol 演化 QA ④ IFT 指令跟随 ⑤ 增强推理 QA

### 数据流全景图
    
    
    # ============ 阶段 A：时间序列 + 精确标签 ============
     metric_set.json (567 个真实指标名)
            │  随机采样一个指标 (如 "CPU 使用率")
            ▼
     GPT Selector ── 依据指标物理含义，从"全属性集"挑选合理的属性子集
            │        (例: CPU→可能有周期+突增, 不太可能是负值)
            ▼
     Attribute Sampler ── 在子集内随机采样属性组合并赋具体数值
            │             (类型 / 位置 position / 幅度 amplitude / 周期 period)
            ▼
     ┌─────────────── Attribute Pool（属性池，记录全部细节）───────────────┐
     │  seasonal / trend / frequency / noise / local[] / statistics       │
     └────────────────────────────┬──────────────────────────────────────┘
            │ rule-based 确定性生成            │ 模板 + LLM 精修
            ▼                                 ▼
       Time Series (np.ndarray)         Precise Text Label（100% 准确）
            │                                 │
            └──────────────┬──────────────────┘
                           ▼
    # ============ 阶段 B：问答对 ============
       ├── 模板 QA   (uts/mts_template_qa)  ── 规则填模板，海量、精确
       ├── LLM 种子 QA (generate_llm_qa)    ── LLM 基于属性池生成初始 Q&A;
       │        ▼
       ├── TSEvol   (generate_tsevol_dataset) ── 深度/广度演化 + 属性消除器去重
       ├── IFT      (generate_ift_dataset)   ── 指令跟随（需对齐标签）
       └── 增强推理  (uts/mts_reason, rewrite) ── 单/多变量推理 + 重写增广
                           ▼
            Alignment 数据集 + SFT 数据集 → 两阶段训练 ChatTS
      

对应论文 **Figure 3（整体流水线）** 与 **Figure 4（属性生成器）** 。GitHub 的 `demo/demo_ts_generator.ipynb` 是这套生成器的交互式演示。 

## 4阶段一：属性驱动的时间序列合成

这是整套方法的基石。核心思想：**时间序列 = 全局趋势 + 周期项 + 局部变化 + 噪声** 的叠加，每一项都由可枚举的"属性"控制。源码见 `chatts/ts_generator/generate.py`。

### 4.1 全属性集（All Attribute Set）

论文口径为「4 趋势 + 7 周期 + 3 噪声 + 19 局部波动」。源码中的真实定义（含被选中概率）如下：

#### ① 趋势 Trend（4 类）

increase 0.3decrease 0.3 keep steady 0.3multiple 0.1

multiple = 多段拼接的分段趋势（先升后降等）。

#### ② 周期 Seasonal（波形 × 频率 = 7 类）

no periodic 0.7sin 0.25 square 0.02triangle 0.03

3 种波形 × {高频, 低频} 组合 + 无周期 ≈ 论文所述 7 类。

#### ③ 噪声 Noise（3 类）

almost no noise 0.8noisy 0.2

noisy 内部再分「高斯随机噪声」与「正弦不规则噪声」两种实现。

#### ④ 局部波动 Local Change（19 类）

upward/downward spikewide spike continuous spikesudden increase/decrease shakeupward/downward convex rapid rise + slow decline …decrease after spike …

数字为被采样权重（如 upward spike=12, sudden increase=10）。

#### 19 种局部波动完整清单

类别| 类型（英文原名）  
---|---  
尖峰类| upward spike / downward spike / wide upward spike / wide downward spike / continuous upward spike / continuous downward spike  
阶跃类| sudden increase / sudden decrease  
凸起类| upward convex / downward convex  
抖动类| shake  
快慢组合| rapid rise followed by slow decline / slow rise followed by rapid decline / rapid decline followed by slow rise / slow decline followed by rapid rise  
尖峰后趋势| decrease after upward spike / increase after downward spike / increase after upward spike / decrease after downward spike  
  
### 4.2 两种属性采样模式

**随机模式** `generate_random_attributes()`

纯随机按概率采样，用于覆盖广泛形态。会依据 `seq_len` 做安全约束（如序列 <24 强制无周期、<=32 无噪声、<=8 去掉 shake/突变）。

**受控模式** `generate_controlled_attributes()`

由 GPT 依据 metric 语义给出属性子集与数值范围（min/max），再在范围内采样。这就是"让曲线符合真实物理含义"的关键（对应 `config/metric_set.json`）。

### 4.3 波形是怎么"算"出来的（关键数学）

核心函数 `generate_time_series(attribute_pool, seq_len)` 按固定顺序叠加各分量，最终 **y = 周期项 + 局部变化 + 趋势项 + 噪声项** 。

#### ① 周期项：多谐波正弦叠加

并非单一正弦，而是**随机数量的谐波叠加** ，且幅度随时间轻微漂移，从而生成丰富的周期形态：
    
    
    # generate_seasonal_wave() —— sin 波形核心
    base_frequency = 1 / period
    num_harmonics  = randint(1, min(period//6, 10))      # 谐波个数随机
    for n in 1..num_harmonics:
        phase = uniform(0, 2π)
        A_n   = amplitude_series / n * (1 + 微小漂移·sin(...))   # 第 n 谐波幅度 ∝ 1/n
        data += A_n · sin(2π · base_frequency · n · t + phase)
    data = data / (max-min) · 目标幅度; data -= mean(data)   # 归一到目标振幅、去均值
      

square（方波）与 triangle（三角波）则按周期内相位 `cycle_pos = (t mod period)/period` 逐点判定，落在 [start, start+duration] 区间内赋幅值。

#### ② 频率：由 period 大小决定高/低频
    
    
    high frequency: period ∈ [seq_len/16, seq_len/8]
    low  frequency: period ∈ [seq_len/8,  seq_len/3]
    

#### ③ 趋势项：4 种趋势的构造
    
    
    increase / decrease : generate_ts_change(seq_len, ±amplitude) + bias   # 平滑单调曲线
    keep steady         : y += bias                                        # 常量
    multiple            : 随机分段点 → generate_trend_curve() 拼多段斜率     # 分段趋势
    amplitude = uniform(0.8, 3.0) · overall_amplitude                      # 趋势幅度
    

#### ④ 噪声项：两种实现
    
    
    noisy 分支：
      · 随机高斯噪声  : std = uniform(0.03, 0.15) · overall_amplitude; noise~N(0, std)
      · 正弦不规则噪声: 200 个随机频率/相位正弦叠加，模拟不规则抖动
      · 再对分段乘以 uniform(0.1, 5.0) 放大 → 局部噪声强弱不均
    almost no noise 分支：std ≈ 0（uniform(0, 0.001)·amplitude），曲线基本光滑
    

#### ⑤ 整体幅度与偏置：跨数量级采样

为让模型见过各种量级（0.x ~ 百万级），幅度指数按概率分布采样：
    
    
    e ~ choice([-2,-1,0,1,2,3,4,5,6,7], p=[.1,.2,.2,.3,.1,.04,.03,.02,.008,.002])
    overall_amplitude = uniform(10^(e-1), 10^(e+1))
    overall_bias      = uniform(-10^(e+1), 10^(e+1))
    

**为什么标签一定准？** 因为波形是**由属性池按规则确定性生成** 的（而非先有波形再去猜属性）。生成时同步把每个属性的 `detail` 文本、位置、幅度、以及最终统计量（mean/std/max/min/max_pos/min_pos）写回属性池。局部变化的数值甚至用占位符 `<|137|>` 在生成后回填为该点真实值，做到**数值级精确对齐** 。 

## 5属性 → 文本描述（对齐语料）

属性池通过两个函数转成自然语言标签，构成对齐训练的"答案"：

函数| 风格| 用途  
---|---|---  
`attribute_to_text()`| 结构化罗列各属性（长度/趋势/周期/频率/噪声/局部/统计）| 属性描述任务  
`attribute_to_caption()`| 更自然流畅，把趋势与局部变化按时间顺序串成一段叙述| 整体 caption 描述  
  
#### 生成的文本长这样（节选自源码模板）
    
    
    "The length of the time series is 256. From the perspective of the slope, the overall
     trend is increasing. The value starts from around 12.3 and ends at around 48.7 ...
     The time series is showing sin periodic fluctuation: the amplitude ... is 5.2 between
     point 0 and point 255. Each fluctuation period is approximately 32.0 points ...
     In terms of local characteristics, there is an upward spike with amplitude 8.4 near
     point 137, forming a upward spike. Specific data details: divided into 32 segments,
     the approximate mean values are [...]. The maximum is 51.2, the minimum is 10.8."
    

**Value-Preserved Encoding：** 文本里同时保留了分段均值、最大/最小值、起止值、随机抽点数值，让模型不仅懂"形状"还能答"数值"。这对应论文的**数值保留归一化** ——每条序列独立 min-max 归一，缩放/偏移参数以文本形式注入 prompt。 

#### 三类对齐语料（`chatts/align/`）

产线| 文件| 侧重  
---|---|---  
UTS| `uts_template_qa.py` / `uts_llm_qa.py`| 单变量全局+局部属性  
MTS-Shape| `mts_shape_template_qa.py` / `..._llm_qa.py`| 多变量全局趋势相关性  
MTS-Local| `mts_local_template_qa.py` / `..._llm_qa.py`| 多变量局部波动相关性  
  
每类都有「template（规则填模板，海量且 100% 准确）」和「llm（LLM 改写，更自然多样）」两条子产线。

## 6阶段二：QA 合成与 TSEvol 详解

对齐语料让模型"看懂"时间序列，但要"会推理"还需多样、复杂的问答。ChatTS 借鉴 **Evol-Instruct / MMEvol** ，提出时序版 **TSEvol** （源码 `chatts/sft/generate_tsevol_dataset.py` \+ `utils/evol_prompt.py`）。

### 6.1 三步走

  1. **种子 QA** （`generate_llm_qa.py`）：LLM 基于属性池，先生成一批初始简单问答作为演化起点。
  2. **演化 QA** （`generate_tsevol_dataset.py`）：对每条种子做 DFS 深度演化（`DFS_K=3`），每步随机挑一种演化算子把问题变难/变新。
  3. **去重校验** ：用"属性消除器"（comparison eliminator）判断新 QA 是否与旧 QA 等价/是否可从上下文推出，剔除重复与幻觉。

### 6.2 演化算子（源码里的 7 种 prompt）

算子| 方向| 作用  
---|---|---  
Constraints| 深度| 依据属性再加一条约束/条件  
Deepen| 深度| 加大问题的深度与广度  
Concretizing| 深度| 把泛化概念替换成更具体概念  
Complex Reasoning| 广度| 改写成需多步推理的难题  
Situation| 广度| 构造真实业务情境（电商大促/工厂能耗等）+ 提问  
Deductive Reasoning| 广度| Yes/No 演绎推理（给定阈值规则判断是否异常）  
Causal Reasoning| 广度| 因果多选（成因识别/效应预测/异常解释/时序关联）  
  
### 6.3 属性字段演化机制（evol 的巧妙处）

每条种子 QA 记录它当前"用到了哪些属性字段"（`fields`）。每次 `evol()` 会从**尚未用到的字段** 中随机拉入一个新属性作为上下文，逼迫问题覆盖新维度：
    
    
    all_fields = {trend, seasonal, noise, local, statistic, correlation}
    evol(): 从 (all_fields − 已用 fields) 中随机选一个新字段加入上下文
            → 保证问答逐步覆盖更多属性、避免总问同一个点
    

**correlation pool（相关性池）：** 多变量场景下，系统额外记录"哪些序列之间具有相关属性"，作为一个特殊字段参与演化，专门训练模型的**跨序列关联分析** 能力。 

### 6.4 关键约束（防止数据污染）

  * **问题里不能泄露具体属性** ：像"noise=0.5""spike at point 100"这类细节**只能出现在答案** ，问题需用泛化语言 —— 逼模型自己从序列里读出来。
  * **只能用 CONTEXT 里的属性** ：禁止 LLM 编造上下文外的信息，从源头压制幻觉。
  * **交叉校验 + 等价消除** ：生成后与属性池比对、与前一条 QA 比对，判定 Equal/Invalid/Valid。

### 6.5 其它 SFT 增强产线

脚本| 作用  
---|---  
`generate_ift_dataset.py`| 指令跟随（IFT）数据，需先有对齐标签  
`generate_uts_reason.py` / `_cn.py`| 单变量推理增强（英 / 中文，中文含一致性检查）  
`generate_mts_reason.py`| 多变量推理增强  
`generate_rewrite_dataset.py`| 通过重写做数据增广  
  
## 7数据集规模统计

### 论文口径（实际训练用量）

| 数值 | 含义 |
| --- | --- |
| 567 | 真实指标名 |
| 33+ | 属性类型 (4+7+3+19) |
| 105k | 对齐样本 (UTS/MTS 各 35k) |
| 24,270 | TSEvol 指令 |
| 5,050 | Follow 指令 |
| 64–1024 | 序列长度范围 |


### 开源仓库默认配置口径（`config/datagen_config.yaml`，可自行放大）

配置项| 默认值| 含义  
---|---|---  
num_data_template_qa| 20000| 模板 QA  
num_data_llm_qa| 15000| LLM 种子 QA  
num_data_tsevol| 10000| TSEvol 演化  
num_data_ift| 10000| 指令跟随  
num_data_uts/mts_reason| 各 10000| 推理增强  
num_data_rewrite| 10000| 重写增广  
seq_len| 256（可设 null 随机）| 序列长度  
  
## 8GitHub 仓库结构映射（数据合成相关）
    
    
    ChatTS/
    ├── config/
    │   ├── datagen_config.yaml        # ★ 数据生成总配置（LLM 路径/数量/长度）
    │   └── metric_set.json            # ★ 567 个真实指标名 + 属性约束
    ├── chatts/
    │   ├── ts_generator/              # ★★ 阶段A：时间序列合成器
    │   │   ├── generate.py            #   主逻辑：属性采样 + 波形生成 + 属性转文本
    │   │   ├── trend_utils.py         #   分段趋势曲线
    │   │   ├── local_changes.py       #   19 种局部波动的类实现
    │   │   └── change_utils.py        #   基础变化/尖峰工具
    │   ├── align/                     # ★ 阶段B-①：对齐 QA（template + llm）
    │   │   ├── uts_template_qa.py / uts_llm_qa.py
    │   │   ├── mts_shape_template_qa.py / mts_shape_llm_qa.py
    │   │   └── mts_local_template_qa.py / mts_local_llm_qa.py
    │   └── sft/                       # ★★ 阶段B-②~⑤：SFT 数据
    │       ├── generate_llm_qa.py         # 种子 QA
    │       ├── generate_tsevol_dataset.py # TSEvol 演化（核心）
    │       ├── generate_ift_dataset.py    # 指令跟随
    │       ├── generate_uts_reason.py / _cn.py / generate_mts_reason.py
    │       ├── generate_rewrite_dataset.py
    │       └── utils/
    │           ├── evol_prompt.py     # ★ 7 种演化算子 + EvolPrompt 类
    │           ├── evol_attributes.py  # 属性→prompt（含 correlation）
    │           └── rewrite_prompt.py
    ├── scripts/
    │   ├── generate_align_datasets.sh          # 一键生成对齐集
    │   └── generate_enhanced_sft_datasets.sh   # 一键生成增强 SFT
    └── demo/
        └── demo_ts_generator.ipynb    # ★ 生成器交互演示（最适合上手）
      

实际模型微调在**独立仓库** [ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training)（改自 LLaMA-Factory）；官方合成训练集已开源在 [HuggingFace](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset)。 

## 9一键复现命令
    
    
    # 0. 环境（推理需 python==3.11）
    pip install -r requirements.txt
    
    # 1. 配置：编辑 config/datagen_config.yaml
    #    local_llm_path: 指向本地 LLM（推荐 Qwen2.5-32B-Instruct）
    #    num_gpus / seq_len / num_data_* 按需设置
    
    # 2. 生成对齐数据集（阶段 A + 对齐 QA）
    bash scripts/generate_align_datasets.sh
    
    # 3. 生成 SFT 数据集
    python3 -m chatts.sft.generate_llm_qa          # ① LLM 种子 QA
    python3 -m chatts.sft.generate_tsevol_dataset  # ② TSEvol 演化（本地 LLM 或远程 API）
    python3 -m chatts.sft.generate_ift_dataset     # ③ 指令跟随（需先有对齐标签）
    
    # 4.（可选）增强 SFT：单/多变量推理 + 重写
    bash scripts/generate_enhanced_sft_datasets.sh
    
    # 5. 只想看时间序列生成器？打开：
    demo/demo_ts_generator.ipynb
    

**注意：** TSEvol / LLM QA 需要一个本地或远程 LLM 作为"数据生成器"（论文用 Qwen2.5-32B）。生成阶段与训练阶段解耦：数据生成产物为 `.jsonl`，再喂给 ChatTS-Training 微调。 

## 10要点评析

#### ✅ 方法的高明之处

  * **标签零成本且 100% 精确** ：先属性后波形，天然对齐到数值级。
  * **模式可控可枚举** ：33+ 属性自由组合 + 正弦谐波叠加 → 理论上无限形态。
  * **GPT 注入语义** ：567 指标让合成曲线符合真实物理含义，缩小 sim-to-real gap。
  * **TSEvol 保证多样性与难度** ：字段演化 + 去重消除器，避免同质化。
  * **数值保留编码** ：能答"尖峰多大"这类定量问题，超越纯视觉 MLLM。

#### ⚠️ 潜在局限

  * 合成分布与真实分布仍有差距，极端真实场景可能未覆盖。
  * 合成 QA 质量受"数据生成器 LLM"能力上限约束。
  * 波形是"加性叠加"模型，难以刻画强非线性耦合的复杂系统。
  * 聚焦理解与推理，**不做** 预测/异常检测/分类。

**一句话总结：** ChatTS 用"属性驱动的确定性生成"把标注问题变成了生成问题，再用 TSEvol 把简单标签演化成复杂推理数据，从而**完全靠合成数据** 训出一个能理解并推理多元时间序列的原生多模态大模型——在对齐任务上比 GPT-4o 提升约 46%，推理任务提升约 26%。 

报告基于 arXiv:2412.03104 (VLDB'25) 论文与 github.com/NetManAIOps/ChatTS 源码整理 · 数值以论文与源码为准 
