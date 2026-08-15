# ChatTS × Qwen3-4B-Instruct-2507 适配

本目录把 [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
接入现有 ChatTS + Chronos-2 两阶段训练框架。

Qwen3-4B 在这里替换的是 ChatTS 的语言模型主干，不是时间序列编码器；Chronos-2
继续负责时间序列特征提取。原始 Qwen checkpoint 不能直接传给 ChatTS-Training，
因为它没有 `ts_encoder`、ChatTS Processor 和 `<ts>/<ts/>` token 元数据。

## 目录内容

这些文件保持 ChatTS-Training 的相对路径，可直接复制：

- [`scripts/download_prepare_qwen3_4b_instruct_2507.sh`](./scripts/download_prepare_qwen3_4b_instruct_2507.sh)：联网机器下载与转换入口。
- [`scripts/prepare_qwen3_chatts_base.py`](./scripts/prepare_qwen3_chatts_base.py)：不加载权重张量的初始化目录转换器。
- [`scripts/full/run_chronos2_qwen3_4b_2507_two_stage.sh`](./scripts/full/run_chronos2_qwen3_4b_2507_two_stage.sh)：内网容器两阶段训练入口。
- [`tests/pipeline/test_prepare_qwen3_chatts_base.py`](./tests/pipeline/test_prepare_qwen3_chatts_base.py)：token、配置、权重索引和冲突拒绝测试。

复制到服务器的 ChatTS-Training：

```bash
rsync -av scripts/ /workspace/ChatTS-Training/scripts/
rsync -av tests/ /workspace/ChatTS-Training/tests/
```

## 1. 联网机器下载并转换

转换器保留 Qwen3-4B 的语言模型权重，只执行以下修改：

1. `Qwen3ForCausalLM` 改为 `Qwen3TSForCausalLM`。
2. 保留 4B 的 `hidden_size=2560` 和 36 层结构。
3. 添加 `<ts>` 与 `<ts/>`，固定 ID 为 `151669/151670`。
4. 添加 ChatTS Processor 和 remote-code 映射。
5. 写入 `CHATTS_BASE_MANIFEST.json`，训练前验证模型身份。

```bash
cd /workspace/ChatTS-Training

bash scripts/download_prepare_qwen3_4b_instruct_2507.sh \
  --raw-model-dir /share/airesearch/data/finiverse/model/Qwen3-4B-Instruct-2507 \
  --chatts-template /share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B \
  --output-dir /share/airesearch/data/finiverse/model/ChatTS-Qwen3-4B-Instruct-2507
```

`--chatts-template` 只读取现有官方 ChatTS-Qwen3-8B 目录中的三个 Python 模型代码文件，
不会复制 8B 权重。默认对4B权重创建 hardlink；若不在同一文件系统，会自动退化为复制。
需要完全复制时传入 `--weight-mode copy`。

已经下载原始 Qwen 时：

```bash
bash scripts/download_prepare_qwen3_4b_instruct_2507.sh \
  --skip-download \
  --raw-model-dir /path/to/Qwen3-4B-Instruct-2507 \
  --chatts-template /path/to/ChatTS-Qwen3-8B \
  --output-dir /path/to/ChatTS-Qwen3-4B-Instruct-2507
```

## 2. 内网容器预检

Qwen3-4B-Instruct-2507 要求 `transformers>=4.51.0`：

```bash
python3 -c 'import transformers; print(transformers.__version__)'
```

加载 Dataset Studio 的 datav3 注册信息：

```bash
cd /workspace/ChatTS-Training

set -a
source /share/airesearch/data/finiverse/traindata/chatts-data-versions/datav3/training.env
set +a

bash scripts/full/run_chronos2_qwen3_4b_2507_two_stage.sh \
  --model-path /share/airesearch/data/finiverse/model/ChatTS-Qwen3-4B-Instruct-2507 \
  --output-root /share/airesearch/data/finiverse/output/ChatTS-msxf-4B-Instruct-2507-datav3 \
  --chronos2-path /workspace/chronos2 \
  --preflight-only
```

## 3. 正式训练

```bash
bash scripts/full/run_chronos2_qwen3_4b_2507_two_stage.sh \
  --model-path /share/airesearch/data/finiverse/model/ChatTS-Qwen3-4B-Instruct-2507 \
  --output-root /share/airesearch/data/finiverse/output/ChatTS-msxf-4B-Instruct-2507-datav3 \
  --chronos2-path /workspace/chronos2
```

默认配方：

| 阶段 | LLM LR | TS-to-text projector LR | Chronos-2 | 训练模块 |
|---|---:|---:|---|---|
| Stage 1 | `1e-5` | `3e-5` | 冻结 | 完整 Qwen3-4B + projector |
| Stage 2 | `1e-5` | `1e-5` | 冻结 | 完整 Qwen3-4B + projector |

8B 的 Stage1 checkpoint 不能作为4B的 Stage2 输入：两者 hidden size、Transformer
权重和 projector 形状不同。4B 必须重新完整训练 Stage1，再进入 Stage2。

Qwen3-4B-Instruct-2507 是 non-thinking 模型；评测时保持 `ENABLE_THINKING=0`。

## 验证状态

- `bash -n`：两个 shell 入口通过。
- `py_compile`：转换器与测试通过。
- 微型模型 fixture：成功转换与冲突 token 拒绝两个测试通过。
- 未在本地下载约 8GB 的真实权重，也未执行8卡训练；这两项应在服务器完成。
