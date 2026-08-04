# 上游基线记录

本目录不是凭空重写 ChatTS。2026-08-04 使用 Hugging Face Hub 下载了
[`bytedance-research/ChatTS-8B`](https://huggingface.co/bytedance-research/ChatTS-8B/tree/main)
除模型 tensor 文件以外的官方目录，快照 revision 为：

```text
7216a84034063c2d394901fe45d20a1e8d773a1f
```

三个关键 Python 文件的官方 SHA-256：

| 文件 | 官方 SHA-256 | 本项目处理 |
|---|---|---|
| `configuration_qwen3_ts.py` | `9140c0fb47f43bd4ef8d7dab3e3e3b28c2fbec9d99247161a112531d874529ad` | 只增加 Chronos-2 自包含配置字段与校验 |
| `modeling_qwen3_ts.py` | `05476d81fd48c92b05f24e8896fedcdec4e1df8087bcd1cf6575d67f996a629a` | 保留原 Qwen3、时序占位符展开、generation 逻辑；增加注册式 Chronos-2 encoder，并由配置选择 |
| `processing_qwen3_ts.py` | `cc6fb71c6e6a7c9cf4ef9e5df2b563cd9d0e201a8af8d4ec0199d2a71f640505` | 原样保留，避免改变训练时归一化与 `<ts>` 预处理语义 |

其他下载到并核对过的官方非 tensor 文件包括：

```text
README.md
added_tokens.json
chat_template.jinja
config.json
generation_config.json
merges.txt
processor_config.json
pytorch_model.bin.index.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
```

转换时不会直接使用官方 8B 的 `config.json`，因为你的语言模型是 Qwen3-1.7B。
脚本会从你的训练检查点复制 1.7B 的层数、hidden size、tokenizer 和 generation
配置，只用上述官方 Python 文件作为实现基线，再写入 Chronos-2 配置。这一点可以避免把
8B 配置错误套到 1.7B 权重上。

Chronos-2 的模型类与内嵌配置依据：

- [`amazon/chronos-2` config.json](https://huggingface.co/amazon/chronos-2/blob/main/config.json)
- [`amazon-science/chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting)

