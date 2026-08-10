# Official benchmark source snapshots

这里保存 ChatTS 外部评测适配所依赖的官方 benchmark 源码快照，便于从本仓库
一次下载后复制到离线服务器。两个目录都去除了上游嵌套 `.git`，不是 Git
submodule；上游 LICENSE 文件保持原样。

| 目录 | 上游 | 固定 revision | 许可证 | 数据状态 |
|---|---|---|---|---|
| `TS-Haystack/` | <https://github.com/AI-X-Labs/TS-Haystack> | `5e8d3162d47176e19fbd9c5f3d3cf9c6e0e9a7d4` | CC BY-NC 4.0 | Git 仓库只含 `data/.gitkeep`；正式数据和 sidecar 仍需按官方 dataset card 下载到 `data/` |
| `TimeSeriesExam/` | <https://github.com/moment-timeseries-foundation-model/TimeSeriesExam> | `384cf50864860c65e962b441eaa4c201857a06f8` | MIT | 已包含 `output/round_0_folder` 至 `round_3_folder`；默认评测 round 3 的 763 条样本 |

快照日期：2026-08-10。

建议复制到服务器：

```text
/share/airesearch/data/finiverse/TS-Haystack
/share/airesearch/data/finiverse/TimeSeriesExam
```

TS-Haystack 的最终结构至少应包含：

```text
TS-Haystack/
├── src/datasets/registry.py
└── data/
    ├── capture24/
    ├── sleep_psg/
    ├── ltafdb/
    └── uk_dale/
```

TimeSeriesExam 默认运行所需的两个关键文件已经在快照内：

```text
TimeSeriesExam/evaluate/concepts.py
TimeSeriesExam/output/round_3_folder/qa_dataset.json
```

本目录仅用于上游源码与公开数据的再分发/离线复制。使用 TS-Haystack 时必须遵守
其 CC BY-NC 4.0 的非商业限制；具体条款以各子目录的 `LICENSE` 为准。
