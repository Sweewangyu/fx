from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .state import StateStore


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _fmt_delta(current: Any, baseline: Any, digits: int = 4) -> str:
    if current is None or baseline is None:
        return "—"
    return f"{float(current) - float(baseline):+.{digits}f}"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
    """Keep report ordering identical to Autoresearch._rank."""
    metrics = item.get("metrics") or {}
    score = metrics.get("primary_score")
    gpu = metrics.get("gpu_hours")
    loss = metrics.get("validation_loss")
    return (
        -(float(score) if score is not None else -1.0),
        float(gpu) if gpu is not None else float("inf"),
        float(loss) if loss is not None else float("inf"),
    )


def _assessment(item: dict[str, Any]) -> str:
    metrics = item.get("metrics") or {}
    gate = metrics.get("gate_pass")
    if item.get("status") != "completed" or gate is None:
        return "—"
    if item.get("phase") == "proxy":
        return "main-only" if gate is not False else "main-fail"
    if gate is True:
        return "guard-pass"
    if gate is False:
        return "guard-fail"
    return "—"


def _badcase_count(metrics: dict[str, Any]) -> Any:
    summary = metrics.get("badcase_summary")
    return summary.get("badcases") if isinstance(summary, dict) else None


def _tiny_tasks(metrics: dict[str, Any]) -> dict[str, Any]:
    tasks = (metrics.get("suites") or {}).get("tinybenchmarks", {}).get("tasks")
    return tasks if isinstance(tasks, dict) else {}


def _tiny_tasks_cell(metrics: dict[str, Any]) -> str:
    tasks = _tiny_tasks(metrics)
    if not tasks:
        return "—"
    return "; ".join(f"{_md(task)}={_fmt(value)}" for task, value in sorted(tasks.items()))


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_experiment_ids(manifest: dict[str, Any]) -> list[str]:
    references: list[Any] = [manifest.get("baseline")]
    references.extend(manifest.get("proxies") or [])
    references.extend(manifest.get("finalists") or [])
    result: list[str] = []
    for reference in references:
        if not isinstance(reference, dict) or not isinstance(reference.get("id"), str):
            continue
        if reference["id"] not in result:
            result.append(reference["id"])
    return result


def leaderboard_svg(experiments: list[dict[str, Any]], destination: Path) -> None:
    scored = [
        item
        for item in experiments
        if item["phase"] != "final-test"
        and item["status"] == "completed"
        and (item.get("metrics") or {}).get("primary_score") is not None
    ]
    scored.sort(key=_rank_key)
    scored = scored[:12]
    width = 1200
    row_height = 46
    height = max(220, 130 + row_height * len(scored))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><linearGradient id="bar" x1="0" x2="1"><stop offset="0" stop-color="#536dfe"/><stop offset="1" stop-color="#00b8a9"/></linearGradient></defs>',
        '<rect width="100%" height="100%" rx="20" fill="#0f172a"/>',
        '<text x="42" y="48" fill="#f8fafc" font-size="25" font-family="Inter,Arial" font-weight="700">ChatTS Autoresearch — observed search-dev scores</text>',
        '<text x="42" y="76" fill="#94a3b8" font-size="14" font-family="Inter,Arial">Primary descending, GPU-hours ascending, validation loss ascending</text>',
    ]
    if not scored:
        lines.append(
            '<text x="42" y="138" fill="#cbd5e1" font-size="18" font-family="Inter,Arial">No completed scored experiments.</text>'
        )
    else:
        max_score = max(float(item["metrics"]["primary_score"]) for item in scored) or 1.0
        for index, item in enumerate(scored):
            y = 112 + index * row_height
            score = float(item["metrics"]["primary_score"])
            bar_width = max(2.0, 610.0 * score / max_score)
            assessment = _assessment(item)
            assessment_color = (
                "#34d399"
                if assessment == "guard-pass"
                else "#fb7185"
                if assessment in {"guard-fail", "main-fail"}
                else "#94a3b8"
            )
            lines.extend(
                [
                    f'<text x="42" y="{y + 22}" fill="#e2e8f0" font-size="14" font-family="Inter,Arial">{html.escape(item["id"][:36])}</text>',
                    f'<rect x="342" y="{y + 5}" width="{bar_width:.2f}" height="24" rx="7" fill="url(#bar)"/>',
                    f'<text x="{min(970, 354 + bar_width):.2f}" y="{y + 22}" fill="#f8fafc" font-size="14" font-family="Inter,Arial" font-weight="700">{score:.4f}</text>',
                    f'<text x="1050" y="{y + 22}" fill="{assessment_color}" font-size="13" font-family="Inter,Arial">{assessment}</text>',
                ]
            )
    lines.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _search_table(lines: list[str], experiments: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Experiment | Phase | Status | Primary | TSR strict | TSR flexible | TSE strict | TSE flexible | Coverage | Haystack IoU | tiny avg | tiny tasks | Badcases | Evaluation | GPU-hours | Val loss |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|",
        ]
    )
    for item in sorted(experiments, key=_rank_key):
        metrics = item.get("metrics") or {}
        suites = metrics.get("suites") or {}
        tsr = suites.get("tsrbench") or {}
        tse = suites.get("timeseriesexam") or {}
        tiny = suites.get("tinybenchmarks") or {}
        lines.append(
            "| {id} | {phase} | {status} | {primary} | {tsr_strict} | {tsr_flex} | "
            "{tse_strict} | {tse_flex} | {coverage} | {haystack} | {tiny_avg} | "
            "{tiny_tasks} | {badcases} | {assessment} | {gpu} | {loss} |".format(
                id=_md(item["id"]),
                phase=_md(item["phase"]),
                status=_md(item["status"]),
                primary=_fmt(metrics.get("primary_score")),
                tsr_strict=_fmt(tsr.get("strict_accuracy")),
                tsr_flex=_fmt(tsr.get("flexible_accuracy")),
                tse_strict=_fmt(tse.get("strict_accuracy")),
                tse_flex=_fmt(tse.get("flexible_accuracy")),
                coverage=_fmt(metrics.get("coverage")),
                haystack=_fmt((suites.get("ts_haystack") or {}).get("mean_iou")),
                tiny_avg=_fmt(tiny.get("average_accuracy")),
                tiny_tasks=_tiny_tasks_cell(metrics),
                badcases=_fmt(_badcase_count(metrics), 0),
                assessment=_assessment(item),
                gpu=_fmt(metrics.get("gpu_hours"), 3),
                loss=_fmt(metrics.get("validation_loss"), 6),
            )
        )


def _formal_table(
    lines: list[str], experiments: list[dict[str, Any]], *, same_frozen_model: bool
) -> None:
    def role(item: dict[str, Any]) -> str:
        return str((item.get("config") or {}).get("role", "formal"))

    ordered = sorted(
        experiments,
        key=lambda item: ("baseline" not in role(item), "champion" not in role(item), item["id"]),
    )
    lines.extend(
        [
            "",
            "## 正式测试：baseline vs champion",
            "",
            "| Role | Experiment | Primary | TSR strict | TSR flexible | TSE strict | TSE flexible | Coverage | Haystack IoU | tiny avg | Badcases | Guard |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    by_role: dict[str, dict[str, Any]] = {}
    for item in ordered:
        metrics = item.get("metrics") or {}
        suites = metrics.get("suites") or {}
        tsr = suites.get("tsrbench") or {}
        tse = suites.get("timeseriesexam") or {}
        tiny = suites.get("tinybenchmarks") or {}
        item_role = role(item)
        by_role[item_role] = item
        lines.append(
            "| {role} | {id} | {primary} | {tsr_strict} | {tsr_flex} | {tse_strict} | "
            "{tse_flex} | {coverage} | {haystack} | {tiny_avg} | {badcases} | {gate} |".format(
                role=_md(item_role),
                id=_md(item["id"]),
                primary=_fmt(metrics.get("primary_score")),
                tsr_strict=_fmt(tsr.get("strict_accuracy")),
                tsr_flex=_fmt(tsr.get("flexible_accuracy")),
                tse_strict=_fmt(tse.get("strict_accuracy")),
                tse_flex=_fmt(tse.get("flexible_accuracy")),
                coverage=_fmt(metrics.get("coverage")),
                haystack=_fmt((suites.get("ts_haystack") or {}).get("mean_iou")),
                tiny_avg=_fmt(tiny.get("average_accuracy")),
                badcases=_fmt(_badcase_count(metrics), 0),
                gate=_assessment(item),
            )
        )
    baseline = next((item for name, item in by_role.items() if "baseline" in name), None)
    champion = next((item for name, item in by_role.items() if "champion" in name), None)
    if baseline is not None and champion is None and same_frozen_model:
        champion = baseline
        lines.extend(
            [
                "",
                "- 冻结 champion 与 baseline 是同一实验；正式评测只执行一次，"
                "下方差值因此均为 0。",
            ]
        )
    if baseline is None or champion is None:
        return
    base_metrics = baseline.get("metrics") or {}
    champ_metrics = champion.get("metrics") or {}
    base_suites = base_metrics.get("suites") or {}
    champ_suites = champ_metrics.get("suites") or {}
    base_tsr = base_suites.get("tsrbench") or {}
    champ_tsr = champ_suites.get("tsrbench") or {}
    base_tse = base_suites.get("timeseriesexam") or {}
    champ_tse = champ_suites.get("timeseriesexam") or {}
    lines.append(
        "| Δ champion-baseline | — | {primary} | {tsr_strict} | {tsr_flex} | "
        "{tse_strict} | {tse_flex} | {coverage} | {haystack} | {tiny_avg} | "
        "{badcases} | — |".format(
            primary=_fmt_delta(champ_metrics.get("primary_score"), base_metrics.get("primary_score")),
            tsr_strict=_fmt_delta(
                champ_tsr.get("strict_accuracy"), base_tsr.get("strict_accuracy")
            ),
            tsr_flex=_fmt_delta(
                champ_tsr.get("flexible_accuracy"), base_tsr.get("flexible_accuracy")
            ),
            tse_strict=_fmt_delta(
                champ_tse.get("strict_accuracy"), base_tse.get("strict_accuracy")
            ),
            tse_flex=_fmt_delta(
                champ_tse.get("flexible_accuracy"), base_tse.get("flexible_accuracy")
            ),
            coverage=_fmt_delta(champ_metrics.get("coverage"), base_metrics.get("coverage")),
            haystack=_fmt_delta(
                (champ_suites.get("ts_haystack") or {}).get("mean_iou"),
                (base_suites.get("ts_haystack") or {}).get("mean_iou"),
            ),
            tiny_avg=_fmt_delta(
                (champ_suites.get("tinybenchmarks") or {}).get("average_accuracy"),
                (base_suites.get("tinybenchmarks") or {}).get("average_accuracy"),
            ),
            badcases=_fmt_delta(
                _badcase_count(champ_metrics), _badcase_count(base_metrics), 0
            ),
        )
    )
    base_tasks = _tiny_tasks(base_metrics)
    champ_tasks = _tiny_tasks(champ_metrics)
    task_names = sorted(set(base_tasks) | set(champ_tasks))
    if task_names:
        lines.extend(
            [
                "",
                "### 正式 tinyBench 子任务",
                "",
                "| Task | Baseline | Champion | Δ champion-baseline |",
                "|---|---:|---:|---:|",
            ]
        )
        for task in task_names:
            lines.append(
                f"| {_md(task)} | {_fmt(base_tasks.get(task))} | "
                f"{_fmt(champ_tasks.get(task))} | "
                f"{_fmt_delta(champ_tasks.get(task), base_tasks.get(task))} |"
            )


def _manifest_section(
    lines: list[str], output_root: Path, manifest: dict[str, Any] | None, manifest_exists: bool
) -> int:
    analysis_paths = sorted((output_root / "analysis").glob("round-*.json"))
    if not manifest_exists and not analysis_paths:
        return 0
    lines.extend(["", "## 搜索冻结清单与轮次分析", ""])
    if manifest_exists:
        lines.append("- 搜索清单：[SEARCH_COMPLETE.json](SEARCH_COMPLETE.json)")
    if manifest is None and manifest_exists:
        lines.append("- `SEARCH_COMPLETE.json` 当前无法解析；本报告未从其生成排名结论。")
    elif manifest is not None:
        ranking = manifest.get("ranking") or []
        selected = manifest.get("selected_proxy_ids") or []
        lines.extend(
            [
                f"- Search hash：`{_md(manifest.get('search_hash', '—'))}`",
                "- Manifest proxy ranking："
                + (" → ".join(f"`{_md(item)}`" for item in ranking) if ranking else "—"),
                "- Selected proxies："
                + (", ".join(f"`{_md(item)}`" for item in selected) if selected else "—"),
            ]
        )
    if analysis_paths:
        lines.extend(["", "### Analysis rounds", ""])
        bound = set((manifest or {}).get("analysis_hashes") or {})
        for path in analysis_paths:
            payload = _read_object(path) or {}
            source = payload.get("source_experiment_id", "—")
            count = payload.get("sampled_badcases", "—")
            family = payload.get("recommended_family", "—")
            status = "manifest-bound" if path.name in bound else "historical/unbound"
            lines.append(
                f"- [{path.name}](analysis/{path.name}) — source `{_md(source)}`, "
                f"sampled badcases `{_md(count)}`, recommendation `{_md(family)}`, {status}"
            )
    return len(analysis_paths)


def generate_report(state: StateStore, output_root: Path, freeze: dict[str, Any] | None) -> Path:
    experiments = state.list_experiments()
    experiment_by_id = {item["id"]: item for item in experiments}
    manifest_path = output_root / "SEARCH_COMPLETE.json"
    manifest = _read_object(manifest_path)
    all_search = [item for item in experiments if item["phase"] != "final-test"]
    stale_count = 0
    if manifest_path.is_file() and manifest is not None:
        official_ids = _manifest_experiment_ids(manifest)
        search_experiments = [experiment_by_id[item_id] for item_id in official_ids if item_id in experiment_by_id]
        stale_count = sum(item["id"] not in set(official_ids) for item in all_search)
    elif manifest_path.is_file():
        search_experiments = []
        stale_count = len(all_search)
    else:
        search_experiments = all_search
    formal_experiments = [item for item in experiments if item["phase"] == "final-test"]
    report_experiments = [*search_experiments, *formal_experiments]
    scored = [
        item
        for item in report_experiments
        if item["status"] == "completed" and (item.get("metrics") or {}).get("primary_score") is not None
    ]
    svg_path = output_root / "figures" / "leaderboard.svg"
    leaderboard_svg(search_experiments, svg_path)
    lines = [
        "# ChatTS Chronos-2 Autoresearch 报告",
        "",
        "> 本报告只汇总实际产生并被解析器验证的结果；空缺项不会由模型推测或补写。",
        "",
        "## 结论",
        "",
    ]
    if freeze:
        lines.extend(
            [
                f"- 冻结冠军：`{freeze['champion']['experiment_id']}`",
                f"- 冻结模型：`{freeze['champion']['model_path']}`",
                f"- 冻结时间：`{freeze['frozen_at']}`",
            ]
        )
    else:
        lines.append("- 尚未冻结冠军；正式测试仍处于锁定状态。")
    lines.extend(
        [
            "- 当前实验固定 Chronos-2 与 seed 42；单 seed 是本报告的明确限制。",
            "",
            "## 实验排行榜（search-dev）",
            "",
            "![Observed leaderboard](figures/leaderboard.svg)",
            "",
            "排序规则与搜索器一致：Primary 降序，然后 GPU-hours 升序，再按 val loss 升序。",
            "Proxy 只跑主指标，因此标记为 `main-only`，不代表已通过 guard。",
            "",
        ]
    )
    if manifest is not None:
        lines.append("Search 表仅包含 `SEARCH_COMPLETE.json` 绑定的 baseline/proxies/finalists。")
        if stale_count:
            lines.append(f"已排除 {stale_count} 个未被清单引用的历史/陈旧实验。")
        lines.append("")
    elif manifest_path.is_file():
        lines.extend(
            [
                "`SEARCH_COMPLETE.json` 无法解析；为避免把 SQLite 遗留实验当作正式搜索结果，",
                f"本表已将 {stale_count} 个未验证搜索实验全部排除。",
                "",
            ]
        )
    _search_table(lines, search_experiments)
    if formal_experiments:
        same_frozen_model = bool(
            freeze
            and freeze.get("baseline", {}).get("experiment_id")
            == freeze.get("champion", {}).get("experiment_id")
        )
        _formal_table(lines, formal_experiments, same_frozen_model=same_frozen_model)
    analysis_count = _manifest_section(lines, output_root, manifest, manifest_path.is_file())
    lines.extend(["", "## 失败与门槛", ""])
    failures = [item for item in report_experiments if item["status"] == "failed"]
    guard_rejected = [
        item
        for item in scored
        if item["phase"] != "proxy" and (item.get("metrics") or {}).get("gate_pass") is False
    ]
    proxy_main_failed = [
        item
        for item in scored
        if item["phase"] == "proxy" and (item.get("metrics") or {}).get("gate_pass") is False
    ]
    if not failures and not guard_rejected and not proxy_main_failed:
        lines.append("- 当前没有已记录的执行失败、主指标失败或 guard gate 淘汰。")
    for item in failures:
        lines.append(f"- `{item['id']}` 执行失败：{item.get('error') or '未提供错误信息'}")
    for item in proxy_main_failed:
        lines.append(
            f"- `{item['id']}` 未通过 proxy 主指标检查："
            + "; ".join((item.get("metrics") or {}).get("gate_reasons", []))
        )
    for item in guard_rejected:
        lines.append(
            f"- `{item['id']}` 未通过 guard gate："
            + "; ".join((item.get("metrics") or {}).get("gate_reasons", []))
        )
    lines.extend(
        [
            "",
            "## 可复现性",
            "",
            "- 完整配置、数据快照、评测协议和命令分别由 SHA256 绑定。",
            "- 实验状态见 `state.sqlite3`，交换格式见 `experiments.jsonl` 与 `leaderboard.csv`。",
            "- 每个实验的原始训练/评测日志、统一 badcase 和 resolved config 均保留在本目录。",
            "- DeepSeek 只参与标签、错误分析和下一轮白名单补丁建议，不产生或改写评测分数。",
            "",
        ]
    )
    report_path = output_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "completed": sum(item["status"] == "completed" for item in experiments),
        "failed": len(failures),
        "scored": len(scored),
        "frozen": freeze is not None,
        "search_manifest_exists": manifest_path.is_file(),
        "search_manifest_valid_json": manifest is not None,
        "official_search_experiments": len(search_experiments),
        "stale_search_experiments": stale_count,
        "analysis_rounds": analysis_count,
    }
    (output_root / "report_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report_path
