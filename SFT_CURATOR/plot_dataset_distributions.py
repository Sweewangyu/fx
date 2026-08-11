#!/usr/bin/env python3
"""Plot per-dataset quality and difficulty distributions from merge reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

QUALITY_LEVELS = ("unusable", "weak", "acceptable", "good", "excellent")
DIFFICULTY_LEVELS = ("very_easy", "easy", "moderate", "hard", "very_hard")

QUALITY_COLORS = {
    "unusable": "#C62828",
    "weak": "#EF6C00",
    "acceptable": "#F9A825",
    "good": "#43A047",
    "excellent": "#1565C0",
}

DIFFICULTY_COLORS = {
    "very_easy": "#2E7D32",
    "easy": "#66BB6A",
    "moderate": "#F9A825",
    "hard": "#EF6C00",
    "very_hard": "#C62828",
}


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; install it with `python -m pip install matplotlib`"
        ) from exc
    return plt


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "dataset"


def load_distributions(
    path: Path,
) -> tuple[list[str], dict[str, Counter], dict[str, Counter]]:
    quality: dict[str, Counter] = defaultdict(Counter)
    difficulty: dict[str, Counter] = defaultdict(Counter)
    dataset_order: list[str] = []
    seen_datasets = set()
    seen_cells = set()

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source", "ability_label", "quality", "difficulty", "count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"input CSV is missing columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, 2):
            source = str(row["source"]).strip()
            ability = str(row["ability_label"]).strip()
            quality_label = str(row["quality"]).strip()
            difficulty_label = str(row["difficulty"]).strip()
            if not source:
                raise ValueError(f"empty source at {path}:{line_number}")
            if quality_label not in QUALITY_LEVELS:
                raise ValueError(
                    f"unknown quality `{quality_label}` at {path}:{line_number}"
                )
            if difficulty_label not in DIFFICULTY_LEVELS:
                raise ValueError(
                    f"unknown difficulty `{difficulty_label}` at {path}:{line_number}"
                )
            try:
                count = int(row["count"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid count at {path}:{line_number}") from exc
            if count < 0:
                raise ValueError(f"negative count at {path}:{line_number}")

            cell = (source, ability, quality_label, difficulty_label)
            if cell in seen_cells:
                raise ValueError(f"duplicate distribution cell at {path}:{line_number}")
            seen_cells.add(cell)
            if source not in seen_datasets:
                seen_datasets.add(source)
                dataset_order.append(source)
            quality[source][quality_label] += count
            difficulty[source][difficulty_label] += count

    if not dataset_order:
        raise ValueError(f"input CSV has no data rows: {path}")
    for source in dataset_order:
        quality_total = sum(quality[source].values())
        difficulty_total = sum(difficulty[source].values())
        if quality_total != difficulty_total:
            raise ValueError(
                f"quality/difficulty totals disagree for {source}: "
                f"{quality_total} != {difficulty_total}"
            )
        if quality_total <= 0:
            raise ValueError(f"dataset has no samples: {source}")
    return dataset_order, quality, difficulty


def draw_distribution(
    plt: Any,
    dataset: str,
    axis_name: str,
    levels: tuple[str, ...],
    counts: Counter,
    colors: dict[str, str],
    output_path: Path,
    dpi: int,
) -> None:
    values = [int(counts[level]) for level in levels]
    total = sum(values)
    percentages = [100.0 * value / total for value in values]
    labels = [level.replace("_", " ").title() for level in levels]

    figure, axes = plt.subplots(figsize=(10, 6))
    bars = axes.bar(
        labels,
        values,
        color=[colors[level] for level in levels],
        edgecolor="#333333",
        linewidth=0.7,
        width=0.72,
    )
    axes.set_title(
        f"{dataset}\n{axis_name} Distribution (n={total:,})",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    axes.set_xlabel(axis_name, fontsize=12)
    axes.set_ylabel("Number of QA Samples", fontsize=12)
    axes.grid(axis="y", linestyle="--", alpha=0.3)
    axes.set_axisbelow(True)
    axes.tick_params(axis="x", labelsize=11)
    axes.tick_params(axis="y", labelsize=10)

    maximum = max(values)
    axes.set_ylim(0, max(1.0, maximum * 1.24))
    for bar, value, percent in zip(bars, values, percentages):
        if value == 0:
            continue
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + maximum * 0.025,
            f"{value:,}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_summary(
    path: Path,
    datasets: list[str],
    quality: dict[str, Counter],
    difficulty: dict[str, Counter],
) -> list[dict[str, Any]]:
    rows = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "dataset",
                "distribution",
                "level",
                "count",
                "percent",
                "total",
            ),
        )
        writer.writeheader()
        for dataset in datasets:
            total = sum(quality[dataset].values())
            dataset_summary = {
                "dataset": dataset,
                "total": total,
                "quality": {},
                "difficulty": {},
            }
            for distribution, levels, values in (
                ("quality", QUALITY_LEVELS, quality[dataset]),
                ("difficulty", DIFFICULTY_LEVELS, difficulty[dataset]),
            ):
                for level in levels:
                    count = int(values[level])
                    percent = round(100.0 * count / total, 4)
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "distribution": distribution,
                            "level": level,
                            "count": count,
                            "percent": percent,
                            "total": total,
                        }
                    )
                    dataset_summary[distribution][level] = {
                        "count": count,
                        "percent": percent,
                    }
            rows.append(dataset_summary)
    return rows


def write_wide_summary(
    path: Path,
    datasets: list[str],
    quality: dict[str, Counter],
    difficulty: dict[str, Counter],
) -> None:
    fields = ["dataset", "total"]
    for prefix, levels in (
        ("quality", QUALITY_LEVELS),
        ("difficulty", DIFFICULTY_LEVELS),
    ):
        for level in levels:
            fields.extend((f"{prefix}_{level}_count", f"{prefix}_{level}_percent"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            total = sum(quality[dataset].values())
            row: dict[str, Any] = {"dataset": dataset, "total": total}
            for prefix, levels, values in (
                ("quality", QUALITY_LEVELS, quality[dataset]),
                ("difficulty", DIFFICULTY_LEVELS, difficulty[dataset]),
            ):
                for level in levels:
                    count = int(values[level])
                    row[f"{prefix}_{level}_count"] = count
                    row[f"{prefix}_{level}_percent"] = round(100.0 * count / total, 4)
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为每个数据集分别绘制质量分布图和难度分布图"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("merged_labels/reports/source_ability_quality_difficulty.csv"),
        help="merge_tsqa_annotations.py 生成的按来源联合分布 CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("merged_labels/reports/dataset_plots"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="只绘制指定数据集；可重复传入，不传则绘制全部",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = args.input.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if args.dpi < 72:
            raise ValueError("--dpi must be at least 72")

        datasets, quality, difficulty = load_distributions(input_path)
        if args.dataset:
            selected = set(args.dataset)
            unknown = selected - set(datasets)
            if unknown:
                raise ValueError(f"unknown dataset(s): {', '.join(sorted(unknown))}")
            datasets = [dataset for dataset in datasets if dataset in selected]

        plt = load_matplotlib()
        used_names = set()
        outputs = []
        for dataset in datasets:
            directory_name = safe_name(dataset)
            if directory_name in used_names:
                raise ValueError(
                    f"dataset output directory collision: {directory_name}"
                )
            used_names.add(directory_name)
            dataset_dir = output_dir / directory_name
            quality_path = dataset_dir / "quality_distribution.png"
            difficulty_path = dataset_dir / "difficulty_distribution.png"
            draw_distribution(
                plt,
                dataset,
                "Quality",
                QUALITY_LEVELS,
                quality[dataset],
                QUALITY_COLORS,
                quality_path,
                args.dpi,
            )
            draw_distribution(
                plt,
                dataset,
                "Difficulty",
                DIFFICULTY_LEVELS,
                difficulty[dataset],
                DIFFICULTY_COLORS,
                difficulty_path,
                args.dpi,
            )
            outputs.append(
                {
                    "dataset": dataset,
                    "quality_plot": str(quality_path),
                    "difficulty_plot": str(difficulty_path),
                }
            )
            print(
                json.dumps(
                    {"event": "dataset_plotted", "dataset": dataset},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )

        long_summary_path = output_dir / "dataset_distribution_summary.csv"
        wide_summary_path = output_dir / "dataset_distribution_wide.csv"
        summary_rows = write_summary(
            long_summary_path,
            datasets,
            quality,
            difficulty,
        )
        write_wide_summary(
            wide_summary_path,
            datasets,
            quality,
            difficulty,
        )
        manifest = {
            "schema_version": "tsqa-dataset-distribution-plots-v1",
            "input": str(input_path),
            "dataset_count": len(datasets),
            "datasets": summary_rows,
            "summary_csv": str(long_summary_path),
            "wide_summary_csv": str(wide_summary_path),
            "outputs": outputs,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports concise errors.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
