#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUMMARIES = (
    REPO_ROOT / "runtime/research/synthetic-v2-univariate-capabilities-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-time-varying-seasonality-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-multitarget-capabilities-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-hierarchical-coherence-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-covariate-capabilities-experiment/summary.json",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/superpowers/baselines/figures/synthetic-v2-capabilities"
DEFAULT_INDEX = REPO_ROOT / "docs/superpowers/baselines/2026-07-02-synthetic-v2-capability-metric-plots.md"

CAPABILITY_ORDER = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
    "common_factor",
    "hierarchical_coherence",
    "covariate_response",
)
MODEL_ORDER = (
    "naive",
    "seasonal_naive",
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "timesfm2.5",
    "AutoARIMA",
    "Holt-Winters",
)
MODEL_COLORS = {
    "naive": "#8a8f98",
    "seasonal_naive": "#5f6673",
    "Timer-3.5": "#2563eb",
    "Timer-3.0": "#0891b2",
    "Chronos-2": "#16a34a",
    "moirai2": "#9333ea",
    "toto2.0": "#f97316",
    "timesfm2.5": "#dc2626",
    "AutoARIMA": "#7c2d12",
    "Holt-Winters": "#a16207",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic v2 capability metrics by intensity.")
    parser.add_argument("--summary", action="append", type=Path, dest="summaries", help="Summary JSON path. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metric", action="append", dest="metrics", help="Metric to plot for every capability. Can be repeated. Defaults to mae and mase.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [load_summary(path) for path in (args.summaries or list(DEFAULT_SUMMARIES))]
    metrics = args.metrics or ["mae", "mase"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, str]] = []
    for metric in metrics:
        for capability_id in capabilities_in_order(summaries):
            if not has_metric(summaries, capability_id, metric):
                continue
            path = args.output_dir / f"{capability_id}-{metric}.png"
            plot_capability_metric(summaries, capability_id, metric, path)
            generated.append({"capability_id": capability_id, "metric": metric, "path": markdown_path(path, args.index.parent)})

    if has_metric(summaries, "hierarchical_coherence", "coherence_mae"):
        path = args.output_dir / "hierarchical_coherence-coherence_mae.png"
        plot_capability_metric(summaries, "hierarchical_coherence", "coherence_mae", path)
        generated.append({"capability_id": "hierarchical_coherence", "metric": "coherence_mae", "path": markdown_path(path, args.index.parent)})

    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(render_index(summaries, generated, metrics), encoding="utf-8")
    print(f"wrote {len(generated)} plots to {args.output_dir}")
    print(f"wrote index: {args.index}")
    return 0


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_source_path"] = display_path(path)
    return payload


def capabilities_in_order(summaries: list[dict[str, Any]]) -> list[str]:
    seen = {
        capability_id
        for summary in summaries
        for capability_id in summary.get("requested_capabilities", [])
    }
    ordered = [capability_id for capability_id in CAPABILITY_ORDER if capability_id in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def plot_capability_metric(summaries: list[dict[str, Any]], capability_id: str, metric: str, path: Path) -> None:
    rows = rows_for_capability(summaries, capability_id)
    values_by_model: dict[str, dict[int, float]] = {}
    for row in rows:
        value = row.get("metrics", {}).get(metric)
        if value is None:
            continue
        values_by_model.setdefault(row["model_id"], {})[row_intensity(row)] = float(value)
    if not values_by_model:
        raise RuntimeError(f"no metric [{metric}] values for capability [{capability_id}]")

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=160)
    for model_id in sorted(values_by_model, key=model_sort_key):
        values = values_by_model[model_id]
        xs = sorted(values)
        ys = [values[x] for x in xs]
        is_baseline = model_id in {"naive", "seasonal_naive"}
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8 if not is_baseline else 1.4,
            linestyle="--" if is_baseline else "-",
            color=MODEL_COLORS.get(model_id),
            label=model_id,
            alpha=0.95 if not is_baseline else 0.8,
        )

    ax.set_title(f"{capability_id} - {metric.upper()} by intensity", fontsize=13, pad=12)
    ax.set_xlabel("Intensity")
    ax.set_ylabel(metric.upper())
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, axis="both", color="#d8dee8", linewidth=0.8, alpha=0.75)
    ax.set_facecolor("#fbfcfe")
    fig.patch.set_facecolor("white")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(path)
    plt.close(fig)


def rows_for_capability(summaries: list[dict[str, Any]], capability_id: str) -> list[dict[str, Any]]:
    rows = [
        row
        for summary in summaries
        for row in summary.get("summaries", [])
        if row.get("capability_id") == capability_id
    ]
    return rows


def has_metric(summaries: list[dict[str, Any]], capability_id: str, metric: str) -> bool:
    return any(metric in row.get("metrics", {}) for row in rows_for_capability(summaries, capability_id))


def render_index(summaries: list[dict[str, Any]], generated: list[dict[str, str]], metrics: list[str]) -> str:
    lines = [
        "# Synthetic v2 能力维度指标曲线",
        "",
        "日期：2026-07-02",
        "",
        f"结果指标：{', '.join(f'`{metric}`' for metric in metrics)}。每张图横坐标为 intensity，纵坐标为指标值，曲线为模型；`naive` 和 `seasonal_naive` 使用虚线。",
        "",
        "## 输入数据",
        "",
        "| Source | Capabilities |",
        "| --- | --- |",
    ]
    for summary in summaries:
        capabilities = ", ".join(f"`{capability_id}`" for capability_id in summary.get("requested_capabilities", []))
        lines.append(f"| `{summary['_source_path']}` | {capabilities} |")
    lines.extend(["", "## 图表", ""])
    for item in generated:
        lines.extend(
            [
                f"### `{item['capability_id']}` / `{item['metric']}`",
                "",
                f"![{item['capability_id']} {item['metric']}]({item['path']})",
                "",
            ]
        )
    return "\n".join(lines)


def model_sort_key(model_id: str) -> tuple[int, str]:
    if model_id in MODEL_ORDER:
        return MODEL_ORDER.index(model_id), model_id
    return len(MODEL_ORDER), model_id


def row_intensity(row: dict[str, Any]) -> int:
    return int(row.get("intensity", row.get("difficulty")))


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def markdown_path(path: Path, relative_to: Path) -> str:
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return display_path(path)


if __name__ == "__main__":
    raise SystemExit(main())
