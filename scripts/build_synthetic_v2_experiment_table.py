#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUMMARIES = (
    REPO_ROOT / "runtime/research/synthetic-v2-univariate-capabilities-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-time-varying-seasonality-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-multitarget-capabilities-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-hierarchical-coherence-experiment/summary.json",
    REPO_ROOT / "runtime/research/synthetic-v2-covariate-capabilities-experiment/summary.json",
)
DEFAULT_OUTPUT = REPO_ROOT / "docs/superpowers/baselines/2026-07-01-synthetic-v2-all-capabilities-experiment-table.md"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined Markdown table from synthetic v2 experiment summaries.")
    parser.add_argument("--summary", action="append", type=Path, dest="summaries", help="Summary JSON path. Can be repeated.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [load_summary(path) for path in (args.summaries or list(DEFAULT_SUMMARIES))]
    output = render_markdown(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"wrote table: {args.output}")
    return 0


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_source_path"] = display_path(path)
    return payload


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Synthetic v2 全能力真实模型实验表",
            "",
            "日期：2026-07-01",
            "",
            "## 输入实验",
            "",
            *experiment_lines(summaries),
            "",
            "## 模型支持边界",
            "",
            *support_lines(summaries),
            "",
            "## 主要观察",
            "",
            *observation_lines(summaries),
            "",
            "## 指标长表",
            "",
            *metric_table_lines(summaries),
            "",
        ]
    )


def experiment_lines(summaries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Source | Capabilities | Target dim | Covariate dim | Samples / intensity | Selected models |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        requirements = summary.get("requirements") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['_source_path']}`",
                    ", ".join(f"`{capability}`" for capability in summary.get("requested_capabilities", [])),
                    str(requirements.get("target_dim", 1)),
                    str(requirements.get("covariate_dim", 0)),
                    str(summary.get("sample_count_per_capability_intensity", summary.get("sample_count_per_capability_difficulty", "-"))),
                    ", ".join(f"`{model}`" for model in summary.get("selected_models", [])) or "none",
                ]
            )
            + " |"
        )
    return lines


def support_lines(summaries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Experiment | Requested model | Status | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for summary in summaries:
        experiment = ", ".join(summary.get("requested_capabilities", []))
        selected = set(summary.get("selected_models", []))
        skipped = {item["model_id"]: item.get("reason", "-") for item in summary.get("skipped_models", [])}
        for model_id in summary.get("requested_models", []):
            if model_id in selected:
                status = "selected"
                reason = "-"
            else:
                status = "skipped"
                reason = skipped.get(model_id, "-")
            lines.append(f"| {experiment} | `{model_id}` | {status} | {reason} |")
    return lines


def observation_lines(summaries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for capability_id in CAPABILITY_ORDER:
        summary = summary_for_capability(summaries, capability_id)
        if summary is None:
            continue
        best = best_model(summary, capability_id)
        if best is None:
            lines.append(f"- `{capability_id}`：没有真实模型成功结果。")
            continue
        mae, model_id = best
        lines.append(f"- `{capability_id}`：平均 MAE 最低的是 `{model_id}`（{fmt(mae)}）。")
    lines.extend(
        [
            "- 单变量 6 个维度本轮 6 个真实模型全部成功；Timer 修复后 `regime_switching` 不再 failed request。",
            "- 新增 `time_varying_seasonality` 和 `hierarchical_coherence` 都能跑通；后者额外记录 `coherence_mae`，用于检查预测是否满足 parent-child 加总关系。",
            "- 多目标维度当前只有 `toto2.0` 声明支持 `target_dim=3`，因此这些维度更像 toto 与 naive baselines 的 sanity check，还不能做横向模型排名。",
            "- `covariate_response` 当前按单目标 known-future covariates 跑，只有 `Chronos-2` 纳入主实验；AutoARIMA/Holt-Winters 小样本 dry run 显示慢或失败，未进入主表。",
            "- `regime_switching` 使用历史可观察的重复切换时钟，预测期切换不再是无先兆冲击。",
            "- `nonlinear_persistence` 明确测试稳定的多滞后非线性依赖，不再声称 fractional long memory。",
            "- `predictable_intermittency` 使用历史重复脉冲时钟，强度只控制脉冲显著性。",
        ]
    )
    return lines


def metric_table_lines(summaries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Capability | Intensity | Model | Target dim | Cov dim | Samples | Fail | MAE | MASE | MSE | MAE / SNaive | Coherence MAE |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = combined_rows(summaries)
    for row in sorted(rows, key=sort_key):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['capability_id']}`",
                    str(row["intensity"]),
                    f"`{row['model_id']}`",
                    str(row.get("target_dim", 1)),
                    str(row.get("covariate_dim", 0)),
                    str(row.get("sample_count", 0)),
                    str(row.get("failed_count", 0)),
                    fmt(row.get("mae")),
                    fmt(row.get("mase")),
                    fmt(row.get("mse")),
                    fmt(row.get("mae_vs_seasonal_naive")),
                    fmt(row.get("coherence_mae")),
                ]
            )
            + " |"
        )
    return lines


def combined_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        comparisons = {
            (row["model_id"], row["capability_id"], row_intensity(row)): row
            for row in summary.get("comparisons", [])
        }
        failures = {
            (row["model_id"], row["capability_id"], row_intensity(row)): int(row.get("failed_count", 0))
            for row in summary.get("failure_counts", [])
        }
        seen: set[tuple[str, str, int]] = set()
        for item in summary.get("summaries", []):
            intensity = row_intensity(item)
            key = (item["model_id"], item["capability_id"], intensity)
            seen.add(key)
            comparison = comparisons.get(key, {})
            rows.append(
                {
                    "capability_id": item["capability_id"],
                    "intensity": intensity,
                    "difficulty": intensity,
                    "model_id": item["model_id"],
                    "target_dim": item.get("target_dim", summary.get("requirements", {}).get("target_dim", 1)),
                    "covariate_dim": item.get("covariate_dim", summary.get("requirements", {}).get("covariate_dim", 0)),
                    "sample_count": item.get("sample_count", 0),
                    "failed_count": failures.get(key, 0),
                    "mae": item.get("metrics", {}).get("mae"),
                    "mase": item.get("metrics", {}).get("mase"),
                    "mse": item.get("metrics", {}).get("mse"),
                    "coherence_mae": item.get("metrics", {}).get("coherence_mae"),
                    "mae_vs_seasonal_naive": comparison.get("mae_vs_seasonal_naive"),
                }
            )
        for key, failed_count in failures.items():
            if key in seen:
                continue
            model_id, capability_id, intensity = key
            rows.append(
                {
                    "capability_id": capability_id,
                    "intensity": intensity,
                    "difficulty": intensity,
                    "model_id": model_id,
                    "target_dim": summary.get("requirements", {}).get("target_dim", 1),
                    "covariate_dim": summary.get("requirements", {}).get("covariate_dim", 0),
                    "sample_count": 0,
                    "failed_count": failed_count,
                    "mae": None,
                    "mase": None,
                    "mse": None,
                    "mae_vs_seasonal_naive": None,
                }
            )
    return rows


def best_model(summary: dict[str, Any], capability_id: str) -> tuple[float, str] | None:
    values_by_model: dict[str, list[float]] = {}
    for row in summary.get("summaries", []):
        if row["capability_id"] != capability_id or row["model_id"] in {"naive", "seasonal_naive"}:
            continue
        value = row.get("metrics", {}).get("mae")
        if value is None:
            continue
        values_by_model.setdefault(row["model_id"], []).append(float(value))
    if not values_by_model:
        return None
    return min((sum(values) / len(values), model_id) for model_id, values in values_by_model.items())


def summary_for_capability(summaries: list[dict[str, Any]], capability_id: str) -> dict[str, Any] | None:
    return next((summary for summary in summaries if capability_id in summary.get("requested_capabilities", [])), None)


def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    capability_rank = CAPABILITY_ORDER.index(row["capability_id"]) if row["capability_id"] in CAPABILITY_ORDER else len(CAPABILITY_ORDER)
    model_rank = MODEL_ORDER.index(row["model_id"]) if row["model_id"] in MODEL_ORDER else len(MODEL_ORDER)
    return capability_rank, row_intensity(row), model_rank


def row_intensity(row: dict[str, Any]) -> int:
    return int(row.get("intensity", row.get("difficulty")))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
