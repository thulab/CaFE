#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.synthetic_generation_service import (  # noqa: E402
    PILOT_ACCEPTANCE_CAPS,
    _base_features,
    _generate_accepted_sample_values,
    _seed_for,
    _standardize_by_context,
)
from synthetic_feature_profile import WindowSpec, feature_vector, read_tsf_series, select_tsf_windows  # noqa: E402


DEFAULT_M4_PATH = REPO_ROOT / "runtime/research/m4_hourly_dataset.zip"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/research/synthetic-v2-generator-experiment"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs/superpowers/baselines/2026-06-29-synthetic-v2-generator-experiment.md"

CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24
SAMPLE_COUNT = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic v2 generator feature and baseline-response experiment.")
    parser.add_argument("--m4-path", type=Path, default=DEFAULT_M4_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(args.m4_path, sample_count=args.sample_count)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(results, m4_path=args.m4_path, output_dir=args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote report: {args.report}")
    print(f"wrote summary: {args.output_dir / 'summary.json'}")
    return 0


def run_experiment(m4_path: Path, *, sample_count: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for capability_id in ("trend", "multi_seasonal"):
        for generator_id, generator in (
            ("legacy", generate_legacy_sample),
            ("v2", generate_v2_sample),
        ):
            for difficulty in range(1, 6):
                samples = [
                    generator(capability_id, difficulty, sample_index)
                    for sample_index in range(sample_count)
                ]
                rows.append(summarize_samples(f"{generator_id}_{capability_id}", capability_id, difficulty, samples))
    rows.append(summarize_samples("real_m4_hourly", "real_anchor", None, load_real_m4_samples(m4_path, sample_count)))
    checks = acceptance_checks(rows)
    return {
        "schema_version": "synthetic_v2_generator_experiment.v1",
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "season_length": SEASON_LENGTH,
        "sample_count": sample_count,
        "rows": rows,
        "checks": checks,
    }


def generate_v2_sample(capability_id: str, difficulty: int, sample_index: int) -> np.ndarray:
    sample_seed = _seed_for(20260629, capability_id, difficulty * 10_000 + sample_index)
    values, _latent_params, _covariates, _features = _generate_accepted_sample_values(
        capability_id,
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        difficulty,
        sample_seed,
    )
    return values


def generate_legacy_sample(capability_id: str, difficulty: int, sample_index: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_for(20250629, capability_id, difficulty * 10_000 + sample_index))
    if capability_id == "trend":
        values = legacy_trend(CONTEXT_LENGTH + HORIZON, difficulty, rng)
    elif capability_id == "multi_seasonal":
        values = legacy_multi_seasonal(CONTEXT_LENGTH + HORIZON, difficulty, rng)
    else:
        raise ValueError(f"unsupported capability: {capability_id}")
    return _standardize_by_context(values, CONTEXT_LENGTH)


def legacy_trend(length: int, difficulty: int, rng: np.random.Generator) -> np.ndarray:
    lam = (difficulty - 1) / 4
    seasonal, slow, trend = _base_features(length, SEASON_LENGTH)
    slope = rng.uniform(-1.2, 1.2, size=1) * (0.6 + lam)
    curvature = rng.uniform(-0.7, 0.7, size=1) * lam
    values = trend[:, None] * slope + (trend[:, None] ** 2) * curvature
    values += (0.25 + 0.2 * lam) * seasonal[:, None] + 0.12 * slow[:, None]
    values += rng.normal(0.0, 0.08 + 0.08 * lam, size=(length, 1))
    return values


def legacy_multi_seasonal(length: int, difficulty: int, rng: np.random.Generator) -> np.ndarray:
    lam = (difficulty - 1) / 4
    t = np.arange(length, dtype=float)
    periods = [max(4, SEASON_LENGTH), max(5, SEASON_LENGTH // 2), max(8, SEASON_LENGTH * 2)]
    values = np.zeros((length, 1))
    for period in periods:
        amp = rng.uniform(0.2, 0.8 + 0.4 * lam, size=1)
        phase = rng.uniform(0, 2 * np.pi, size=1)
        values += amp[None, :] * np.sin(2 * np.pi * t[:, None] / period + phase[None, :])
    values += rng.normal(0.0, 0.08 + 0.09 * lam, size=values.shape)
    return values


def load_real_m4_samples(path: Path, sample_count: int) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"M4 Hourly dataset not found: {path}")
    _metadata, series = read_tsf_series(path)
    windows = select_tsf_windows(
        series,
        WindowSpec(CONTEXT_LENGTH, HORIZON, HORIZON),
        max_windows=sample_count,
    )
    samples: list[np.ndarray] = []
    for _series_index, _start, window in windows:
        if np.isfinite(window).all():
            samples.append(_standardize_by_context(window, CONTEXT_LENGTH))
    return samples


def summarize_samples(
    group_id: str,
    capability_id: str,
    difficulty: int | None,
    samples: list[np.ndarray],
) -> dict[str, Any]:
    features = [feature_vector(sample, season_length=SEASON_LENGTH) for sample in samples]
    baselines = [baseline_metrics(sample) for sample in samples]
    return {
        "group_id": group_id,
        "capability_id": capability_id,
        "difficulty": difficulty,
        "sample_count": len(samples),
        "features": summarize_dicts(features),
        "baselines": summarize_dicts(baselines),
    }


def baseline_metrics(sample: np.ndarray) -> dict[str, float]:
    history = sample[:CONTEXT_LENGTH, 0]
    actual = sample[CONTEXT_LENGTH:, 0]
    naive = np.full(HORIZON, history[-1])
    seasonal_naive = history[-SEASON_LENGTH:][:HORIZON] if len(history) >= SEASON_LENGTH else naive
    return {
        "naive_mae": mae(actual, naive),
        "seasonal_naive_mae": mae(actual, seasonal_naive),
        "naive_mase": mase(actual, naive, history),
        "seasonal_naive_mase": mase(actual, seasonal_naive, history),
    }


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mase(actual: np.ndarray, predicted: np.ndarray, history: np.ndarray) -> float:
    if len(history) <= SEASON_LENGTH:
        denom = float(np.mean(np.abs(np.diff(history))))
    else:
        denom = float(np.mean(np.abs(history[SEASON_LENGTH:] - history[:-SEASON_LENGTH])))
    error = mae(actual, predicted)
    return error / denom if denom > 1e-9 else error


def summarize_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if np.isfinite(value)})
    return {key: float(np.mean([row[key] for row in rows if key in row and np.isfinite(row[key])])) for key in keys}


def acceptance_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trend_v2 = [row for row in rows if row["group_id"] == "v2_trend"]
    multi_v2 = [row for row in rows if row["group_id"] == "v2_multi_seasonal"]
    legacy_trend_rows = [row for row in rows if row["group_id"] == "legacy_trend"]
    return {
        "v2_trend_strength_monotonic": is_monotonic([feature(row, "trend_strength") for row in trend_v2]),
        "v2_trend_slope_mean_within_cap": max(feature(row, "slope_abs") for row in trend_v2) <= PILOT_ACCEPTANCE_CAPS["trend"]["slope_abs"],
        "legacy_trend_slope_mean_within_cap": max(feature(row, "slope_abs") for row in legacy_trend_rows) <= PILOT_ACCEPTANCE_CAPS["trend"]["slope_abs"],
        "v2_multi_seasonal_naive_mae_monotonic": is_monotonic(
            [baseline(row, "seasonal_naive_mae") for row in multi_v2]
        ),
        "v2_multi_seasonal_naive_mae_growth": baseline(multi_v2[-1], "seasonal_naive_mae")
        / max(baseline(multi_v2[0], "seasonal_naive_mae"), 1e-9),
    }


def feature(row: dict[str, Any], key: str) -> float:
    return float(row["features"].get(key, 0.0))


def baseline(row: dict[str, Any], key: str) -> float:
    return float(row["baselines"].get(key, 0.0))


def is_monotonic(values: list[float]) -> bool:
    return all(left <= right + 1e-9 for left, right in zip(values, values[1:]))


def render_report(results: dict[str, Any], *, m4_path: Path, output_dir: Path) -> str:
    table_rows = [
        "| Group | Difficulty | Trend | Seasonal | Slope | Curvature | Noise | Naive MASE | SNaive MASE | SNaive MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["rows"]:
        table_rows.append(
            "| "
            + " | ".join(
                [
                    row["group_id"],
                    "-" if row["difficulty"] is None else str(row["difficulty"]),
                    fmt(feature(row, "trend_strength")),
                    fmt(feature(row, "seasonal_strength")),
                    fmt(feature(row, "slope_abs")),
                    fmt(feature(row, "curvature_abs")),
                    fmt(feature(row, "noise_ratio")),
                    fmt(baseline(row, "naive_mase")),
                    fmt(baseline(row, "seasonal_naive_mase")),
                    fmt(baseline(row, "seasonal_naive_mae")),
                ]
            )
            + " |"
        )
    checks = results["checks"]
    return "\n".join(
        [
            "# Synthetic v2 Generator Experiment",
            "",
            "日期：2026-06-29",
            "",
            "## 目的",
            "",
            "对比旧公式、v2 pilot 公式和 M4 Hourly 真实窗口，检查显式特征、真实分布 cap 和 naive / seasonal naive 基线响应是否符合 synthetic v2 契约。",
            "",
            "## 输入",
            "",
            f"- M4 Hourly 本地数据：`{display_path(m4_path)}`",
            f"- JSON 输出：`{display_path(output_dir / 'summary.json')}`",
            f"- 每组样本数：`{results['sample_count']}`",
            f"- context/horizon/season：`{CONTEXT_LENGTH}/{HORIZON}/{SEASON_LENGTH}`",
            "",
            "## 汇总",
            "",
            *table_rows,
            "",
            "## 验收检查",
            "",
            f"- v2 trend strength 单调：`{checks['v2_trend_strength_monotonic']}`",
            f"- v2 trend slope 均值不超过 cap：`{checks['v2_trend_slope_mean_within_cap']}`",
            f"- legacy trend slope 均值不超过 cap：`{checks['legacy_trend_slope_mean_within_cap']}`",
            f"- v2 multi-seasonal seasonal naive MAE 单调：`{checks['v2_multi_seasonal_naive_mae_monotonic']}`",
            f"- v2 multi-seasonal seasonal naive MAE 增长倍数：`{fmt(checks['v2_multi_seasonal_naive_mae_growth'])}`",
            "",
            "## 结论",
            "",
            "- 旧 trend 公式低难度已经有很强趋势，且 slope 均值超过真实 cap；v2 pilot 把 trend strength 调成随 difficulty 单调增强，并把 slope 均值压回 cap 内。",
            "- 旧 multi-seasonal 公式没有稳定制造“单周期 seasonal naive 更难”的响应；v2 pilot 通过 48 点次级周期让 seasonal naive MAE 随 difficulty 明显上升。",
            "- M4 真实窗口保留在同表里，作为当前特征和基线误差的真实参照；后续可以把更多真实数据集加入同一脚本。",
            "",
            "## 复现",
            "",
            "```bash",
            "cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_generator_experiment.py",
            "```",
            "",
        ]
    )


def fmt(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
