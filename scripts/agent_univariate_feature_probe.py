#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.synthetic_generation_service import (  # noqa: E402
    PILOT_ACCEPTANCE_CAPS,
    _generate_accepted_sample_values,
    _seed_for,
    _standardize_by_context,
)
from synthetic_feature_profile import (  # noqa: E402
    WindowSpec,
    feature_vector,
    read_tsf_series,
    robust_scale,
    select_tsf_windows,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/research/agent-univariate-feature-probe"
DEFAULT_M4_PATH = REPO_ROOT / "runtime/research/m4_hourly_dataset.zip"
DEFAULT_US_BIRTHS_PATH = REPO_ROOT / "runtime/research/us_births_dataset.zip"
DEFAULT_M4_PROFILE = REPO_ROOT / "runtime/research/m4_hourly_168ctx_profile.json"
DEFAULT_US_BIRTHS_PROFILE = REPO_ROOT / "runtime/research/us_births_weekly_profile_v2.json"

CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "long_memory_nonlinear",
    "intermittent_heteroskedastic",
)
CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24
DIFFICULTIES = (1, 2, 3, 4, 5)

CURRENT_FEATURES = (
    "trend_strength",
    "seasonal_strength",
    "slope_abs",
    "curvature_abs",
    "noise_ratio",
    "acf1",
    "acf_abs_mean",
    "outlier_rate",
    "spike_rate",
)
DIAGNOSTIC_FEATURES = (
    "multi_period_score",
    "spectral_entropy",
    "seasonal_drift_score",
    "seasonal_amplitude_cv",
    "regime_shift_score",
    "acf_long_mean",
    "nonlinear_lag1_gain",
    "volatility_cv",
)
PRIMARY_CHECKS = {
    "trend": ("trend_strength", "slope_abs", "curvature_abs"),
    "multi_seasonal": ("multi_period_score", "seasonal_strength"),
    "time_varying_seasonality": ("seasonal_drift_score", "seasonal_amplitude_cv"),
    "regime_switching": ("regime_shift_score",),
    "long_memory_nonlinear": ("acf_abs_mean", "acf_long_mean", "nonlinear_lag1_gain"),
    "intermittent_heteroskedastic": ("spike_rate", "outlier_rate", "volatility_cv", "noise_ratio"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a univariate synthetic feature calibration probe without model scoring."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--m4-path", type=Path, default=DEFAULT_M4_PATH)
    parser.add_argument("--us-births-path", type=Path, default=DEFAULT_US_BIRTHS_PATH)
    parser.add_argument("--real-max-windows", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, sample_rows = run_probe(
        sample_count=args.sample_count,
        seed=args.seed,
        m4_path=args.m4_path,
        us_births_path=args.us_births_path,
        real_max_windows=args.real_max_windows,
    )
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "sample_features.jsonl", sample_rows)
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"wrote summary: {display_path(args.output_dir / 'summary.json')}")
    print(f"wrote samples: {display_path(args.output_dir / 'sample_features.jsonl')}")
    print(f"wrote report: {display_path(args.output_dir / 'report.md')}")
    return 0


def run_probe(
    *,
    sample_count: int,
    seed: int,
    m4_path: Path,
    us_births_path: Path,
    real_max_windows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    for capability_id in CAPABILITIES:
        for difficulty in DIFFICULTIES:
            for sample_index in range(sample_count):
                sample_seed = _seed_for(seed, capability_id, difficulty * 10_000 + sample_index)
                values, latent_params, _covariates, realized = _generate_accepted_sample_values(
                    capability_id,
                    CONTEXT_LENGTH + HORIZON,
                    CONTEXT_LENGTH,
                    1,
                    SEASON_LENGTH,
                    difficulty,
                    sample_seed,
                )
                diagnostics = diagnostic_features(values[:, 0], season_length=SEASON_LENGTH)
                acceptance = latent_params.get("acceptance", {})
                sample_rows.append(
                    {
                        "capability_id": capability_id,
                        "difficulty": difficulty,
                        "sample_index": sample_index,
                        "accepted": bool(acceptance.get("accepted", True)),
                        "attempts": int(acceptance.get("attempts", 1)),
                        "failed_features": list(acceptance.get("failed_features") or []),
                        "current_features": select_features(realized, CURRENT_FEATURES),
                        "diagnostic_features": select_features(diagnostics, DIAGNOSTIC_FEATURES),
                        "generator_latent": compact_latent_params(latent_params),
                    }
                )

    generated_summary = summarize_generated(sample_rows)
    monotonicity = monotonicity_checks(generated_summary["by_capability_difficulty"])
    real_anchors = load_real_anchors(
        m4_path=m4_path,
        us_births_path=us_births_path,
        real_max_windows=real_max_windows,
    )
    summary = {
        "schema_version": "agent_univariate_feature_probe.v1",
        "config": {
            "capabilities": list(CAPABILITIES),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "season_length": SEASON_LENGTH,
            "difficulty": list(DIFFICULTIES),
            "sample_count_per_cell": sample_count,
            "seed": seed,
            "model_scoring": "not_run",
        },
        "current_acceptance_caps": PILOT_ACCEPTANCE_CAPS,
        "generated": generated_summary,
        "monotonicity": monotonicity,
        "real_anchors": real_anchors,
        "feature_recommendation": recommended_feature_dimensions(),
        "adequacy_notes": adequacy_notes(monotonicity),
    }
    return summary, sample_rows


def compact_latent_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key == "acceptance":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list):
            out[key] = value[:8]
    return out


def diagnostic_features(values: np.ndarray, *, season_length: int) -> dict[str, float]:
    y = robust_scale(np.asarray(values, dtype=float))
    return {
        "multi_period_score": multi_period_score(y, season_length),
        "spectral_entropy": spectral_entropy(y),
        "seasonal_drift_score": seasonal_drift_score(y, season_length),
        "seasonal_amplitude_cv": seasonal_amplitude_cv(y, season_length),
        "regime_shift_score": regime_shift_score(y, season_length),
        "acf_long_mean": acf_long_mean(y, season_length),
        "nonlinear_lag1_gain": nonlinear_lag1_gain(y),
        "volatility_cv": volatility_cv(y, season_length),
    }


def multi_period_score(values: np.ndarray, season_length: int) -> float:
    centered = values - float(np.mean(values))
    if centered.size < 8 or np.allclose(centered, 0.0):
        return 0.0
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0)
    if spectrum.size <= 2:
        return 0.0
    spectrum[0] = 0.0
    primary_idx = nearest_frequency_index(freqs, 1.0 / max(2, season_length))
    primary_power = float(spectrum[primary_idx]) if primary_idx < spectrum.size else 0.0
    periods = [max(2, season_length // 2), season_length * 2, season_length * 3]
    secondary = 0.0
    for period in periods:
        idx = nearest_frequency_index(freqs, 1.0 / period)
        if idx != primary_idx and idx < spectrum.size:
            secondary += float(spectrum[idx])
    total = float(np.sum(spectrum))
    if total <= 1e-12:
        return 0.0
    if primary_power <= 1e-12:
        return secondary / total
    return float(secondary / (secondary + primary_power))


def nearest_frequency_index(freqs: np.ndarray, frequency: float) -> int:
    return int(np.argmin(np.abs(freqs - frequency)))


def spectral_entropy(values: np.ndarray) -> float:
    centered = values - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size <= 2:
        return 0.0
    power = power[1:]
    total = float(np.sum(power))
    if total <= 1e-12:
        return 0.0
    probs = power / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy / math.log(float(probs.size)) if probs.size > 1 else 0.0


def seasonal_drift_score(values: np.ndarray, season_length: int) -> float:
    if values.size < season_length * 4:
        return 0.0
    trend = polynomial_fit(values)
    detrended = values - trend
    midpoint = values.size // 2
    first = seasonal_profile(detrended[:midpoint], season_length)
    second = seasonal_profile(detrended[midpoint:], season_length)
    denom = rms(first) + rms(second) + 1e-12
    return float(rms(second - first) / denom)


def seasonal_amplitude_cv(values: np.ndarray, season_length: int) -> float:
    if values.size < season_length * 3:
        return 0.0
    amplitudes: list[float] = []
    for start in range(0, values.size - season_length + 1, season_length):
        segment = values[start : start + season_length]
        amplitudes.append(float(np.percentile(segment, 90) - np.percentile(segment, 10)))
    if not amplitudes:
        return 0.0
    mean = float(np.mean(amplitudes))
    return float(np.std(amplitudes) / mean) if mean > 1e-12 else 0.0


def regime_shift_score(values: np.ndarray, season_length: int) -> float:
    if values.size < 12:
        return 0.0
    trend = polynomial_fit(values)
    seasonal = repeat_profile(seasonal_profile(values - trend, season_length), values.size)
    residual = values - trend - seasonal
    scale = float(np.std(residual))
    if scale <= 1e-12:
        scale = float(np.std(values)) or 1.0
    margin = max(8, min(season_length, values.size // 4))
    scores: list[float] = []
    for cut in range(margin, values.size - margin):
        left = residual[:cut]
        right = residual[cut:]
        pooled = math.sqrt(float(np.var(left)) + float(np.var(right)) + 1e-12)
        denom = max(scale, pooled / math.sqrt(2.0), 1e-12)
        scores.append(abs(float(np.mean(right) - np.mean(left))) / denom)
    return float(max(scores)) if scores else 0.0


def acf_long_mean(values: np.ndarray, season_length: int) -> float:
    max_lag = min(values.size // 2, max(season_length * 2, 2))
    min_lag = min(max(2, season_length), max_lag)
    vals = [abs(autocorrelation(values, lag)) for lag in range(min_lag, max_lag + 1)]
    return float(np.mean(vals)) if vals else 0.0


def nonlinear_lag1_gain(values: np.ndarray) -> float:
    if values.size < 12:
        return 0.0
    x = values[:-1]
    y = values[1:]
    linear = np.column_stack([np.ones_like(x), x])
    nonlinear = np.column_stack([np.ones_like(x), x, np.sin(x), x * x])
    return max(0.0, r2(y, nonlinear) - r2(y, linear))


def volatility_cv(values: np.ndarray, season_length: int) -> float:
    if values.size < 8:
        return 0.0
    diffs = np.diff(values)
    window = max(4, min(season_length, diffs.size // 3 if diffs.size >= 12 else diffs.size))
    if window <= 1:
        return 0.0
    vols = []
    for start in range(0, diffs.size - window + 1, max(1, window // 2)):
        vols.append(float(np.std(diffs[start : start + window])))
    mean = float(np.mean(vols)) if vols else 0.0
    return float(np.std(vols) / mean) if mean > 1e-12 else 0.0


def polynomial_fit(values: np.ndarray) -> np.ndarray:
    if values.size < 4:
        return np.full_like(values, float(np.mean(values)))
    t = np.linspace(-1.0, 1.0, values.size)
    coeffs = np.polyfit(t, values, min(2, values.size - 1))
    return np.polyval(coeffs, t)


def seasonal_profile(values: np.ndarray, season_length: int) -> np.ndarray:
    period = max(2, int(season_length))
    profile = np.zeros(period, dtype=float)
    for phase in range(period):
        phase_values = values[np.arange(values.size) % period == phase]
        profile[phase] = float(np.mean(phase_values)) if phase_values.size else 0.0
    return profile - float(np.mean(profile))


def repeat_profile(profile: np.ndarray, length: int) -> np.ndarray:
    if profile.size == 0:
        return np.zeros(length)
    reps = int(math.ceil(length / profile.size))
    return np.tile(profile, reps)[:length]


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or values.size <= lag:
        return 0.0
    a = values[:-lag] - float(np.mean(values[:-lag]))
    b = values[lag:] - float(np.mean(values[lag:]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def r2(y: np.ndarray, design: np.ndarray) -> float:
    try:
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    fitted = design @ coeffs
    total = float(np.sum((y - float(np.mean(y))) ** 2))
    if total <= 1e-12:
        return 0.0
    residual = float(np.sum((y - fitted) ** 2))
    return max(0.0, min(1.0, 1.0 - residual / total))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def summarize_generated(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: list[dict[str, Any]] = []
    for capability_id in CAPABILITIES:
        for difficulty in DIFFICULTIES:
            rows = [
                row
                for row in sample_rows
                if row["capability_id"] == capability_id and row["difficulty"] == difficulty
            ]
            feature_rows = [flatten_features(row) for row in rows]
            failed = Counter(feature for row in rows for feature in row["failed_features"])
            by_cell.append(
                {
                    "capability_id": capability_id,
                    "difficulty": difficulty,
                    "sample_count": len(rows),
                    "acceptance_pass_rate": mean([1.0 if row["accepted"] else 0.0 for row in rows]),
                    "attempts_mean": mean([float(row["attempts"]) for row in rows]),
                    "attempts_max": max([row["attempts"] for row in rows], default=0),
                    "failed_features": dict(sorted(failed.items())),
                    "features": summarize_feature_rows(feature_rows),
                    "generator_latent": summarize_feature_rows([numeric_latent(row) for row in rows]),
                }
            )
    return {
        "by_capability_difficulty": by_cell,
        "sample_count": len(sample_rows),
    }


def flatten_features(row: dict[str, Any]) -> dict[str, float]:
    return {
        **{f"current.{key}": value for key, value in row["current_features"].items()},
        **{f"diagnostic.{key}": value for key, value in row["diagnostic_features"].items()},
    }


def numeric_latent(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in row["generator_latent"].items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def summarize_feature_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key, value in row.items() if math.isfinite(value)})
    return {key: summarize_values([row[key] for row in rows if key in row and math.isfinite(row[key])]) for key in keys}


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def monotonicity_checks(by_cell: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for capability_id, features in PRIMARY_CHECKS.items():
        cells = sorted(
            [cell for cell in by_cell if cell["capability_id"] == capability_id],
            key=lambda item: item["difficulty"],
        )
        for feature in features:
            key = feature_key(feature)
            means = [float(cell["features"].get(key, {}).get("mean", float("nan"))) for cell in cells]
            finite = [value for value in means if math.isfinite(value)]
            out.append(
                {
                    "capability_id": capability_id,
                    "feature": feature,
                    "feature_key": key,
                    "means_by_difficulty": means,
                    "d5_over_d1": finite[-1] / finite[0] if len(finite) >= 2 and abs(finite[0]) > 1e-12 else None,
                    "delta_d5_d1": finite[-1] - finite[0] if len(finite) >= 2 else None,
                    "adjacent_non_decreasing": adjacent_non_decreasing(finite),
                    "spearman_difficulty": spearman_against_difficulty(finite),
                    "directional_pass": directional_pass(finite),
                }
            )
    return out


def feature_key(feature: str) -> str:
    if feature in CURRENT_FEATURES:
        return f"current.{feature}"
    return f"diagnostic.{feature}"


def adjacent_non_decreasing(values: list[float]) -> bool:
    return len(values) >= 2 and all(left <= right + 1e-9 for left, right in zip(values, values[1:]))


def spearman_against_difficulty(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    ranks = rank_values(values)
    x = np.arange(1, len(values) + 1, dtype=float)
    return pearson(x, np.asarray(ranks, dtype=float))


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = (idx + end + 1) / 2.0
        for pos in range(idx, end):
            ranks[order[pos]] = rank
        idx = end
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size != right.size or left.size < 2:
        return None
    a = left - float(np.mean(left))
    b = right - float(np.mean(right))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else None


def directional_pass(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    rho = spearman_against_difficulty(values)
    return bool(values[-1] > values[0] and rho is not None and rho >= 0.7)


def load_real_anchors(*, m4_path: Path, us_births_path: Path, real_max_windows: int) -> dict[str, Any]:
    anchors: dict[str, Any] = {
        "profile_files": {},
        "diagnostic_feature_runs": {},
    }
    for name, path in (("m4_hourly_168ctx", DEFAULT_M4_PROFILE), ("us_births_weekly", DEFAULT_US_BIRTHS_PROFILE)):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            anchors["profile_files"][name] = {
                "path": display_path(path),
                "window_count": data.get("window_count"),
                "features": {
                    feature: {
                        "mean": stats.get("mean"),
                        "p95": stats.get("p95"),
                        "max": stats.get("max"),
                    }
                    for feature, stats in data.get("features", {}).items()
                    if feature in CURRENT_FEATURES
                },
            }
    if m4_path.exists():
        anchors["diagnostic_feature_runs"]["m4_hourly_168ctx"] = summarize_real_tsf(
            m4_path,
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            season_length=SEASON_LENGTH,
            max_windows=real_max_windows,
        )
    if us_births_path.exists():
        anchors["diagnostic_feature_runs"]["us_births_weekly"] = summarize_real_tsf(
            us_births_path,
            context_length=365,
            horizon=30,
            season_length=7,
            max_windows=min(real_max_windows, 500),
        )
    return anchors


def summarize_real_tsf(
    path: Path,
    *,
    context_length: int,
    horizon: int,
    season_length: int,
    max_windows: int,
) -> dict[str, Any]:
    _metadata, series = read_tsf_series(path)
    windows = select_tsf_windows(
        series,
        WindowSpec(context_length=context_length, horizon=horizon, stride=horizon),
        max_windows=max_windows,
    )
    rows: list[dict[str, float]] = []
    for _series_index, _start, window in windows:
        if not np.isfinite(window).all():
            continue
        values = _standardize_by_context(window, context_length)[:, 0]
        current = feature_vector(values, season_length=season_length)
        diagnostics = diagnostic_features(values, season_length=season_length)
        rows.append(
            {
                **{f"current.{key}": value for key, value in current.items() if key in CURRENT_FEATURES},
                **{f"diagnostic.{key}": value for key, value in diagnostics.items() if key in DIAGNOSTIC_FEATURES},
            }
        )
    return {
        "path": display_path(path),
        "context_length": context_length,
        "horizon": horizon,
        "season_length": season_length,
        "window_count": len(rows),
        "features": summarize_feature_rows(rows),
    }


def recommended_feature_dimensions() -> list[dict[str, Any]]:
    return [
        {
            "dimension": "trend_shape",
            "primary_features": ["trend_strength"],
            "calibration_features": ["slope_abs", "curvature_abs"],
            "reason": "Captures deterministic trend contribution while slope and curvature keep generated trend shapes inside real-profile ranges.",
        },
        {
            "dimension": "seasonal_structure",
            "primary_features": ["multi_period_score", "seasonal_drift_score"],
            "calibration_features": ["seasonal_strength", "seasonal_amplitude_cv"],
            "reason": "Separates multiple periods and non-stationary seasonal profiles; seasonal_strength is useful as a sanity check but collapses these cases.",
        },
        {
            "dimension": "structural_breaks",
            "primary_features": ["change_point_count_or_shift_energy"],
            "calibration_features": ["regime_shift_score", "generator switch_count"],
            "reason": "The generator control is monotone, but the current largest-split diagnostic is not; paper use needs a real-data change-point count or cumulative shift-energy extractor.",
        },
        {
            "dimension": "persistence",
            "primary_features": ["nonlinear_lag1_gain"],
            "calibration_features": ["acf_abs_mean", "acf_long_mean"],
            "reason": "The current capability behaves more like nonlinear persistence than strict long memory; autocorrelation summaries are useful context but were not monotone in this run.",
        },
        {
            "dimension": "intermittency_and_volatility",
            "primary_features": ["spike_rate", "outlier_rate", "noise_ratio"],
            "calibration_features": ["volatility_cv"],
            "reason": "Covers sparse bursts, heavy tails, and residual irregularity without using model errors; the tested volatility_cv diagnostic was not monotone after robust scaling.",
        },
    ]


def adequacy_notes(monotonicity: list[dict[str, Any]]) -> list[str]:
    passes = {(row["capability_id"], row["feature"]): row["directional_pass"] for row in monotonicity}
    notes = [
        "Current extraction is adequate for trend caps because trend_strength, slope_abs, and curvature_abs are explicit and already validated against M4 Hourly.",
        "Current extraction is only partially adequate for multi_seasonal because seasonal_strength is high but does not measure secondary periods; multi_period_score is needed.",
        "Current extraction is not adequate for time_varying_seasonality because no current feature measures seasonal profile drift.",
        "Current extraction is not adequate for regime_switching because no current feature measures structural break magnitude.",
        "Current extraction is not adequate for long_memory_nonlinear as named; acf summaries were not monotone, while nonlinear_lag1_gain was.",
        "Current extraction is partially adequate for intermittent_heteroskedastic through spike_rate/outlier_rate/noise_ratio; the tested volatility_cv should not be used as-is.",
    ]
    weak = [
        f"{capability_id}:{feature}"
        for (capability_id, feature), did_pass in sorted(passes.items())
        if not did_pass
    ]
    if weak:
        notes.append("Feature-direction checks that did not pass the rho>=0.7 and d5>d1 rule: " + ", ".join(weak))
    return notes


def select_features(features: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    return {name: float(features[name]) for name in names if name in features and math.isfinite(features[name])}


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Agent Univariate Feature Probe",
        "",
        "## Config",
        "",
        f"- sample_count_per_cell: `{summary['config']['sample_count_per_cell']}`",
        f"- context/horizon/season: `{CONTEXT_LENGTH}/{HORIZON}/{SEASON_LENGTH}`",
        "- model scoring: `not_run`",
        "",
        "## Acceptance Pass Rate",
        "",
        "| Capability | d1 | d2 | d3 | d4 | d5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    cells = summary["generated"]["by_capability_difficulty"]
    for capability_id in CAPABILITIES:
        row_cells = [cell for cell in cells if cell["capability_id"] == capability_id]
        lines.append(
            "| "
            + " | ".join(
                [capability_id]
                + [fmt(cell["acceptance_pass_rate"]) for cell in sorted(row_cells, key=lambda item: item["difficulty"])]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Generator Control Means",
            "",
            "| Capability | Control | d1 | d2 | d3 | d4 | d5 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    control_map = {
        "trend": ("slope_abs_mean", "curvature_abs_mean"),
        "multi_seasonal": ("secondary_amplitude_ratio",),
        "time_varying_seasonality": ("amplitude_delta_mean", "phase_drift_cycles"),
        "regime_switching": ("switch_count", "forecast_switch"),
        "long_memory_nonlinear": ("ar_phi", "nonlinear_strength"),
        "intermittent_heteroskedastic": ("event_probability", "burst_count"),
    }
    for capability_id, controls in control_map.items():
        capability_cells = sorted(
            [cell for cell in cells if cell["capability_id"] == capability_id],
            key=lambda item: item["difficulty"],
        )
        for control in controls:
            values = [
                cell.get("generator_latent", {}).get(control, {}).get("mean")
                for cell in capability_cells
            ]
            lines.append(
                "| "
                + " | ".join([capability_id, control, *[fmt(value) for value in values]])
                + " |"
            )
    lines.extend(
        [
            "",
            "## Directional Feature Checks",
            "",
            "| Capability | Feature | d1 | d2 | d3 | d4 | d5 | rho | pass |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["monotonicity"]:
        means = row["means_by_difficulty"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["capability_id"],
                    row["feature"],
                    *[fmt(value) for value in means],
                    fmt(row["spearman_difficulty"]),
                    str(row["directional_pass"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Recommended Feature Dimensions", ""])
    for item in summary["feature_recommendation"]:
        features = ", ".join(item["primary_features"])
        lines.append(f"- `{item['dimension']}`: {features}. {item['reason']}")
    lines.extend(["", "## Adequacy Notes", ""])
    lines.extend(f"- {note}" for note in summary["adequacy_notes"])
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
