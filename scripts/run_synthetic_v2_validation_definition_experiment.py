#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
    CAPABILITIES_BY_ID,
    _generate_accepted_sample_values,
    _realized_features,
    _seed_for,
)
from synthetic_feature_profile import WindowSpec, feature_vector, read_tsf_series, select_tsf_windows  # noqa: E402


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24
TARGET_DIM_MULTI = 3
DEFAULT_CAPABILITIES = (
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
UNIVARIATE_CAPABILITIES = {
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
    "covariate_response",
}
MULTIVARIATE_CAPABILITIES = {
    "common_factor",
    "hierarchical_coherence",
}
BASE_FEATURES = (
    "trend_strength",
    "seasonal_strength",
    "acf_abs_mean",
    "noise_ratio",
    "spike_rate",
)
EXTENDED_FEATURES = (
    "slope_abs",
    "curvature_abs",
    "multi_period_score",
    "seasonal_drift_score",
    "seasonal_amplitude_modulation",
    "seasonal_phase_variation",
    "change_point_shift_energy",
    "level_shift_strength",
    "volatility_shift_strength",
    "burst_rate",
    "nonlinear_lag1_gain",
    "nonlinear_multi_lag_gain",
    "heteroskedastic_strength",
    "avg_abs_target_corr",
    "pca_top1_explained",
    "pca_top2_explained",
    "effective_factor_rank",
    "lead_lag_peak_abs",
    "hierarchy_residual_mean_abs",
    "hierarchy_child_heterogeneity",
    "avg_abs_covariate_target_corr",
    "future_abs_covariate_target_corr",
    "event_lift_abs",
    "covariate_incremental_r2",
)
SELECTED_UNIVARIATE_FEATURES = (
    "trend_strength",
    "multi_period_score",
    "seasonal_amplitude_modulation",
    "change_point_shift_energy",
    "nonlinear_multi_lag_gain",
    "spike_rate",
)
SELECTED_MULTI_COV_FEATURES = (
    "pca_top1_explained",
    "effective_factor_rank",
    "covariate_incremental_r2",
    "hierarchy_child_heterogeneity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation-definition experiments for synthetic v2 paper metrics.")
    parser.add_argument("--m4-path", type=Path, default=REPO_ROOT / "runtime/research/m4_hourly_dataset.zip")
    parser.add_argument("--real-max-windows", type=int, default=2000)
    parser.add_argument("--sample-count", type=int, default=80, help="Synthetic samples per capability and intensity level.")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runtime/research/synthetic-v2-validation-definition-experiment")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "docs/superpowers/baselines/2026-07-07-synthetic-v2-validation-definition-experiment.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_windows = load_m4_windows(args.m4_path, args.real_max_windows)
    real_train, real_holdout = split_real_windows(real_windows)
    synthetic_rows = generate_synthetic_rows(args.sample_count, args.seed)

    real_feature_rows = [feature_row_from_window(window, "real_anchor", None, None, idx) for idx, window in enumerate(real_windows)]
    train_feature_rows = [feature_row_from_window(window, "real_train", None, None, idx) for idx, window in enumerate(real_train)]
    holdout_feature_rows = [feature_row_from_window(window, "real_holdout", None, None, idx) for idx, window in enumerate(real_holdout)]

    calibration = calibrate_novelty(real_train, real_holdout, synthetic_rows)
    feature_summary = summarize_groups(synthetic_rows)
    real_feature_summary = summarize_feature_dicts([row["features"] for row in real_feature_rows])
    monotonicity = monotonicity_checks(feature_summary)
    distribution_checks = controlled_distribution_checks(
        [row["features"] for row in train_feature_rows],
        synthetic_rows,
    )
    selection = feature_selection_summary(feature_summary, real_feature_summary)

    summary = {
        "schema_version": "synthetic_v2_validation_definition_experiment.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "season_length": SEASON_LENGTH,
            "real_anchor": "m4_hourly",
            "real_window_count": len(real_windows),
            "real_train_count": len(real_train),
            "real_holdout_count": len(real_holdout),
            "synthetic_sample_count_per_capability_intensity": args.sample_count,
            "seed": args.seed,
        },
        "real_feature_summary": real_feature_summary,
        "synthetic_feature_summary": feature_summary,
        "monotonicity": monotonicity,
        "novelty_calibration": calibration,
        "distribution_checks": distribution_checks,
        "feature_selection": selection,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "synthetic_features.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in synthetic_rows) + "\n",
        encoding="utf-8",
    )
    report = render_report(summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote summary: {args.output_dir / 'summary.json'}")
    print(f"wrote report: {args.report}")
    return 0


def load_m4_windows(path: Path, max_windows: int) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"M4 Hourly dataset not found: {path}")
    _metadata, series = read_tsf_series(path)
    selected = select_tsf_windows(
        series,
        WindowSpec(CONTEXT_LENGTH, HORIZON, HORIZON),
        max_windows=max_windows,
    )
    windows = [window.astype(float) for _series_index, _start, window in selected if np.isfinite(window).all()]
    if len(windows) < 100:
        raise RuntimeError(f"too few finite real windows: {len(windows)}")
    return windows


def split_real_windows(windows: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train = [window for idx, window in enumerate(windows) if idx % 5 != 4]
    holdout = [window for idx, window in enumerate(windows) if idx % 5 == 4]
    return train, holdout


def generate_synthetic_rows(sample_count: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capability_id in DEFAULT_CAPABILITIES:
        target_dim = TARGET_DIM_MULTI if capability_id in MULTIVARIATE_CAPABILITIES else 1
        for intensity in range(1, 6):
            for sample_index in range(sample_count):
                sample_seed = _seed_for(seed, capability_id, sample_index)
                target, latent, covariates, realized = _generate_accepted_sample_values(
                    capability_id,
                    CONTEXT_LENGTH + HORIZON,
                    CONTEXT_LENGTH,
                    target_dim,
                    SEASON_LENGTH,
                    intensity,
                    sample_seed,
                )
                features = extended_feature_vector(target, covariates)
                features.update({key: float(value) for key, value in realized.items() if is_finite(value)})
                rows.append(
                    {
                        "capability_id": capability_id,
                        "intensity": intensity,
                        "difficulty": intensity,
                        "sample_index": sample_index,
                        "sample_seed": sample_seed,
                        "target_dim": target_dim,
                        "covariate_dim": 0 if covariates is None else int(covariates.shape[1]),
                        "features": clean_features(features),
                        "latent_params": latent,
                        "target": target.astype(float).tolist() if capability_id in UNIVARIATE_CAPABILITIES else None,
                    }
                )
    return rows


def feature_row_from_window(window: np.ndarray, kind: str, capability: str | None, intensity: int | None, index: int) -> dict[str, Any]:
    features = extended_feature_vector(window, None)
    return {
        "kind": kind,
        "capability_id": capability,
        "intensity": intensity,
        "index": index,
        "features": clean_features(features),
    }


def extended_feature_vector(target: np.ndarray, covariates: np.ndarray | None) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    features = feature_vector(target, season_length=SEASON_LENGTH)
    features.update(_realized_features(target, covariates, SEASON_LENGTH, min(CONTEXT_LENGTH, target.shape[0])))
    mean_series = np.mean(target, axis=1)
    features.update(structural_univariate_features(mean_series))
    if target.shape[1] > 1:
        features.update(multivariate_features(target))
    if covariates is not None and np.asarray(covariates).size:
        features.update(covariate_features(target, np.asarray(covariates, dtype=float)))
    return clean_features(features)


def structural_univariate_features(values: np.ndarray) -> dict[str, float]:
    y = robust_scale_1d(values)
    n = y.size
    if n < 12:
        return {}
    min_seg = max(8, min(24, n // 8))
    level_scores = []
    volatility_scores = []
    std_all = float(np.std(y)) or 1.0
    for cut in range(min_seg, n - min_seg):
        left = y[:cut]
        right = y[cut:]
        level_scores.append(abs(float(np.mean(left) - np.mean(right))) / std_all)
        volatility_scores.append(abs(float(np.std(left) - np.std(right))) / std_all)
    diff = np.diff(y)
    roll = rolling_std(y, max(6, min(SEASON_LENGTH // 2, n // 8)))
    return {
        "level_shift_strength": float(max(level_scores)) if level_scores else 0.0,
        "volatility_shift_strength": float(max(volatility_scores)) if volatility_scores else 0.0,
        "burst_rate": float(np.mean(np.abs(y) > 3.0)),
        "heteroskedastic_strength": float(np.std(roll) / (np.mean(roll) + 1e-9)) if roll.size else 0.0,
        "diff_spike_rate": float(np.mean(np.abs(robust_scale_1d(diff)) > 3.0)) if diff.size else 0.0,
    }


def multivariate_features(values: np.ndarray) -> dict[str, float]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    corr = np.nan_to_num(np.corrcoef(centered.T), nan=0.0)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    var = singular**2
    total = float(np.sum(var))
    explained = var / total if total > 1e-12 else np.zeros_like(var)
    hierarchy_residual = values[:, 0] - np.sum(values[:, 1:], axis=1) if values.shape[1] > 2 else np.asarray([])
    return {
        "avg_abs_target_corr": float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0,
        "pca_top1_explained": float(explained[0]) if explained.size else 0.0,
        "pca_top2_explained": float(np.sum(explained[:2])) if explained.size else 0.0,
        "lead_lag_peak_abs": lead_lag_peak(values),
        "hierarchy_residual_mean_abs": float(np.mean(np.abs(hierarchy_residual))) if hierarchy_residual.size else 0.0,
    }


def covariate_features(target: np.ndarray, covariates: np.ndarray) -> dict[str, float]:
    scores = []
    for cov_idx in range(covariates.shape[1]):
        for target_idx in range(target.shape[1]):
            corr = safe_corr(covariates[:, cov_idx], target[:, target_idx])
            if is_finite(corr):
                scores.append(abs(corr))
    event_lifts = []
    for cov_idx in range(covariates.shape[1]):
        column = covariates[:, cov_idx]
        unique = np.unique(column)
        if unique.size <= 3 and np.any(column > 0):
            active = column > 0
            inactive = ~active
            if active.any() and inactive.any():
                event_lifts.append(abs(float(np.mean(target[active]) - np.mean(target[inactive]))))
    return {
        "avg_abs_covariate_target_corr": float(np.mean(scores)) if scores else 0.0,
        "event_lift_abs": float(np.mean(event_lifts)) if event_lifts else 0.0,
    }


def lead_lag_peak(values: np.ndarray, max_lag: int = 12) -> float:
    if values.shape[1] < 2:
        return 0.0
    peaks = []
    for left in range(values.shape[1]):
        for right in range(values.shape[1]):
            if left == right:
                continue
            for lag in range(1, min(max_lag, values.shape[0] // 4) + 1):
                peaks.append(abs(safe_corr(values[:-lag, left], values[lag:, right])))
    finite = [value for value in peaks if is_finite(value)]
    return float(max(finite)) if finite else 0.0


def calibrate_novelty(real_train: list[np.ndarray], real_holdout: list[np.ndarray], synthetic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    real_train_raw = np.vstack([flatten_window(window) for window in real_train])
    real_holdout_raw = np.vstack([flatten_window(window) for window in real_holdout])
    real_train_features = np.vstack([feature_array(extended_feature_vector(window, None), SELECTED_UNIVARIATE_FEATURES) for window in real_train])
    real_holdout_features = np.vstack([feature_array(extended_feature_vector(window, None), SELECTED_UNIVARIATE_FEATURES) for window in real_holdout])
    feature_center = np.nanmedian(real_train_features, axis=0)
    feature_scale = robust_feature_scale(real_train_features)
    real_train_features_z = (real_train_features - feature_center) / feature_scale
    real_holdout_features_z = (real_holdout_features - feature_center) / feature_scale

    holdout_raw = nearest_distances(real_holdout_raw, real_train_raw)
    holdout_feat = nearest_distances(real_holdout_features_z, real_train_features_z)
    raw_q05 = float(np.quantile(holdout_raw["d1"], 0.05))
    feat_q05 = float(np.quantile(holdout_feat["d1"], 0.05))

    synth_univariate = [row for row in synthetic_rows if row["target"] is not None]
    synth_raw = np.vstack([flatten_window(np.asarray(row["target"], dtype=float)) for row in synth_univariate])
    synth_feat = np.vstack([feature_array(row["features"], SELECTED_UNIVARIATE_FEATURES) for row in synth_univariate])
    synth_feat_z = (synth_feat - feature_center) / feature_scale
    synth_raw_nn = nearest_distances(synth_raw, real_train_raw)
    synth_feat_nn = nearest_distances(synth_feat_z, real_train_features_z)

    by_capability: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(synth_univariate):
        by_capability[row["capability_id"]].append(idx)

    capability_rows = {}
    for capability, indexes in sorted(by_capability.items()):
        raw_ratios = synth_raw_nn["d1"][indexes] / max(raw_q05, 1e-9)
        feat_ratios = synth_feat_nn["d1"][indexes] / max(feat_q05, 1e-9)
        nndr = synth_raw_nn["ratio"][indexes]
        capability_rows[capability] = {
            "raw_novelty_ratio_q05": float(np.quantile(raw_ratios, 0.05)),
            "raw_novelty_ratio_median": float(np.median(raw_ratios)),
            "feature_novelty_ratio_q05": float(np.quantile(feat_ratios, 0.05)),
            "feature_novelty_ratio_median": float(np.median(feat_ratios)),
            "near_duplicate_rate_raw": float(np.mean(raw_ratios < 1.0)),
            "near_duplicate_rate_feature": float(np.mean(feat_ratios < 1.0)),
            "nndr_q05": float(np.quantile(nndr, 0.05)),
            "nndr_median": float(np.median(nndr)),
        }

    return {
        "real_holdout_baseline": {
            "raw_dcr_q05": raw_q05,
            "raw_dcr_median": float(np.median(holdout_raw["d1"])),
            "feature_dcr_q05": feat_q05,
            "feature_dcr_median": float(np.median(holdout_feat["d1"])),
        },
        "synthetic_vs_real_train": capability_rows,
    }


def controlled_distribution_checks(real_features: list[dict[str, float]], synthetic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = np.vstack([feature_array(row, SELECTED_UNIVARIATE_FEATURES) for row in real_features])
    real_a, real_b = real[::2], real[1::2]
    reference = {
        "mmd_real_vs_real": rbf_mmd(real_a, real_b),
        "swd_real_vs_real": sliced_wasserstein(real_a, real_b),
    }
    by_capability = {}
    for capability in sorted({row["capability_id"] for row in synthetic_rows if row["target"] is not None}):
        rows = [row for row in synthetic_rows if row["capability_id"] == capability and row["target"] is not None]
        synth = np.vstack([feature_array(row["features"], SELECTED_UNIVARIATE_FEATURES) for row in rows])
        by_capability[capability] = {
            "mmd_vs_real": rbf_mmd(real, synth),
            "swd_vs_real": sliced_wasserstein(real, synth),
        }
    return {"reference": reference, "by_capability": by_capability}


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["capability_id"], int(row.get("intensity", row.get("difficulty"))))].append(row)
    summaries = []
    for (capability, intensity), group in sorted(grouped.items()):
        summaries.append(
            {
                "capability_id": capability,
                "intensity": intensity,
                "difficulty": intensity,
                "sample_count": len(group),
                "target_dim": max(int(row["target_dim"]) for row in group),
                "covariate_dim": max(int(row["covariate_dim"]) for row in group),
                "features": summarize_feature_dicts([row["features"] for row in group]),
                "acceptance_rate": float(
                    np.mean([
                        1.0 if (row.get("latent_params", {}).get("acceptance", {}).get("accepted", True)) else 0.0
                        for row in group
                    ])
                ),
            }
        )
    return summaries


def summarize_feature_dicts(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key, value in row.items() if is_finite(value)})
    out = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row and is_finite(row[key])], dtype=float)
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return out


def monotonicity_checks(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_cap = defaultdict(list)
    for row in summaries:
        by_cap[row["capability_id"]].append(row)
    checks = {}
    for capability, rows in by_cap.items():
        rows = sorted(rows, key=lambda item: item["intensity"])
        feature_checks = {}
        keys = sorted({key for row in rows for key in row["features"]})
        for key in keys:
            values = [row["features"][key]["mean"] for row in rows if key in row["features"]]
            if len(values) == 5:
                feature_checks[key] = {
                    "values": [float(value) for value in values],
                    "spearman_with_intensity": spearman(values),
                    "nondecreasing": bool(all(values[idx] <= values[idx + 1] + 1e-9 for idx in range(len(values) - 1))),
                    "nonincreasing": bool(all(values[idx] >= values[idx + 1] - 1e-9 for idx in range(len(values) - 1))),
                }
        checks[capability] = feature_checks
    return checks


def feature_selection_summary(summaries: list[dict[str, Any]], real_summary: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = []
    for feature in [*SELECTED_UNIVARIATE_FEATURES, *SELECTED_MULTI_COV_FEATURES]:
        real_stats = real_summary.get(feature, {})
        evidence = []
        for capability in sorted({row["capability_id"] for row in summaries}):
            values = [
                row["features"][feature]["mean"]
                for row in summaries
                if row["capability_id"] == capability and feature in row["features"]
            ]
            if len(values) == 5:
                rho = spearman(values)
                if abs(rho) >= 0.7:
                    evidence.append({"capability_id": capability, "spearman": rho, "d1": values[0], "d5": values[-1]})
        rows.append({"feature": feature, "real_anchor": real_stats, "monotonic_evidence": evidence[:8]})
    return {
        "selected_univariate_features": list(SELECTED_UNIVARIATE_FEATURES),
        "selected_multi_covariate_features": list(SELECTED_MULTI_COV_FEATURES),
        "feature_evidence": rows,
    }


def render_report(summary: dict[str, Any]) -> str:
    config = summary["config"]
    lines = [
        "# Synthetic v2 Validation Definition Experiment",
        "",
        f"日期：{datetime.now().date().isoformat()}",
        "",
        "## Purpose",
        "",
        "为论文中的真实分布抽取、特征维度和生成后检验规则提供可复现实验证据；本实验不包含 discriminative score 或 predictive score。",
        "",
        "## Config",
        "",
        f"- Anchor: M4 Hourly, windows={config['real_window_count']} train={config['real_train_count']} holdout={config['real_holdout_count']}",
        f"- Window: context={config['context_length']}, horizon={config['horizon']}, season={config['season_length']}",
        f"- Synthetic: {config['synthetic_sample_count_per_capability_intensity']} samples per capability/intensity",
        "",
        "## Real Anchor Feature Quantiles",
        "",
        "| Feature | p05 | p50 | p95 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for feature in SELECTED_UNIVARIATE_FEATURES:
        stats = summary["real_feature_summary"].get(feature, {})
        lines.append(f"| `{feature}` | {fmt(stats.get('p05'))} | {fmt(stats.get('p50'))} | {fmt(stats.get('p95'))} |")
    lines.extend(["", "## Synthetic Intensity Evidence", "", "| Capability | Feature | i1 | i3 | i5 | Spearman |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for capability, feature in (
        ("trend", "trend_strength"),
        ("multi_seasonal", "multi_period_score"),
        ("time_varying_seasonality", "seasonal_amplitude_modulation"),
        ("regime_switching", "change_point_shift_energy"),
        ("nonlinear_persistence", "nonlinear_multi_lag_gain"),
        ("predictable_intermittency", "spike_rate"),
        ("common_factor", "pca_top1_explained"),
        ("hierarchical_coherence", "hierarchy_child_heterogeneity"),
        ("covariate_response", "covariate_incremental_r2"),
    ):
        check = summary["monotonicity"].get(capability, {}).get(feature)
        if not check:
            continue
        vals = check["values"]
        lines.append(f"| `{capability}` | `{feature}` | {fmt(vals[0])} | {fmt(vals[2])} | {fmt(vals[4])} | {fmt(check['spearman_with_intensity'])} |")
    lines.extend(["", "## Novelty Calibration", ""])
    baseline = summary["novelty_calibration"]["real_holdout_baseline"]
    lines.extend(
        [
            f"- real holdout raw DCR q05/median: `{fmt(baseline['raw_dcr_q05'])}` / `{fmt(baseline['raw_dcr_median'])}`",
            f"- real holdout feature DCR q05/median: `{fmt(baseline['feature_dcr_q05'])}` / `{fmt(baseline['feature_dcr_median'])}`",
            "",
            "| Capability | raw novelty q05 | feature novelty q05 | raw near-dup | feature near-dup | NNDR q05 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for capability, stats in summary["novelty_calibration"]["synthetic_vs_real_train"].items():
        lines.append(
            f"| `{capability}` | {fmt(stats['raw_novelty_ratio_q05'])} | {fmt(stats['feature_novelty_ratio_q05'])} | "
            f"{fmt(stats['near_duplicate_rate_raw'])} | {fmt(stats['near_duplicate_rate_feature'])} | {fmt(stats['nndr_q05'])} |"
        )
    lines.extend(["", "## Controlled Distribution Distances", ""])
    ref = summary["distribution_checks"]["reference"]
    lines.extend(
        [
            f"- real-vs-real MMD / SWD reference: `{fmt(ref['mmd_real_vs_real'])}` / `{fmt(ref['swd_real_vs_real'])}`",
            "",
            "| Capability | MMD vs real | SWD vs real |",
            "| --- | ---: | ---: |",
        ]
    )
    for capability, stats in summary["distribution_checks"]["by_capability"].items():
        lines.append(f"| `{capability}` | {fmt(stats['mmd_vs_real'])} | {fmt(stats['swd_vs_real'])} |")
    lines.extend(["", "## Recommended Feature Set", ""])
    lines.append("- Univariate core scalars: " + ", ".join(f"`{feature}`" for feature in SELECTED_UNIVARIATE_FEATURES))
    lines.append("- Multi/covariate core scalars: " + ", ".join(f"`{feature}`" for feature in SELECTED_MULTI_COV_FEATURES))
    lines.append("- Multi/covariate secondary diagnostics: `lead_lag_peak_abs`, `avg_abs_covariate_target_corr`, `event_lift_abs`.")
    lines.append("")
    lines.append("Full JSON summary: `runtime/research/synthetic-v2-validation-definition-experiment/summary.json`.")
    return "\n".join(lines) + "\n"


def flatten_window(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] != 1:
        arr = arr[:, :1]
    return robust_scale_1d(arr[:, 0])


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    d1 = np.empty(query.shape[0], dtype=float)
    d2 = np.empty(query.shape[0], dtype=float)
    for start in range(0, query.shape[0], 256):
        block = query[start : start + 256]
        distances = np.mean(np.abs(block[:, None, :] - reference[None, :, :]), axis=2)
        part = np.partition(distances, kth=min(1, distances.shape[1] - 1), axis=1)
        d1[start : start + block.shape[0]] = part[:, 0]
        d2[start : start + block.shape[0]] = part[:, 1] if distances.shape[1] > 1 else part[:, 0]
    return {"d1": d1, "d2": d2, "ratio": d1 / np.maximum(d2, 1e-9)}


def feature_array(features: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) if is_finite(features.get(name, 0.0)) else 0.0 for name in names], dtype=float)


def robust_feature_scale(values: np.ndarray) -> np.ndarray:
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    scale = q75 - q25
    std = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-9, scale, std)
    return np.where(scale > 1e-9, scale, 1.0)


def rbf_mmd(left: np.ndarray, right: np.ndarray) -> float:
    left, right = standardize_pair(left, right)
    combined = np.vstack([left, right])
    subset = combined[:: max(1, combined.shape[0] // 600)]
    pdists = np.sqrt(np.sum((subset[:, None, :] - subset[None, :, :]) ** 2, axis=2))
    positive = pdists[pdists > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(median, 1e-6) ** 2)
    k_xx = np.exp(-gamma * np.sum((left[:, None, :] - left[None, :, :]) ** 2, axis=2)).mean()
    k_yy = np.exp(-gamma * np.sum((right[:, None, :] - right[None, :, :]) ** 2, axis=2)).mean()
    k_xy = np.exp(-gamma * np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def sliced_wasserstein(left: np.ndarray, right: np.ndarray, projections: int = 64) -> float:
    left, right = standardize_pair(left, right)
    rng = np.random.default_rng(17)
    n = min(left.shape[0], right.shape[0], 1000)
    left = left[rng.choice(left.shape[0], size=n, replace=False)]
    right = right[rng.choice(right.shape[0], size=n, replace=False)]
    dirs = rng.normal(size=(projections, left.shape[1]))
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    distances = []
    for direction in dirs:
        distances.append(float(np.mean(np.abs(np.sort(left @ direction) - np.sort(right @ direction)))))
    return float(np.mean(distances))


def standardize_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([left, right])
    center = np.nanmedian(combined, axis=0)
    scale = robust_feature_scale(combined)
    return (left - center) / scale, (right - center) / scale


def robust_scale_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-9:
        return (values - median) / iqr
    std = float(np.std(values))
    if std > 1e-9:
        return (values - median) / std
    return values - median


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return np.asarray([], dtype=float)
    return np.asarray([float(np.std(values[idx : idx + window])) for idx in range(values.size - window + 1)], dtype=float)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size != right.size or left.size < 3:
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def spearman(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ranks = np.argsort(np.argsort(np.asarray(values, dtype=float))).astype(float)
    x = np.arange(len(values), dtype=float)
    return safe_corr(x, ranks)


def clean_features(features: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in features.items() if is_finite(value)}


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value: Any) -> str:
    if not is_finite(value):
        return "-"
    return f"{float(value):.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
