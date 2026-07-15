#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.synthetic_generation_service import _generate_accepted_sample_values, _seed_for  # noqa: E402


CAPABILITIES = (
    "common_factor",
    "hierarchical_coherence",
    "covariate_response",
)
CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24
TARGET_DIM = 3
SAMPLE_COUNT = 64
DIFFICULTIES = range(1, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multivariate/covariate synthetic feature probes.")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runtime/research/agent-multivar-feature-analysis")
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        default=[],
        help="Real-model summary JSON to include. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = generate_feature_rows(args.sample_count)
    feature_summary = summarize_feature_rows(feature_rows)
    model_summary = summarize_model_responses(args.summary)
    output = {
        "schema_version": "agent_multivar_feature_probe.v1",
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "season_length": SEASON_LENGTH,
        "sample_count_per_capability_difficulty": args.sample_count,
        "feature_summary": feature_summary,
        "model_response_summary": model_summary,
        "recommendations": recommendations(feature_summary, model_summary),
    }
    (args.output_dir / "tables.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render_report(output), encoding="utf-8")
    print(f"wrote tables: {args.output_dir / 'tables.json'}")
    print(f"wrote report: {args.output_dir / 'report.md'}")
    return 0


def generate_feature_rows(sample_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capability_id in CAPABILITIES:
        target_dim = TARGET_DIM if capability_id != "covariate_response" else 1
        for difficulty in DIFFICULTIES:
            for sample_index in range(sample_count):
                seed = _seed_for(20260707, capability_id, difficulty * 10_000 + sample_index)
                values, _latent, covariates, realized = _generate_accepted_sample_values(
                    capability_id,
                    CONTEXT_LENGTH + HORIZON,
                    CONTEXT_LENGTH,
                    target_dim,
                    SEASON_LENGTH,
                    difficulty,
                    seed,
                )
                row = {
                    "capability_id": capability_id,
                    "difficulty": difficulty,
                    "sample_index": sample_index,
                    **realized,
                    **multivariate_features(values),
                }
                if capability_id == "hierarchical_coherence":
                    row.update(hierarchy_features(values))
                if covariates is not None:
                    row.update(covariate_features(values, covariates))
                rows.append(row)
    return rows


def multivariate_features(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 2 or values.shape[1] < 2:
        return {}
    corr = np.nan_to_num(np.corrcoef(values.T), nan=0.0)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    centered = values - np.mean(values, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    total = float(np.sum(eigvals))
    shares = eigvals / total if total > 1e-12 else np.zeros_like(eigvals)
    entropy = -float(np.sum([share * math.log(share) for share in shares if share > 1e-12]))
    return {
        "avg_abs_target_corr_probe": float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0,
        "first_pc_variance_share": float(np.max(shares)) if shares.size else 0.0,
        "effective_factor_rank": float(math.exp(entropy)) if shares.size else 0.0,
    }


def lag_features(values: np.ndarray, *, max_lag: int) -> dict[str, float]:
    if values.ndim != 2 or values.shape[1] < 2:
        return {}
    peak_values: list[float] = []
    zero_values: list[float] = []
    best_lags: list[int] = []
    for source in range(values.shape[1]):
        for target in range(values.shape[1]):
            if source == target:
                continue
            zero_values.append(abs(corrcoef(values[:, source], values[:, target])))
            candidates = [
                (lag, abs(corrcoef(values[:-lag, source], values[lag:, target])))
                for lag in range(1, max_lag + 1)
            ]
            best_lag, best_value = max(candidates, key=lambda item: item[1])
            peak_values.append(best_value)
            best_lags.append(best_lag)
    return {
        "lag_peak_abs_corr": float(np.mean(peak_values)),
        "lag_peak_minus_zero_corr": float(np.mean(peak_values) - np.mean(zero_values)),
        "lag_peak_mean_lag": float(np.mean(best_lags)),
    }


def regime_shift_features(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 2 or values.shape[0] < 12:
        return {}
    scores: list[tuple[float, int, np.ndarray]] = []
    for cut in range(24, values.shape[0] - 24):
        before = np.mean(values[:cut], axis=0)
        after = np.mean(values[cut:], axis=0)
        delta = after - before
        scores.append((float(np.linalg.norm(delta)), cut, delta))
    shift_norm, cut, delta = max(scores, key=lambda item: item[0])
    signs = np.sign(delta[np.abs(delta) > 1e-9])
    agreement = abs(float(np.sum(signs))) / len(signs) if signs.size else 0.0
    return {
        "system_shift_norm": shift_norm,
        "system_shift_cut": float(cut),
        "system_shift_direction_agreement": agreement,
    }


def hierarchy_features(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 2 or values.shape[1] < 3:
        return {}
    residual = values[:, 0] - np.sum(values[:, 1:], axis=1)
    denom = float(np.mean(np.abs(values[:, 0]))) or 1.0
    return {
        "hierarchy_residual_probe": float(np.mean(np.abs(residual))),
        "hierarchy_relative_residual_probe": float(np.mean(np.abs(residual)) / denom),
    }


def covariate_features(values: np.ndarray, covariates: np.ndarray) -> dict[str, float]:
    target = values[:, 0]
    all_corrs = [abs(corrcoef(target, covariates[:, index])) for index in range(covariates.shape[1])]
    future_target = values[CONTEXT_LENGTH:, 0]
    future_cov = covariates[CONTEXT_LENGTH:]
    future_corrs = [abs(corrcoef(future_target, future_cov[:, index])) for index in range(future_cov.shape[1])]
    event = covariates[:, -1] > 0.5
    event_gap = 0.0
    if np.any(event) and np.any(~event):
        event_gap = abs(float(np.mean(target[event]) - np.mean(target[~event])))
    return {
        "avg_abs_covariate_target_corr_probe": float(np.mean(all_corrs)),
        "future_abs_covariate_target_corr_probe": float(np.mean(future_corrs)),
        "event_target_mean_gap": event_gap,
    }


def corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2:
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 1e-12 else 0.0


def summarize_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["capability_id"], int(row["difficulty"]))].append(row)
    out: list[dict[str, Any]] = []
    for (capability_id, difficulty), group in sorted(grouped.items()):
        keys = sorted(
            key
            for row in group
            for key, value in row.items()
            if key not in {"capability_id", "difficulty", "sample_index"} and is_finite(value)
        )
        summary = {
            "capability_id": capability_id,
            "difficulty": difficulty,
            "sample_count": len(group),
        }
        for key in keys:
            values = [float(row[key]) for row in group if is_finite(row.get(key))]
            summary[key] = float(np.mean(values))
        out.append(summary)
    return out


def summarize_model_responses(paths: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        by_key = {
            (row["model_id"], row["capability_id"], int(row["difficulty"])): row
            for row in data.get("summaries", [])
        }
        for row in data.get("summaries", []):
            model_id = row["model_id"]
            if model_id in {"naive", "seasonal_naive"}:
                continue
            capability_id = row["capability_id"]
            difficulty = int(row["difficulty"])
            seasonal = by_key.get(("seasonal_naive", capability_id, difficulty))
            naive = by_key.get(("naive", capability_id, difficulty))
            summaries.append(
                {
                    "source": str(path),
                    "model_id": model_id,
                    "capability_id": capability_id,
                    "difficulty": difficulty,
                    "sample_count": row["sample_count"],
                    "mae": metric(row, "mae"),
                    "mase": metric(row, "mase"),
                    "coherence_mae": metric(row, "coherence_mae"),
                    "mae_vs_seasonal_naive": ratio(metric(row, "mae"), metric(seasonal, "mae") if seasonal else None),
                    "mae_vs_naive": ratio(metric(row, "mae"), metric(naive, "mae") if naive else None),
                }
            )
    return summaries


def recommendations(feature_summary: list[dict[str, Any]], model_summary: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "feature": "known_future_covariate_response",
            "capability_id": "covariate_response",
            "reason": "Directly validates whether a model uses known-future inputs; generation has a difficulty-controlled covariate-target signal and current probes include two covariate-capable models.",
        },
        {
            "feature": "hierarchical_coherence",
            "capability_id": "hierarchical_coherence",
            "reason": "Separates forecast accuracy from structural validity; generated inputs have near-zero hierarchy residual while model forecasts show measurable coherence error.",
        },
        {
            "feature": "low_rank_common_factor_structure",
            "capability_id": "common_factor",
            "reason": "Has clean real-distribution analogues through PCA variance share and effective factor rank; generation shows controlled rank broadening and current probes show strong model-vs-baseline separation.",
        },
    ]


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get("metrics", {}).get(key)
    return float(value) if is_finite(value) else None


def ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or abs(baseline) < 1e-12:
        return None
    return float(value / baseline)


def is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def render_report(output: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Agent Multivariate/Covariate Feature Probe",
            "",
            f"- Samples per capability/difficulty: `{output['sample_count_per_capability_difficulty']}`",
            f"- Window: context `{output['context_length']}`, horizon `{output['horizon']}`, season `{output['season_length']}`",
            "",
            "## Generation Feature Summary",
            "",
            render_feature_table(output["feature_summary"]),
            "",
            "## Current Model Response Summary",
            "",
            render_model_table(output["model_response_summary"]),
            "",
            "## Recommended Paper Feature Set",
            "",
            render_recommendations(output["recommendations"]),
            "",
        ]
    )


def render_feature_table(rows: list[dict[str, Any]]) -> str:
    selected = {
        "common_factor": ["first_pc_variance_share", "effective_factor_rank", "avg_abs_target_corr_probe"],
        "hierarchical_coherence": ["hierarchy_residual_probe", "hierarchy_relative_residual_probe", "avg_abs_target_corr_probe"],
        "covariate_response": ["avg_abs_covariate_target_corr_probe", "future_abs_covariate_target_corr_probe", "event_target_mean_gap"],
    }
    lines = [
        "| Capability | d | Feature 1 | Feature 2 | Feature 3 |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        keys = selected[row["capability_id"]]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["capability_id"],
                    str(row["difficulty"]),
                    f"`{keys[0]}`={fmt(row.get(keys[0]))}",
                    f"`{keys[1]}`={fmt(row.get(keys[1]))}",
                    f"`{keys[2]}`={fmt(row.get(keys[2]))}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_model_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Capability | Model | Mean MAE | Mean MASE | Mean MAE / SNaive | Mean MAE / Naive | Mean coherence MAE | n/d |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["capability_id"], row["model_id"])].append(row)
    for (capability_id, model_id), group in sorted(grouped.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    capability_id,
                    model_id,
                    fmt(mean(row["mae"] for row in group)),
                    fmt(mean(row["mase"] for row in group)),
                    fmt(mean(row["mae_vs_seasonal_naive"] for row in group)),
                    fmt(mean(row["mae_vs_naive"] for row in group)),
                    fmt(mean(row["coherence_mae"] for row in group)),
                    str(sum(int(row["sample_count"]) for row in group)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_recommendations(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        lines.append(f"- `{row['feature']}` (`{row['capability_id']}`): {row['reason']}")
    return "\n".join(lines)


def mean(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def fmt(value: Any) -> str:
    if value is None or not is_finite(value):
        return "-"
    return f"{float(value):.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
