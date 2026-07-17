#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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
    CONTROL_FEATURES_BY_CAPABILITY,
    INTENSITY_FEATURE_DIRECTIONS,
    PAPER_UNIVARIATE_CAPABILITY_IDS,
    TARGET_FEATURES_BY_CAPABILITY,
    _seed_for,
)
from run_synthetic_v2_near_distance_calibration import (  # noqa: E402
    BUCKET_SPECS,
    BucketSpec,
    load_real_bucket,
)


DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "backend/app/data/synthetic_v2_feature_gate_artifact.json"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "runtime/research/synthetic-v2-feature-gate-calibration/summary.json"
DEFAULT_MAX_WINDOWS = 600
DEFAULT_CALIBRATION_FRACTION = 0.20
DEFAULT_GATE_REFERENCE_FRACTION = 0.45
DEFAULT_COVERAGE = 0.95
DEFAULT_SEED = 20260715
QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
FEATURE_GATE_BUCKET_SPECS = (
    *BUCKET_SPECS,
    BucketSpec(
        "electricity_hourly_daily_2048ctx_24h",
        "tsf_univariate",
        "electricity_hourly_dataset.zip",
        2048,
        24,
        24,
        24,
        synthetic_capabilities=PAPER_UNIVARIATE_CAPABILITY_IDS,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the real-only joint control-feature support artifact for capts-paper-v1."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument("--gate-reference-fraction", type=float, default=DEFAULT_GATE_REFERENCE_FRACTION)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--buckets", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact, summary = build_artifact(
        data_dir=args.data_dir,
        max_windows=args.max_windows,
        calibration_fraction=args.calibration_fraction,
        gate_reference_fraction=args.gate_reference_fraction,
        coverage=args.coverage,
        seed=args.seed,
        bucket_ids=tuple(args.buckets) if args.buckets else None,
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote artifact: {args.artifact}")
    print(f"wrote summary: {args.summary}")
    return 0


def build_artifact(
    *,
    data_dir: Path,
    max_windows: int,
    calibration_fraction: float,
    gate_reference_fraction: float = DEFAULT_GATE_REFERENCE_FRACTION,
    coverage: float,
    seed: int,
    bucket_ids: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    if not 0.0 < gate_reference_fraction < 0.5:
        raise ValueError("gate_reference_fraction must be between 0 and 0.5")
    if not 0.5 < coverage < 1.0:
        raise ValueError("coverage must be between 0.5 and 1.0")
    selected = [
        spec
        for spec in FEATURE_GATE_BUCKET_SPECS
        if bucket_ids is None or spec.profile_id in bucket_ids
    ]
    if not selected:
        raise ValueError("no bucket specs selected")

    created_at = datetime.now(timezone.utc).isoformat()
    online_buckets: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for spec in selected:
        rows = load_real_bucket(spec, data_dir / spec.asset_name, max_windows=max_windows)
        parameter_rows, reference, calibration, split_summary = split_real_rows_three_way(
            rows,
            spec,
            calibration_fraction=calibration_fraction,
            gate_reference_fraction=gate_reference_fraction,
            seed=_seed_for(seed, spec.profile_id, 0),
        )
        capabilities = {
            capability_id: calibrate_capability(
                capability_id,
                reference,
                calibration,
                coverage=coverage,
            )
            for capability_id in spec.synthetic_capabilities
        }
        online_buckets[spec.profile_id] = {
            "profile_id": spec.profile_id,
            "context_length": int(spec.context_length),
            "horizon": int(spec.horizon),
            "season_length": int(spec.season_length),
            "target_dim": int(spec.target_dim),
            "covariate_dim": int(spec.covariate_dim),
            "split": split_summary,
            "capabilities": capabilities,
        }
        summaries.append(
            {
                "profile_id": spec.profile_id,
                "real_window_count": len(rows),
                "generator_parameter_count": len(parameter_rows),
                "reference_count": len(reference),
                "calibration_count": len(calibration),
                "split": split_summary,
                "capabilities": {
                    capability_id: {
                        "control_features": list(config["control_support"]["feature_names"]),
                        "threshold": config["control_support"]["threshold"],
                        "calibration_acceptance_rate": config["control_support"]["calibration_acceptance_rate"],
                    }
                    for capability_id, config in capabilities.items()
                },
            }
        )

    try:
        data_dir_label = str(data_dir.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        data_dir_label = str(data_dir)
    config = {
        "data_dir": data_dir_label,
        "max_windows_per_bucket": int(max_windows),
        "calibration_fraction": float(calibration_fraction),
        "gate_reference_fraction_of_development": float(gate_reference_fraction),
        "coverage": float(coverage),
        "seed": int(seed),
        "support_method": "median_iqr_standardization + shrunk_robust_mahalanobis",
        "threshold_method": "finite_sample_split_conformal_quantile_from_real_calibration_only",
        "split_policy": "three-way series/group split; single-series temporal split with C+H embargo",
        "target_features": "diagnostic empirical quantiles only; not an online rejection condition",
    }
    artifact = {
        "schema_version": "synthetic_v2_feature_gate_online.v1",
        "created_at": created_at,
        "config": config,
        "buckets": online_buckets,
    }
    summary = {
        "schema_version": "synthetic_v2_feature_gate_calibration.v1",
        "created_at": created_at,
        "config": config,
        "buckets": summaries,
    }
    return artifact, summary


def split_real_rows(
    rows: list[dict[str, Any]],
    spec: BucketSpec,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) < 60:
        raise ValueError(f"bucket {spec.profile_id} needs at least 60 windows, got {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or "single-series")
        groups.setdefault(group_id, []).append(row)

    if len(groups) >= 2:
        group_ids = np.asarray(sorted(groups), dtype=object)
        rng = np.random.default_rng(seed)
        rng.shuffle(group_ids)
        target_calibration_count = max(20, int(math.ceil(len(rows) * calibration_fraction)))
        calibration_groups: list[str] = []
        calibration_count = 0
        for group_id in group_ids:
            remaining_group_count = len(group_ids) - len(calibration_groups) - 1
            if calibration_count >= target_calibration_count and calibration_groups:
                break
            if remaining_group_count < 1:
                break
            resolved = str(group_id)
            calibration_groups.append(resolved)
            calibration_count += len(groups[resolved])
        calibration_group_set = set(calibration_groups)
        reference = [row for group_id, group_rows in groups.items() if group_id not in calibration_group_set for row in group_rows]
        calibration = [row for group_id, group_rows in groups.items() if group_id in calibration_group_set for row in group_rows]
        policy = "group"
        embargo = 0
    else:
        ordered = sorted(rows, key=lambda row: int(row.get("window_start") or 0))
        calibration_count = max(20, int(math.ceil(len(ordered) * calibration_fraction)))
        calibration = ordered[-calibration_count:]
        first_calibration_start = int(calibration[0].get("window_start") or 0)
        embargo = int(spec.context_length + spec.horizon)
        reference = [
            row
            for row in ordered[:-calibration_count]
            if int(row.get("window_start") or 0) + embargo <= first_calibration_start
        ]
        policy = "temporal_embargo"
        calibration_group_set = set(groups)

    if len(reference) < 30 or len(calibration) < 20:
        raise ValueError(
            f"bucket {spec.profile_id} produced an undersized leakage-safe split: "
            f"reference={len(reference)}, calibration={len(calibration)}"
        )
    return reference, calibration, {
        "policy": policy,
        "group_count": len(groups),
        "calibration_group_count": len(calibration_group_set),
        "reference_count": len(reference),
        "calibration_count": len(calibration),
        "embargo_steps": embargo,
    }


def split_real_rows_three_way(
    rows: list[dict[str, Any]],
    spec: BucketSpec,
    *,
    calibration_fraction: float,
    gate_reference_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    development, gate_calibration, outer_summary = split_real_rows(
        rows,
        spec,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    generator_parameters, gate_reference, inner_summary = split_real_rows(
        development,
        spec,
        calibration_fraction=gate_reference_fraction,
        seed=_seed_for(seed, spec.profile_id, 1),
    )
    return generator_parameters, gate_reference, gate_calibration, {
        "policy": f"three_way_{outer_summary['policy']}",
        "group_count": outer_summary["group_count"],
        "generator_parameter_count": len(generator_parameters),
        "gate_reference_count": len(gate_reference),
        "gate_calibration_count": len(gate_calibration),
        "generator_gate_embargo_steps": inner_summary["embargo_steps"],
        "gate_calibration_embargo_steps": outer_summary["embargo_steps"],
        "generator_split": inner_summary,
        "gate_calibration_split": outer_summary,
    }


def calibrate_capability(
    capability_id: str,
    reference: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    *,
    coverage: float,
) -> dict[str, Any]:
    control_names = CONTROL_FEATURES_BY_CAPABILITY[capability_id]
    if control_names:
        reference_controls = feature_matrix(reference, control_names)
        calibration_controls = feature_matrix(calibration, control_names)
        center = np.median(reference_controls, axis=0)
        scale = robust_scale(reference_controls)
        reference_z = (reference_controls - center) / scale
        calibration_z = (calibration_controls - center) / scale
        robust_location = np.median(reference_z, axis=0)
        clipped_reference = np.clip(reference_z, -8.0, 8.0)
        covariance = np.atleast_2d(np.cov(clipped_reference, rowvar=False))
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            0.75 * covariance
            + 0.25 * diagonal
            + 1e-4 * np.eye(len(control_names))
        )
        precision = np.linalg.pinv(covariance)
        calibration_scores = mahalanobis_scores(
            calibration_z,
            robust_location,
            precision,
        )
        threshold = max(
            float(conformal_quantile(calibration_scores, coverage)),
            1e-6,
        )
        control_support = {
            "method": "shrunk_robust_mahalanobis",
            "feature_names": list(control_names),
            "feature_center": round_nested(center),
            "feature_scale": round_nested(scale),
            "robust_location_z": round_nested(robust_location),
            "precision": round_nested(precision),
            "threshold": round_float(threshold),
            "coverage": float(coverage),
            "reference_count": len(reference),
            "calibration_count": len(calibration),
            "calibration_acceptance_rate": round_float(
                float(np.mean(calibration_scores <= threshold))
            ),
            "marginal_quantiles": {
                name: quantile_map(
                    reference_controls[:, index],
                    levels=(0.01, 0.50, 0.99),
                )
                for index, name in enumerate(control_names)
            },
        }
    else:
        control_support = {
            "method": "not_applicable_no_independent_observable_controls",
            "feature_names": [],
            "feature_center": [],
            "feature_scale": [],
            "robust_location_z": [],
            "precision": [],
            "threshold": 0.0,
            "coverage": 1.0,
            "reference_count": len(reference),
            "calibration_count": len(calibration),
            "calibration_acceptance_rate": 1.0,
            "marginal_quantiles": {},
        }

    target_reference: dict[str, Any] = {}
    for name in TARGET_FEATURES_BY_CAPABILITY[capability_id]:
        values = finite_feature_values(reference, name)
        if values.size == 0:
            continue
        target_reference[name] = {
            "direction": INTENSITY_FEATURE_DIRECTIONS[capability_id].get(name, "increase"),
            "quantiles": quantile_map(values),
        }

    return {
        "control_support": control_support,
        "target_reference": target_reference,
    }


def feature_matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row["features"].get(name, float("nan"))) for name in names]
            for row in rows
        ],
        dtype=float,
    )
    if not np.isfinite(matrix).all():
        missing = [
            name
            for index, name in enumerate(names)
            if not np.isfinite(matrix[:, index]).all()
        ]
        raise ValueError(f"non-finite real control features: {missing}")
    return matrix


def finite_feature_values(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = np.asarray([row["features"].get(name, float("nan")) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def robust_scale(values: np.ndarray) -> np.ndarray:
    q75 = np.quantile(values, 0.75, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    scale = q75 - q25
    standard_deviation = np.std(values, axis=0)
    scale = np.where(scale > 1e-9, scale, standard_deviation)
    return np.where(scale > 1e-9, scale, 1.0)


def mahalanobis_scores(
    standardized: np.ndarray,
    location: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    delta = standardized - location
    squared = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return np.sqrt(np.maximum(squared, 0.0) / max(1, standardized.shape[1]))


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = int(math.ceil((ordered.size + 1) * coverage))
    index = min(max(rank - 1, 0), ordered.size - 1)
    return float(ordered[index])


def quantile_map(
    values: np.ndarray,
    *,
    levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> dict[str, float]:
    quantiles = np.quantile(np.asarray(values, dtype=float), levels)
    return {
        f"p{int(round(level * 100)):02d}": round_float(value)
        for level, value in zip(levels, quantiles)
    }


def round_nested(values: np.ndarray, digits: int = 8) -> list[Any]:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def round_float(value: float, digits: int = 8) -> float:
    return float(round(float(value), digits))


if __name__ == "__main__":
    raise SystemExit(main())
