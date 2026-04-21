from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .anchor import load_anchor_artifacts
from .baselines import baseline_by_name
from .calibration import FamilyCalibration, calibrate_family
from .constants import FEATURE_COLUMNS
from .domain import BenchmarkV1Config, SeriesMeta, SeriesSample, SeriesSpec
from .families import FAMILY_GENERATORS
from .features import extract_features, feature_vector
from .metrics import mase
from .utils import adjacent_meta_path, seeded_rng, write_json, write_parquet
from .validation import validate_benchmark


def _standardize(features: dict[str, float], means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (feature_vector(features) - means) / stds


def _sample_anchor_target(meta: dict, rng: np.random.Generator) -> tuple[int, dict[str, float], int, int]:
    weights = np.asarray(meta["cluster_weights"], dtype=float)
    cluster_id = int(rng.choice(len(weights), p=weights / weights.sum()))
    prototype = meta["prototypes"][cluster_id]
    target_features = {name: float(prototype[name]) for name in FEATURE_COLUMNS}
    return cluster_id, target_features, int(prototype["season_length"]), int(prototype["dominant_scale"])


def _baseline_type(features: dict[str, float]) -> str:
    if features["seasonal_strength"] >= 0.45:
        return "seasonal_naive"
    if features["trend_strength"] < 0.1 and features["spectral_entropy"] > 0.92:
        return "last_value"
    return "auto_theta"


def _objective_for_family(family: str) -> str:
    return {
        "trend": "trend_strength",
        "multi_seasonal": "seasonal_strength",
        "regime_switching": "changepoint_density",
        "long_memory_nonlinear": "acf_half_life",
        "intermittent_heteroskedastic": "intermittency",
    }.get(family, "seasonal_strength")


def _within_tolerance(features: dict[str, float], target_features: dict[str, float], means: np.ndarray, stds: np.ndarray, tolerance: float, objective: str) -> bool:
    standardized = _standardize(features, means, stds)
    target = _standardize(target_features, means, stds)
    deltas = np.abs(standardized - target)
    objective_index = FEATURE_COLUMNS.index(objective) if objective in FEATURE_COLUMNS else None
    non_target = [delta for idx, delta in enumerate(deltas) if idx != objective_index]
    if np.median(non_target) > 0.3:
        return False
    return float(np.sqrt(np.mean((standardized - target) ** 2))) <= tolerance


def _distance_to_target(features: dict[str, float], target_features: dict[str, float], means: np.ndarray, stds: np.ndarray) -> float:
    standardized = _standardize(features, means, stds)
    target = _standardize(target_features, means, stds)
    return float(np.sqrt(np.mean((standardized - target) ** 2)))


def _generate_candidate(
    family: str,
    difficulty_bin: int,
    horizon_ratio: float,
    seed: int,
    meta: dict,
    calibration: FamilyCalibration | None = None,
) -> SeriesSample:
    rng = seeded_rng(seed)
    cluster_id, target_features, season_length, dominant_scale = _sample_anchor_target(meta, rng)
    horizon = int(np.clip(round(horizon_ratio * dominant_scale), 12, 96))
    context = int(min(8 * horizon, 512))
    burn_in = int(max(4 * dominant_scale, 200))
    total_length = burn_in + context + horizon
    if calibration is None:
        control_lambda = float(rng.uniform(0.0, 1.0))
    else:
        target_bin = max(1, min(5, difficulty_bin))
        control_lambda = float(rng.uniform((target_bin - 1) / 5, target_bin / 5))
    output = FAMILY_GENERATORS[family](
        length=total_length,
        season_length=max(1, season_length),
        control_lambda=control_lambda,
        rng=rng,
        anchor_features=target_features,
    )
    series = output.values[-(context + horizon) :]
    realized_features = extract_features(output.values)
    spec = SeriesSpec(
        track="diagnostic",
        family=family,
        difficulty=difficulty_bin,
        horizon_ratio=horizon_ratio,
        seed=seed,
        anchor_cluster_id=cluster_id,
    )
    return SeriesSample(
        id=f"{spec.track}:{family}:{difficulty_bin}:{horizon_ratio:.2f}:{seed}",
        context=series[:context].tolist(),
        target=series[context:].tolist(),
        horizon=horizon,
        spec=spec,
        meta=SeriesMeta(
            latent_params={**output.latent_params, "control_lambda": control_lambda},
            realized_features=realized_features,
            season_length=max(1, season_length),
            dominant_scale=dominant_scale,
            baseline_type=_baseline_type(realized_features),
        ),
    )


def _compute_baseline_mase(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        history = np.asarray(row.context, dtype=float)
        target = np.asarray(row.target, dtype=float)
        predictor = baseline_by_name(str(row.baseline_type))
        forecast = predictor.predict(history, len(target), int(row.season_length))
        rows.append({"id": row.id, "baseline_mase": mase(history, target, forecast, int(row.season_length))})
    return pd.DataFrame(rows)


def build_benchmark_artifacts(
    anchor_stats_path: Path,
    output_path: Path,
    anchor_track_size: int,
    diagnostic_per_cell: int,
    seed: int,
    version: str | None = None,
) -> Path:
    _, anchor_meta = load_anchor_artifacts(anchor_stats_path)
    meta = {
        "feature_columns": anchor_meta.feature_columns,
        "feature_means": anchor_meta.feature_means,
        "feature_stds": anchor_meta.feature_stds,
        "covariance": anchor_meta.covariance,
        "prototypes": anchor_meta.prototypes,
        "cluster_weights": anchor_meta.cluster_weights,
        "anchor_mode": anchor_meta.anchor_mode,
    }
    means = np.asarray(meta["feature_means"], dtype=float)
    stds = np.asarray(meta["feature_stds"], dtype=float)
    config = BenchmarkV1Config(artifact_root=output_path.parent, anchor_track_size=anchor_track_size, diagnostic_per_cell=diagnostic_per_cell, random_seed=seed)
    calibration_candidates = int(np.clip(max(80, diagnostic_per_cell * 15), 80, config.calibration_candidates_per_family))
    calibrations: dict[str, FamilyCalibration] = {}
    calibration_rows = []
    for family in config.diagnostic_families:
        calibration, frame = calibrate_family(
            family=family,
            n_candidates=calibration_candidates,
            horizon_ratio=0.5,
            seed=config.random_seed + len(calibrations) * 17,
        )
        calibrations[family] = calibration
        calibration_rows.append(frame)
    records = []
    rng = seeded_rng(config.random_seed)
    target_families = config.diagnostic_families
    for idx in range(config.anchor_track_size):
        family = target_families[idx % len(target_families)]
        sample = None
        best_distance = float("inf")
        best_candidate = None
        for attempt in range(24):
            candidate = _generate_candidate(
                family=family,
                difficulty_bin=3,
                horizon_ratio=float(rng.choice(config.horizon_ratios)),
                seed=config.random_seed + idx * 101 + attempt,
                meta=meta,
                calibration=None,
            )
            target_features = {name: meta["prototypes"][candidate.spec.anchor_cluster_id][name] for name in FEATURE_COLUMNS}
            distance = _distance_to_target(candidate.meta.realized_features, target_features, means, stds)
            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate
            if _within_tolerance(candidate.meta.realized_features, target_features, means, stds, config.anchor_tolerance_sigma, _objective_for_family(family)):
                candidate.spec.track = "anchor"
                candidate.id = f"anchor:{idx:05d}"
                sample = candidate
                break
        if sample is None:
            sample = best_candidate
            if sample is None:
                raise RuntimeError(f"failed to generate anchor sample {idx}")
            sample.spec.track = "anchor"
            sample.id = f"anchor:{idx:05d}"
        records.append(sample.to_record())
    for family in config.diagnostic_families:
        objective = _objective_for_family(family)
        calibration = calibrations[family]
        for difficulty in config.difficulty_levels:
            for horizon_ratio in config.horizon_ratios:
                for replica in range(config.diagnostic_per_cell):
                    sample = None
                    best_distance = float("inf")
                    best_candidate = None
                    for attempt in range(28):
                        seed_value = config.random_seed + difficulty * 100_000 + replica * 101 + attempt
                        candidate = _generate_candidate(
                            family=family,
                            difficulty_bin=difficulty,
                            horizon_ratio=horizon_ratio,
                            seed=seed_value,
                            meta=meta,
                            calibration=calibration,
                        )
                        target_features = {name: meta["prototypes"][candidate.spec.anchor_cluster_id][name] for name in FEATURE_COLUMNS}
                        distance = _distance_to_target(candidate.meta.realized_features, target_features, means, stds)
                        if distance < best_distance:
                            best_distance = distance
                            best_candidate = candidate
                        if _within_tolerance(candidate.meta.realized_features, target_features, means, stds, config.diagnostic_tolerance_sigma, objective):
                            candidate.spec.track = "diagnostic"
                            candidate.id = f"diagnostic:{family}:{difficulty}:{horizon_ratio:.2f}:{replica:03d}"
                            sample = candidate
                            break
                    if sample is None:
                        sample = best_candidate
                        if sample is None:
                            raise RuntimeError(f"failed to generate diagnostic sample for {family}")
                        sample.spec.track = "diagnostic"
                        sample.id = f"diagnostic:{family}:{difficulty}:{horizon_ratio:.2f}:{replica:03d}"
                    records.append(sample.to_record())
    benchmark_version = version or f"v1-s{seed}"
    frame = pd.DataFrame(records)
    frame["benchmark_version"] = benchmark_version
    frame = frame.merge(_compute_baseline_mase(frame), on="id", how="left")
    write_parquet(frame, output_path)
    calibration_path = output_path.with_name(f"{output_path.stem}_calibration.parquet")
    write_parquet(pd.concat(calibration_rows, ignore_index=True), calibration_path)
    meta_payload = {
        "benchmark_version": benchmark_version,
        "benchmark_seed": seed,
        "anchor_mode": anchor_meta.anchor_mode,
        "known_limitations": [
            "Anchor Track still borrows diagnostic generators in this v1 implementation.",
            "Gaussian copula anchor prior is not implemented yet.",
            "Difficulty and realism constraints are heuristic and should not be treated as publication-grade evidence.",
        ],
        "config": {
            "anchor_track_size": config.anchor_track_size,
            "diagnostic_per_cell": config.diagnostic_per_cell,
            "diagnostic_families": config.diagnostic_families,
            "difficulty_levels": config.difficulty_levels,
            "horizon_ratios": config.horizon_ratios,
        },
        "anchor_meta": meta,
        "calibration_path": str(calibration_path),
        "baseline_cached": True,
    }
    write_json(meta_payload, adjacent_meta_path(output_path))
    meta_payload["validation_summary"] = validate_benchmark(output_path)
    write_json(meta_payload, adjacent_meta_path(output_path))
    return output_path
