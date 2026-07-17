#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_VERSION = "v1"
EXPERIMENT_ID = "E3_model_capability_profiles"
SCHEMA_VERSION = "paper_e3_model_capability_profiles.v1"
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "runtime/paper_exp" / EXPERIMENT_VERSION / "E2_dynamic_stability"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp" / EXPERIMENT_VERSION / EXPERIMENT_ID
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-17-paper-e3-model-capability-profiling-protocol.md"
)
EXPECTED_E2_MANIFEST_SHA256 = (
    "5e91a4a4dadba842939754c8ad3e2efa22c8af3e247bf169c94ef0afbf27cfe0"
)
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2026071703
SEASONAL_BASELINE = "seasonal_naive"

MODEL_ORDER = (
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "timesfm2.5",
    "tirex2",
)
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
UNIVARIATE_CAPABILITIES = CAPABILITY_ORDER[:6]
STRUCTURED_CAPABILITIES = CAPABILITY_ORDER[6:]
INTENSITIES = (1, 2, 3, 4, 5)
INTENSITY_AXIS = np.linspace(0.0, 1.0, len(INTENSITIES))
TASK_PROTOCOL_BY_CAPABILITY = {
    **{capability: "univariate" for capability in UNIVARIATE_CAPABILITIES},
    "common_factor": "multi_target",
    "hierarchical_coherence": "multi_target",
    "covariate_response": "known_future_covariates",
}
CAPABILITY_LABELS = {
    "trend": "Trend",
    "multi_seasonal": "Multi-seasonal",
    "time_varying_seasonality": "Time-varying seasonality",
    "regime_switching": "Regime switching",
    "nonlinear_persistence": "Nonlinear persistence",
    "predictable_intermittency": "Predictable intermittency",
    "common_factor": "Common factor",
    "hierarchical_coherence": "Hierarchical coherence",
    "covariate_response": "Covariate response",
}
MODEL_COLORS = {
    "Timer-3.5": "#0072B2",
    "Timer-3.0": "#E69F00",
    "Chronos-2": "#56B4E9",
    "moirai2": "#009E73",
    "toto2.0": "#7A5195",
    "timesfm2.5": "#CC79A7",
    "tirex2": "#D55E00",
}


@dataclass(frozen=True)
class SampleInfo:
    sample_id: str
    profile_id: str
    capability_id: str
    intensity: int
    round_index: int
    sample_index: int
    context_length: int
    horizon: int
    target_dim: int
    target_abs_sum: float
    future_point_count: int
    parent_abs_sum: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-v1 E3 model capability profiles from sealed E2 output."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument(
        "--allow-existing-empty",
        action="store_true",
        help="Only intended for isolated tests; completed output is always immutable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    verify_e2_source(source_dir)
    prepare_output_dir(output_dir, allow_existing_empty=args.allow_existing_empty)

    source_config = read_json(source_dir / "config.json")
    validate_source_config(source_config)
    config = experiment_config(
        source_dir=source_dir,
        source_config=source_config,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    write_json(output_dir / "config.json", config)

    samples = load_sample_info(source_dir / "samples.jsonl")
    observations = load_prediction_observations(
        source_dir,
        samples=samples,
        foundation_models=tuple(source_config["requested_models"]),
    )
    validate_observation_design(observations, samples=samples, config=source_config)

    cell_scores = profile_intensity_score_frame(observations)
    intensity_curves = intensity_curve_frame(cell_scores)
    bucket_scores = bucket_score_frame(cell_scores)
    capability_profiles = capability_profile_frame(intensity_curves, bucket_scores)

    bootstrap = capability_bootstrap_results(
        observations,
        cell_scores=cell_scores,
        replicates=args.bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    intensity_curves = attach_intensity_bootstrap_ci(intensity_curves, bootstrap)
    capability_profiles = attach_capability_bootstrap_ci(capability_profiles, bootstrap)
    capability_profiles = add_capability_ranks(capability_profiles)
    model_summary = model_summary_frame(capability_profiles, bootstrap)

    outputs = {
        "profile_intensity_scores.csv": cell_scores,
        "intensity_curves.csv": intensity_curves,
        "bucket_scores.csv": bucket_scores,
        "capability_profiles.csv": capability_profiles,
        "model_summary.csv": model_summary,
    }
    for filename, frame in outputs.items():
        write_dataframe(output_dir / filename, frame)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    figure_files = render_figures(
        figures_dir,
        intensity_curves=intensity_curves,
        capability_profiles=capability_profiles,
        model_summary=model_summary,
    )

    summary = summarize_results(
        config=config,
        capability_profiles=capability_profiles,
        model_summary=model_summary,
        cell_scores=cell_scores,
        table_rows={name: len(frame) for name, frame in outputs.items()},
        figure_files=figure_files,
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "paper_tables.md").write_text(
        render_paper_tables(capability_profiles, model_summary), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    write_manifest(output_dir, source_dir=source_dir)

    print(
        f"E3 profiles: {len(capability_profiles)} model-capability rows; "
        f"figures={len(figure_files)}",
        flush=True,
    )
    print(f"E3 output: {output_dir}", flush=True)
    return 0


def experiment_config(
    *,
    source_dir: Path,
    source_config: dict[str, Any],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment": str(source_dir.relative_to(REPO_ROOT)),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "canonical_scale_id": source_config["canonical_scale_id"],
        "canonical_scale_fingerprint": source_config["canonical_scale_fingerprint"],
        "foundation_models": list(source_config["requested_models"]),
        "seasonal_baseline": SEASONAL_BASELINE,
        "capabilities": list(CAPABILITY_ORDER),
        "univariate_capabilities": list(UNIVARIATE_CAPABILITIES),
        "structured_reporting": {
            "common_factor": "multi_target",
            "hierarchical_coherence": "multi_target",
            "covariate_response": "known_future_covariates",
            "combined_global_score": False,
        },
        "intensities": list(INTENSITIES),
        "aggregation": {
            "within_profile_intensity": "equal sample weight across 5 rounds x 32 samples",
            "profiles": "equal macro weight",
            "intensities": "equal macro weight",
            "mase_auc": "trapezoidal integral on x=(intensity-1)/4",
            "worst_level": "maximum observed profile-macro MASE among five levels",
            "relative_skill": (
                "1-model_mean_mase/seasonal_naive_mean_mase within each "
                "profile-intensity, then equal macro average"
            ),
            "nmae_abs": "sum_abs_error/sum_abs_future_target within cell, then macro average",
            "cross_bucket_variance": "sample variance ddof=1 of bucket five-level MASE",
        },
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "round_cluster_resampling": True,
            "sample_index_resampling_within_round": True,
            "paired_across_models_intensities_and_baseline": True,
            "profiles_fixed_not_resampled": True,
        },
        "primary_metric": "seasonal_mase",
        "secondary_metric": "nmae_abs",
        "ranking_direction": "lower_is_better",
    }


def validate_source_config(config: dict[str, Any]) -> None:
    if tuple(config.get("requested_models", ())) != MODEL_ORDER:
        raise ValueError("E2 requested model order does not match frozen E3 model set")
    if tuple(config.get("intensities", ())) != INTENSITIES:
        raise ValueError("E2 intensity levels do not match E3 protocol")
    if int(config.get("expected_generated_sample_count", 0)) != 18_400:
        raise ValueError("E2 sample count is not the frozen 18,400 design")
    if len(config.get("round_seeds", ())) != 5:
        raise ValueError("E2 must contain five rounds")
    if int(config.get("samples_per_round_per_cell", 0)) != 32:
        raise ValueError("E2 must contain 32 samples per round and cell")


def verify_e2_source(source_dir: Path) -> None:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing sealed E2 manifest: {manifest_path}")
    observed_manifest_hash = sha256_file(manifest_path)
    if source_dir == DEFAULT_SOURCE_DIR.resolve() and (
        observed_manifest_hash != EXPECTED_E2_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "formal E2 manifest hash changed: "
            f"{observed_manifest_hash} != {EXPECTED_E2_MANIFEST_SHA256}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("experiment_id") != "E2_dynamic_stability":
        raise ValueError("source manifest is not E2_dynamic_stability")
    for relative, expected in manifest.get("files", {}).items():
        path = source_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"E2 manifest file is missing: {relative}")
        observed_size = path.stat().st_size
        if observed_size != int(expected["bytes"]):
            raise RuntimeError(
                f"E2 size mismatch for {relative}: {observed_size} != {expected['bytes']}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != expected["sha256"]:
            raise RuntimeError(f"E2 SHA-256 mismatch for {relative}")


def prepare_output_dir(path: Path, *, allow_existing_empty: bool) -> None:
    if path.exists():
        entries = list(path.iterdir()) if path.is_dir() else [path]
        if entries or not allow_existing_empty:
            raise FileExistsError(f"E3 output is immutable and already exists: {path}")
    else:
        path.mkdir(parents=True)


def load_sample_info(path: Path) -> dict[str, SampleInfo]:
    samples: dict[str, SampleInfo] = {}
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in samples:
            raise ValueError(f"duplicate E2 sample id: {sample_id}")
        target = np.asarray(row["target"], dtype=float)
        context = int(row["context_length"])
        horizon = int(row["horizon"])
        target_dim = int(row["target_dim"])
        if target.shape != (context + horizon, target_dim):
            raise ValueError(f"invalid target shape for {sample_id}: {target.shape}")
        future = target[context:]
        target_abs_sum = float(np.sum(np.abs(future)))
        if not math.isfinite(target_abs_sum) or target_abs_sum <= 0:
            raise ValueError(f"NMAE denominator is non-positive for {sample_id}")
        parent_abs_sum = (
            float(np.sum(np.abs(future[:, 0])))
            if row["capability_id"] == "hierarchical_coherence"
            else None
        )
        if parent_abs_sum is not None and parent_abs_sum <= 0:
            raise ValueError(f"hierarchy parent denominator is non-positive for {sample_id}")
        samples[sample_id] = SampleInfo(
            sample_id=sample_id,
            profile_id=str(row["profile_id"]),
            capability_id=str(row["capability_id"]),
            intensity=int(row["intensity"]),
            round_index=int(row["round_index"]),
            sample_index=int(row["sample_index"]),
            context_length=context,
            horizon=horizon,
            target_dim=target_dim,
            target_abs_sum=target_abs_sum,
            future_point_count=int(future.size),
            parent_abs_sum=parent_abs_sum,
        )
    if not samples:
        raise ValueError("E2 samples.jsonl is empty")
    return samples


def load_prediction_observations(
    source_dir: Path,
    *,
    samples: dict[str, SampleInfo],
    foundation_models: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in (*foundation_models, SEASONAL_BASELINE):
        path = source_dir / "predictions" / f"{safe_filename(model_id)}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing E2 predictions for {model_id}: {path}")
        seen: set[str] = set()
        for prediction in iter_jsonl(path):
            sample_id = str(prediction["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate prediction for {model_id}/{sample_id}")
            seen.add(sample_id)
            if sample_id not in samples:
                raise ValueError(f"prediction references unknown sample: {sample_id}")
            sample = samples[sample_id]
            validate_prediction_identity(prediction, sample)
            metrics = prediction.get("metrics", {})
            mase = finite_float(metrics.get("mase"), f"{model_id}/{sample_id}/mase")
            mae = finite_float(metrics.get("mae"), f"{model_id}/{sample_id}/mae")
            coherence_mae: float | None = None
            if sample.capability_id == "hierarchical_coherence":
                coherence_mae = finite_float(
                    metrics.get("coherence_mae"),
                    f"{model_id}/{sample_id}/coherence_mae",
                )
            rows.append(
                {
                    "model_id": model_id,
                    "model_group": str(prediction["model_group"]),
                    "sample_id": sample_id,
                    "profile_id": sample.profile_id,
                    "capability_id": sample.capability_id,
                    "intensity": sample.intensity,
                    "round_index": sample.round_index,
                    "sample_index": sample.sample_index,
                    "mase": mase,
                    "mae": mae,
                    "abs_error_sum": mae * sample.future_point_count,
                    "target_abs_sum": sample.target_abs_sum,
                    "future_point_count": sample.future_point_count,
                    "coherence_abs_sum": (
                        coherence_mae * sample.horizon
                        if coherence_mae is not None
                        else np.nan
                    ),
                    "coherence_point_count": (
                        sample.horizon if coherence_mae is not None else np.nan
                    ),
                    "parent_abs_sum": (
                        sample.parent_abs_sum
                        if sample.parent_abs_sum is not None
                        else np.nan
                    ),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("no E2 prediction observations were loaded")
    return frame


def validate_prediction_identity(prediction: dict[str, Any], sample: SampleInfo) -> None:
    expected = {
        "profile_id": sample.profile_id,
        "capability_id": sample.capability_id,
        "intensity": sample.intensity,
        "round_index": sample.round_index,
        "sample_index": sample.sample_index,
    }
    for name, value in expected.items():
        if prediction.get(name) != value:
            raise ValueError(
                f"prediction/sample mismatch for {sample.sample_id}: "
                f"{name}={prediction.get(name)!r} != {value!r}"
            )


def validate_observation_design(
    observations: pd.DataFrame,
    *,
    samples: dict[str, SampleInfo],
    config: dict[str, Any],
) -> None:
    if len(samples) != int(config["expected_generated_sample_count"]):
        raise ValueError("sample count does not match E2 config")
    expected_per_cell = len(config["round_seeds"]) * int(
        config["samples_per_round_per_cell"]
    )
    grouped = observations.groupby(
        ["model_id", "profile_id", "capability_id", "intensity"], sort=False
    )
    sizes = grouped.size()
    if not bool((sizes == expected_per_cell).all()):
        bad = sizes[sizes != expected_per_cell].head().to_dict()
        raise ValueError(f"incomplete E2 prediction cells: {bad}")
    duplicate_count = int(
        observations.duplicated(subset=["model_id", "sample_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"found {duplicate_count} duplicate model/sample observations")
    capability_set = set(observations["capability_id"].unique())
    if capability_set != set(CAPABILITY_ORDER):
        raise ValueError(
            f"capability set mismatch: {sorted(capability_set)} != {sorted(CAPABILITY_ORDER)}"
        )
    model_groups = observations.groupby("model_id")["model_group"].unique().to_dict()
    for model_id in config["requested_models"]:
        if list(model_groups.get(model_id, ())) != ["timer_service"]:
            raise ValueError(f"invalid model_group for {model_id}: {model_groups.get(model_id)}")
    if list(model_groups.get(SEASONAL_BASELINE, ())) != ["baseline"]:
        raise ValueError("seasonal_naive is not marked as a baseline")


def profile_intensity_score_frame(observations: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "model_group", "profile_id", "capability_id", "intensity"]
    rows: list[dict[str, Any]] = []
    for key, group in observations.groupby(keys, sort=True):
        model_id, model_group, profile_id, capability_id, intensity = key
        coherence_mae = np.nan
        coherence_nmae = np.nan
        if capability_id == "hierarchical_coherence":
            coherence_mae = float(
                group["coherence_abs_sum"].sum() / group["coherence_point_count"].sum()
            )
            coherence_nmae = float(
                group["coherence_abs_sum"].sum() / group["parent_abs_sum"].sum()
            )
        rows.append(
            {
                "model_id": model_id,
                "model_group": model_group,
                "profile_id": profile_id,
                "capability_id": capability_id,
                "task_protocol": TASK_PROTOCOL_BY_CAPABILITY[capability_id],
                "intensity": int(intensity),
                "sample_count": int(len(group)),
                "round_count": int(group["round_index"].nunique()),
                "mase_mean": float(group["mase"].mean()),
                "mase_std": float(group["mase"].std(ddof=1)),
                "mae_mean": float(group["mae"].mean()),
                "nmae_abs": float(
                    group["abs_error_sum"].sum() / group["target_abs_sum"].sum()
                ),
                "coherence_mae": coherence_mae,
                "coherence_nmae": coherence_nmae,
            }
        )
    all_cells = pd.DataFrame.from_records(rows)
    baseline = all_cells[all_cells["model_id"] == SEASONAL_BASELINE].copy()
    baseline = baseline[
        [
            "profile_id",
            "capability_id",
            "intensity",
            "mase_mean",
            "nmae_abs",
        ]
    ].rename(
        columns={
            "mase_mean": "seasonal_naive_mase_mean",
            "nmae_abs": "seasonal_naive_nmae_abs",
        }
    )
    foundation = all_cells[all_cells["model_group"] == "timer_service"].copy()
    foundation = foundation.merge(
        baseline,
        on=["profile_id", "capability_id", "intensity"],
        how="left",
        validate="many_to_one",
    )
    if foundation[["seasonal_naive_mase_mean", "seasonal_naive_nmae_abs"]].isna().any().any():
        raise ValueError("missing seasonal-naive reference cell")
    if bool((foundation["seasonal_naive_mase_mean"] <= 0).any()):
        raise ValueError("seasonal-naive MASE must be positive")
    if bool((foundation["seasonal_naive_nmae_abs"] <= 0).any()):
        raise ValueError("seasonal-naive NMAE must be positive")
    foundation["seasonal_naive_skill_mase"] = 1.0 - (
        foundation["mase_mean"] / foundation["seasonal_naive_mase_mean"]
    )
    foundation["seasonal_naive_skill_nmae"] = 1.0 - (
        foundation["nmae_abs"] / foundation["seasonal_naive_nmae_abs"]
    )
    return sort_frame(foundation, include_profile=True, include_intensity=True)


def intensity_curve_frame(cell_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_id", "capability_id", "task_protocol", "intensity"]
    for key, group in cell_scores.groupby(keys, sort=True):
        model_id, capability_id, task_protocol, intensity = key
        rows.append(
            {
                "model_id": model_id,
                "capability_id": capability_id,
                "task_protocol": task_protocol,
                "intensity": int(intensity),
                "profile_count": int(group["profile_id"].nunique()),
                "sample_count": int(group["sample_count"].sum()),
                "mase_mean": float(group["mase_mean"].mean()),
                "mase_profile_variance": sample_variance(group["mase_mean"]),
                "nmae_abs_mean": float(group["nmae_abs"].mean()),
                "seasonal_naive_mase_mean": float(
                    group["seasonal_naive_mase_mean"].mean()
                ),
                "seasonal_naive_skill_mase": float(
                    group["seasonal_naive_skill_mase"].mean()
                ),
                "seasonal_naive_skill_nmae": float(
                    group["seasonal_naive_skill_nmae"].mean()
                ),
                "coherence_mae": finite_mean_or_nan(group["coherence_mae"]),
                "coherence_nmae": finite_mean_or_nan(group["coherence_nmae"]),
            }
        )
    return sort_frame(pd.DataFrame.from_records(rows), include_intensity=True)


def bucket_score_frame(cell_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_id", "profile_id", "capability_id", "task_protocol"]
    for key, group in cell_scores.groupby(keys, sort=True):
        model_id, profile_id, capability_id, task_protocol = key
        ordered = group.sort_values("intensity")
        require_five_intensities(ordered, key)
        mase = ordered["mase_mean"].to_numpy(dtype=float)
        nmae = ordered["nmae_abs"].to_numpy(dtype=float)
        skill = ordered["seasonal_naive_skill_mase"].to_numpy(dtype=float)
        worst_position = int(np.argmax(mase))
        best_position = int(np.argmin(mase))
        rows.append(
            {
                "model_id": model_id,
                "profile_id": profile_id,
                "capability_id": capability_id,
                "task_protocol": task_protocol,
                "sample_count": int(ordered["sample_count"].sum()),
                "five_level_mase_mean": float(np.mean(mase)),
                "mase_auc": normalized_auc(mase),
                "worst_level_intensity": INTENSITIES[worst_position],
                "worst_level_mase": float(mase[worst_position]),
                "best_level_intensity": INTENSITIES[best_position],
                "best_level_mase": float(mase[best_position]),
                "mase_intensity_1": float(mase[0]),
                "mase_intensity_5": float(mase[-1]),
                "mase_endpoint_delta": float(mase[-1] - mase[0]),
                "mase_endpoint_relative_change": relative_change(mase[0], mase[-1]),
                "mase_intensity_slope": linear_intensity_slope(mase),
                "mase_intensity_spearman": spearman_five_levels(mase),
                "five_level_nmae_abs_mean": float(np.mean(nmae)),
                "five_level_skill_mase_mean": float(np.mean(skill)),
                "five_level_skill_nmae_mean": float(
                    ordered["seasonal_naive_skill_nmae"].mean()
                ),
                "five_level_coherence_mae_mean": finite_mean_or_nan(
                    ordered["coherence_mae"]
                ),
                "five_level_coherence_nmae_mean": finite_mean_or_nan(
                    ordered["coherence_nmae"]
                ),
            }
        )
    return sort_frame(pd.DataFrame.from_records(rows), include_profile=True)


def capability_profile_frame(
    intensity_curves: pd.DataFrame,
    bucket_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_id", "capability_id", "task_protocol"]
    for key, group in intensity_curves.groupby(keys, sort=True):
        model_id, capability_id, task_protocol = key
        ordered = group.sort_values("intensity")
        require_five_intensities(ordered, key)
        mase = ordered["mase_mean"].to_numpy(dtype=float)
        nmae = ordered["nmae_abs_mean"].to_numpy(dtype=float)
        skill = ordered["seasonal_naive_skill_mase"].to_numpy(dtype=float)
        model_buckets = bucket_scores[
            (bucket_scores["model_id"] == model_id)
            & (bucket_scores["capability_id"] == capability_id)
        ]
        bucket_values = model_buckets["five_level_mase_mean"].to_numpy(dtype=float)
        bucket_skill = model_buckets["five_level_skill_mase_mean"].to_numpy(dtype=float)
        worst_position = int(np.argmax(mase))
        best_position = int(np.argmin(mase))
        rows.append(
            {
                "model_id": model_id,
                "capability_id": capability_id,
                "task_protocol": task_protocol,
                "profile_count": int(ordered["profile_count"].iloc[0]),
                "sample_count": int(ordered["sample_count"].sum()),
                "five_level_mase_mean": float(np.mean(mase)),
                "mase_auc": normalized_auc(mase),
                "worst_level_intensity": INTENSITIES[worst_position],
                "worst_level_mase": float(mase[worst_position]),
                "best_level_intensity": INTENSITIES[best_position],
                "best_level_mase": float(mase[best_position]),
                "mase_intensity_1": float(mase[0]),
                "mase_intensity_5": float(mase[-1]),
                "mase_endpoint_delta": float(mase[-1] - mase[0]),
                "mase_endpoint_relative_change": relative_change(mase[0], mase[-1]),
                "mase_intensity_slope": linear_intensity_slope(mase),
                "mase_intensity_spearman": spearman_five_levels(mase),
                "five_level_nmae_abs_mean": float(np.mean(nmae)),
                "five_level_skill_mase_mean": float(np.mean(skill)),
                "five_level_skill_nmae_mean": float(
                    ordered["seasonal_naive_skill_nmae"].mean()
                ),
                "bucket_mase_variance": sample_variance(bucket_values),
                "bucket_mase_std": sample_std(bucket_values),
                "bucket_mase_cv": coefficient_of_variation(bucket_values),
                "bucket_mase_range": value_range(bucket_values),
                "bucket_skill_variance": sample_variance(bucket_skill),
                "five_level_coherence_mae_mean": finite_mean_or_nan(
                    ordered["coherence_mae"]
                ),
                "five_level_coherence_nmae_mean": finite_mean_or_nan(
                    ordered["coherence_nmae"]
                ),
            }
        )
    return sort_frame(pd.DataFrame.from_records(rows))


def capability_bootstrap_results(
    observations: pd.DataFrame,
    *,
    cell_scores: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    grouped = {
        (str(model), str(profile), str(capability), int(intensity)): group.sort_values(
            ["round_index", "sample_index"]
        )
        for (model, profile, capability, intensity), group in observations.groupby(
            ["model_id", "profile_id", "capability_id", "intensity"], sort=False
        )
    }
    results: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for capability_id in CAPABILITY_ORDER:
        cap_cells = cell_scores[cell_scores["capability_id"] == capability_id]
        profiles = sorted(cap_cells["profile_id"].unique())
        models = [model for model in MODEL_ORDER if model in set(cap_cells["model_id"])]
        if not profiles or not models:
            raise ValueError(f"no cells for capability {capability_id}")
        for model_id in models:
            model_profiles = set(
                cap_cells[cap_cells["model_id"] == model_id]["profile_id"].unique()
            )
            if model_profiles != set(profiles):
                raise ValueError(
                    f"partial profile compatibility for {model_id}/{capability_id}: "
                    f"{sorted(model_profiles)} != {profiles}"
                )

        first = grouped[(SEASONAL_BASELINE, profiles[0], capability_id, INTENSITIES[0])]
        round_indexes = sorted(first["round_index"].unique())
        sample_indexes = sorted(first["sample_index"].unique())
        round_count = len(round_indexes)
        sample_count = len(sample_indexes)
        capability_seed = stable_seed(seed, capability_id)
        rng = np.random.default_rng(capability_seed)
        round_draws = rng.integers(
            0, round_count, size=(replicates, round_count), endpoint=False
        )
        sample_draws = rng.integers(
            0,
            sample_count,
            size=(replicates, len(profiles), round_count, sample_count),
            endpoint=False,
        )

        baseline_mase = np.empty((replicates, len(profiles), len(INTENSITIES)))
        denominators = np.empty_like(baseline_mase)
        baseline_nmae_numerators = np.empty_like(baseline_mase)
        for profile_position, profile_id in enumerate(profiles):
            draws = sample_draws[:, profile_position]
            for intensity_position, intensity in enumerate(INTENSITIES):
                baseline_group = grouped[
                    (SEASONAL_BASELINE, profile_id, capability_id, intensity)
                ]
                baseline_mase[:, profile_position, intensity_position] = bootstrap_mean(
                    metric_grid(
                        baseline_group,
                        "mase",
                        round_indexes=round_indexes,
                        sample_indexes=sample_indexes,
                    ),
                    round_draws,
                    draws,
                )
                denominators[:, profile_position, intensity_position] = bootstrap_sum(
                    metric_grid(
                        baseline_group,
                        "target_abs_sum",
                        round_indexes=round_indexes,
                        sample_indexes=sample_indexes,
                    ),
                    round_draws,
                    draws,
                )
                baseline_nmae_numerators[
                    :, profile_position, intensity_position
                ] = bootstrap_sum(
                    metric_grid(
                        baseline_group,
                        "abs_error_sum",
                        round_indexes=round_indexes,
                        sample_indexes=sample_indexes,
                    ),
                    round_draws,
                    draws,
                )

        for model_id in models:
            model_mase = np.empty_like(baseline_mase)
            model_nmae_numerators = np.empty_like(baseline_mase)
            coherence_abs = (
                np.empty_like(baseline_mase)
                if capability_id == "hierarchical_coherence"
                else None
            )
            coherence_points = np.empty_like(baseline_mase) if coherence_abs is not None else None
            parent_denominators = (
                np.empty_like(baseline_mase) if coherence_abs is not None else None
            )
            for profile_position, profile_id in enumerate(profiles):
                draws = sample_draws[:, profile_position]
                for intensity_position, intensity in enumerate(INTENSITIES):
                    group = grouped[(model_id, profile_id, capability_id, intensity)]
                    model_mase[:, profile_position, intensity_position] = bootstrap_mean(
                        metric_grid(
                            group,
                            "mase",
                            round_indexes=round_indexes,
                            sample_indexes=sample_indexes,
                        ),
                        round_draws,
                        draws,
                    )
                    model_nmae_numerators[
                        :, profile_position, intensity_position
                    ] = bootstrap_sum(
                        metric_grid(
                            group,
                            "abs_error_sum",
                            round_indexes=round_indexes,
                            sample_indexes=sample_indexes,
                        ),
                        round_draws,
                        draws,
                    )
                    if coherence_abs is not None:
                        assert coherence_points is not None and parent_denominators is not None
                        coherence_abs[
                            :, profile_position, intensity_position
                        ] = bootstrap_sum(
                            metric_grid(
                                group,
                                "coherence_abs_sum",
                                round_indexes=round_indexes,
                                sample_indexes=sample_indexes,
                            ),
                            round_draws,
                            draws,
                        )
                        coherence_points[
                            :, profile_position, intensity_position
                        ] = bootstrap_sum(
                            metric_grid(
                                group,
                                "coherence_point_count",
                                round_indexes=round_indexes,
                                sample_indexes=sample_indexes,
                            ),
                            round_draws,
                            draws,
                        )
                        parent_denominators[
                            :, profile_position, intensity_position
                        ] = bootstrap_sum(
                            metric_grid(
                                group,
                                "parent_abs_sum",
                                round_indexes=round_indexes,
                                sample_indexes=sample_indexes,
                            ),
                            round_draws,
                            draws,
                        )

            cell_nmae = model_nmae_numerators / denominators
            cell_skill = 1.0 - model_mase / baseline_mase
            curve_mase = np.mean(model_mase, axis=1)
            curve_nmae = np.mean(cell_nmae, axis=1)
            curve_skill = np.mean(cell_skill, axis=1)
            result = {
                "curve_mase": curve_mase,
                "curve_nmae_abs": curve_nmae,
                "curve_skill_mase": curve_skill,
                "five_level_mase_mean": np.mean(curve_mase, axis=1),
                "mase_auc": np.trapezoid(curve_mase, x=INTENSITY_AXIS, axis=1),
                "worst_level_mase": np.max(curve_mase, axis=1),
                "five_level_nmae_abs_mean": np.mean(curve_nmae, axis=1),
                "five_level_skill_mase_mean": np.mean(curve_skill, axis=1),
            }
            if coherence_abs is not None:
                assert coherence_points is not None and parent_denominators is not None
                curve_coherence_mae = np.mean(coherence_abs / coherence_points, axis=1)
                curve_coherence_nmae = np.mean(
                    coherence_abs / parent_denominators, axis=1
                )
                result["curve_coherence_mae"] = curve_coherence_mae
                result["curve_coherence_nmae"] = curve_coherence_nmae
                result["five_level_coherence_mae_mean"] = np.mean(
                    curve_coherence_mae, axis=1
                )
                result["five_level_coherence_nmae_mean"] = np.mean(
                    curve_coherence_nmae, axis=1
                )
            results[(model_id, capability_id)] = result
    return results


def attach_intensity_bootstrap_ci(
    frame: pd.DataFrame,
    bootstrap: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    result = frame.copy()
    mappings = {
        "mase": "curve_mase",
        "nmae_abs": "curve_nmae_abs",
        "skill_mase": "curve_skill_mase",
        "coherence_mae": "curve_coherence_mae",
        "coherence_nmae": "curve_coherence_nmae",
    }
    for prefix in mappings:
        result[f"{prefix}_ci_low"] = np.nan
        result[f"{prefix}_ci_high"] = np.nan
    for index, row in result.iterrows():
        values = bootstrap[(row["model_id"], row["capability_id"])]
        position = INTENSITIES.index(int(row["intensity"]))
        for prefix, name in mappings.items():
            if name not in values:
                continue
            low, high = percentile_ci(values[name][:, position])
            result.at[index, f"{prefix}_ci_low"] = low
            result.at[index, f"{prefix}_ci_high"] = high
    return result


def attach_capability_bootstrap_ci(
    frame: pd.DataFrame,
    bootstrap: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    result = frame.copy()
    metrics = (
        "five_level_mase_mean",
        "mase_auc",
        "worst_level_mase",
        "five_level_nmae_abs_mean",
        "five_level_skill_mase_mean",
        "five_level_coherence_mae_mean",
        "five_level_coherence_nmae_mean",
    )
    for metric in metrics:
        result[f"{metric}_ci_low"] = np.nan
        result[f"{metric}_ci_high"] = np.nan
    for index, row in result.iterrows():
        values = bootstrap[(row["model_id"], row["capability_id"])]
        for metric in metrics:
            if metric not in values:
                continue
            low, high = percentile_ci(values[metric])
            result.at[index, f"{metric}_ci_low"] = low
            result.at[index, f"{metric}_ci_high"] = high
    return result


def add_capability_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rank_five_level_mase"] = result.groupby("capability_id")[
        "five_level_mase_mean"
    ].rank(method="min", ascending=True)
    result["rank_mase_auc"] = result.groupby("capability_id")["mase_auc"].rank(
        method="min", ascending=True
    )
    result["rank_worst_level_mase"] = result.groupby("capability_id")[
        "worst_level_mase"
    ].rank(method="min", ascending=True)
    result["rank_seasonal_naive_skill"] = result.groupby("capability_id")[
        "five_level_skill_mase_mean"
    ].rank(method="min", ascending=False)
    hierarchy = result["capability_id"] == "hierarchical_coherence"
    result["rank_coherence_nmae"] = np.nan
    result.loc[hierarchy, "rank_coherence_nmae"] = result.loc[hierarchy].groupby(
        "capability_id"
    )["five_level_coherence_nmae_mean"].rank(method="min", ascending=True)
    return sort_frame(result)


def model_summary_frame(
    capability_profiles: pd.DataFrame,
    bootstrap: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        group = capability_profiles[
            (capability_profiles["model_id"] == model_id)
            & (capability_profiles["capability_id"].isin(UNIVARIATE_CAPABILITIES))
        ]
        if set(group["capability_id"]) != set(UNIVARIATE_CAPABILITIES):
            raise ValueError(f"incomplete univariate profile for {model_id}")
        ordered = group.set_index("capability_id").loc[list(UNIVARIATE_CAPABILITIES)]
        mase_boot = np.mean(
            np.stack(
                [
                    bootstrap[(model_id, capability)]["five_level_mase_mean"]
                    for capability in UNIVARIATE_CAPABILITIES
                ],
                axis=1,
            ),
            axis=1,
        )
        skill_boot = np.mean(
            np.stack(
                [
                    bootstrap[(model_id, capability)]["five_level_skill_mase_mean"]
                    for capability in UNIVARIATE_CAPABILITIES
                ],
                axis=1,
            ),
            axis=1,
        )
        nmae_boot = np.mean(
            np.stack(
                [
                    bootstrap[(model_id, capability)]["five_level_nmae_abs_mean"]
                    for capability in UNIVARIATE_CAPABILITIES
                ],
                axis=1,
            ),
            axis=1,
        )
        strongest = ordered.sort_values(
            ["rank_five_level_mase", "five_level_skill_mase_mean"],
            ascending=[True, False],
        ).iloc[0]
        weakest = ordered.sort_values(
            ["rank_five_level_mase", "five_level_skill_mase_mean"],
            ascending=[False, True],
        ).iloc[0]
        mase_low, mase_high = percentile_ci(mase_boot)
        skill_low, skill_high = percentile_ci(skill_boot)
        nmae_low, nmae_high = percentile_ci(nmae_boot)
        rows.append(
            {
                "model_id": model_id,
                "univariate_capability_count": len(UNIVARIATE_CAPABILITIES),
                "univariate_macro_mase": float(ordered["five_level_mase_mean"].mean()),
                "univariate_macro_mase_ci_low": mase_low,
                "univariate_macro_mase_ci_high": mase_high,
                "univariate_macro_nmae_abs": float(
                    ordered["five_level_nmae_abs_mean"].mean()
                ),
                "univariate_macro_nmae_abs_ci_low": nmae_low,
                "univariate_macro_nmae_abs_ci_high": nmae_high,
                "univariate_macro_skill_mase": float(
                    ordered["five_level_skill_mase_mean"].mean()
                ),
                "univariate_macro_skill_mase_ci_low": skill_low,
                "univariate_macro_skill_mase_ci_high": skill_high,
                "mean_capability_rank": float(ordered["rank_five_level_mase"].mean()),
                "top1_capability_count": int(
                    (ordered["rank_five_level_mase"] == 1).sum()
                ),
                "strongest_relative_capability": str(strongest.name),
                "strongest_relative_rank": int(strongest["rank_five_level_mase"]),
                "weakest_relative_capability": str(weakest.name),
                "weakest_relative_rank": int(weakest["rank_five_level_mase"]),
                "structured_capability_count": int(
                    (
                        (capability_profiles["model_id"] == model_id)
                        & capability_profiles["capability_id"].isin(
                            STRUCTURED_CAPABILITIES
                        )
                    ).sum()
                ),
            }
        )
    result = pd.DataFrame.from_records(rows)
    result["rank_univariate_macro_mase"] = result["univariate_macro_mase"].rank(
        method="min", ascending=True
    )
    result["rank_univariate_macro_skill"] = result[
        "univariate_macro_skill_mase"
    ].rank(method="min", ascending=False)
    return result.sort_values(
        ["rank_univariate_macro_mase", "model_id"], ignore_index=True
    )


def render_figures(
    figures_dir: Path,
    *,
    intensity_curves: pd.DataFrame,
    capability_profiles: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> list[str]:
    configure_plot_style()
    stems: list[Path] = []
    stems.append(plot_skill_heatmap(figures_dir, capability_profiles))
    stems.append(plot_intensity_response(figures_dir, intensity_curves))
    stems.append(plot_univariate_profiles(figures_dir, capability_profiles))
    stems.append(plot_bucket_variability(figures_dir, capability_profiles))
    stems.append(plot_univariate_summary(figures_dir, model_summary))
    return [
        path.relative_to(figures_dir.parent).as_posix()
        for stem in stems
        for path in (stem.with_suffix(".png"), stem.with_suffix(".svg"), stem.with_suffix(".pdf"))
    ]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_skill_heatmap(directory: Path, profiles: pd.DataFrame) -> Path:
    matrix = np.full((len(CAPABILITY_ORDER), len(MODEL_ORDER)), np.nan)
    mase = np.full_like(matrix, np.nan)
    for capability_position, capability in enumerate(CAPABILITY_ORDER):
        for model_position, model in enumerate(MODEL_ORDER):
            match = profiles[
                (profiles["capability_id"] == capability)
                & (profiles["model_id"] == model)
            ]
            if len(match) == 1:
                matrix[capability_position, model_position] = float(
                    match.iloc[0]["five_level_skill_mase_mean"]
                )
                mase[capability_position, model_position] = float(
                    match.iloc[0]["five_level_mase_mean"]
                )
    figure, axis = plt.subplots(figsize=(11.2, 6.8))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#E5E7EB")
    finite = matrix[np.isfinite(matrix)]
    norm = TwoSlopeNorm(
        vmin=min(-0.25, float(np.min(finite))),
        vcenter=0.0,
        vmax=max(0.9, float(np.max(finite))),
    )
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
    axis.grid(False)
    axis.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=30, ha="right")
    axis.set_yticks(
        range(len(CAPABILITY_ORDER)),
        [CAPABILITY_LABELS[capability] for capability in CAPABILITY_ORDER],
    )
    axis.axhline(len(UNIVARIATE_CAPABILITIES) - 0.5, color="#111827", linewidth=1.5)
    for row in range(matrix.shape[0]):
        best_column = int(np.nanargmin(mase[row]))
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                color = "white" if value < -0.05 or value > 0.72 else "#111827"
                axis.text(
                    column,
                    row,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold" if column == best_column else "normal",
                )
            else:
                axis.text(column, row, "N/A", ha="center", va="center", color="#6B7280")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.032, pad=0.025)
    colorbar.set_label("Skill vs seasonal naive (higher is better)")
    figure.suptitle(
        "Model capability profile: seasonal-naive-relative skill",
        y=0.995,
        fontsize=14,
    )
    axis.set_title(
        "Bold = lowest five-level mean MASE within capability; structured tasks below divider",
        color="#4B5563",
        fontsize=8,
        pad=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    stem = directory / "figure_1_capability_skill_heatmap"
    save_figure(figure, stem)
    return stem


def plot_intensity_response(directory: Path, curves: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(14.8, 11.0), sharex=True)
    for axis, capability in zip(axes.flat, CAPABILITY_ORDER, strict=True):
        cap = curves[curves["capability_id"] == capability]
        for model in MODEL_ORDER:
            group = cap[cap["model_id"] == model].sort_values("intensity")
            if group.empty:
                continue
            x = group["intensity"].to_numpy(dtype=float)
            y = group["mase_mean"].to_numpy(dtype=float)
            low = group["mase_ci_low"].to_numpy(dtype=float)
            high = group["mase_ci_high"].to_numpy(dtype=float)
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                color=MODEL_COLORS[model],
                label=model,
            )
            axis.fill_between(x, low, high, color=MODEL_COLORS[model], alpha=0.09)
        baseline = cap.groupby("intensity", as_index=False)[
            "seasonal_naive_mase_mean"
        ].first()
        axis.plot(
            baseline["intensity"],
            baseline["seasonal_naive_mase_mean"],
            color="#111827",
            linestyle="--",
            linewidth=1.3,
            label="Seasonal naive",
        )
        axis.set_title(CAPABILITY_LABELS[capability])
        axis.set_xticks(INTENSITIES)
        axis.set_xlabel("Intensity")
        axis.set_ylabel("MASE (lower is better)")
        axis.margins(x=0.04, y=0.12)
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[model], marker="o", linewidth=1.7, label=model)
        for model in MODEL_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], color="#111827", linestyle="--", label="Seasonal naive")
    )
    figure.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    figure.suptitle(
        "Intensity-response curves (profile-macro MASE, shaded 95% paired bootstrap CI)",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.975))
    stem = directory / "figure_2_intensity_response_mase"
    save_figure(figure, stem)
    return stem


def plot_univariate_profiles(directory: Path, profiles: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(12.6, 6.3))
    x = np.arange(len(UNIVARIATE_CAPABILITIES))
    for model in MODEL_ORDER:
        group = profiles[
            (profiles["model_id"] == model)
            & profiles["capability_id"].isin(UNIVARIATE_CAPABILITIES)
        ].set_index("capability_id").loc[list(UNIVARIATE_CAPABILITIES)]
        y = 100.0 * group["five_level_skill_mase_mean"].to_numpy(dtype=float)
        low = 100.0 * group["five_level_skill_mase_mean_ci_low"].to_numpy(dtype=float)
        high = 100.0 * group["five_level_skill_mase_mean_ci_high"].to_numpy(dtype=float)
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=4,
            color=MODEL_COLORS[model],
            label=model,
        )
        axis.fill_between(x, low, high, color=MODEL_COLORS[model], alpha=0.08)
    axis.axhline(0, color="#111827", linestyle="--", linewidth=1.2)
    axis.set_xticks(
        x,
        [CAPABILITY_LABELS[capability] for capability in UNIVARIATE_CAPABILITIES],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel("Skill vs seasonal naive (%)")
    axis.set_title("Univariate capability fingerprints")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axis.margins(x=0.03, y=0.12)
    figure.tight_layout()
    stem = directory / "figure_3_univariate_capability_fingerprints"
    save_figure(figure, stem)
    return stem


def plot_bucket_variability(directory: Path, profiles: pd.DataFrame) -> Path:
    capabilities = [
        capability
        for capability in CAPABILITY_ORDER
        if int(
            profiles[profiles["capability_id"] == capability]["profile_count"].max()
        )
        >= 2
    ]
    matrix = np.full((len(capabilities), len(MODEL_ORDER)), np.nan)
    for row, capability in enumerate(capabilities):
        for column, model in enumerate(MODEL_ORDER):
            match = profiles[
                (profiles["capability_id"] == capability)
                & (profiles["model_id"] == model)
            ]
            if len(match) == 1:
                matrix[row, column] = float(match.iloc[0]["bucket_mase_cv"])
    finite = matrix[np.isfinite(matrix)]
    vmax = max(0.15, float(np.quantile(finite, 0.95)))
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#E5E7EB")
    image = axis.imshow(
        np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto"
    )
    axis.grid(False)
    axis.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=30, ha="right")
    axis.set_yticks(
        range(len(capabilities)), [CAPABILITY_LABELS[value] for value in capabilities]
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
                color=("white" if np.isfinite(value) and value > 0.65 * vmax else "#111827"),
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.032, pad=0.025)
    colorbar.set_label("CV across bucket five-level MASE (lower is more robust)")
    figure.suptitle(
        "Sensitivity to real-data conditioning profile",
        y=0.995,
        fontsize=14,
    )
    axis.set_title(
        "Descriptive only: capabilities have two or three fixed paper-v1 buckets",
        color="#4B5563",
        fontsize=8,
        pad=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    stem = directory / "figure_4_cross_bucket_variability"
    save_figure(figure, stem)
    return stem


def plot_univariate_summary(directory: Path, summary: pd.DataFrame) -> Path:
    ordered = summary.sort_values("univariate_macro_mase", ascending=True).reset_index(drop=True)
    positions = np.arange(len(ordered))
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.8, 5.5), sharey=True)
    for axis in (left, right):
        axis.set_yticks(positions, ordered["model_id"])
    left.invert_yaxis()
    values = ordered["univariate_macro_mase"].to_numpy(dtype=float)
    left.errorbar(
        values,
        positions,
        xerr=np.vstack(
            [
                values - ordered["univariate_macro_mase_ci_low"].to_numpy(dtype=float),
                ordered["univariate_macro_mase_ci_high"].to_numpy(dtype=float) - values,
            ]
        ),
        fmt="o",
        color="#2563EB",
        capsize=3,
        linewidth=1.4,
    )
    left.set_xlabel("Macro MASE (lower is better)")
    left.set_title("Six-capability univariate macro")
    skill = 100.0 * ordered["univariate_macro_skill_mase"].to_numpy(dtype=float)
    right.errorbar(
        skill,
        positions,
        xerr=np.vstack(
            [
                skill
                - 100.0
                * ordered["univariate_macro_skill_mase_ci_low"].to_numpy(dtype=float),
                100.0
                * ordered["univariate_macro_skill_mase_ci_high"].to_numpy(dtype=float)
                - skill,
            ]
        ),
        fmt="o",
        color="#059669",
        capsize=3,
        linewidth=1.4,
    )
    right.axvline(0, color="#111827", linestyle="--", linewidth=1)
    right.set_xlabel("Macro skill vs seasonal naive (%)")
    right.set_title("Baseline-relative interpretation")
    figure.suptitle("Univariate model summary with 95% paired bootstrap CI", fontsize=13)
    figure.tight_layout()
    stem = directory / "figure_5_univariate_model_summary"
    save_figure(figure, stem)
    return stem


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def summarize_results(
    *,
    config: dict[str, Any],
    capability_profiles: pd.DataFrame,
    model_summary: pd.DataFrame,
    cell_scores: pd.DataFrame,
    table_rows: dict[str, int],
    figure_files: list[str],
) -> dict[str, Any]:
    leaders: dict[str, Any] = {}
    for capability in CAPABILITY_ORDER:
        group = capability_profiles[
            capability_profiles["capability_id"] == capability
        ].sort_values(["five_level_mase_mean", "model_id"])
        best = group.iloc[0]
        worst = group.iloc[-1]
        leaders[capability] = {
            "compatible_model_count": len(group),
            "best_model": best["model_id"],
            "best_five_level_mase": best["five_level_mase_mean"],
            "best_mase_ci": [
                best["five_level_mase_mean_ci_low"],
                best["five_level_mase_mean_ci_high"],
            ],
            "best_relative_skill": best["five_level_skill_mase_mean"],
            "worst_model": worst["model_id"],
            "worst_five_level_mase": worst["five_level_mase_mean"],
        }
    below_baseline = capability_profiles[
        capability_profiles["five_level_skill_mase_mean"] < 0
    ].sort_values("five_level_skill_mase_mean")
    bucket_candidates = capability_profiles[
        capability_profiles["bucket_mase_cv"].notna()
    ].sort_values("bucket_mase_cv", ascending=False)
    strongest_bucket_dependency = bucket_candidates.iloc[0]
    hierarchy = capability_profiles[
        capability_profiles["capability_id"] == "hierarchical_coherence"
    ].sort_values("five_level_coherence_nmae_mean")
    coherence_best = hierarchy.iloc[0]
    endpoint = capability_profiles.assign(
        endpoint_abs=capability_profiles["mase_endpoint_relative_change"].abs()
    ).sort_values("endpoint_abs", ascending=False).iloc[0]
    return clean_for_json(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "coverage": {
                "foundation_model_count": int(
                    capability_profiles["model_id"].nunique()
                ),
                "capability_count": len(CAPABILITY_ORDER),
                "model_capability_rows": len(capability_profiles),
                "profile_intensity_rows": len(cell_scores),
            },
            "univariate_model_summary": model_summary.to_dict(orient="records"),
            "capability_leaders": leaders,
            "diagnostics": {
                "below_seasonal_naive_capability_count": len(below_baseline),
                "below_seasonal_naive": below_baseline[
                    [
                        "model_id",
                        "capability_id",
                        "five_level_skill_mase_mean",
                    ]
                ].to_dict(orient="records"),
                "largest_endpoint_response": {
                    "model_id": endpoint["model_id"],
                    "capability_id": endpoint["capability_id"],
                    "relative_change_i1_to_i5": endpoint[
                        "mase_endpoint_relative_change"
                    ],
                    "mase_i1": endpoint["mase_intensity_1"],
                    "mase_i5": endpoint["mase_intensity_5"],
                },
                "largest_cross_bucket_cv": {
                    "model_id": strongest_bucket_dependency["model_id"],
                    "capability_id": strongest_bucket_dependency["capability_id"],
                    "profile_count": strongest_bucket_dependency["profile_count"],
                    "bucket_mase_cv": strongest_bucket_dependency["bucket_mase_cv"],
                },
                "best_hierarchy_coherence": {
                    "model_id": coherence_best["model_id"],
                    "coherence_nmae": coherence_best[
                        "five_level_coherence_nmae_mean"
                    ],
                    "coherence_mae": coherence_best[
                        "five_level_coherence_mae_mean"
                    ],
                },
            },
            "table_rows": table_rows,
            "figures": figure_files,
            "interpretation_boundary": (
                "E3 is synthetic capability profiling. External validity, common-factor channel "
                "controls, and future-covariate ablations require later experiments."
            ),
        }
    )


def render_report(summary: dict[str, Any]) -> str:
    model_rows = summary["univariate_model_summary"]
    leaders = summary["capability_leaders"]
    diagnostics = summary["diagnostics"]
    lines = [
        "# Paper E3：模型能力画像正式结果",
        "",
        f"日期：{summary['created_at'][:10]}",
        "",
        "## 概览",
        "",
        (
            "E3 只读复用已封存的 E2 样本与预测，按 bucket 等权、intensity 等权形成 "
            f"{summary['coverage']['model_capability_rows']} 个基础模型 × capability 画像。"
        ),
        "结构化任务按 multi-target 与 known-future-covariate 协议分别报告，不把不兼容项计为最差，也不合成结构化全局分数。",
        "",
        "## 六个单变量能力的 macro 总览",
        "",
        "| Rank | Model | Macro MASE [95% CI] | Skill vs SNaive | Mean capability rank | Relative strength | Relative weakness |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in model_rows:
        lines.append(
            "| {rank:.0f} | `{model}` | {mase:.4f} [{low:.4f}, {high:.4f}] | "
            "{skill:.1f}% | {mean_rank:.2f} | `{strong}` | `{weak}` |".format(
                rank=row["rank_univariate_macro_mase"],
                model=row["model_id"],
                mase=row["univariate_macro_mase"],
                low=row["univariate_macro_mase_ci_low"],
                high=row["univariate_macro_mase_ci_high"],
                skill=100 * row["univariate_macro_skill_mase"],
                mean_rank=row["mean_capability_rank"],
                strong=row["strongest_relative_capability"],
                weak=row["weakest_relative_capability"],
            )
        )
    lines.extend(
        [
            "",
            "## 分能力最优模型",
            "",
            "| Capability | Compatible models | Best model | Five-level MASE [95% CI] | Skill vs SNaive |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for capability in CAPABILITY_ORDER:
        row = leaders[capability]
        lines.append(
            "| `{cap}` | {count} | `{model}` | {mase:.4f} [{low:.4f}, {high:.4f}] | {skill:.1f}% |".format(
                cap=capability,
                count=row["compatible_model_count"],
                model=row["best_model"],
                mase=row["best_five_level_mase"],
                low=row["best_mase_ci"][0],
                high=row["best_mase_ci"][1],
                skill=100 * row["best_relative_skill"],
            )
        )
    below = diagnostics["below_seasonal_naive"]
    below_text = (
        "；".join(
            f"`{row['model_id']} / {row['capability_id']}` ({100 * row['five_level_skill_mase_mean']:.1f}%)"
            for row in below
        )
        if below
        else "无"
    )
    endpoint = diagnostics["largest_endpoint_response"]
    bucket = diagnostics["largest_cross_bucket_cv"]
    hierarchy = diagnostics["best_hierarchy_coherence"]
    lines.extend(
        [
            "",
            "## 诊断性发现",
            "",
            f"- capability 宏平均低于 seasonal naive 的模型单元：{below_text}。",
            (
                "- 最大 intensity 端点相对变化为 "
                f"`{endpoint['model_id']} / {endpoint['capability_id']}`："
                f"{100 * endpoint['relative_change_i1_to_i5']:.1f}% "
                f"(MASE {endpoint['mase_i1']:.4f} → {endpoint['mase_i5']:.4f})。"
            ),
            (
                "- 最大跨 bucket CV 为 "
                f"`{bucket['model_id']} / {bucket['capability_id']}`："
                f"{bucket['bucket_mase_cv']:.3f}，基于 {bucket['profile_count']} 个固定 profile；"
                "该值只作基底敏感性描述。"
            ),
            (
                "- 层级预测一致性误差最低的是 "
                f"`{hierarchy['model_id']}`：coherence NMAE={hierarchy['coherence_nmae']:.4f}，"
                f"coherence MAE={hierarchy['coherence_mae']:.4f}。"
            ),
            "",
            "## 解释边界",
            "",
            "intensity 表示结构强度，不表示难度；因此最差档由实际曲线确定，曲线随 intensity 下降并不异常。跨能力比较优先使用 seasonal-naive-relative skill，原始 MASE 只在同一 capability 或明确的 macro 规则下比较。",
            "",
            "本实验可以提出模型能力缺陷假设，但不能独自证明该缺陷会迁移到真实数据。`common_factor` 尚需 channel-independent/permutation 对照，`covariate_response` 尚需 drop/shuffle/event-flip 配对消融；合成—真实对应关系由后续外部效度实验检验。",
            "",
            "## 输出",
            "",
            "- `capability_profiles.csv`：论文主画像表、CI、排名与跨 bucket 统计。",
            "- `intensity_curves.csv`：五档 MASE/NMAE/skill 曲线及 CI。",
            "- `profile_intensity_scores.csv` 与 `bucket_scores.csv`：可追溯的底层汇总。",
            "- `model_summary.csv`：六个单变量能力的模型 macro 总览。",
            "- `paper_tables.md`：可直接审阅的主表。",
            "- `figures/`：5 张图，各保留 PNG、SVG 与 PDF。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_paper_tables(
    capability_profiles: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> str:
    lines = [
        "# E3 paper tables",
        "",
        "## Univariate macro summary",
        "",
        "| Rank | Model | MASE | 95% CI | NMAE | Skill | Mean cap. rank |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model_summary.to_dict(orient="records"):
        lines.append(
            f"| {int(row['rank_univariate_macro_mase'])} | `{row['model_id']}` | "
            f"{row['univariate_macro_mase']:.4f} | "
            f"[{row['univariate_macro_mase_ci_low']:.4f}, {row['univariate_macro_mase_ci_high']:.4f}] | "
            f"{row['univariate_macro_nmae_abs']:.4f} | "
            f"{100 * row['univariate_macro_skill_mase']:.1f}% | "
            f"{row['mean_capability_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Capability profiles",
            "",
            "| Capability | Model | Rank | MASE | 95% CI | AUC | Worst (I) | Skill | Bucket CV |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for capability in CAPABILITY_ORDER:
        group = capability_profiles[
            capability_profiles["capability_id"] == capability
        ].sort_values("rank_five_level_mase")
        for row in group.to_dict(orient="records"):
            bucket_cv = (
                f"{row['bucket_mase_cv']:.3f}"
                if row["bucket_mase_cv"] is not None
                and math.isfinite(float(row["bucket_mase_cv"]))
                else "N/A"
            )
            lines.append(
                f"| `{capability}` | `{row['model_id']}` | "
                f"{int(row['rank_five_level_mase'])} | {row['five_level_mase_mean']:.4f} | "
                f"[{row['five_level_mase_mean_ci_low']:.4f}, {row['five_level_mase_mean_ci_high']:.4f}] | "
                f"{row['mase_auc']:.4f} | {row['worst_level_mase']:.4f} "
                f"(I{int(row['worst_level_intensity'])}) | "
                f"{100 * row['five_level_skill_mase_mean']:.1f}% | {bucket_cv} |"
            )
    lines.extend(
        [
            "",
            "## Hierarchical prediction coherence",
            "",
            "| Model | Forecast MASE rank | Coherence NMAE | Coherence NMAE 95% CI | Coherence rank |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    hierarchy = capability_profiles[
        capability_profiles["capability_id"] == "hierarchical_coherence"
    ].sort_values("rank_coherence_nmae")
    for row in hierarchy.to_dict(orient="records"):
        lines.append(
            f"| `{row['model_id']}` | {int(row['rank_five_level_mase'])} | "
            f"{row['five_level_coherence_nmae_mean']:.4f} | "
            f"[{row['five_level_coherence_nmae_mean_ci_low']:.4f}, "
            f"{row['five_level_coherence_nmae_mean_ci_high']:.4f}] | "
            f"{int(row['rank_coherence_nmae'])} |"
        )
    return "\n".join(lines) + "\n"


def write_manifest(output_dir: Path, *, source_dir: Path) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    payload = {
        "schema_version": "paper_experiment_manifest.v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source": {
            "experiment": str(source_dir.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "all_source_files_verified": True,
        },
        "inputs": {
            "protocol": {
                "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256_file(PROTOCOL_PATH),
            },
            "runner": {
                "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "files": files,
    }
    write_json(output_dir / "manifest.json", payload)


def metric_grid(
    group: pd.DataFrame,
    column: str,
    *,
    round_indexes: list[int],
    sample_indexes: list[int],
) -> np.ndarray:
    expected = pd.MultiIndex.from_product(
        [round_indexes, sample_indexes], names=["round_index", "sample_index"]
    )
    indexed = group.set_index(["round_index", "sample_index"])
    if not indexed.index.is_unique:
        raise ValueError("bootstrap grid contains duplicate round/sample indexes")
    values = indexed.reindex(expected)[column]
    if values.isna().any():
        raise ValueError(f"bootstrap grid for {column} is incomplete or non-finite")
    array = values.to_numpy(dtype=float).reshape(len(round_indexes), len(sample_indexes))
    if not np.isfinite(array).all():
        raise ValueError(f"bootstrap grid for {column} contains non-finite values")
    return array


def bootstrap_mean(
    values: np.ndarray,
    round_draws: np.ndarray,
    sample_draws: np.ndarray,
) -> np.ndarray:
    selected = values[round_draws[:, :, None], sample_draws]
    return np.mean(selected, axis=(1, 2))


def bootstrap_sum(
    values: np.ndarray,
    round_draws: np.ndarray,
    sample_draws: np.ndarray,
) -> np.ndarray:
    selected = values[round_draws[:, :, None], sample_draws]
    return np.sum(selected, axis=(1, 2))


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    if not np.isfinite(finite).all() or finite.size == 0:
        raise ValueError("cannot compute percentile CI from empty or non-finite values")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def normalized_auc(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.shape != (len(INTENSITIES),):
        raise ValueError("intensity AUC requires exactly five values")
    return float(np.trapezoid(array, x=INTENSITY_AXIS))


def linear_intensity_slope(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.shape != (len(INTENSITIES),):
        raise ValueError("intensity slope requires exactly five values")
    centered_x = INTENSITY_AXIS - np.mean(INTENSITY_AXIS)
    return float(np.sum(centered_x * (array - np.mean(array))) / np.sum(centered_x**2))


def spearman_five_levels(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.shape != (len(INTENSITIES),):
        raise ValueError("intensity Spearman requires exactly five values")
    ranks = pd.Series(array).rank(method="average").to_numpy(dtype=float)
    if np.std(ranks) == 0:
        return 0.0
    return float(np.corrcoef(np.arange(1, 6, dtype=float), ranks)[0, 1])


def relative_change(first: float, last: float) -> float:
    if not math.isfinite(first) or first <= 0:
        raise ValueError("relative endpoint change requires a positive first value")
    return float((last - first) / first)


def sample_variance(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.var(array, ddof=1)) if len(array) >= 2 else np.nan


def sample_std(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.std(array, ddof=1)) if len(array) >= 2 else np.nan


def coefficient_of_variation(values: Iterable[float]) -> float:
    array = finite_array(values)
    if len(array) < 2:
        return np.nan
    mean = float(np.mean(array))
    if mean <= 0:
        raise ValueError("bucket CV requires a positive mean")
    return float(np.std(array, ddof=1) / mean)


def value_range(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.max(array) - np.min(array)) if len(array) >= 2 else np.nan


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def finite_mean_or_nan(values: Iterable[float]) -> float:
    array = finite_array(values)
    return float(np.mean(array)) if len(array) else np.nan


def require_five_intensities(group: pd.DataFrame, key: Any) -> None:
    observed = tuple(int(value) for value in group["intensity"])
    if observed != INTENSITIES:
        raise ValueError(f"expected five ordered intensities for {key}, got {observed}")


def sort_frame(
    frame: pd.DataFrame,
    *,
    include_profile: bool = False,
    include_intensity: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    result["_model_order"] = pd.Categorical(
        result["model_id"], categories=list(MODEL_ORDER), ordered=True
    )
    result["_capability_order"] = pd.Categorical(
        result["capability_id"], categories=list(CAPABILITY_ORDER), ordered=True
    )
    columns = ["_capability_order", "_model_order"]
    if include_profile:
        columns.append("profile_id")
    if include_intensity:
        columns.append("intensity")
    result = result.sort_values(columns, ignore_index=True)
    return result.drop(columns=["_model_order", "_capability_order"])


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    cleaned = clean_for_json(payload)
    path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.12g",
        na_rep="",
    )


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
