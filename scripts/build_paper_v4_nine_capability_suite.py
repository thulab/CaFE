#!/usr/bin/env python3
"""Build the paper-v4 four-lookback profiles and qualify all nine generators.

The central invariant is pairing: a synthetic task is generated once at L=504,
H=48 and the L={96,168,336,504} benchmark views are suffixes of that same
master task.  Acceptance requires every view to pass its own real-calibrated
feature-support and near-distance gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.synthetic_feature_gate import evaluate_feature_support_gate  # noqa: E402
from app.services.synthetic_generation_service import (  # noqa: E402
    PAPER_UNIVARIATE_CAPABILITY_IDS,
    TARGET_FEATURES_BY_CAPABILITY,
    _attempt_seed,
    _generate_sample_values,
    _realized_features,
    _seed_for,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    resolve_generator_conditioning,
)
from app.services.synthetic_near_distance_gate import evaluate_near_distance_gate  # noqa: E402
from build_synthetic_v2_feature_gate_artifact import (  # noqa: E402
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_COVERAGE,
    DEFAULT_GATE_REFERENCE_FRACTION,
    calibrate_capability as calibrate_feature_gate_capability,
    split_real_rows_three_way,
)
from build_synthetic_v2_generator_conditioning_artifact import (  # noqa: E402
    calibrate_capability_conditioning,
    derive_profile_nuisance,
    empirical_percentiles,
    finite_values,
    quantiles_for_levels,
    reference_percentile_levels,
    summarize_real_features,
)
from build_paper_v4_profile_suite import SOURCE_SPECS as UNIVARIATE_SOURCE_SPECS  # noqa: E402
from run_synthetic_v2_near_distance_calibration import (  # noqa: E402
    BucketSpec,
    load_real_bucket,
    make_row,
    normalize_covariates,
    online_artifact_bucket,
    standardize_target,
    thresholds_from_split,
)


SCHEMA_VERSION = "paper_v4_nine_capability_suite.v1"
CONTEXT_LENGTHS = (96, 168, 336, 504)
MAX_CONTEXT_LENGTH = max(CONTEXT_LENGTHS)
HORIZON = 48
VALIDATION_EMBARGO = 48
MASTER_LOADER_HORIZON = HORIZON + VALIDATION_EMBARGO
DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_GIFT_EVAL_DIR = Path.home() / "xmy/gift-eval"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp/v4/01_nine_capability_suite"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-18-paper-v4-nine-capability-profile-and-generation-protocol.md"
)
PAPER_V2_CANONICAL_PATH = (
    REPO_ROOT
    / "runtime/paper_exp/v2/00_transfer_protocol_freeze/generator_conditioning_artifact.json"
)
PAPER_V1_CANONICAL_PATH = (
    REPO_ROOT / "backend/app/data/synthetic_v2_generator_conditioning_artifact.json"
)
DEFAULT_MAX_WINDOWS_PER_SOURCE = 120
DEFAULT_CALIBRATION_SAMPLES = 16
DEFAULT_QUALIFICATION_SAMPLES_PER_CELL = 8
DEFAULT_MAX_ATTEMPTS = 64
DEFAULT_SEED = 2026071804

STRUCTURED_CAPABILITY_IDS = (
    "common_factor",
    "hierarchical_coherence",
    "covariate_response",
)
ALL_CAPABILITY_IDS = (*PAPER_UNIVARIATE_CAPABILITY_IDS, *STRUCTURED_CAPABILITY_IDS)
PRIMARY_TARGET_FEATURE = {
    "trend": "trend_strength",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "change_point_shift_energy",
    "nonlinear_persistence": "nonlinear_conditional_gain",
    "predictable_intermittency": "spike_rate",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "covariate_response": "covariate_incremental_r2",
}


@dataclass(frozen=True)
class RealSource:
    source_id: str
    dataset_name: str
    family_id: str
    domain: str
    task_id: str
    kind: str
    asset_name: str
    season_length: int
    frequency: str
    target_dim: int = 1
    covariate_dim: int = 0
    hierarchy: str | None = None
    task: int = 1


@dataclass(frozen=True)
class TaskDesign:
    task_id: str
    target_dim: int
    covariate_dim: int
    hierarchy: str | None
    season_length: int
    frequency: str
    capabilities: tuple[str, ...]


TASK_DESIGNS = {
    "univariate": TaskDesign(
        "univariate",
        1,
        0,
        None,
        24,
        "h",
        tuple(PAPER_UNIVARIATE_CAPABILITY_IDS),
    ),
    "common_factor": TaskDesign(
        "common_factor",
        3,
        0,
        None,
        24,
        "h",
        ("common_factor",),
    ),
    "hierarchy": TaskDesign(
        "hierarchy",
        3,
        0,
        "additive_first",
        7,
        "D",
        ("hierarchical_coherence",),
    ),
    "covariate": TaskDesign(
        "covariate",
        1,
        0,
        None,
        24,
        "h",
        ("covariate_response",),
    ),
}


UNIVARIATE_CALIBRATION_SOURCES = (
    RealSource(
        "m4_hourly",
        "M4 Hourly",
        "m4_hourly",
        "Econ/Fin",
        "univariate",
        "tsf_univariate",
        "m4_hourly_dataset.zip",
        24,
        "h",
    ),
    RealSource(
        "gift_electricity_h",
        "Electricity/H",
        "electricity",
        "Energy",
        "univariate",
        "gift_univariate",
        "electricity/H",
        24,
        "h",
    ),
    RealSource(
        "gift_ett1_h",
        "ETT1/H",
        "ETT",
        "Energy",
        "univariate",
        "gift_univariate",
        "ett1/H",
        24,
        "h",
    ),
    RealSource(
        "gift_loop_seattle_h",
        "Loop Seattle/H",
        "LOOP_SEATTLE",
        "Transport",
        "univariate",
        "gift_univariate",
        "LOOP_SEATTLE/H",
        24,
        "h",
    ),
    RealSource(
        "gift_bitbrains_fast_h",
        "Bitbrains Fast Storage/H",
        "bitbrains",
        "Web/CloudOps",
        "univariate",
        "gift_univariate",
        "bitbrains_fast_storage/H",
        24,
        "h",
    ),
    RealSource(
        "gift_bizitobs_l2c_h",
        "BizITObs L2C/H",
        "bizitobs_l2c",
        "Web/CloudOps",
        "univariate",
        "gift_univariate",
        "bizitobs_l2c/H",
        24,
        "h",
    ),
)

STRUCTURED_SOURCES = (
    RealSource(
        "electricity_hourly_panel",
        "Electricity Hourly",
        "electricity",
        "Energy",
        "common_factor",
        "tsf_panel",
        "electricity_hourly_dataset.zip",
        24,
        "h",
        target_dim=3,
    ),
    RealSource(
        "traffic_hourly_panel",
        "Traffic Hourly",
        "traffic",
        "Transport",
        "common_factor",
        "tsf_panel",
        "traffic_hourly_dataset.zip",
        24,
        "h",
        target_dim=3,
    ),
    RealSource(
        "gift_jena_weather_panel",
        "Jena Weather/H",
        "jena_weather",
        "Nature",
        "common_factor",
        "gift_panel",
        "jena_weather/H",
        24,
        "h",
        target_dim=3,
    ),
    RealSource(
        "gift_bizitobs_l2c_panel",
        "BizITObs L2C/H",
        "bizitobs_l2c",
        "Web/CloudOps",
        "common_factor",
        "gift_panel",
        "bizitobs_l2c/H",
        24,
        "h",
        target_dim=3,
    ),
    RealSource(
        "m5_daily_hierarchy",
        "M5 Daily",
        "m5",
        "Retail",
        "hierarchy",
        "m5_hierarchy",
        "m5-forecasting-accuracy.zip",
        7,
        "D",
        target_dim=3,
        hierarchy="additive_first",
    ),
    RealSource(
        "gefcom2014_load",
        "GEFCom2014 Load",
        "gefcom2014_load",
        "Energy",
        "covariate",
        "gefcom2014_load",
        "GEFCom2014.zip",
        24,
        "h",
        covariate_dim=25,
    ),
    RealSource(
        "gefcom2014_solar",
        "GEFCom2014 Solar",
        "gefcom2014_solar",
        "Energy",
        "covariate",
        "gefcom2014_solar",
        "GEFCom2014.zip",
        24,
        "h",
        covariate_dim=12,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and qualify the paper-v4 nine-capability four-lookback suite."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gift-eval-dir", type=Path, default=DEFAULT_GIFT_EVAL_DIR)
    parser.add_argument(
        "--max-windows-per-source",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_SOURCE,
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
    )
    parser.add_argument(
        "--qualification-samples-per-cell",
        type=int,
        default=DEFAULT_QUALIFICATION_SAMPLES_PER_CELL,
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stage", choices=("all", "build", "qualify"), default="all")
    return parser.parse_args()


def source_asset_path(source: RealSource, *, data_dir: Path, gift_eval_dir: Path) -> Path:
    if source.kind.startswith("gift_"):
        return gift_eval_dir / source.asset_name
    return data_dir / source.asset_name


def master_bucket_spec(source: RealSource, *, max_windows: int) -> BucketSpec:
    return BucketSpec(
        profile_id=f"{source.source_id}__master_L504_H48_E48",
        kind=source.kind,
        asset_name=source.asset_name,
        context_length=MAX_CONTEXT_LENGTH,
        horizon=MASTER_LOADER_HORIZON,
        stride=HORIZON,
        season_length=source.season_length,
        target_dim=source.target_dim,
        covariate_dim=source.covariate_dim,
        hierarchy=source.hierarchy,
        max_series=max(240, max_windows),
        max_groups=20,
        task=source.task,
        synthetic_capabilities=TASK_DESIGNS[source.task_id].capabilities,
    )


def task_bucket_spec(task: TaskDesign, context_length: int) -> BucketSpec:
    return BucketSpec(
        profile_id=gate_profile_id(task.task_id, context_length),
        kind="pooled_real_profile",
        asset_name="multiple",
        context_length=context_length,
        horizon=HORIZON,
        stride=HORIZON,
        season_length=task.season_length,
        target_dim=task.target_dim,
        covariate_dim=task.covariate_dim,
        hierarchy=task.hierarchy,
        synthetic_capabilities=task.capabilities,
    )


def generator_profile_id(task_id: str) -> str:
    return f"paper_v4_{task_id}_global_L504_H48"


def gate_profile_id(task_id: str, context_length: int) -> str:
    return f"paper_v4_{task_id}_global_L{context_length}_H48"


def paired_view(
    target: np.ndarray,
    covariates: np.ndarray | None,
    *,
    context_length: int,
    hierarchy: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Take a suffix view while preserving the master's exact 48-step future."""

    target_array = np.asarray(target, dtype=float)
    expected = MAX_CONTEXT_LENGTH + MASTER_LOADER_HORIZON
    if target_array.shape[0] != expected:
        raise ValueError(f"master target must have {expected} steps")
    start = MAX_CONTEXT_LENGTH - int(context_length)
    stop = MAX_CONTEXT_LENGTH + HORIZON
    view_target = standardize_target(
        target_array[start:stop],
        context_length,
        hierarchy=hierarchy,
    )
    if covariates is None:
        return view_target, None
    view_covariates = normalize_covariates(
        np.asarray(covariates, dtype=float)[start:stop],
        context_length,
    )
    return view_target, view_covariates


def synthetic_paired_view(
    target: np.ndarray,
    covariates: np.ndarray | None,
    *,
    context_length: int,
    hierarchy: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Suffix a 552-step generated task (which has no loader embargo tail)."""

    target_array = np.asarray(target, dtype=float)
    expected = MAX_CONTEXT_LENGTH + HORIZON
    if target_array.shape[0] != expected:
        raise ValueError(f"generated master target must have {expected} steps")
    start = MAX_CONTEXT_LENGTH - int(context_length)
    view_target = standardize_target(
        target_array[start:],
        context_length,
        hierarchy=hierarchy,
    )
    if covariates is None:
        return view_target, None
    view_covariates = normalize_covariates(
        np.asarray(covariates, dtype=float)[start:],
        context_length,
    )
    return view_target, view_covariates


def load_source_views(
    source: RealSource,
    *,
    data_dir: Path,
    gift_eval_dir: Path,
    max_windows: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    loader_spec = master_bucket_spec(source, max_windows=max_windows)
    path = source_asset_path(source, data_dir=data_dir, gift_eval_dir=gift_eval_dir)
    master_rows = load_real_bucket(loader_spec, path, max_windows=max_windows)
    views: dict[int, list[dict[str, Any]]] = {
        context_length: [] for context_length in CONTEXT_LENGTHS
    }
    for row_index, row in enumerate(master_rows):
        original_group = str(row.get("group_id") or "single-series")
        source_group = f"{source.source_id}:{original_group}"
        source_start = int(row.get("window_start") or 0)
        for context_length in CONTEXT_LENGTHS:
            target, covariates = paired_view(
                np.asarray(row["target"], dtype=float),
                row.get("covariates"),
                context_length=context_length,
                hierarchy=source.hierarchy,
            )
            spec = replace(
                loader_spec,
                profile_id=f"{source.source_id}__L{context_length}_H48",
                context_length=context_length,
                horizon=HORIZON,
            )
            view = make_row(target, spec, covariates=covariates, label="real")
            view.update(
                {
                    "group_id": source_group,
                    "window_start": source_start + MAX_CONTEXT_LENGTH - context_length,
                    "source_id": source.source_id,
                    "family_id": source.family_id,
                    "master_row_index": row_index,
                }
            )
            views[context_length].append(view)
    return views, {
        "source": asdict(source),
        "asset_path": relative_or_absolute(path),
        "master_window_count": len(master_rows),
        "master_shape": {
            "context_length": MAX_CONTEXT_LENGTH,
            "benchmark_horizon": HORIZON,
            "validation_embargo": VALIDATION_EMBARGO,
        },
    }


def balanced_three_way_split(
    source_rows: dict[str, list[dict[str, Any]]],
    spec: BucketSpec,
    *,
    seed: int,
    minimum_rows: int = 60,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Split inside each dataset, then pool equal-source partitions.

    A global group split is invalid for heterogeneous pooled profiles: with two
    sources it can compare Load only against Solar only and inflate the natural
    real-real distance.  Source-local splitting keeps every gate threshold
    anchored to within-dataset variation while preserving group/temporal
    leakage protection.
    """

    eligible = {
        source_id: rows
        for source_id, rows in source_rows.items()
        if len(rows) >= minimum_rows
    }
    if not eligible:
        raise ValueError(f"{spec.profile_id} has no source with {minimum_rows} rows")
    source_partitions: dict[
        str,
        tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    source_splits: dict[str, dict[str, Any]] = {}
    excluded_reasons: dict[str, str] = {}
    for source_index, source_id in enumerate(sorted(eligible)):
        source_spec = replace(spec, profile_id=f"{spec.profile_id}__{source_id}")
        try:
            source_parameter, source_reference, source_calibration, summary = (
                split_real_rows_three_way(
                    eligible[source_id],
                    source_spec,
                    calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
                    gate_reference_fraction=DEFAULT_GATE_REFERENCE_FRACTION,
                    seed=_seed_for(seed, source_spec.profile_id, source_index),
                )
            )
        except ValueError as error:
            excluded_reasons[source_id] = str(error)
            continue
        source_partitions[source_id] = (
            source_parameter,
            source_reference,
            source_calibration,
        )
        source_splits[source_id] = summary
    if not source_partitions:
        raise ValueError(f"{spec.profile_id} has no source with a valid three-way split")

    # Balance after source-local splitting.  This keeps the larger temporal
    # history needed for embargo, while giving each source exactly equal mass
    # inside each of the three downstream roles.
    partition_caps = tuple(
        min(len(parts[index]) for parts in source_partitions.values())
        for index in range(3)
    )
    parameter: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for parts in source_partitions.values():
        parameter.extend(sample_evenly(parts[0], partition_caps[0]))
        reference.extend(sample_evenly(parts[1], partition_caps[1]))
        calibration.extend(sample_evenly(parts[2], partition_caps[2]))
    split_summary = {
        "policy": "source_local_three_way_then_equal_source_pool",
        "source_count": len(source_partitions),
        "generator_parameter_count": len(parameter),
        "gate_reference_count": len(reference),
        "gate_calibration_count": len(calibration),
        "balanced_partition_rows_per_source": {
            "generator_parameter": partition_caps[0],
            "gate_reference": partition_caps[1],
            "gate_calibration": partition_caps[2],
        },
        "source_splits": source_splits,
    }
    pool_summary = {
        "eligible_source_ids": sorted(source_partitions),
        "excluded_source_ids": sorted(set(source_rows) - set(source_partitions)),
        "excluded_reasons": {
            **{
                source_id: f"fewer than {minimum_rows} rows"
                for source_id in sorted(set(source_rows) - set(eligible))
            },
            **excluded_reasons,
        },
        "source_input_row_counts": {
            source_id: len(rows) for source_id, rows in source_rows.items()
        },
        "balanced_partition_rows_per_source": {
            "generator_parameter": partition_caps[0],
            "gate_reference": partition_caps[1],
            "gate_calibration": partition_caps[2],
        },
        "pooled_count": len(parameter) + len(reference) + len(calibration),
    }
    return parameter, reference, calibration, split_summary, pool_summary


def calibrate_feature_gate_with_rounding_guard(
    capability_id: str,
    reference: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
) -> dict[str, Any]:
    config = calibrate_feature_gate_capability(
        capability_id,
        reference,
        calibration,
        coverage=DEFAULT_COVERAGE,
    )
    support = config["control_support"]
    if support["feature_names"]:
        # Online artifacts round centers, scales and precision to 8 digits.
        # A 1e-5 guard prevents an exactly-on-boundary point from failing only
        # because the score is evaluated after that serialization.
        support["threshold"] = round_float(float(support["threshold"]) * 1.00001)
        support["serialization_rounding_guard"] = 1.00001
    return config


def source_profile(
    source: RealSource,
    context_length: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "profile_id": f"{source.source_id}__L{context_length}_H48",
        "source_id": source.source_id,
        "dataset_name": source.dataset_name,
        "family_id": source.family_id,
        "domain": source.domain,
        "task_id": source.task_id,
        "context_length": context_length,
        "horizon": HORIZON,
        "target_dim": source.target_dim,
        "covariate_dim": source.covariate_dim,
        "hierarchy": source.hierarchy,
        "season_length": source.season_length,
        "frequency": source.frequency,
        "window_count": len(rows),
        "feature_summary": summarize_real_features(rows),
    }


def compose_canonical_intensity() -> dict[str, Any]:
    paper_v2 = read_json(PAPER_V2_CANONICAL_PATH)["canonical_intensity"]
    paper_v1 = read_json(PAPER_V1_CANONICAL_PATH)["canonical_intensity"]
    capabilities = {
        capability_id: (
            paper_v2["capabilities"][capability_id]
            if capability_id in PAPER_UNIVARIATE_CAPABILITY_IDS
            else paper_v1["capabilities"][capability_id]
        )
        for capability_id in ALL_CAPABILITY_IDS
    }
    fingerprint_payload = json.dumps(
        {
            capability_id: {
                "primary_feature": PRIMARY_TARGET_FEATURE[capability_id],
                "target_values": capabilities[capability_id]["target_values"],
            }
            for capability_id in ALL_CAPABILITY_IDS
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "scale_id": "synthetic-v2-paper-v4-nine-capability-frozen-2026-07-18",
        "scale_fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
        "policy": (
            "paper-v2 final-shape canonical scale for six univariate capabilities; "
            "frozen paper-v1 structured canonical scale for three structured capabilities"
        ),
        "capabilities": capabilities,
    }


def measurement_rows(
    rows: list[dict[str, Any]],
    *,
    context_length: int,
    horizon: int,
    season_length: int,
) -> list[dict[str, Any]]:
    stop = context_length + horizon
    result: list[dict[str, Any]] = []
    for row in rows:
        target = np.asarray(row["target"], dtype=float)[:stop]
        covariates = row.get("covariates")
        covariate_view = (
            np.asarray(covariates, dtype=float)[:stop]
            if covariates is not None
            else None
        )
        result.append(
            {
                **row,
                "features": _realized_features(
                    target,
                    covariate_view,
                    season_length,
                    context_length,
                ),
            }
        )
    return result


def build_suite(
    output_dir: Path,
    *,
    data_dir: Path,
    gift_eval_dir: Path,
    max_windows: int,
    calibration_samples: int,
    seed: int,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    sources = (*UNIVARIATE_CALIBRATION_SOURCES, *STRUCTURED_SOURCES)
    source_views: dict[str, dict[int, list[dict[str, Any]]]] = {}
    inventories: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        print(f"[source {source_index + 1}/{len(sources)}] {source.source_id}", flush=True)
        views, inventory = load_source_views(
            source,
            data_dir=data_dir,
            gift_eval_dir=gift_eval_dir,
            max_windows=max_windows,
        )
        source_views[source.source_id] = views
        inventories.append(inventory)
        profiles.extend(
            source_profile(source, context_length, views[context_length])
            for context_length in CONTEXT_LENGTHS
        )

    canonical_intensity = compose_canonical_intensity()
    generator_profiles: dict[str, dict[str, Any]] = {}
    feature_buckets: dict[str, dict[str, Any]] = {}
    near_buckets: dict[str, dict[str, Any]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    pool_summaries: dict[str, dict[str, Any]] = {}

    for task_index, task in enumerate(TASK_DESIGNS.values()):
        task_sources = [source for source in sources if source.task_id == task.task_id]
        task_parameter_rows: dict[int, list[dict[str, Any]]] = {}
        for context_length in CONTEXT_LENGTHS:
            spec = task_bucket_spec(task, context_length)
            parameter, reference, calibration, split_summary, pool_summary = (
                balanced_three_way_split(
                    {
                        source.source_id: source_views[source.source_id][context_length]
                        for source in task_sources
                    },
                    spec,
                    seed=_seed_for(seed, spec.profile_id, 0),
                )
            )
            task_parameter_rows[context_length] = parameter
            pool_summaries[gate_profile_id(task.task_id, context_length)] = pool_summary
            split_summaries[spec.profile_id] = split_summary
            feature_buckets[spec.profile_id] = {
                "profile_id": spec.profile_id,
                "context_length": context_length,
                "horizon": HORIZON,
                "season_length": task.season_length,
                "target_dim": task.target_dim,
                "covariate_dim": task.covariate_dim,
                "split": split_summary,
                "capabilities": {
                    capability_id: calibrate_feature_gate_with_rounding_guard(
                        capability_id,
                        reference,
                        calibration,
                    )
                    for capability_id in task.capabilities
                },
            }
            thresholds, _diagnostics = thresholds_from_split(reference, calibration)
            near_buckets[spec.profile_id] = online_artifact_bucket(
                spec,
                sample_evenly(reference, min(192, len(reference))),
                thresholds=thresholds,
                split_summary=split_summary,
            )

        master_spec = task_bucket_spec(task, MAX_CONTEXT_LENGTH)
        master_spec = replace(master_spec, profile_id=generator_profile_id(task.task_id))
        parameter = task_parameter_rows[MAX_CONTEXT_LENGTH]
        master_split = split_summaries[gate_profile_id(task.task_id, MAX_CONTEXT_LENGTH)]
        measured = measurement_rows(
            parameter,
            context_length=MAX_CONTEXT_LENGTH,
            horizon=HORIZON,
            season_length=task.season_length,
        )
        real_feature_summary = summarize_real_features(measured)
        nuisance = derive_profile_nuisance(
            real_feature_summary,
            MAX_CONTEXT_LENGTH,
            task.season_length,
        )
        controlled_feature_preconditioning: dict[str, Any] = {
            "method": "none",
            "changes": {},
        }
        if task.task_id == "covariate":
            residual_outlier = real_feature_summary.get(
                "covariate_residual_outlier_rate",
                {},
            )
            if float(residual_outlier.get("p75", 1.0)) <= 0.01:
                # The real long-window residuals have essentially no heavy-tail
                # outliers.  A Student-t residual would create a structural
                # controlled-feature mismatch that retries cannot fix, so select
                # the generator's Gaussian residual mechanism before fitting the
                # capability intensity curve.
                nuisance["noise_degrees_of_freedom"] = 0.0
                controlled_feature_preconditioning = {
                    "method": "real_control_support_rule",
                    "rule": (
                        "use Gaussian residuals when real "
                        "covariate_residual_outlier_rate p75 <= 0.01"
                    ),
                    "observed_p75": residual_outlier.get("p75"),
                    "changes": {"noise_degrees_of_freedom": 0.0},
                }
        capability_configs: dict[str, dict[str, Any]] = {}
        for capability_index, capability_id in enumerate(task.capabilities):
            print(
                f"[conditioning {task_index + 1}/{len(TASK_DESIGNS)}] "
                f"{task.task_id}/{capability_id}",
                flush=True,
            )
            definition = canonical_intensity["capabilities"][capability_id]
            targets = [float(value) for value in definition["target_values"]]
            primary = PRIMARY_TARGET_FEATURE[capability_id]
            primary_values = finite_values(measured, primary)
            if not primary_values.size:
                raise ValueError(f"{task.task_id}/{capability_id} has no finite {primary}")
            local_quantiles = {
                name: quantiles_for_levels(
                    finite_values(measured, name),
                    reference_percentile_levels(capability_id),
                )
                for name in TARGET_FEATURES_BY_CAPABILITY[capability_id]
                if finite_values(measured, name).size
            }
            # Regime and nonlinear observables have materially higher
            # Monte-Carlo variance than the other seven capability features.
            # Doubling their fit bank is pre-registered here; the independent
            # validation bank and the common error tolerance stay unchanged.
            per_grid_samples = (
                calibration_samples * 2
                if capability_id in {"regime_switching", "nonlinear_persistence"}
                else calibration_samples
            )
            parameters, intensity_lambdas, calibration = calibrate_capability_conditioning(
                spec=master_spec,
                capability_id=capability_id,
                profile_nuisance=nuisance,
                real_feature_summary=real_feature_summary,
                canonical_target_values=targets,
                sample_count=per_grid_samples,
                seed=_seed_for(seed, master_spec.profile_id, 100 + capability_index),
                primary_feature=primary,
            )
            if calibration["status"] != "supported":
                raise ValueError(
                    f"unsupported conditioning {task.task_id}/{capability_id}: "
                    f"max_normalized_error={calibration['max_normalized_error']}"
                )
            capability_configs[capability_id] = {
                "parameters": parameters,
                "intensity_lambdas": intensity_lambdas,
                "canonical_reference_percentile_levels": definition[
                    "reference_percentile_levels"
                ],
                "canonical_target_feature": primary,
                "canonical_target_values": targets,
                "canonical_raw_reference_quantile_values": definition[
                    "raw_reference_quantile_values"
                ],
                "calibrated_realized_strengths": calibration["realized_values"],
                "local_real_percentiles_at_canonical_targets": empirical_percentiles(
                    primary_values,
                    targets,
                ),
                "local_real_target_quantiles": local_quantiles,
                "canonical_calibration": calibration,
                "calibration_method": (
                    "frozen canonical target with pooled real-profile inverse calibration"
                ),
            }
        generator_profiles[master_spec.profile_id] = {
            "profile_id": master_spec.profile_id,
            "conditioning_role": "paper_v4_pooled_train_only_master_task",
            "dataset_name": " + ".join(source.dataset_name for source in task_sources),
            "family_id": f"pooled_{task.task_id}",
            "context_length": MAX_CONTEXT_LENGTH,
            "horizon": HORIZON,
            "target_dim": task.target_dim,
            "covariate_dim": task.covariate_dim,
            "hierarchy": task.hierarchy,
            "season_length": task.season_length,
            "feature_measurement_horizon": HORIZON,
            "frequency": task.frequency,
            "selection_weight": 1.0,
            "nuisance_parameters": nuisance,
            "controlled_feature_preconditioning": controlled_feature_preconditioning,
            "real_parameter_feature_summary": real_feature_summary,
            "split": master_split,
            "capabilities": capability_configs,
        }

    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "context_lengths": list(CONTEXT_LENGTHS),
        "max_context_length": MAX_CONTEXT_LENGTH,
        "horizon": HORIZON,
        "validation_embargo": VALIDATION_EMBARGO,
        "pairing_policy": (
            "one L=504,H=48 master task; shorter lookbacks are suffix views with "
            "the identical untouched 48-step future"
        ),
        "profile_role": (
            "real-window empirical calibration; profiles are not synthetic samples"
        ),
        "max_windows_per_source": max_windows,
        "calibration_samples_per_grid_cell": calibration_samples,
        "seed": seed,
    }
    generator_artifact = {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v2",
        "created_at": created_at,
        "config": config,
        "canonical_intensity": canonical_intensity,
        "profiles": generator_profiles,
    }
    feature_artifact = {
        "schema_version": "synthetic_v2_feature_gate_online.v1",
        "created_at": created_at,
        "config": {
            **config,
            "coverage": DEFAULT_COVERAGE,
            "support_method": "median_iqr + shrunk_robust_mahalanobis",
        },
        "buckets": feature_buckets,
    }
    near_artifact = {
        "schema_version": "synthetic_v2_near_distance_online.v2",
        "created_at": created_at,
        "source_summary_schema_version": SCHEMA_VERSION,
        "config": {
            **config,
            "artifact_reference_count": 192,
            "strict_rule": "full-window or context-only DCR below real p01",
            "combined_rule": "DCR below p05 and NNDR below p01",
        },
        "buckets": near_buckets,
    }
    mapping = build_capability_dataset_mapping(profiles)
    profile_suite = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "config": config,
        "source_inventory": inventories,
        "structured_source_profiles": [
            profile for profile in profiles if profile["task_id"] != "univariate"
        ],
        "calibration_source_profiles": profiles,
        "pool_summaries": pool_summaries,
        "split_summaries": split_summaries,
        "capability_dataset_mapping": mapping,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "profile_suite.json", profile_suite)
    write_json(output_dir / "generator_conditioning_artifact.json", generator_artifact)
    write_json(output_dir / "feature_gate_artifact.json", feature_artifact)
    write_json(output_dir / "near_distance_artifact.json", near_artifact)
    write_mapping_csv(output_dir / "capability_dataset_mapping.csv", mapping)


def build_capability_dataset_mapping(
    calibration_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    structured_by_task: dict[str, list[dict[str, Any]]] = {}
    for profile in calibration_profiles:
        structured_by_task.setdefault(str(profile["task_id"]), []).append(profile)
    univariate_datasets = [
        {
            "source_id": source.source_id,
            "dataset_name": source.dataset_name,
            "family_id": source.family_id,
            "domain": source.domain,
        }
        for source in UNIVARIATE_SOURCE_SPECS
    ]
    mapping: list[dict[str, Any]] = []
    for capability_id in ALL_CAPABILITY_IDS:
        task_id = task_id_for_capability(capability_id)
        if task_id == "univariate":
            datasets = univariate_datasets
        else:
            unique: dict[str, dict[str, Any]] = {}
            for profile in structured_by_task[task_id]:
                unique[str(profile["source_id"])] = {
                    "source_id": profile["source_id"],
                    "dataset_name": profile["dataset_name"],
                    "family_id": profile["family_id"],
                    "domain": profile["domain"],
                }
            datasets = [unique[source_id] for source_id in sorted(unique)]
        mapping.append(
            {
                "capability_id": capability_id,
                "task_id": task_id,
                "context_lengths": list(CONTEXT_LENGTHS),
                "horizon": HORIZON,
                "generator_profile_id": generator_profile_id(task_id),
                "gate_profile_ids": [
                    gate_profile_id(task_id, context_length)
                    for context_length in CONTEXT_LENGTHS
                ],
                "datasets": datasets,
            }
        )
    return mapping


def task_id_for_capability(capability_id: str) -> str:
    for task in TASK_DESIGNS.values():
        if capability_id in task.capabilities:
            return task.task_id
    raise KeyError(capability_id)


def qualify_suite(
    output_dir: Path,
    *,
    samples_per_cell: int,
    max_attempts: int,
    seed: int,
) -> dict[str, Any]:
    generator_artifact = read_json(output_dir / "generator_conditioning_artifact.json")
    feature_artifact = read_json(output_dir / "feature_gate_artifact.json")
    near_artifact = read_json(output_dir / "near_distance_artifact.json")
    accepted_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for capability_index, capability_id in enumerate(ALL_CAPABILITY_IDS):
        task = TASK_DESIGNS[task_id_for_capability(capability_id)]
        profile_id = generator_profile_id(task.task_id)
        conditioning = resolve_generator_conditioning(
            capability_id=capability_id,
            profile_id=profile_id,
            context_length=MAX_CONTEXT_LENGTH,
            horizon=HORIZON,
            target_dim=task.target_dim,
            artifact=generator_artifact,
        )
        if conditioning is None:
            raise RuntimeError(f"missing generator conditioning: {profile_id}/{capability_id}")
        for intensity in range(1, 6):
            for sample_index in range(samples_per_cell):
                base_seed = _seed_for(
                    seed,
                    f"{capability_id}:intensity:{intensity}",
                    sample_index,
                )
                accepted: dict[str, Any] | None = None
                last_rejection: dict[str, Any] | None = None
                for attempt in range(max_attempts):
                    attempt_seed = _attempt_seed(base_seed, attempt)
                    rng = np.random.default_rng(attempt_seed)
                    target, metadata, covariates = _generate_sample_values(
                        capability_id,
                        MAX_CONTEXT_LENGTH + HORIZON,
                        MAX_CONTEXT_LENGTH,
                        task.target_dim,
                        task.season_length,
                        intensity,
                        rng,
                        generator_conditioning=conditioning,
                    )
                    view_results: list[dict[str, Any]] = []
                    construction_validated = bool(
                        metadata.get("predictability", {}).get(
                            "construction_validated",
                            False,
                        )
                    )
                    all_passed = construction_validated
                    for context_length in CONTEXT_LENGTHS:
                        view_target, view_covariates = synthetic_paired_view(
                            target,
                            covariates,
                            context_length=context_length,
                            hierarchy=task.hierarchy,
                        )
                        features = _realized_features(
                            view_target,
                            view_covariates,
                            task.season_length,
                            context_length,
                        )
                        view_profile_id = gate_profile_id(task.task_id, context_length)
                        feature_gate = evaluate_feature_support_gate(
                            capability_id=capability_id,
                            features=features,
                            profile_ids=(view_profile_id,),
                            context_length=context_length,
                            horizon=HORIZON,
                            target_dim=task.target_dim,
                            artifact=feature_artifact,
                        )
                        near_gate = evaluate_near_distance_gate(
                            target=view_target,
                            features=features,
                            profile_ids=(view_profile_id,),
                            context_length=context_length,
                            horizon=HORIZON,
                            artifact=near_artifact,
                        )
                        view_passed = bool(
                            feature_gate["enforced"]
                            and feature_gate["accepted"]
                            and near_gate["enforced"]
                            and near_gate["accepted"]
                        )
                        all_passed = all_passed and view_passed
                        view_results.append(
                            {
                                "context_length": context_length,
                                "profile_id": view_profile_id,
                                "passed": view_passed,
                                "feature_gate_status": feature_gate["status"],
                                "feature_gate_normalized_score": feature_gate.get(
                                    "normalized_score"
                                ),
                                "near_distance_status": near_gate["status"],
                                "strict_risk": near_gate.get("strict_risk"),
                                "combined_risk": near_gate.get("combined_risk"),
                                "primary_feature": PRIMARY_TARGET_FEATURE[capability_id],
                                "primary_feature_value": features.get(
                                    PRIMARY_TARGET_FEATURE[capability_id]
                                ),
                            }
                        )
                    candidate = {
                        "capability_id": capability_id,
                        "task_id": task.task_id,
                        "intensity": intensity,
                        "sample_index": sample_index,
                        "base_seed": base_seed,
                        "attempt": attempt,
                        "attempt_seed": attempt_seed,
                        "construction_validated": construction_validated,
                        "all_four_views_passed": all_passed,
                        "target_sha256": array_sha256(target),
                        "future_sha256": array_sha256(
                            np.asarray(target, dtype=float)[MAX_CONTEXT_LENGTH:]
                        ),
                        "covariates_sha256": (
                            array_sha256(covariates)
                            if covariates is not None
                            else None
                        ),
                        "views": view_results,
                    }
                    if all_passed:
                        accepted = candidate
                        break
                    last_rejection = candidate
                if accepted is None:
                    failures.append(last_rejection or {
                        "capability_id": capability_id,
                        "intensity": intensity,
                        "sample_index": sample_index,
                        "reason": "no_attempt",
                    })
                else:
                    accepted_rows.append(accepted)
            print(
                f"[qualify {capability_index + 1}/{len(ALL_CAPABILITY_IDS)}] "
                f"{capability_id} intensity={intensity}",
                flush=True,
            )
    expected = len(ALL_CAPABILITY_IDS) * 5 * samples_per_cell
    by_capability = {
        capability_id: {
            "expected": 5 * samples_per_cell,
            "accepted": sum(
                row["capability_id"] == capability_id for row in accepted_rows
            ),
            "failed": sum(
                row.get("capability_id") == capability_id for row in failures
            ),
            "mean_attempts": round_float(
                np.mean(
                    [
                        row["attempt"] + 1
                        for row in accepted_rows
                        if row["capability_id"] == capability_id
                    ]
                )
                if any(
                    row["capability_id"] == capability_id for row in accepted_rows
                )
                else math.nan
            ),
        }
        for capability_id in ALL_CAPABILITY_IDS
    }
    result = {
        "schema_version": "paper_v4_nine_capability_qualification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "context_lengths": list(CONTEXT_LENGTHS),
            "horizon": HORIZON,
            "intensities": [1, 2, 3, 4, 5],
            "samples_per_cell": samples_per_cell,
            "max_attempts": max_attempts,
            "seed": seed,
            "acceptance_rule": (
                "construction predictability and both real-calibrated gates pass "
                "at all four paired lookback views"
            ),
        },
        "expected_sample_count": expected,
        "accepted_sample_count": len(accepted_rows),
        "failed_sample_count": len(failures),
        "evaluated_view_count": expected * len(CONTEXT_LENGTHS),
        "accepted_view_count": len(accepted_rows) * len(CONTEXT_LENGTHS),
        "all_nine_capabilities_qualified": (
            len(failures) == 0
            and all(row["accepted"] == row["expected"] for row in by_capability.values())
        ),
        "by_capability": by_capability,
        "accepted_samples": accepted_rows,
        "failures": failures,
    }
    write_json(output_dir / "qualification.json", result)
    (output_dir / "report.md").write_text(
        render_report(
            read_json(output_dir / "profile_suite.json"),
            result,
        ),
        encoding="utf-8",
    )
    write_manifest(output_dir)
    return result


def render_report(profile_suite: dict[str, Any], qualification: dict[str, Any]) -> str:
    lines = [
        "# Paper v4 九能力四档 lookback profile 与生成验收",
        "",
        f"- Lookback：`{CONTEXT_LENGTHS}`",
        f"- Prediction length：`H={HORIZON}`",
        "- 配对：同一条 L=504 母样本截取四个后缀视图，48 步 future 完全相同",
        f"- 合格样本：`{qualification['accepted_sample_count']}` / "
        f"`{qualification['expected_sample_count']}`",
        f"- 九能力全部合格：`{qualification['all_nine_capabilities_qualified']}`",
        "",
        "## 能力验收",
        "",
        "| 能力 | 合格/期望 | 失败 | 平均尝试次数 |",
        "|---|---:|---:|---:|",
    ]
    for capability_id, row in qualification["by_capability"].items():
        lines.append(
            f"| `{capability_id}` | {row['accepted']}/{row['expected']} | "
            f"{row['failed']} | {row['mean_attempts']} |"
        )
    lines.extend(
        [
            "",
            "## Profile 与数据集",
            "",
            "| 能力 | 数据集 |",
            "|---|---|",
        ]
    )
    for row in profile_suite["capability_dataset_mapping"]:
        datasets = ", ".join(dataset["dataset_name"] for dataset in row["datasets"])
        lines.append(f"| `{row['capability_id']}` | {datasets} |")
    return "\n".join(lines) + "\n"


def write_mapping_csv(path: Path, mapping: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "capability_id",
                "task_id",
                "context_lengths",
                "horizon",
                "generator_profile_id",
                "gate_profile_ids",
                "datasets",
            ),
        )
        writer.writeheader()
        for row in mapping:
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames if key in row},
                    "context_lengths": json.dumps(row["context_lengths"]),
                    "gate_profile_ids": json.dumps(row["gate_profile_ids"]),
                    "datasets": json.dumps(row["datasets"], ensure_ascii=False),
                }
            )


def write_manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "paper_v4_nine_capability_manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "builder_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "files": files,
        },
    )


def sample_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indices]


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def round_float(value: Any, digits: int = 8) -> float | None:
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def main() -> int:
    args = parse_args()
    if args.max_windows_per_source < 60:
        raise ValueError("max-windows-per-source must be at least 60")
    if args.calibration_samples < 4:
        raise ValueError("calibration-samples must be at least 4")
    if args.qualification_samples_per_cell < 1:
        raise ValueError("qualification-samples-per-cell must be positive")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    output_dir = args.output_dir.resolve()
    if args.stage in {"all", "build"}:
        build_suite(
            output_dir,
            data_dir=args.data_dir.resolve(),
            gift_eval_dir=args.gift_eval_dir.resolve(),
            max_windows=int(args.max_windows_per_source),
            calibration_samples=int(args.calibration_samples),
            seed=int(args.seed),
        )
    if args.stage in {"all", "qualify"}:
        qualification = qualify_suite(
            output_dir,
            samples_per_cell=int(args.qualification_samples_per_cell),
            max_attempts=int(args.max_attempts),
            seed=int(args.seed),
        )
        if not qualification["all_nine_capabilities_qualified"]:
            raise RuntimeError(
                f"qualification failed for {qualification['failed_sample_count']} samples"
            )
    print(f"paper-v4 nine-capability suite: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
