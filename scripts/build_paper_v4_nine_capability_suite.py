#!/usr/bin/env python3
"""Build dataset-local Paper v7 profiles and qualify supported generators.

The central invariant is pairing: a synthetic task is generated once at L=504,
H=48 and the L={96,168,336,504} benchmark views are suffixes of that same
master task.  Acceptance requires every view to pass its own real-calibrated
feature-support and near-distance gates.  Datasets are never pooled: each
dataset/task view owns its split, relative-intensity targets, conditioning, and
gate artifacts.  Unsupported capability cells remain explicit audit records.
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
    PAPER_GENERATOR_VERSION,
    PAPER_UNIVARIATE_CAPABILITY_IDS,
    _attempt_seed,
    _generate_sample_values,
    _realized_features,
    _regime_clock_history_incremental_r2,
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
    annotate_regime_clock_rows,
    calibrate_capability_conditioning,
    derive_profile_nuisance,
    finite_values,
    qualify_regime_reference_rows,
    summarize_real_features,
)
from build_paper_v4_profile_suite import (  # noqa: E402
    DATASET_SPECS as UNIVARIATE_DATASET_SPECS,
)
from run_synthetic_v2_near_distance_calibration import (  # noqa: E402
    BucketSpec,
    load_real_bucket,
    make_row,
    normalize_covariates,
    online_artifact_bucket,
    standardize_target,
    thresholds_from_split,
)
from synthetic_feature_profile import (  # noqa: E402
    GEFCOM2014_WIND_COVARIATE_PROVENANCE,
    GEFCOM2014_WIND_NWP_COLUMNS,
    M5_COVARIATE_PROVENANCE,
    M5_KNOWN_FUTURE_COVARIATES,
    PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS,
    PAPER_V7_GEFCOM2012_COVARIATE_PROVENANCE,
    PAPER_V7_SWISS_COVARIATE_PROVENANCE,
    PAPER_V7_SWISS_NWP_COLUMNS,
)


SCHEMA_VERSION = "paper_v7_nine_capability_suite.v1"
TASK_VIEW_ID_SEPARATOR = "::"
CONTEXT_LENGTHS = (96, 168, 336, 504)
MAX_CONTEXT_LENGTH = max(CONTEXT_LENGTHS)
HORIZON = 48
VALIDATION_EMBARGO = 48
MASTER_LOADER_HORIZON = HORIZON + VALIDATION_EMBARGO
DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_GIFT_EVAL_DIR = Path.home() / "xmy/gift-eval"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp/v7/01_nine_capability_suite"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-21-paper-v7-structured-dataset-expansion-protocol.md"
)
DEFAULT_MAX_WINDOWS_PER_DATASET = 120
DEFAULT_CALIBRATION_SAMPLES = 16
DEFAULT_QUALIFICATION_SAMPLES_PER_CELL = 8
DEFAULT_MAX_ATTEMPTS = 64
DEFAULT_SEED = 2026071804
RELATIVE_INTENSITY_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
REAL_TOLERANCE_LOWER_QUANTILE = 0.05
REAL_TOLERANCE_UPPER_QUANTILE = 0.95
REAL_TOLERANCE_UPPER_MULTIPLIER = 1.20
REAL_DIAGNOSTIC_QUANTILE_LEVELS = (0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95)
MIN_TARGET_SPAN_RELATIVE_TO_MAGNITUDE = 1e-6
MIN_ADJACENT_GAP_FRACTION_OF_SPAN = 0.02
SOURCE_ROW_AUDIT_FIELDS = (
    "native_target_dim",
    "canonical_target_dim",
    "sensitivity_target_dims",
    "target_selection_policy",
    "target_channel_indices",
    "leaf_item_ids",
    "panel_semantics",
    "zone_ids",
    "known_future_covariates",
    "covariate_provenance",
    "forecast_release_id",
    "forecast_release_valid_start",
    "forecast_release_valid_end",
    "forecast_window_valid_start",
    "forecast_available_future_steps",
    "forecast_stitching",
    "issue_time",
    "issue_time_semantics",
    "source_segment_id",
    "source_tail_excluded_steps",
    "target_column_names",
    "covariate_column_names",
    "source_frequency",
    "processed_npz_sha256",
    "processed_metadata_sha256",
    "source_provenance",
    "hierarchy_provenance",
    "benchmark_covariate_vintage_count",
    "embargo_covariate_policy",
)

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
    "regime_switching": "regime_clock_history_incremental_r2",
    "nonlinear_persistence": "nonlinear_conditional_gain",
    "predictable_intermittency": "spike_rate",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "covariate_response": "covariate_incremental_r2",
}


@dataclass(frozen=True)
class RealSource:
    dataset_id: str
    dataset_name: str
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
    native_target_dim: int | None = None
    sensitivity_target_dims: tuple[int, ...] = ()
    target_selection_policy: str | None = None
    known_future_covariates: tuple[str, ...] = ()
    covariate_provenance: str | None = None
    target_column_names: tuple[str, ...] = ()
    covariate_column_names: tuple[str, ...] = ()
    hierarchy_provenance: str | None = None


TaskViewKey = tuple[str, str]


def task_view_id(dataset_id: str, task_id: str) -> str:
    """Return the stable public identity of one dataset/task view."""

    return f"{dataset_id}{TASK_VIEW_ID_SEPARATOR}{task_id}"


def source_task_view_id(source: RealSource) -> str:
    return task_view_id(source.dataset_id, source.task_id)


def source_task_view_key(source: RealSource) -> TaskViewKey:
    """Return the collision-safe in-memory key for one source view."""

    return source.dataset_id, source.task_id


def record_task_view_id(record: dict[str, Any]) -> str:
    """Read new artifacts while remaining compatible with old artifacts."""

    stored = record.get("task_view_id")
    if stored:
        return str(stored)
    available_task_id = record.get("available_task_id", record["task_id"])
    return task_view_id(
        str(record["dataset_id"]),
        str(available_task_id),
    )


def index_sources_by_task_view(
    sources: tuple[RealSource, ...],
) -> dict[TaskViewKey, RealSource]:
    indexed: dict[TaskViewKey, RealSource] = {}
    for source in sources:
        key = source_task_view_key(source)
        if key in indexed:
            raise ValueError(
                "duplicate dataset/task view: "
                f"{source_task_view_id(source)}"
            )
        indexed[key] = source
    return indexed


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


UNIVARIATE_CALIBRATION_SOURCES = tuple(
    RealSource(
        dataset.dataset_id,
        dataset.dataset_name,
        dataset.domain,
        "univariate",
        dataset.kind,
        dataset.asset_name,
        24,
        dataset.frequency,
    )
    for dataset in UNIVARIATE_DATASET_SPECS
)

STRUCTURED_SOURCES = (
    RealSource(
        "swiss_hierarchical_demand",
        "Swiss Hierarchical Demand",
        "Energy",
        "common_factor",
        "paper_v7_swiss",
        "v7-p0-data/processed/swiss_hierarchical_demand.npz",
        48,
        "30min",
        target_dim=3,
        native_target_dim=24,
        target_selection_policy=(
            "three native meter leaves at canonical indices 0,6,12; aggregate "
            "nodes are excluded from the common-factor target"
        ),
    ),
    RealSource(
        "swiss_hierarchical_demand",
        "Swiss Hierarchical Demand",
        "Energy",
        "hierarchy",
        "paper_v7_swiss",
        "v7-p0-data/processed/swiss_hierarchical_demand.npz",
        48,
        "30min",
        target_dim=3,
        hierarchy="additive_first",
        native_target_dim=24,
        target_selection_policy="native all,S1,S2 strict hierarchy",
        target_column_names=("all", "S1", "S2"),
        hierarchy_provenance=(
            "official native Swiss hierarchy, validated before and after "
            "complete-bin 30-minute aggregation"
        ),
    ),
    RealSource(
        "swiss_hierarchical_demand",
        "Swiss Hierarchical Demand",
        "Energy",
        "covariate",
        "paper_v7_swiss",
        "v7-p0-data/processed/swiss_hierarchical_demand.npz",
        48,
        "30min",
        target_dim=1,
        covariate_dim=6,
        native_target_dim=24,
        target_selection_policy="native all aggregate as the scalar load target",
        target_column_names=("all",),
        covariate_column_names=PAPER_V7_SWISS_NWP_COLUMNS,
        known_future_covariates=PAPER_V7_SWISS_NWP_COLUMNS,
        covariate_provenance=PAPER_V7_SWISS_COVARIATE_PROVENANCE,
    ),
    RealSource(
        "gefcom2012_load",
        "GEFCom2012 Load",
        "Energy",
        "common_factor",
        "paper_v7_gefcom2012",
        "v7-p0-data/processed/gefcom2012_load.npz",
        24,
        "h",
        target_dim=3,
        native_target_dim=20,
        target_selection_policy=(
            "three synchronized native load zones at canonical indices "
            "0,9,19; total and subtotals are excluded"
        ),
        target_column_names=("zone_1", "zone_10", "zone_20"),
    ),
    RealSource(
        "gefcom2012_load",
        "GEFCom2012 Load",
        "Energy",
        "hierarchy",
        "paper_v7_gefcom2012",
        "v7-p0-data/processed/gefcom2012_load.npz",
        24,
        "h",
        target_dim=3,
        hierarchy="additive_first",
        native_target_dim=20,
        target_selection_policy=(
            "canonical total,sum_zones_1_10,sum_zones_11_20 projection"
        ),
        target_column_names=(
            "total",
            "sum_zones_1_10",
            "sum_zones_11_20",
        ),
        hierarchy_provenance=(
            "derived two-subtotal projection; official solution Zone21 "
            "validates the 20-zone total with zero residual"
        ),
    ),
    RealSource(
        "gefcom2012_load",
        "GEFCom2012 Load",
        "Energy",
        "covariate",
        "paper_v7_gefcom2012",
        "v7-p0-data/processed/gefcom2012_load.npz",
        24,
        "h",
        target_dim=1,
        covariate_dim=6,
        native_target_dim=20,
        target_selection_policy="derived total as the scalar load target",
        target_column_names=("total",),
        covariate_column_names=PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS,
        known_future_covariates=PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS,
        covariate_provenance=PAPER_V7_GEFCOM2012_COVARIATE_PROVENANCE,
    ),
    RealSource(
        "gift_ett1_h",
        "ETT1/H",
        "Energy",
        "common_factor",
        "gift_panel",
        "ett1/H",
        24,
        "h",
        target_dim=3,
        native_target_dim=7,
        sensitivity_target_dims=(7,),
        target_selection_policy=(
            "canonical first 3 channels in native Arrow order; all 7 channels "
            "are reserved for a dimension-sensitivity view"
        ),
    ),
    RealSource(
        "electricity_hourly_panel",
        "Electricity Hourly",
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
        "Web/CloudOps",
        "common_factor",
        "gift_panel",
        "bizitobs_l2c/H",
        24,
        "h",
        target_dim=3,
    ),
    RealSource(
        "m5_daily",
        "M5 Daily",
        "Retail",
        "common_factor",
        "m5_sibling_panel",
        "m5-forecasting-accuracy.zip",
        7,
        "D",
        target_dim=3,
        target_selection_policy=(
            "three disjoint item leaves sharing store_id and dept_id; "
            "aggregate rows are excluded"
        ),
    ),
    RealSource(
        "m5_daily",
        "M5 Daily",
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
        "m5_daily",
        "M5 Daily",
        "Retail",
        "covariate",
        "m5_covariate",
        "m5-forecasting-accuracy.zip",
        7,
        "D",
        covariate_dim=4,
        known_future_covariates=M5_KNOWN_FUTURE_COVARIATES,
        covariate_provenance=M5_COVARIATE_PROVENANCE,
    ),
    RealSource(
        "gefcom2014_wind",
        "GEFCom2014 Wind",
        "Energy",
        "common_factor",
        "gefcom2014_wind_panel",
        "GEFCom2014.zip",
        24,
        "h",
        target_dim=3,
        task=15,
        native_target_dim=10,
        sensitivity_target_dims=(10,),
        target_selection_policy=(
            "disjoint canonical groups of 3 synchronized wind-farm zones; "
            "all 10 zones are reserved for a dimension-sensitivity view"
        ),
    ),
    RealSource(
        "gefcom2014_wind",
        "GEFCom2014 Wind",
        "Energy",
        "covariate",
        "gefcom2014_wind_covariate",
        "GEFCom2014.zip",
        24,
        "h",
        target_dim=3,
        covariate_dim=12,
        task=0,
        native_target_dim=10,
        sensitivity_target_dims=(10,),
        target_selection_policy=(
            "same disjoint canonical 3-zone panels as common_factor"
        ),
        known_future_covariates=tuple(
            f"target_{target_index}_{column}"
            for target_index in range(3)
            for column in GEFCOM2014_WIND_NWP_COLUMNS
        ),
        covariate_provenance=GEFCOM2014_WIND_COVARIATE_PROVENANCE,
    ),
    RealSource(
        "gefcom2014_load",
        "GEFCom2014 Load",
        "Energy",
        "covariate",
        "gefcom2014_load",
        "GEFCom2014.zip",
        24,
        "h",
        covariate_dim=25,
        covariate_provenance=(
            "GEFCom2014 Load training weather columns; retained existing v6 view"
        ),
    ),
    RealSource(
        "gefcom2014_solar",
        "GEFCom2014 Solar",
        "Energy",
        "covariate",
        "gefcom2014_solar",
        "GEFCom2014.zip",
        24,
        "h",
        covariate_dim=12,
        covariate_provenance=(
            "official GEFCom2014 Solar predictor file joined by zone/timestamp"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and qualify the Paper v7 nine-capability four-lookback suite "
            "(legacy builder filename retained for compatibility)."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gift-eval-dir", type=Path, default=DEFAULT_GIFT_EVAL_DIR)
    parser.add_argument(
        "--max-windows-per-dataset",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_DATASET,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset_id subset; every selected dataset still emits all nine rows.",
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
        profile_id=(
            f"{source.dataset_id}__{source.task_id}__master_L504_H48_E48"
        ),
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
        native_target_dim=source.native_target_dim,
        sensitivity_target_dims=source.sensitivity_target_dims,
        target_selection_policy=source.target_selection_policy,
        known_future_covariates=source.known_future_covariates,
        covariate_provenance=source.covariate_provenance,
        frequency=source.frequency,
        target_column_names=source.target_column_names,
        covariate_column_names=source.covariate_column_names,
        hierarchy_provenance=source.hierarchy_provenance,
    )


def dataset_bucket_spec(source: RealSource, context_length: int) -> BucketSpec:
    task = TASK_DESIGNS[source.task_id]
    return BucketSpec(
        profile_id=gate_profile_id(
            source.dataset_id,
            source.task_id,
            context_length,
        ),
        kind=source.kind,
        asset_name=source.asset_name,
        context_length=context_length,
        horizon=HORIZON,
        stride=HORIZON,
        season_length=source.season_length,
        target_dim=source.target_dim,
        covariate_dim=source.covariate_dim,
        hierarchy=source.hierarchy,
        synthetic_capabilities=task.capabilities,
        native_target_dim=source.native_target_dim,
        sensitivity_target_dims=source.sensitivity_target_dims,
        target_selection_policy=source.target_selection_policy,
        known_future_covariates=source.known_future_covariates,
        covariate_provenance=source.covariate_provenance,
        frequency=source.frequency,
        target_column_names=source.target_column_names,
        covariate_column_names=source.covariate_column_names,
        hierarchy_provenance=source.hierarchy_provenance,
    )


def generator_profile_id(dataset_id: str, task_id: str) -> str:
    return f"{dataset_id}__{task_id}__L504_H48"


def gate_profile_id(dataset_id: str, task_id: str, context_length: int) -> str:
    return f"{dataset_id}__{task_id}__L{context_length}_H48"


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


def synthetic_view_features(
    *,
    capability_id: str,
    target: np.ndarray,
    covariates: np.ndarray | None,
    season_length: int,
    context_length: int,
    latent: dict[str, Any],
) -> dict[str, float]:
    """Measure a generated suffix view, including latent-aligned diagnostics."""

    features = _realized_features(
        target,
        covariates,
        season_length,
        context_length,
    )
    if capability_id == "regime_switching":
        view_start = MAX_CONTEXT_LENGTH - int(context_length)
        cut_points = [
            int(point) - view_start
            for point in latent["cut_points"]
            if int(point) > view_start
        ]
        features["regime_clock_history_incremental_r2"] = (
            _regime_clock_history_incremental_r2(
                target,
                context_length=context_length,
                season_length=season_length,
                cut_points=cut_points,
                dwell_length=int(latent["dwell_length"]),
            )
        )
    return features


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
        source_group = f"{source_task_view_id(source)}:{original_group}"
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
                profile_id=gate_profile_id(
                    source.dataset_id,
                    source.task_id,
                    context_length,
                ),
                context_length=context_length,
                horizon=HORIZON,
            )
            view = make_row(target, spec, covariates=covariates, label="real")
            view.update(
                {
                    "group_id": source_group,
                    "window_start": source_start + MAX_CONTEXT_LENGTH - context_length,
                    "dataset_id": source.dataset_id,
                    "task_id": source.task_id,
                    "task_view_id": source_task_view_id(source),
                    "master_row_index": row_index,
                }
            )
            view.update(
                {
                    field: row[field]
                    for field in SOURCE_ROW_AUDIT_FIELDS
                    if field in row
                }
            )
            views[context_length].append(view)
    return views, {
        "task_view_id": source_task_view_id(source),
        "dataset": {
            **asdict(source),
            "task_view_id": source_task_view_id(source),
        },
        "asset_path": relative_or_absolute(path),
        "source_audit": (
            {
                field: master_rows[0][field]
                for field in SOURCE_ROW_AUDIT_FIELDS
                if field in master_rows[0]
            }
            if master_rows
            else {}
        ),
        "master_window_count": len(master_rows),
        "master_shape": {
            "context_length": MAX_CONTEXT_LENGTH,
            "benchmark_horizon": HORIZON,
            "validation_embargo": VALIDATION_EMBARGO,
        },
    }


def dataset_three_way_split(
    rows: list[dict[str, Any]],
    spec: BucketSpec,
    *,
    seed: int,
    minimum_rows: int = 60,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Create the three leakage-protected roles inside exactly one dataset."""

    if len(rows) < minimum_rows:
        raise ValueError(
            f"{spec.profile_id} has {len(rows)} rows; at least {minimum_rows} required"
        )
    parameter, reference, calibration, split_summary = split_real_rows_three_way(
        rows,
        spec,
        calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
        gate_reference_fraction=DEFAULT_GATE_REFERENCE_FRACTION,
        seed=seed,
    )
    return (
        parameter,
        reference,
        calibration,
        {
            **split_summary,
            "policy": "dataset_local_three_way_no_pooling",
            "dataset_id": rows[0].get("dataset_id"),
            "task_id": rows[0].get("task_id"),
            "task_view_id": rows[0].get("task_view_id"),
        },
    )


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
    support["reference_control_z"] = standardized_control_vectors(
        reference,
        support,
    )
    support["calibration_control_z"] = standardized_control_vectors(
        calibration,
        support,
    )
    return config


def standardized_control_vectors(
    rows: list[dict[str, Any]],
    support: dict[str, Any],
) -> list[list[float]]:
    """Freeze the exact dataset-local real controls used by paper E1.

    E1 should consume the already split and calibrated real reference rather
    than reconstructing a potentially different split from raw source files.
    Empty vectors are retained for capabilities whose construction changes all
    available observables and therefore has no valid nuisance control.
    """

    names = tuple(str(name) for name in support["feature_names"])
    if not names:
        return [[] for _row in rows]
    center = np.asarray(support["feature_center"], dtype=float)
    scale = np.maximum(np.asarray(support["feature_scale"], dtype=float), 1e-9)
    result: list[list[float]] = []
    for row in rows:
        features = row["features"]
        values = np.asarray([float(features[name]) for name in names], dtype=float)
        normalized = (values - center) / scale
        if not np.all(np.isfinite(normalized)):
            raise ValueError("non-finite standardized control vector")
        result.append([round_float(float(value)) for value in normalized])
    return result


def source_profile(
    source: RealSource,
    context_length: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_row = rows[0] if rows else {}
    return {
        "profile_id": gate_profile_id(
            source.dataset_id,
            source.task_id,
            context_length,
        ),
        "dataset_id": source.dataset_id,
        "task_view_id": source_task_view_id(source),
        "dataset_name": source.dataset_name,
        "domain": source.domain,
        "task_id": source.task_id,
        "context_length": context_length,
        "horizon": HORIZON,
        "target_dim": source.target_dim,
        "native_target_dim": source.native_target_dim or source.target_dim,
        "sensitivity_target_dims": list(source.sensitivity_target_dims),
        "target_selection_policy": source.target_selection_policy,
        "target_column_names": list(
            first_row.get("target_column_names")
            or source.target_column_names
        ),
        "covariate_dim": source.covariate_dim,
        "covariate_column_names": list(
            first_row.get("covariate_column_names")
            or source.covariate_column_names
        ),
        "known_future_covariates": list(source.known_future_covariates),
        "covariate_provenance": source.covariate_provenance,
        "hierarchy": source.hierarchy,
        "hierarchy_provenance": (
            first_row.get("hierarchy_provenance")
            or source.hierarchy_provenance
        ),
        "processed_npz_sha256": first_row.get("processed_npz_sha256"),
        "processed_metadata_sha256": first_row.get(
            "processed_metadata_sha256"
        ),
        "season_length": source.season_length,
        "frequency": source.frequency,
        "window_count": len(rows),
        "feature_summary": summarize_real_features(rows),
    }


def intensity_policy() -> dict[str, Any]:
    return {
        "policy_id": "dataset-local-real-bounded-generator-feasible-v1",
        # Retained under the v4 artifact field name for reader compatibility;
        # these values are relative positions, not empirical percentiles.
        "percentile_levels": list(RELATIVE_INTENSITY_LEVELS),
        "relative_dose_levels": list(RELATIVE_INTENSITY_LEVELS),
        "real_tolerance": {
            "lower_quantile": REAL_TOLERANCE_LOWER_QUANTILE,
            "upper_quantile": REAL_TOLERANCE_UPPER_QUANTILE,
            "upper_multiplier": REAL_TOLERANCE_UPPER_MULTIPLIER,
        },
        "definition": (
            "For every dataset/task/capability cell, real data defines a "
            "dataset-local q05 to 1.2*q95 tolerance interval. Intensity levels "
            "1..5 are five evenly spaced relative doses inside its intersection "
            "with the generator response range. Values are not comparable "
            "across datasets."
        ),
    }


def real_tolerance_audit(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "supported": False,
            "reason_code": "no_finite_primary_feature",
            "quantiles": {},
        }
    quantile_values = np.quantile(
        finite,
        REAL_DIAGNOSTIC_QUANTILE_LEVELS,
    )
    quantiles = {
        f"q{int(round(level * 100)):02d}": round_float(value)
        for level, value in zip(
            REAL_DIAGNOSTIC_QUANTILE_LEVELS,
            quantile_values,
            strict=True,
        )
    }
    lower = float(np.quantile(finite, REAL_TOLERANCE_LOWER_QUANTILE))
    raw_upper = float(np.quantile(finite, REAL_TOLERANCE_UPPER_QUANTILE))
    tolerated_upper = REAL_TOLERANCE_UPPER_MULTIPLIER * raw_upper
    magnitude = max(abs(lower), abs(tolerated_upper), 1.0)
    minimum_span = MIN_TARGET_SPAN_RELATIVE_TO_MAGNITUDE * magnitude
    reason_code = (
        None
        if tolerated_upper - lower > minimum_span
        else "insufficient_local_real_tolerance_range"
    )
    return {
        "supported": reason_code is None,
        "reason_code": reason_code,
        "sample_count": int(finite.size),
        "lower_quantile": REAL_TOLERANCE_LOWER_QUANTILE,
        "upper_quantile": REAL_TOLERANCE_UPPER_QUANTILE,
        "upper_multiplier": REAL_TOLERANCE_UPPER_MULTIPLIER,
        "lower": round_float(lower),
        "raw_upper": round_float(raw_upper),
        "tolerated_upper": round_float(tolerated_upper),
        "span": round_float(tolerated_upper - lower),
        "minimum_span": round_float(minimum_span),
        "quantiles": quantiles,
    }


def target_spacing_audit(values: list[float]) -> dict[str, Any]:
    targets = np.asarray(values, dtype=float)
    if targets.shape != (5,) or not np.isfinite(targets).all():
        return {
            "supported": False,
            "reason_code": "missing_or_nonfinite_local_targets",
            "target_values": [round_float(value) for value in targets],
        }
    gaps = np.diff(targets)
    span = float(targets[-1] - targets[0])
    magnitude = max(float(np.max(np.abs(targets))), 1.0)
    absolute_floor = MIN_TARGET_SPAN_RELATIVE_TO_MAGNITUDE * magnitude
    adjacent_floor = max(
        absolute_floor,
        MIN_ADJACENT_GAP_FRACTION_OF_SPAN * max(span, 0.0),
    )
    reason_code: str | None = None
    if span <= absolute_floor:
        reason_code = "insufficient_local_target_range"
    elif np.any(gaps < adjacent_floor):
        reason_code = "insufficient_local_intensity_spacing"
    supported = reason_code is None
    return {
        "supported": supported,
        "reason_code": reason_code,
        "target_values": [round_float(value) for value in targets],
        "target_span": round_float(span),
        "adjacent_gaps": [round_float(value) for value in gaps],
        "minimum_target_span": round_float(absolute_floor),
        "minimum_adjacent_gap": round_float(adjacent_floor),
    }


def structural_support_audit(
    source: RealSource,
    capability_id: str,
) -> dict[str, Any]:
    reason: str | None = None
    if capability_id == "common_factor" and source.target_dim < 3:
        reason = "requires_at_least_three_synchronous_targets"
    elif capability_id == "hierarchical_coherence" and (
        source.target_dim < 3 or source.hierarchy != "additive_first"
    ):
        reason = "requires_explicit_additive_hierarchy"
    elif capability_id == "covariate_response" and source.covariate_dim < 1:
        reason = "requires_known_future_covariates"
    return {
        "supported": reason is None,
        "reason_code": reason,
        "target_dim": source.target_dim,
        "covariate_dim": source.covariate_dim,
        "known_future_covariates": list(source.known_future_covariates),
        "covariate_provenance": source.covariate_provenance,
        "hierarchy": source.hierarchy,
    }


def task_view_support_audit(
    source: RealSource,
    capability_id: str,
) -> dict[str, Any]:
    required_task_id = task_id_for_capability(capability_id)
    variable_structure = structural_support_audit(source, capability_id)
    required_structure = {
        "univariate": {
            "minimum_target_dim": 1,
            "known_future_covariates_required": False,
            "hierarchy_required": None,
        },
        "common_factor": {
            "minimum_target_dim": 3,
            "known_future_covariates_required": False,
            "hierarchy_required": None,
        },
        "hierarchy": {
            "minimum_target_dim": 3,
            "known_future_covariates_required": False,
            "hierarchy_required": "additive_first",
        },
        "covariate": {
            "minimum_target_dim": 1,
            "known_future_covariates_required": True,
            "hierarchy_required": None,
        },
    }[required_task_id]
    reason: str | None = None
    if source.task_id != required_task_id:
        reason = (
            "variable_structure_not_supported"
            if not variable_structure["supported"]
            else "missing_required_task_view"
        )
    elif not variable_structure["supported"]:
        reason = "variable_structure_not_supported"
    return {
        "supported": reason is None,
        "reason_code": reason,
        "required_task_id": required_task_id,
        "available_task_id": source.task_id,
        "required_structure": required_structure,
        "available_structure": {
            "target_dim": source.target_dim,
            "native_target_dim": source.native_target_dim or source.target_dim,
            "covariate_dim": source.covariate_dim,
            "known_future_covariates": list(source.known_future_covariates),
            "covariate_provenance": source.covariate_provenance,
            "hierarchy": source.hierarchy,
        },
        "variable_structure_audit": variable_structure,
    }


def support_matrix_row(
    source: RealSource,
    capability_id: str,
    *,
    status: str,
    reason_codes: tuple[str, ...],
    view_support: dict[int, dict[str, Any]],
    bucket_failures: dict[int, str],
    structure_audit: dict[str, Any] | None = None,
    real_tolerance: dict[str, Any] | None = None,
    target_spacing: dict[str, Any] | None = None,
    conditioning_calibration: dict[str, Any] | None = None,
    task_view_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_task_id = task_id_for_capability(capability_id)
    task_view_audit = task_view_audit or task_view_support_audit(
        source,
        capability_id,
    )
    return {
        "dataset_id": source.dataset_id,
        "task_view_id": source_task_view_id(source),
        "required_task_view_id": task_view_id(
            source.dataset_id,
            required_task_id,
        ),
        "dataset_name": source.dataset_name,
        "domain": source.domain,
        "task_id": required_task_id,
        "available_task_id": source.task_id,
        "capability_id": capability_id,
        "status": status,
        "supported": status == "supported",
        "reason_codes": list(reason_codes),
        "generator_profile_id": generator_profile_id(
            source.dataset_id,
            required_task_id,
        ),
        "gate_profile_ids": [
            gate_profile_id(source.dataset_id, required_task_id, context_length)
            for context_length in CONTEXT_LENGTHS
        ],
        "structure_audit": structure_audit,
        "task_view_audit": task_view_audit,
        "real_tolerance": real_tolerance,
        "target_spacing": target_spacing,
        "conditioning_calibration": conditioning_calibration,
        "view_support": {
            str(context_length): view_support.get(
                context_length,
                {
                    "supported": False,
                    "reason_code": "bucket_not_built",
                },
            )
            for context_length in CONTEXT_LENGTHS
        },
        "bucket_failures": {
            str(context_length): detail
            for context_length, detail in sorted(bucket_failures.items())
        },
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
    dataset_ids: tuple[str, ...] | None = None,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    all_sources = (*UNIVARIATE_CALIBRATION_SOURCES, *STRUCTURED_SOURCES)
    index_sources_by_task_view(all_sources)
    available_ids = {source.dataset_id for source in all_sources}
    unknown_ids = sorted(set(dataset_ids or ()) - available_ids)
    if unknown_ids:
        raise ValueError("unknown dataset ids: " + ", ".join(unknown_ids))
    sources = tuple(
        source
        for source in all_sources
        if dataset_ids is None or source.dataset_id in dataset_ids
    )
    if not sources:
        raise ValueError("no datasets selected")
    source_views: dict[TaskViewKey, dict[int, list[dict[str, Any]]]] = {}
    inventories: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        print(
            f"[task view {source_index + 1}/{len(sources)}] "
            f"{source_task_view_id(source)}",
            flush=True,
        )
        views, inventory = load_source_views(
            source,
            data_dir=data_dir,
            gift_eval_dir=gift_eval_dir,
            max_windows=max_windows,
        )
        source_views[source_task_view_key(source)] = views
        inventories.append(inventory)
        profiles.extend(
            source_profile(source, context_length, views[context_length])
            for context_length in CONTEXT_LENGTHS
        )

    generator_profiles: dict[str, dict[str, Any]] = {}
    feature_buckets: dict[str, dict[str, Any]] = {}
    near_buckets: dict[str, dict[str, Any]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    support_matrix: list[dict[str, Any]] = []

    for dataset_index, source in enumerate(sources):
        task = TASK_DESIGNS[source.task_id]
        for capability_id in ALL_CAPABILITY_IDS:
            if capability_id in task.capabilities:
                continue
            task_view_audit = task_view_support_audit(source, capability_id)
            support_matrix.append(
                support_matrix_row(
                    source,
                    capability_id,
                    status="unsupported",
                    reason_codes=(str(task_view_audit["reason_code"]),),
                    view_support={},
                    bucket_failures={},
                    structure_audit=task_view_audit[
                        "variable_structure_audit"
                    ],
                    task_view_audit=task_view_audit,
                )
            )
        parameter_rows: dict[int, list[dict[str, Any]]] = {}
        view_support: dict[str, dict[int, dict[str, Any]]] = {
            capability_id: {} for capability_id in task.capabilities
        }
        bucket_failures: dict[int, str] = {}
        for context_length in CONTEXT_LENGTHS:
            spec = dataset_bucket_spec(source, context_length)
            try:
                parameter, reference, calibration, split_summary = (
                    dataset_three_way_split(
                        source_views[source_task_view_key(source)][context_length],
                        spec,
                        seed=_seed_for(seed, source_task_view_id(source), 0),
                    )
                )
            except ValueError as error:
                bucket_failures[context_length] = str(error)
                for capability_id in task.capabilities:
                    view_support[capability_id][context_length] = {
                        "supported": False,
                        "reason_code": "dataset_split_failed",
                        "detail": str(error),
                    }
                continue
            parameter_rows[context_length] = parameter
            split_summaries[spec.profile_id] = split_summary
            capability_gates: dict[str, dict[str, Any]] = {}
            for capability_id in task.capabilities:
                try:
                    capability_gates[capability_id] = (
                        calibrate_feature_gate_with_rounding_guard(
                            capability_id,
                            reference,
                            calibration,
                        )
                    )
                    view_support[capability_id][context_length] = {
                        "supported": True,
                        "reason_code": None,
                        "feature_gate": "supported",
                    }
                except Exception as error:
                    view_support[capability_id][context_length] = {
                        "supported": False,
                        "reason_code": "feature_gate_calibration_failed",
                        "detail": str(error),
                    }
            feature_buckets[spec.profile_id] = {
                "profile_id": spec.profile_id,
                "dataset_id": source.dataset_id,
                "task_id": source.task_id,
                "task_view_id": source_task_view_id(source),
                "context_length": context_length,
                "horizon": HORIZON,
                "season_length": source.season_length,
                "target_dim": source.target_dim,
                "covariate_dim": source.covariate_dim,
                "known_future_covariates": list(
                    source.known_future_covariates
                ),
                "covariate_provenance": source.covariate_provenance,
                "split": split_summary,
                "capabilities": capability_gates,
            }
            try:
                thresholds, _diagnostics = thresholds_from_split(reference, calibration)
                near_buckets[spec.profile_id] = online_artifact_bucket(
                    spec,
                    sample_evenly(reference, min(192, len(reference))),
                    thresholds=thresholds,
                    split_summary=split_summary,
                )
            except Exception as error:
                bucket_failures[context_length] = str(error)
                for capability_id in task.capabilities:
                    current = view_support[capability_id][context_length]
                    current.update(
                        {
                            "supported": False,
                            "reason_code": "near_distance_calibration_failed",
                            "detail": str(error),
                        }
                    )

        master_spec = dataset_bucket_spec(source, MAX_CONTEXT_LENGTH)
        master_spec = replace(
            master_spec,
            profile_id=generator_profile_id(source.dataset_id, source.task_id),
        )
        master_parameter = parameter_rows.get(MAX_CONTEXT_LENGTH)
        if master_parameter is None:
            for capability_id in task.capabilities:
                support_matrix.append(
                    support_matrix_row(
                        source,
                        capability_id,
                        status="unsupported",
                        reason_codes=("dataset_split_failed",),
                        view_support=view_support[capability_id],
                        bucket_failures=bucket_failures,
                    )
                )
            continue
        master_gate_id = gate_profile_id(
            source.dataset_id,
            source.task_id,
            MAX_CONTEXT_LENGTH,
        )
        master_split = split_summaries[master_gate_id]
        measured = measurement_rows(
            master_parameter,
            context_length=MAX_CONTEXT_LENGTH,
            horizon=HORIZON,
            season_length=source.season_length,
        )
        real_feature_summary = summarize_real_features(measured)
        try:
            nuisance = derive_profile_nuisance(
                real_feature_summary,
                MAX_CONTEXT_LENGTH,
                source.season_length,
            )
        except Exception as error:
            for capability_id in task.capabilities:
                support_matrix.append(
                    support_matrix_row(
                        source,
                        capability_id,
                        status="unsupported",
                        reason_codes=("conditioning_profile_failed",),
                        view_support=view_support[capability_id],
                        bucket_failures=bucket_failures,
                        conditioning_calibration={
                            "status": "unsupported",
                            "reason_code": "conditioning_profile_failed",
                            "detail": str(error),
                        },
                    )
                )
            continue
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
                f"[conditioning {dataset_index + 1}/{len(sources)}] "
                f"{source.dataset_id}/{source.task_id}/{capability_id}",
                flush=True,
            )
            primary = PRIMARY_TARGET_FEATURE[capability_id]
            structure_audit = structural_support_audit(source, capability_id)
            capability_measurements = measured
            if capability_id == "regime_switching":
                annotated, audits = annotate_regime_clock_rows(
                    measured,
                    master_spec,
                )
                try:
                    _qualified_rows, qualification = qualify_regime_reference_rows(
                        annotated,
                        master_spec,
                        audits=audits,
                    )
                    structure_audit = {
                        **structure_audit,
                        "recurring_regime_qualification": {
                            "status": "detected",
                            "hard_requirement": False,
                            **qualification,
                        },
                    }
                except ValueError as error:
                    structure_audit = {
                        **structure_audit,
                        "recurring_regime_qualification": {
                            "status": "not_detected",
                            "hard_requirement": False,
                            "detail": str(error),
                        },
                    }
                # The recurring two-state clock is the synthetic stress
                # mechanism, not a structure that real data must already
                # contain.  All real parameter windows define its observable
                # primary-feature tolerance; the qualification above remains
                # diagnostic only.
                capability_measurements = annotated
            primary_values = finite_values(capability_measurements, primary)
            real_tolerance = real_tolerance_audit(primary_values)
            if not structure_audit["supported"]:
                spacing_audit = {
                    "supported": False,
                    "reason_code": None,
                    "target_values": [],
                    "status": "not_evaluated_due_to_structure",
                }
            else:
                spacing_audit = {
                    "supported": False,
                    "reason_code": None,
                    "target_values": [],
                    "status": "pending_generator_response_calibration",
                }
            view_failures = [
                row["reason_code"]
                for row in view_support[capability_id].values()
                if not row["supported"]
            ]
            pre_calibration_reasons = [
                reason
                for reason in (
                    structure_audit["reason_code"],
                    real_tolerance["reason_code"],
                    *view_failures,
                )
                if reason
            ]
            if pre_calibration_reasons:
                support_matrix.append(
                    support_matrix_row(
                        source,
                        capability_id,
                        status="unsupported",
                        reason_codes=tuple(dict.fromkeys(pre_calibration_reasons)),
                        view_support=view_support[capability_id],
                        bucket_failures=bucket_failures,
                        structure_audit=structure_audit,
                        real_tolerance=real_tolerance,
                        target_spacing=spacing_audit,
                    )
                )
                continue
            # Regime, time-varying seasonal, and nonlinear observables have
            # materially higher Monte-Carlo variance than the other six
            # capability features.
            # Doubling their fit bank is pre-registered here; the independent
            # validation bank and the common error tolerance stay unchanged.
            per_grid_samples = (
                calibration_samples * 2
                if capability_id
                in {
                    "regime_switching",
                    "time_varying_seasonality",
                    "nonlinear_persistence",
                }
                else calibration_samples
            )
            try:
                parameters, intensity_lambdas, calibration = (
                    calibrate_capability_conditioning(
                        spec=master_spec,
                        capability_id=capability_id,
                        profile_nuisance=nuisance,
                        real_feature_summary=real_feature_summary,
                        target_values=None,
                        real_tolerance_bounds=(
                            float(real_tolerance["lower"]),
                            float(real_tolerance["tolerated_upper"]),
                        ),
                        relative_dose_levels=RELATIVE_INTENSITY_LEVELS,
                        sample_count=per_grid_samples,
                        seed=_seed_for(
                            seed,
                            master_spec.profile_id,
                            100 + capability_index,
                        ),
                        primary_feature=primary,
                    )
                )
            except Exception as error:
                support_matrix.append(
                    support_matrix_row(
                        source,
                        capability_id,
                        status="unsupported",
                        reason_codes=("conditioning_calibration_failed",),
                        view_support=view_support[capability_id],
                        bucket_failures=bucket_failures,
                        structure_audit=structure_audit,
                        real_tolerance=real_tolerance,
                        target_spacing=spacing_audit,
                        conditioning_calibration={
                            "status": "unsupported",
                            "reason_code": "conditioning_calibration_failed",
                            "detail": str(error),
                        },
                    )
                )
                continue
            if calibration["status"] != "supported":
                targets = [
                    float(value)
                    for value in calibration.get("target_values", [])
                ]
                spacing_audit = (
                    target_spacing_audit(targets)
                    if targets
                    else {
                        "supported": False,
                        "reason_code": calibration.get(
                            "reason_code",
                            "conditioning_calibration_failed",
                        ),
                        "target_values": [],
                    }
                )
                calibration_reason = str(
                    calibration.get(
                        "reason_code",
                        "conditioning_calibration_failed",
                    )
                )
                support_matrix.append(
                    support_matrix_row(
                        source,
                        capability_id,
                        status="unsupported",
                        reason_codes=(calibration_reason,),
                        view_support=view_support[capability_id],
                        bucket_failures=bucket_failures,
                        structure_audit=structure_audit,
                        real_tolerance=real_tolerance,
                        target_spacing=spacing_audit,
                        conditioning_calibration=calibration,
                    )
                )
                continue
            targets = [
                float(value) for value in calibration["target_values"]
            ]
            spacing_audit = target_spacing_audit(targets)
            if not spacing_audit["supported"]:
                support_matrix.append(
                    support_matrix_row(
                        source,
                        capability_id,
                        status="unsupported",
                        reason_codes=(
                            str(spacing_audit["reason_code"]),
                        ),
                        view_support=view_support[capability_id],
                        bucket_failures=bucket_failures,
                        structure_audit=structure_audit,
                        real_tolerance=real_tolerance,
                        target_spacing=spacing_audit,
                        conditioning_calibration=calibration,
                    )
                )
                continue
            capability_configs[capability_id] = {
                "parameters": parameters,
                "intensity_lambdas": intensity_lambdas,
                "target_feature": primary,
                "target_percentile_levels": list(RELATIVE_INTENSITY_LEVELS),
                "target_relative_levels": list(RELATIVE_INTENSITY_LEVELS),
                "target_values": targets,
                "calibrated_realized_strengths": calibration["realized_values"],
                "calibration": calibration,
                "calibration_method": (
                    "dataset-local real-bounded generator-feasible inverse calibration"
                ),
            }
            support_matrix.append(
                support_matrix_row(
                    source,
                    capability_id,
                    status="supported",
                    reason_codes=(),
                    view_support=view_support[capability_id],
                    bucket_failures=bucket_failures,
                    structure_audit=structure_audit,
                    real_tolerance=real_tolerance,
                    target_spacing=spacing_audit,
                    conditioning_calibration=calibration,
                )
            )
        generator_profiles[master_spec.profile_id] = {
            "profile_id": master_spec.profile_id,
            "dataset_id": source.dataset_id,
            "task_view_id": source_task_view_id(source),
            "conditioning_role": "paper_v7_dataset_local_train_only_master_task",
            "dataset_name": source.dataset_name,
            "task_id": source.task_id,
            "context_length": MAX_CONTEXT_LENGTH,
            "horizon": HORIZON,
            "target_dim": source.target_dim,
            "native_target_dim": source.native_target_dim or source.target_dim,
            "sensitivity_target_dims": list(source.sensitivity_target_dims),
            "target_selection_policy": source.target_selection_policy,
            "covariate_dim": source.covariate_dim,
            "known_future_covariates": list(source.known_future_covariates),
            "covariate_provenance": source.covariate_provenance,
            "hierarchy": source.hierarchy,
            "season_length": source.season_length,
            "feature_measurement_horizon": HORIZON,
            "frequency": source.frequency,
            "selection_weight": 1.0,
            "nuisance_parameters": nuisance,
            "controlled_feature_preconditioning": controlled_feature_preconditioning,
            "real_parameter_feature_summary": real_feature_summary,
            "split": master_split,
            "capabilities": capability_configs,
        }

    support_matrix.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            str(row["task_view_id"]),
            ALL_CAPABILITY_IDS.index(str(row["capability_id"])),
        )
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": PAPER_GENERATOR_VERSION,
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
            "dataset-local real-window empirical calibration; no cross-dataset pooling"
        ),
        "dataset_is_independent_unit": True,
        "task_view_is_calibration_unit": True,
        "max_windows_per_dataset": max_windows,
        "calibration_samples_per_grid_cell": calibration_samples,
        "seed": seed,
    }
    generator_artifact = {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v4",
        "generator_version": PAPER_GENERATOR_VERSION,
        "created_at": created_at,
        "config": config,
        "intensity_policy": intensity_policy(),
        "profiles": generator_profiles,
    }
    feature_artifact = {
        "schema_version": "synthetic_v2_feature_gate_online.v1",
        "generator_version": PAPER_GENERATOR_VERSION,
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
        "generator_version": PAPER_GENERATOR_VERSION,
        "created_at": created_at,
        "dataset_summary_schema_version": SCHEMA_VERSION,
        "config": {
            **config,
            "artifact_reference_count": 192,
            "strict_rule": "full-window or context-only DCR below real p01",
            "combined_rule": "DCR below p05 and NNDR below p01",
        },
        "buckets": near_buckets,
    }
    profile_suite = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "config": config,
        "dataset_inventory": inventories,
        "structured_dataset_profiles": [
            profile for profile in profiles if profile["task_id"] != "univariate"
        ],
        "calibration_dataset_profiles": profiles,
        "split_summaries": split_summaries,
        "support_matrix": support_matrix,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "profile_suite.json", profile_suite)
    write_json(output_dir / "generator_conditioning_artifact.json", generator_artifact)
    write_json(output_dir / "feature_gate_artifact.json", feature_artifact)
    write_json(output_dir / "near_distance_artifact.json", near_artifact)
    write_json(
        output_dir / "dataset_capability_support_matrix.json",
        {
            # Additive task-view identity fields remain readable by existing
            # v1 consumers, including the frozen Paper E1 runner.
            "schema_version": "paper_v7_dataset_capability_support_matrix.v1",
            "created_at": created_at,
            "intensity_policy": intensity_policy(),
            "cells": support_matrix,
        },
    )
    write_support_matrix_csv(
        output_dir / "dataset_capability_support_matrix.csv",
        support_matrix,
    )


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
    support_artifact = read_json(
        output_dir / "dataset_capability_support_matrix.json"
    )
    supported_cells = [
        cell for cell in support_artifact["cells"] if cell["status"] == "supported"
    ]
    accepted_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(supported_cells):
        capability_id = str(cell["capability_id"])
        dataset_id = str(cell["dataset_id"])
        task_id = str(cell["task_id"])
        cell_task_view_id = record_task_view_id(cell)
        profile_id = str(cell["generator_profile_id"])
        profile = generator_artifact["profiles"][profile_id]
        target_dim = int(profile["target_dim"])
        season_length = int(profile["season_length"])
        hierarchy = profile.get("hierarchy")
        conditioning = resolve_generator_conditioning(
            capability_id=capability_id,
            profile_id=profile_id,
            context_length=MAX_CONTEXT_LENGTH,
            horizon=HORIZON,
            target_dim=target_dim,
            artifact=generator_artifact,
        )
        if conditioning is None:
            raise RuntimeError(
                f"support matrix claims missing conditioning: "
                f"{profile_id}/{capability_id}"
            )
        for intensity in range(1, 6):
            for sample_index in range(samples_per_cell):
                base_seed = _seed_for(
                    seed,
                    f"{dataset_id}:{task_id}:{capability_id}:intensity:{intensity}",
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
                        target_dim,
                        season_length,
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
                            hierarchy=hierarchy,
                        )
                        features = synthetic_view_features(
                            capability_id=capability_id,
                            target=view_target,
                            covariates=view_covariates,
                            season_length=season_length,
                            context_length=context_length,
                            latent=metadata,
                        )
                        view_profile_id = gate_profile_id(
                            dataset_id,
                            task_id,
                            context_length,
                        )
                        feature_gate = evaluate_feature_support_gate(
                            capability_id=capability_id,
                            features=features,
                            profile_ids=(view_profile_id,),
                            context_length=context_length,
                            horizon=HORIZON,
                            target_dim=target_dim,
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
                        "dataset_id": dataset_id,
                        "task_view_id": cell_task_view_id,
                        "capability_id": capability_id,
                        "task_id": task_id,
                        "generator_profile_id": profile_id,
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
                        "dataset_id": dataset_id,
                        "task_view_id": cell_task_view_id,
                        "capability_id": capability_id,
                        "task_id": task_id,
                        "intensity": intensity,
                        "sample_index": sample_index,
                        "reason": "no_attempt",
                    })
                else:
                    accepted_rows.append(accepted)
            print(
                f"[qualify {cell_index + 1}/{len(supported_cells)}] "
                f"{dataset_id}/{task_id}/{capability_id} intensity={intensity}",
                flush=True,
            )
    expected = len(supported_cells) * 5 * samples_per_cell
    by_cell = {
        f"{record_task_view_id(cell)}::{cell['capability_id']}": {
            "dataset_id": cell["dataset_id"],
            "task_view_id": record_task_view_id(cell),
            "task_id": cell["task_id"],
            "capability_id": cell["capability_id"],
            "expected": 5 * samples_per_cell,
            "accepted": sum(
                row["dataset_id"] == cell["dataset_id"]
                and row["task_id"] == cell["task_id"]
                and row["capability_id"] == cell["capability_id"]
                for row in accepted_rows
            ),
            "failed": sum(
                row.get("dataset_id") == cell["dataset_id"]
                and row.get("task_id") == cell["task_id"]
                and row.get("capability_id") == cell["capability_id"]
                for row in failures
            ),
            "mean_attempts": round_float(
                np.mean(
                    [
                        row["attempt"] + 1
                        for row in accepted_rows
                        if row["dataset_id"] == cell["dataset_id"]
                        and row["task_id"] == cell["task_id"]
                        and row["capability_id"] == cell["capability_id"]
                    ]
                )
                if any(
                    row["dataset_id"] == cell["dataset_id"]
                    and row["task_id"] == cell["task_id"]
                    and row["capability_id"] == cell["capability_id"]
                    for row in accepted_rows
                )
                else math.nan
            ),
        }
        for cell in supported_cells
    }
    by_capability = {
        capability_id: {
            "supported_dataset_count": sum(
                cell["capability_id"] == capability_id for cell in supported_cells
            ),
            "unsupported_dataset_count": sum(
                cell["capability_id"] == capability_id
                and cell["status"] != "supported"
                for cell in support_artifact["cells"]
            ),
            "expected": sum(
                row["capability_id"] == capability_id
                for row in by_cell.values()
            )
            * 5
            * samples_per_cell,
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
        "schema_version": "paper_v7_nine_capability_qualification.v1",
        "generator_version": PAPER_GENERATOR_VERSION,
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
        "supported_cell_count": len(supported_cells),
        "unsupported_cell_count": len(support_artifact["cells"]) - len(supported_cells),
        "accepted_sample_count": len(accepted_rows),
        "failed_sample_count": len(failures),
        "evaluated_view_count": expected * len(CONTEXT_LENGTHS),
        "accepted_view_count": len(accepted_rows) * len(CONTEXT_LENGTHS),
        "all_supported_cells_qualified": (
            len(failures) == 0
            and all(row["accepted"] == row["expected"] for row in by_cell.values())
        ),
        "by_cell": by_cell,
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
        "# Paper v7 dataset-local 九能力四档 lookback profile 与生成验收",
        "",
        f"- Lookback：`{CONTEXT_LENGTHS}`",
        f"- Prediction length：`H={HORIZON}`",
        "- 配对：同一条 L=504 母样本截取四个后缀视图，48 步 future 完全相同",
        "- 校准：每个 dataset/task 独立三路切分，不做跨数据集 pooling",
        "- 强度：dataset-local p10/p30/p50/p70/p90 相对强度，不跨数据集比较绝对值",
        f"- Supported cells：`{qualification['supported_cell_count']}`",
        f"- Unsupported cells：`{qualification['unsupported_cell_count']}`",
        f"- 合格样本：`{qualification['accepted_sample_count']}` / "
        f"`{qualification['expected_sample_count']}`",
        f"- 所有 supported cells 合格："
        f"`{qualification['all_supported_cells_qualified']}`",
        "",
        "## 能力验收",
        "",
        "| 能力 | 支持/不支持 dataset cells | 合格/期望 | 失败 |",
        "|---|---:|---:|---:|",
    ]
    for capability_id, row in qualification["by_capability"].items():
        lines.append(
            f"| `{capability_id}` | {row['supported_dataset_count']}/"
            f"{row['unsupported_dataset_count']} | "
            f"{row['accepted']}/{row['expected']} | {row['failed']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset × task × capability 支持矩阵",
            "",
            "| Dataset | Task | 能力 | 状态 | 原因 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in profile_suite["support_matrix"]:
        reasons = ", ".join(row["reason_codes"]) or "-"
        lines.append(
            f"| `{row['dataset_id']}` | `{row['task_id']}` | "
            f"`{row['capability_id']}` | `{row['status']}` | {reasons} |"
        )
    return "\n".join(lines) + "\n"


def write_support_matrix_csv(
    path: Path,
    support_matrix: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset_id",
                "task_view_id",
                "required_task_view_id",
                "dataset_name",
                "domain",
                "task_id",
                "available_task_id",
                "capability_id",
                "status",
                "reason_codes",
                "task_view_audit",
                "generator_profile_id",
                "gate_profile_ids",
                "target_feature",
                "target_relative_levels",
                "real_tolerance_lower",
                "real_tolerance_upper",
                "target_values",
            ),
        )
        writer.writeheader()
        for row in support_matrix:
            target_spacing = row.get("target_spacing") or {}
            writer.writerow(
                {
                    "dataset_id": row["dataset_id"],
                    "task_view_id": record_task_view_id(row),
                    "required_task_view_id": row.get(
                        "required_task_view_id",
                        task_view_id(
                            str(row["dataset_id"]),
                            str(row["task_id"]),
                        ),
                    ),
                    "dataset_name": row["dataset_name"],
                    "domain": row["domain"],
                    "task_id": row["task_id"],
                    "available_task_id": row["available_task_id"],
                    "capability_id": row["capability_id"],
                    "status": row["status"],
                    "reason_codes": json.dumps(row["reason_codes"]),
                    "task_view_audit": json.dumps(row["task_view_audit"]),
                    "generator_profile_id": row["generator_profile_id"],
                    "gate_profile_ids": json.dumps(row["gate_profile_ids"]),
                    "target_feature": PRIMARY_TARGET_FEATURE[row["capability_id"]],
                    "target_relative_levels": json.dumps(
                        RELATIVE_INTENSITY_LEVELS
                    ),
                    "real_tolerance_lower": (
                        (row.get("real_tolerance") or {}).get("lower")
                    ),
                    "real_tolerance_upper": (
                        (row.get("real_tolerance") or {}).get(
                            "tolerated_upper"
                        )
                    ),
                    "target_values": json.dumps(
                        target_spacing.get("target_values", [])
                    ),
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
            "schema_version": "paper_v7_nine_capability_manifest.v1",
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
    if args.max_windows_per_dataset < 60:
        raise ValueError("max-windows-per-dataset must be at least 60")
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
            max_windows=int(args.max_windows_per_dataset),
            calibration_samples=int(args.calibration_samples),
            seed=int(args.seed),
            dataset_ids=tuple(args.datasets) if args.datasets else None,
        )
    if args.stage in {"all", "qualify"}:
        qualification = qualify_suite(
            output_dir,
            samples_per_cell=int(args.qualification_samples_per_cell),
            max_attempts=int(args.max_attempts),
            seed=int(args.seed),
        )
        if not qualification["all_supported_cells_qualified"]:
            raise RuntimeError(
                "qualification failed for "
                f"{qualification['failed_sample_count']} supported-cell samples"
            )
    print(f"paper-v7 nine-capability suite: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
