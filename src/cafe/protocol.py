from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

from cafe.analysis.metrics import seasonal_period_for_frequency
from cafe.generation.normalization import (
    normalize_covariates,
    standardize_by_context,
    standardize_hierarchy_by_context,
)
from cafe.calibration.conditioning import (
    GeneratorConditioning,
    REAL_BOUNDED_INTENSITY_POLICY_ID,
)
from cafe.generation.families import (
    CROSS_SERIES_MIN_INCREMENTAL_HOLDOUT_GAIN,
    GENERATOR_VERSION,
    PRIMARY_FAMILY_BY_CAPABILITY,
    REQUIRED_REAL_FEATURES_BY_CAPABILITY,
    SECONDARY_FAMILY_BY_CAPABILITY,
    add_observation_noise_to_history,
    derive_deterministic_parameters,
    generate_deterministic_sample,
    standardize_common_factor_counterfactual_member,
    standardize_cross_series_counterfactual_member,
)
from cafe.validation.mechanisms import (
    structural_calibration_member_gate,
    summarize_structural_calibration_reachability,
)
from cafe.data.imputation import impute_observed_window
from cafe.features.profile import (
    FEATURE_SCHEMA_VERSION,
    cafe_feature_vector,
)
from cafe.data.real import (
    RealSeriesRecord,
    load_real_dataset,
)
from cafe.features.primitives import (
    adjusted_r2,
    file_sha256,
    robust_scale,
)


SCHEMA_VERSION = "cafe.pipeline.v1"
REAL_CALIBRATION_CONTEXT_LENGTH = 168
CONTEXT_LENGTH = 336
HORIZON = 48
MASTER_LENGTH = CONTEXT_LENGTH + HORIZON
REAL_FORECAST_MASTER_LENGTH = REAL_CALIBRATION_CONTEXT_LENGTH + HORIZON
VIEW_CONTEXT_LENGTHS = (96, 168, 336)
FIXED_CONTEXT_LENGTH = 168
MIN_REAL_FEATURE_COUNT = 12
INTENSITIES = (1, 2, 3, 4, 5)
QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CALIBRATION_SAMPLE_SEED = 2026072401
GENERATION_PATH_SEED = 2026072403
QUALIFICATION_PATH_SEED = 2026072601
ROBUSTNESS_SEED = 2026072404
ROBUSTNESS_NOISE_RATIO = 0.15
DEFAULT_CALIBRATION_PATH_COUNT = 32
MAX_CALIBRATION_PATH_COUNT = 96
CALIBRATION_PATH_EXPANSION_STEP = 32
NONLINEAR_FEATURE_LAG_FRACTIONS = (
    1.0 / 6.0,
    1.0 / 5.0,
    1.0 / 4.0,
    1.0 / 3.0,
    1.0 / 2.0,
)
NONLINEAR_FEATURE_MAX_LAG = 32
COUNTERFACTUAL_CAPABILITIES = frozenset(
    {
        "common_factor",
        "cross_series_dependence",
        "covariate_response",
    }
)
MAIN_COUNTERFACTUAL_CAPABILITIES = frozenset({"covariate_response"})
STRICT_COUNTERFACTUAL_CAPABILITIES = frozenset(
    {"common_factor", "cross_series_dependence"}
)
PRIMARY_MECHANISM_COUNTERFACTUAL_CAPABILITIES = (
    STRICT_COUNTERFACTUAL_CAPABILITIES
)
INPUT_ABLATION_CAPABILITIES = STRICT_COUNTERFACTUAL_CAPABILITIES
STRUCTURAL_CAPABILITIES = frozenset(
    {
        "common_factor",
        "hierarchical_coherence",
        "cross_series_dependence",
        "covariate_response",
    }
)
GIFT_EVAL_REAL_DATA_ADAPTERS = frozenset(
    {"gift_arrow", "gift_hierarchical_sales"}
)
REAL_INTENSITY_SOURCE_BY_CAPABILITY = {
    "trend": ("features", "real_univariate"),
    "multi_seasonal": ("features", "real_univariate"),
    "time_varying_seasonality": ("features", "real_univariate"),
    "regime_switching": ("features", "real_univariate"),
    "nonlinear_persistence": ("features", "real_univariate"),
    "predictable_intermittency": ("features", "real_univariate"),
    "common_factor": (
        "native_multivariate_features",
        "real_native_multivariate",
    ),
    "hierarchical_coherence": (
        "hierarchy_children_features",
        "real_hierarchy_children",
    ),
    "cross_series_dependence": (
        "native_multivariate_features",
        "real_native_multivariate",
    ),
    "covariate_response": (
        "known_future_covariate_features",
        "real_known_future_covariates",
    ),
}
MINIMUM_REAL_OVERLAP_FRACTION = 0.10
MINIMUM_MAPPED_LAMBDA_SPAN_FRACTION = 0.25
MINIMUM_ADJACENT_NON_DECREASING_PATH_FRACTION = 0.75
MINIMUM_PAIRED_OBSERVABLE_RESPONSE_Z = 3.0
OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY = {
    capability_id: {
        "minimum_adjacent_non_decreasing_path_fraction": (
            MINIMUM_ADJACENT_NON_DECREASING_PATH_FRACTION
        ),
        "minimum_paired_standardized_separation": (
            MINIMUM_PAIRED_OBSERVABLE_RESPONSE_Z
        ),
    }
    for capability_id in (
        "multi_seasonal",
        "time_varying_seasonality",
        "regime_switching",
        "nonlinear_persistence",
        "predictable_intermittency",
        "common_factor",
        "cross_series_dependence",
    )
}


CAPABILITIES = tuple(PRIMARY_FAMILY_BY_CAPABILITY)
PREPARATION_CAPABILITY_PRIORITY = (
    "cross_series_dependence",
    "common_factor",
    "nonlinear_persistence",
    "hierarchical_coherence",
    "covariate_response",
    "multi_seasonal",
    "regime_switching",
    "predictable_intermittency",
    "time_varying_seasonality",
    "trend",
)
PRIMARY_TARGET_FEATURE = {
    "trend": "local_polynomial_energy_share_w96",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "regime_sparse_transition_score",
    # A fixed history-only lag search is shared by real and synthetic windows.
    # Generator-known strength remains a separate construction diagnostic.
    "nonlinear_persistence": "nonlinear_conditional_effect_size",
    # Positive residual energy after a fixed local baseline is observable
    # without event labels. Clock recoverability remains a separate
    # predictability diagnostic and the generator-known energy share remains a
    # construction audit.
    "predictable_intermittency": (
        "event_positive_residual_energy_share"
    ),
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    # Dose is the additional history-only predictive value contributed by
    # other channels. Peak lag correlation remains a structural diagnostic,
    # but is not allowed to saturate and stand in for forecast utilization.
    "cross_series_dependence": "cross_series_incremental_r2",
    # This history-only incremental R² is computed identically on real and
    # synthetic target/covariate histories. Counterfactual response remains the
    # mechanism gate rather than entering intensity calibration.
    "covariate_response": "covariate_incremental_r2",
}


def preparation_capability_order(
    capability_ids: Iterable[str],
) -> tuple[str, ...]:
    priority = {
        capability_id: index
        for index, capability_id in enumerate(
            PREPARATION_CAPABILITY_PRIORITY
        )
    }
    return tuple(
        sorted(
            capability_ids,
            key=lambda capability_id: (
                priority.get(capability_id, len(priority)),
                capability_id,
            ),
        )
    )
TARGET_DIM_BY_CAPABILITY = {
    capability: (
        5
        if capability == "common_factor"
        else (
            3
            if capability
            in {
                "hierarchical_coherence",
                "cross_series_dependence",
            }
            else 1
        )
    )
    for capability in CAPABILITIES
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    logical_name: str
    config_id: str
    asset_name: str
    domain: str
    task_view_id: str = "univariate_background"
    real_data_adapter: str = "gift_arrow"


DATASET_REGISTRY = {
    spec.dataset_id: spec
    for spec in (
        DatasetSpec(
            "gift_electricity_h",
            "Electricity",
            "electricity/H",
            "electricity/H",
            "Energy",
        ),
        DatasetSpec(
            "gift_solar_h",
            "Solar",
            "solar/H",
            "solar/H",
            "Energy",
        ),
        DatasetSpec(
            "gift_ett1_h",
            "ETT1",
            "ett1/H",
            "ett1/H",
            "Energy",
        ),
        DatasetSpec(
            "gift_ett2_h",
            "ETT2",
            "ett2/H",
            "ett2/H",
            "Energy",
        ),
        DatasetSpec(
            "gift_jena_weather_h",
            "Jena Weather",
            "jena_weather/H",
            "jena_weather/H",
            "Nature",
        ),
        DatasetSpec(
            "gift_kdd_cup_h",
            "KDD Cup 2018",
            "kdd_cup_2018_with_missing/H",
            "kdd_cup_2018_with_missing/H",
            "Nature",
        ),
        DatasetSpec(
            "gift_loop_seattle_h",
            "Loop Seattle",
            "LOOP_SEATTLE/H",
            "LOOP_SEATTLE/H",
            "Transport",
        ),
        DatasetSpec(
            "gift_sz_taxi_h",
            "SZ-Taxi",
            "SZ_TAXI/H",
            "SZ_TAXI/H",
            "Transport",
        ),
        DatasetSpec(
            "gift_m_dense_h",
            "M_DENSE",
            "M_DENSE/H",
            "M_DENSE/H",
            "Transport",
        ),
        DatasetSpec(
            "gift_bitbrains_fast_h",
            "Bitbrains Fast Storage",
            "bitbrains_fast_storage/H",
            "bitbrains_fast_storage/H",
            "Web/CloudOps",
        ),
        DatasetSpec(
            "gift_bitbrains_rnd_h",
            "Bitbrains RND",
            "bitbrains_rnd/H",
            "bitbrains_rnd/H",
            "Web/CloudOps",
        ),
        DatasetSpec(
            "gift_bizitobs_l2c_h",
            "BizITObs L2C",
            "bizitobs_l2c/H",
            "bizitobs_l2c/H",
            "Web/CloudOps",
        ),
        DatasetSpec(
            "gift_bizitobs_application",
            "BizITObs Application",
            "bizitobs_application",
            "bizitobs_application",
            "Web/CloudOps",
        ),
        DatasetSpec(
            "gift_bizitobs_service",
            "BizITObs Service",
            "bizitobs_service",
            "bizitobs_service",
            "Web/CloudOps",
        ),
        DatasetSpec(
            "gift_restaurant_d",
            "Restaurant",
            "restaurant",
            "restaurant",
            "Business",
        ),
        DatasetSpec(
            "gift_hierarchical_sales_d",
            "Hierarchical Sales",
            "hierarchical_sales/D",
            "hierarchical_sales/D",
            "Business",
            task_view_id="hierarchy_sibling_pairs",
            real_data_adapter="gift_hierarchical_sales",
        ),
        DatasetSpec(
            "gift_m4_hourly",
            "M4 Hourly",
            "m4_hourly",
            "m4_hourly",
            "Mixed",
        ),
        DatasetSpec(
            "gift_us_births_d",
            "US Births",
            "us_births/D",
            "us_births/D",
            "Nature",
        ),
        DatasetSpec(
            "gift_saugeenday_d",
            "Saugeen River Flow",
            "saugeenday/D",
            "saugeenday/D",
            "Nature",
        ),
        DatasetSpec(
            "gift_temperature_rain_d",
            "Temperature Rain",
            "temperature_rain_with_missing",
            "temperature_rain_with_missing",
            "Nature",
        ),
        DatasetSpec(
            "fev_ett_1h",
            "FEV ETT",
            "fev/ETT_1H",
            "ETT_1H",
            "Energy",
            task_view_id="native_multivariate_targets",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_jena_weather_1h",
            "FEV Jena Weather",
            "fev/jena_weather_1H",
            "jena_weather_1H",
            "Nature",
            task_view_id="native_multivariate_targets",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_boomlet_1282",
            "FEV BOOMLET 1282",
            "fev/boomlet_1282",
            "boomlet_1282",
            "Web/CloudOps",
            task_view_id="native_multivariate_targets",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_uci_air_quality_1h",
            "FEV UCI Air Quality",
            "fev/uci_air_quality_1H",
            "uci_air_quality_1H",
            "Nature",
            task_view_id=(
                "native_multivariate_targets_known_future_covariates"
            ),
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_solar_with_weather_1h",
            "FEV Solar with Weather",
            "fev/solar_with_weather_1H",
            "solar_with_weather_1H",
            "Energy",
            task_view_id="univariate_known_future_covariates",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_proenfo_gfc14",
            "FEV GFC14",
            "fev/proenfo_gfc14",
            "proenfo_gfc14",
            "Energy",
            task_view_id="univariate_known_future_covariates",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_rohlik_orders_1d",
            "FEV Rohlik Orders",
            "fev/rohlik_orders_1D",
            "rohlik_orders_1D",
            "Business",
            task_view_id="univariate_known_future_covariates",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_rossmann_1d",
            "FEV Rossmann",
            "fev/rossmann_1D",
            "rossmann_1D",
            "Business",
            task_view_id="univariate_known_future_covariates",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "fev_hospital_admissions_1d",
            "FEV Hospital Admissions",
            "fev/hospital_admissions_1D",
            "hospital_admissions_1D",
            "Healthcare",
            real_data_adapter="fev_parquet",
        ),
        DatasetSpec(
            "m5_daily",
            "M5 Forecasting Accuracy",
            "m5/D",
            ".",
            "Business",
            real_data_adapter="m5_csv",
        ),
    )
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(*parts: Any, base: int = 0) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int((base + int.from_bytes(digest[:8], "big")) % (2**32 - 1))


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def resolve_dataset(dataset_id: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as error:
        raise ValueError(
            f"unknown cafe dataset {dataset_id!r}; registered="
            f"{sorted(DATASET_REGISTRY)}"
        ) from error


def expand_native_records(
    records: Iterable[tuple[str, np.ndarray]],
) -> list[tuple[str, str, int, np.ndarray]]:
    output: list[tuple[str, str, int, np.ndarray]] = []
    for item_id, values in records:
        array = np.asarray(values, dtype=float)
        channels = array if array.ndim == 2 else array.reshape(1, -1)
        for channel_index, channel in enumerate(channels):
            series_id = (
                str(item_id)
                if array.ndim == 1
                else f"{item_id}:dim:{channel_index}"
            )
            output.append(
                (
                    series_id,
                    str(item_id),
                    int(channel_index),
                    np.asarray(channel, dtype=float),
                )
            )
    return output


def nonoverlapping_strata(
    series_length: int,
    *,
    window_length: int = REAL_FORECAST_MASTER_LENGTH,
) -> list[tuple[int, int]]:
    capacity = int(series_length) // int(window_length)
    if capacity <= 0:
        return []
    boundaries = np.floor(
        np.linspace(0, int(series_length), capacity + 1)
    ).astype(int)
    strata: list[tuple[int, int]] = []
    for index in range(capacity):
        segment_start = int(boundaries[index])
        latest_start = int(boundaries[index + 1] - window_length)
        if latest_start < segment_start:
            raise ValueError("invalid cafe calibration stratum")
        strata.append((segment_start, latest_start))
    return strata


def standardize_history(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(array))
    scale = float(np.std(array))
    if not math.isfinite(scale) or scale <= 1e-10:
        raise ValueError("uninformative calibration window")
    return (array - mean) / scale, mean, scale


def _v8_nonlinear_feature_inputs(
    values: np.ndarray,
    season_length: int,
) -> tuple[np.ndarray, int, int]:
    series = np.asarray(values, dtype=float).reshape(-1)
    series = series[np.isfinite(series)]
    if series.size < 12:
        return np.asarray([], dtype=float), 0, 0
    series = robust_scale(series)
    seasonal_lag = int(
        np.clip(
            max(4, int(round(season_length))),
            1,
            max(1, series.size - 2),
        )
    )
    maximum_lag = min(
        NONLINEAR_FEATURE_MAX_LAG,
        max(2, series.size // 4),
    )
    return series, seasonal_lag, maximum_lag


def _v8_nonlinear_gain_at_lag(
    series: np.ndarray,
    seasonal_lag: int,
    nonlinear_lag: int,
) -> float:
    start = max(seasonal_lag, nonlinear_lag, 1)
    if series.size - start < 8:
        return math.nan
    target = series[start:]
    lag1 = series[start - 1 : -1]
    lag_seasonal = series[: series.size - seasonal_lag]
    if lag_seasonal.size > target.size:
        lag_seasonal = lag_seasonal[-target.size :]
    lag_nonlinear = series[
        start - nonlinear_lag : series.size - nonlinear_lag
    ]
    linear = np.column_stack(
        [
            np.ones_like(target),
            lag1,
            lag_seasonal,
            lag_nonlinear,
        ]
    )
    nonlinear = np.column_stack(
        [
            linear,
            lag_nonlinear**2,
            lag_nonlinear**3,
        ]
    )
    return float(
        adjusted_r2(target, nonlinear)
        - adjusted_r2(target, linear)
    )


def cafe_nonlinear_conditional_gain(
    values: np.ndarray,
    season_length: int,
) -> tuple[float, int, tuple[int, ...]]:
    """Measure generic nonlinear persistence over plausible relative lags.

    The pre-v4 proxy fixed the nonlinear lag to ``season_length // 2`` and
    added one sine-squared term.  CaFE generators intentionally randomize their
    actual lag and use either shifted-tanh or rational responses, so that proxy
    could react more strongly at I3 than I5 merely because a correlated,
    incorrect lag happened to fit one trajectory.

    Search a small, fixed set of season-relative lags and use quadratic/cubic
    terms.  This remains family-neutral and is also available for real windows,
    while covering the smooth bounded response families used by CaFE without
    consulting generator metadata.
    """

    series, seasonal_lag, maximum_lag = _v8_nonlinear_feature_inputs(
        values,
        season_length,
    )
    if not series.size:
        return 0.0, 0, ()
    candidate_lags = tuple(
        sorted(
            {
                int(
                    np.clip(
                        round(seasonal_lag * fraction),
                        2,
                        maximum_lag,
                    )
                )
                for fraction in NONLINEAR_FEATURE_LAG_FRACTIONS
            }
        )
    )
    best_gain = -math.inf
    best_lag = 0
    for nonlinear_lag in candidate_lags:
        gain = _v8_nonlinear_gain_at_lag(
            series,
            seasonal_lag,
            nonlinear_lag,
        )
        if math.isfinite(gain) and gain > best_gain:
            best_gain = gain
            best_lag = nonlinear_lag
    if not math.isfinite(best_gain):
        return 0.0, 0, candidate_lags
    return best_gain, best_lag, candidate_lags


def cafe_nonlinear_actual_lag_gain(
    values: np.ndarray,
    season_length: int,
    nonlinear_lag: int,
) -> float:
    """Return the family-neutral nonlinear gain at a known generated lag."""

    series, seasonal_lag, maximum_lag = _v8_nonlinear_feature_inputs(
        values,
        season_length,
    )
    if not series.size:
        return 0.0
    lag = int(
        np.clip(
            int(nonlinear_lag),
            2,
            maximum_lag,
        )
    )
    gain = _v8_nonlinear_gain_at_lag(
        series,
        seasonal_lag,
        lag,
    )
    return gain if math.isfinite(gain) else 0.0


def calibration_period_policy(
    frequency: str,
    standardized_history: np.ndarray,
) -> dict[str, Any]:
    """Separate calendar, feature, generator-profile, and MASE periods.

    A sub-daily calendar season may be much longer than L168.  In that case it
    remains provenance, the real-window spectral peak supplies an observable
    feature period, and non-seasonal MASE uses lag one.
    """

    history = np.asarray(standardized_history, dtype=float).reshape(-1)
    calendar_period = int(seasonal_period_for_frequency(frequency))
    provisional = cafe_feature_vector(
        history[:, None],
        None,
    )
    raw_dominant = float(provisional.get("dominant_period", 24.0))
    if not math.isfinite(raw_dominant) or raw_dominant <= 0.0:
        raw_dominant = 24.0
    profile_period = int(
        round(
            np.clip(
                raw_dominant,
                8.0,
                REAL_CALIBRATION_CONTEXT_LENGTH / 3.0,
            )
        )
    )
    calendar_feature_observable = bool(
        calendar_period >= 2
        and 2 * calendar_period <= REAL_CALIBRATION_CONTEXT_LENGTH
    )
    feature_period = (
        calendar_period
        if calendar_feature_observable
        else profile_period
    )
    mase_period = (
        calendar_period
        if 1 <= calendar_period < REAL_CALIBRATION_CONTEXT_LENGTH
        else 1
    )
    return {
        "calendar_season_length": calendar_period,
        "calendar_season_feature_observable": (
            calendar_feature_observable
        ),
        "calendar_cycles_in_calibration_history": float(
            REAL_CALIBRATION_CONTEXT_LENGTH / max(calendar_period, 1)
        ),
        "raw_profile_dominant_period": raw_dominant,
        "profile_period": profile_period,
        "feature_period": int(feature_period),
        "feature_period_source": (
            "calendar_season"
            if calendar_feature_observable
            else "observable_profile_dominant_period"
        ),
        "mase_period": int(mase_period),
        "mase_period_source": (
            "calendar_season"
            if mase_period == calendar_period
            else "nonseasonal_lag1_calendar_unobservable_in_l168"
        ),
    }


def summarize_feature_rows(
    rows: Iterable[dict[str, float]],
) -> dict[str, dict[str, float]]:
    materialized = list(rows)
    names = sorted({name for row in materialized for name in row})
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray(
            [
                float(row[name])
                for row in materialized
                if name in row and math.isfinite(float(row[name]))
            ],
            dtype=float,
        )
        if not values.size:
            continue
        quantiles = np.quantile(values, QUANTILE_LEVELS)
        summary[name] = {
            f"p{int(round(level * 100)):02d}": float(value)
            for level, value in zip(QUANTILE_LEVELS, quantiles, strict=True)
        }
        summary[name]["finite_count"] = int(values.size)
        summary[name]["minimum"] = float(np.min(values))
        summary[name]["maximum"] = float(np.max(values))
    return summary


def _real_forecast_mase_scale(
    standardized_history: np.ndarray,
    requested_period: int,
) -> tuple[int, float]:
    history = np.asarray(standardized_history, dtype=float).reshape(-1)
    period = int(requested_period)
    if not 1 <= period < history.size:
        period = 1
    scale = float(np.mean(np.abs(history[period:] - history[:-period])))
    if not math.isfinite(scale) or scale <= 1e-12:
        period = 1
        scale = float(np.mean(np.abs(np.diff(history))))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("real forecast anchor has no valid MASE scale")
    return period, scale


def _native_multivariate_features(
    values: np.ndarray,
    *,
    start: int,
    feature_period: int,
    minimum_observed_fraction: float,
) -> dict[str, float]:
    native = np.asarray(values, dtype=float)
    if native.ndim != 2 or native.shape[0] < 2:
        return {}
    history = native[
        :,
        start : start + REAL_CALIBRATION_CONTEXT_LENGTH,
    ].T
    if history.shape != (
        REAL_CALIBRATION_CONTEXT_LENGTH,
        native.shape[0],
    ):
        return {}
    standardized_channels: list[np.ndarray] = []
    for channel in range(min(history.shape[1], 5)):
        imputed, _observed = impute_observed_window(
            history[:, channel],
            minimum_observed_fraction=minimum_observed_fraction,
        )
        if imputed is None:
            continue
        try:
            standardized, _location, _scale = standardize_history(imputed)
        except ValueError:
            continue
        standardized_channels.append(standardized)
    if len(standardized_channels) < 2:
        return {}
    panel = np.column_stack(standardized_channels)
    features = cafe_feature_vector(
        panel,
        feature_period,
        include_cross_series_predictability=False,
    )
    if panel.shape[1] >= 2:
        cross_features = cafe_feature_vector(
            panel[:, : min(panel.shape[1], 3)],
            feature_period,
            include_cross_series_predictability=True,
            cross_series_max_lag=24,
        )
        for name in (
            "lead_lag_peak_abs",
            "lead_lag_peak_lag_abs",
            "cross_series_incremental_r2",
            "cross_series_effect_memory",
        ):
            if name in cross_features:
                features[name] = cross_features[name]
    return features


def _hierarchy_children_feature_row(
    children_history: np.ndarray,
    *,
    feature_period: int,
) -> dict[str, float]:
    """Build hierarchy coordinates from children and a constructed parent."""

    children = np.asarray(children_history, dtype=float)
    if (
        children.ndim != 2
        or children.shape[0] != REAL_CALIBRATION_CONTEXT_LENGTH
        or children.shape[1] < 2
    ):
        return {}
    selected_children = children[:, :2]
    constructed = np.column_stack(
        [
            np.sum(selected_children, axis=1),
            selected_children,
        ]
    )
    standardized = standardize_hierarchy_by_context(
        constructed,
        REAL_CALIBRATION_CONTEXT_LENGTH,
    )
    features = cafe_feature_vector(
        standardized,
        feature_period,
        hierarchy="additive_first",
        include_cross_series_predictability=False,
    )
    return {
        str(name): float(features[name])
        for name in {
            *REQUIRED_REAL_FEATURES_BY_CAPABILITY[
                "hierarchical_coherence"
            ],
            "hierarchy_residual_mean_abs",
            "hierarchy_aggregation_ratio",
        }
        if name in features
        if math.isfinite(float(features[name]))
    }


def _known_future_covariate_features(
    target_values: np.ndarray,
    covariates: np.ndarray | None,
    *,
    covariate_kind: str | None,
    channel_index: int,
    start: int,
    feature_period: int,
    minimum_observed_fraction: float,
) -> dict[str, float]:
    if covariates is None or covariate_kind != "known_future":
        return {}
    native = np.asarray(target_values, dtype=float)
    channels = native if native.ndim == 2 else native.reshape(1, -1)
    if not 0 <= channel_index < channels.shape[0]:
        return {}
    raw_history = channels[
        channel_index,
        start : start + REAL_CALIBRATION_CONTEXT_LENGTH,
    ]
    covariate_history = np.asarray(covariates, dtype=float)[
        start : start + REAL_CALIBRATION_CONTEXT_LENGTH
    ]
    if (
        raw_history.shape != (REAL_CALIBRATION_CONTEXT_LENGTH,)
        or covariate_history.ndim != 2
        or covariate_history.shape[0] != REAL_CALIBRATION_CONTEXT_LENGTH
    ):
        return {}
    imputed, _observed = impute_observed_window(
        raw_history,
        minimum_observed_fraction=minimum_observed_fraction,
    )
    if imputed is None or not np.isfinite(covariate_history).all():
        return {}
    try:
        standardized, _location, _scale = standardize_history(imputed)
    except ValueError:
        return {}
    normalized_covariates = normalize_covariates(
        covariate_history,
        REAL_CALIBRATION_CONTEXT_LENGTH,
    )
    return cafe_feature_vector(
        standardized[:, None],
        feature_period,
        covariates=normalized_covariates,
    )


def _declared_hierarchy_features(
    hierarchy_values: np.ndarray | None,
    hierarchy_kind: str | None,
    *,
    start: int,
    feature_period: int,
    minimum_observed_fraction: float,
) -> dict[str, float]:
    if hierarchy_values is None or hierarchy_kind is None:
        return {}
    native = np.asarray(hierarchy_values, dtype=float)
    if native.ndim != 2:
        return {}
    if hierarchy_kind == "children_only_additive":
        if native.shape[0] != 2:
            return {}
        child_indexes = slice(None)
    elif hierarchy_kind == "additive_first":
        if native.shape[0] < 3:
            return {}
        child_indexes = slice(1, None)
    else:
        return {}
    history = native[
        :,
        start : start + REAL_CALIBRATION_CONTEXT_LENGTH,
    ].T
    if history.shape != (
        REAL_CALIBRATION_CONTEXT_LENGTH,
        native.shape[0],
    ):
        return {}
    imputed_channels: list[np.ndarray] = []
    for channel in range(history.shape[1]):
        imputed, _observed = impute_observed_window(
            history[:, channel],
            minimum_observed_fraction=minimum_observed_fraction,
        )
        if imputed is None:
            return {}
        imputed_channels.append(imputed)
    # The observed parent is not needed for intensity calibration.  Both
    # contracts resolve to declared synchronized children and reconstruct the
    # parent using the same exact sum used by the synthetic generator.
    return _hierarchy_children_feature_row(
        np.column_stack(imputed_channels)[..., child_indexes],
        feature_period=feature_period,
    )


def build_calibration_anchors(
    dataset: DatasetSpec,
    *,
    gift_eval_dir: Path | None = None,
    source_root: Path | None = None,
    maximum_anchors: int,
    sample_seed: int = CALIBRATION_SAMPLE_SEED,
    minimum_observed_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_root = source_root or gift_eval_dir
    if resolved_root is None:
        raise ValueError("CaFE calibration requires a real-data source root")
    asset_path = resolved_root / dataset.asset_name
    record_limit = (
        max(64, min(256, int(math.ceil(maximum_anchors / 4))))
        if dataset.real_data_adapter == "m5_csv"
        else None
    )
    real_bundle = load_real_dataset(
        dataset.real_data_adapter,
        asset_path,
        record_limit=record_limit,
    )
    frequency = real_bundle.frequency
    native_records = [
        (record.item_id, record.values)
        for record in real_bundle.records
    ]
    record_by_item: dict[str, RealSeriesRecord] = {
        record.item_id: record for record in real_bundle.records
    }
    native_by_item = {
        str(item_id): np.asarray(values, dtype=float)
        for item_id, values in native_records
    }
    series = expand_native_records(native_records)
    candidates: list[tuple[int, int, int]] = []
    for series_index, (_series_id, _item_id, _channel, values) in enumerate(series):
        for lower, upper in nonoverlapping_strata(len(values)):
            candidates.append((series_index, lower, upper))
    target_count = min(len(candidates), int(maximum_anchors))
    rng = np.random.default_rng(
        stable_seed(dataset.dataset_id, sample_seed, base=sample_seed)
    )
    order = rng.permutation(len(candidates)) if candidates else np.asarray([], dtype=int)
    anchors: list[dict[str, Any]] = []
    rejected_missing = 0
    rejected_uninformative = 0
    for candidate_index in order:
        series_index, lower, upper = candidates[int(candidate_index)]
        series_id, item_id, channel_index, values = series[series_index]
        start = int(rng.integers(lower, upper + 1)) if upper > lower else int(lower)
        raw_master = np.asarray(
            values[start : start + REAL_FORECAST_MASTER_LENGTH],
            dtype=float,
        )
        if raw_master.size != REAL_FORECAST_MASTER_LENGTH:
            rejected_missing += 1
            continue
        history, history_observed_fraction = impute_observed_window(
            raw_master[:REAL_CALIBRATION_CONTEXT_LENGTH],
            minimum_observed_fraction=minimum_observed_fraction,
        )
        future, future_observed_fraction = impute_observed_window(
            raw_master[REAL_CALIBRATION_CONTEXT_LENGTH:],
            minimum_observed_fraction=minimum_observed_fraction,
        )
        if history is None or future is None:
            rejected_missing += 1
            continue
        try:
            standardized_history, location, scale = standardize_history(history)
        except ValueError:
            rejected_uninformative += 1
            continue
        period_policy = calibration_period_policy(
            frequency,
            standardized_history,
        )
        features = cafe_feature_vector(
            standardized_history[:, None],
            int(period_policy["feature_period"]),
        )
        (
            nonlinear_gain,
            nonlinear_detected_lag,
            nonlinear_candidate_lags,
        ) = cafe_nonlinear_conditional_gain(
            standardized_history,
            int(period_policy["feature_period"]),
        )
        features["nonlinear_conditional_gain"] = nonlinear_gain
        features["nonlinear_conditional_effect_size"] = math.sqrt(
            max(nonlinear_gain, 0.0)
        )
        features["nonlinear_proxy_best_lag"] = float(
            nonlinear_detected_lag
        )
        features["nonlinear_proxy_candidate_lag_count"] = float(
            len(nonlinear_candidate_lags)
        )
        finite_features = {
            str(name): float(value)
            for name, value in features.items()
            if math.isfinite(float(value))
            and name != "future_abs_covariate_target_corr"
        }
        native_features = _native_multivariate_features(
            native_by_item[item_id],
            start=start,
            feature_period=int(period_policy["feature_period"]),
            minimum_observed_fraction=minimum_observed_fraction,
        )
        finite_native_features = {
            str(name): float(value)
            for name, value in native_features.items()
            if math.isfinite(float(value))
        }
        source_record = record_by_item[item_id]
        covariate_features = _known_future_covariate_features(
            source_record.values,
            source_record.covariates,
            covariate_kind=source_record.covariate_kind,
            channel_index=channel_index,
            start=start,
            feature_period=int(period_policy["feature_period"]),
            minimum_observed_fraction=minimum_observed_fraction,
        )
        finite_covariate_features = {
            str(name): float(value)
            for name, value in covariate_features.items()
            if name
            in set(
                REQUIRED_REAL_FEATURES_BY_CAPABILITY[
                    "covariate_response"
                ]
            )
            if math.isfinite(float(value))
            and name != "future_abs_covariate_target_corr"
        }
        hierarchy_features = _declared_hierarchy_features(
            source_record.hierarchy_values,
            source_record.hierarchy_kind,
            start=start,
            feature_period=int(period_policy["feature_period"]),
            minimum_observed_fraction=minimum_observed_fraction,
        )
        finite_hierarchy_features = {
            str(name): float(value)
            for name, value in hierarchy_features.items()
            if name
            in {
                *REQUIRED_REAL_FEATURES_BY_CAPABILITY[
                    "hierarchical_coherence"
                ],
                "hierarchy_residual_mean_abs",
                "hierarchy_aggregation_ratio",
            }
            if math.isfinite(float(value))
        }
        hierarchy_children_features = dict(finite_hierarchy_features)
        standardized_master = np.concatenate([history, future])
        standardized_master = (standardized_master - location) / scale
        try:
            real_mase_period, real_mase_scale = _real_forecast_mase_scale(
                standardized_history,
                int(period_policy["mase_period"]),
            )
        except ValueError:
            rejected_uninformative += 1
            continue
        anchor_id = (
            f"{safe_id(dataset.dataset_id)}__{safe_id(item_id)}__"
            f"c{channel_index}__t{start}"
        )
        anchors.append(
            {
                "schema_version": "cafe.calibration_anchor.v1",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "anchor_id": anchor_id,
                "dataset_id": dataset.dataset_id,
                "config_id": dataset.config_id,
                "task_view_id": dataset.task_view_id,
                "profile_id": (
                    f"{dataset.dataset_id}__{dataset.task_view_id}__"
                    f"L{REAL_CALIBRATION_CONTEXT_LENGTH}"
                ),
                "frequency": frequency,
                # ``season_length`` remains the literal calendar period for
                # external provenance. CaFE generation and evaluation consume
                # the explicit fields below instead of overloading it.
                "season_length": int(
                    period_policy["calendar_season_length"]
                ),
                **period_policy,
                "item_id": item_id,
                "series_id": series_id,
                "channel_id": channel_index,
                "window_start": start,
                "context_length": REAL_CALIBRATION_CONTEXT_LENGTH,
                "horizon": HORIZON,
                "observed_fraction": history_observed_fraction,
                "future_observed_fraction": future_observed_fraction,
                "history_location": location,
                "history_scale": scale,
                "history_sha256": hashlib.sha256(
                    np.asarray(history, dtype="<f8").tobytes()
                ).hexdigest(),
                "features": finite_features,
                "native_multivariate_features": finite_native_features,
                "known_future_covariate_features": finite_covariate_features,
                "declared_hierarchy_features": finite_hierarchy_features,
                "hierarchy_children_features": (
                    hierarchy_children_features
                ),
                "structural_group_id": source_record.structural_group_id,
                "covariate_column_names": list(
                    source_record.covariate_names
                ),
                "covariate_kind": source_record.covariate_kind,
                "hierarchy_kind": source_record.hierarchy_kind,
                "feature_provenance": {
                    **{
                        name: "real_univariate_history_l168"
                        for name in finite_features
                    },
                    **{
                        name: "real_native_multivariate_history_l168"
                        for name in finite_native_features
                        if name not in finite_features
                    },
                    **{
                        name: "real_known_future_covariates_history_l168"
                        for name in finite_covariate_features
                    },
                    **{
                        name: "real_declared_hierarchy_history_l168"
                        for name in finite_hierarchy_features
                    },
                    **{
                        name: "real_hierarchy_children_history_l168"
                        for name in hierarchy_children_features
                    },
                },
                "feature_provenance_by_scope": {
                    "real_univariate": sorted(finite_features),
                    "real_native_multivariate": sorted(
                        finite_native_features
                    ),
                    "real_known_future_covariates": sorted(
                        finite_covariate_features
                    ),
                    "real_declared_hierarchy": sorted(
                        finite_hierarchy_features
                    ),
                    "real_hierarchy_children": sorted(
                        hierarchy_children_features
                    ),
                },
                "real_forecast_master": {
                    "schema_version": (
                        "cafe.real_anchor_forecast_master.v1"
                    ),
                    "sample_id": f"v8real__{anchor_id}",
                    "dataset_id": dataset.dataset_id,
                    "config_id": dataset.config_id,
                    "task_view_id": dataset.task_view_id,
                    "anchor_id": anchor_id,
                    "context_length": REAL_CALIBRATION_CONTEXT_LENGTH,
                    "horizon": HORIZON,
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "target": standardized_master[:, None].tolist(),
                    "covariates": None,
                    "frequency": frequency,
                    "calendar_season_length": int(
                        period_policy["calendar_season_length"]
                    ),
                    "feature_period": int(period_policy["feature_period"]),
                    "mase_period": real_mase_period,
                    "mase_scale": real_mase_scale,
                    "standardization": {
                        "scope": "history_only_l168",
                        "location": location,
                        "scale": scale,
                    },
                    "history_sha256": hashlib.sha256(
                        np.asarray(
                            standardized_master[
                                :REAL_CALIBRATION_CONTEXT_LENGTH
                            ],
                            dtype="<f8",
                        ).tobytes()
                    ).hexdigest(),
                    "future_sha256": hashlib.sha256(
                        np.asarray(
                            standardized_master[
                                REAL_CALIBRATION_CONTEXT_LENGTH:
                            ],
                            dtype="<f8",
                        ).tobytes()
                    ).hexdigest(),
                },
            }
        )
        if len(anchors) >= target_count:
            break
    if not anchors:
        raise ValueError(f"{dataset.dataset_id} produced no valid cafe anchors")
    univariate_profile = summarize_feature_rows(
        anchor["features"] for anchor in anchors
    )
    native_multivariate_profile = summarize_feature_rows(
        anchor["native_multivariate_features"]
        for anchor in anchors
        if anchor["native_multivariate_features"]
    )
    known_future_covariate_profile = summarize_feature_rows(
        anchor["known_future_covariate_features"]
        for anchor in anchors
        if anchor["known_future_covariate_features"]
    )
    declared_hierarchy_profile = summarize_feature_rows(
        anchor["declared_hierarchy_features"]
        for anchor in anchors
        if anchor["declared_hierarchy_features"]
    )
    hierarchy_children_profile = summarize_feature_rows(
        anchor["hierarchy_children_features"]
        for anchor in anchors
        if anchor["hierarchy_children_features"]
    )
    feature_support: dict[str, dict[str, Any]] = {}
    for scope, profile in (
        ("real_univariate", univariate_profile),
        ("real_native_multivariate", native_multivariate_profile),
        ("real_known_future_covariates", known_future_covariate_profile),
        ("real_declared_hierarchy", declared_hierarchy_profile),
        ("real_hierarchy_children", hierarchy_children_profile),
    ):
        for name, summary in profile.items():
            finite_count = int(summary["finite_count"])
            candidate = {
                "scope": scope,
                "finite_count": finite_count,
                "minimum_finite_count": MIN_REAL_FEATURE_COUNT,
                "usable": finite_count >= MIN_REAL_FEATURE_COUNT,
                "fallback_reason": (
                    None
                    if finite_count >= MIN_REAL_FEATURE_COUNT
                    else "insufficient_finite_real_windows"
                ),
            }
            existing = feature_support.get(name)
            if existing is None or (
                candidate["scope"] != "real_univariate"
                and candidate["usable"]
            ):
                feature_support[name] = candidate
    for anchor in anchors:
        anchor["feature_support"] = feature_support
    metadata = {
        "dataset": asdict(dataset),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "real_data_adapter": real_bundle.adapter_id,
        "asset_path": str(asset_path),
        "asset_files": [
            file_record(path) for path in real_bundle.asset_files
        ],
        "adapter_metadata": real_bundle.metadata,
        "frequency": frequency,
        "season_length": int(
            seasonal_period_for_frequency(frequency)
        ),
        "period_policy": {
            "calendar_season_length": int(
                seasonal_period_for_frequency(frequency)
            ),
            "calendar_feature_observable_rule": (
                "at_least_two_complete_calendar_cycles_in_l168"
            ),
            "profile_period_bounds": [
                8,
                REAL_CALIBRATION_CONTEXT_LENGTH // 3,
            ],
            "mase_fallback": (
                "lag1_when_calendar_period_is_not_defined_inside_l168"
            ),
        },
        "accepted_feature_period_counts": {
            str(period): sum(
                int(anchor["feature_period"]) == period
                for anchor in anchors
            )
            for period in sorted(
                {int(anchor["feature_period"]) for anchor in anchors}
            )
        },
        "accepted_mase_period_counts": {
            str(period): sum(
                int(anchor["mase_period"]) == period
                for anchor in anchors
            )
            for period in sorted(
                {int(anchor["mase_period"]) for anchor in anchors}
            )
        },
        "native_record_count": len(native_records),
        "expanded_series_count": len(series),
        "stratum_count": len(candidates),
        "requested_anchor_limit": int(maximum_anchors),
        "accepted_anchor_count": len(anchors),
        "rejected_missing_count": rejected_missing,
        "rejected_uninformative_count": rejected_uninformative,
        "minimum_observed_fraction": minimum_observed_fraction,
        "sample_seed": sample_seed,
        "feature_profiles": {
            "univariate": univariate_profile,
            "native_multivariate": native_multivariate_profile,
            "known_future_covariates": known_future_covariate_profile,
            "declared_hierarchy": declared_hierarchy_profile,
            "hierarchy_children": hierarchy_children_profile,
            "support": feature_support,
        },
        "window_policy": (
            "forecastable L168+H48 non-overlapping capacity strata with "
            "deterministic without-replacement selection and within-stratum "
            "jitter"
        ),
    }
    return anchors, metadata


def anchor_summary(features: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        name: {"p50": float(value)}
        for name, value in features.items()
        if math.isfinite(float(value))
    }


def anchor_feature_values(
    anchor: dict[str, Any],
    *,
    capability_id: str | None = None,
) -> dict[str, float]:
    values = {
        str(name): float(value)
        for name, value in anchor.get("features", {}).items()
        if anchor.get("feature_support", {})
        .get(name, {"usable": True})
        .get("usable", True)
        if math.isfinite(float(value))
    }
    scoped_field = {
        "common_factor": "native_multivariate_features",
        "cross_series_dependence": "native_multivariate_features",
        "hierarchical_coherence": "hierarchy_children_features",
        "covariate_response": "known_future_covariate_features",
    }.get(capability_id)
    scoped_feature_names = (
        set(REQUIRED_REAL_FEATURES_BY_CAPABILITY.get(capability_id, ()))
        if scoped_field is not None
        else set()
    )
    if scoped_field is not None:
        values.update(
            {
                str(name): float(value)
                for name, value in anchor.get(scoped_field, {}).items()
                if name in scoped_feature_names
                if anchor.get("feature_support", {})
                .get(name, {"usable": True})
                .get("usable", True)
                if math.isfinite(float(value))
            }
        )
    return values


def real_intensity_feature_summary(
    anchors: Iterable[dict[str, Any]],
    *,
    capability_id: str,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Resolve a semantically matched real feature range for one capability.

    Each capability reads only its semantically matched real task view.  A
    missing coordinate is reported as unavailable and is never replaced by a
    generator-known dose or a marginal feature with a similar name.
    """

    target_feature = PRIMARY_TARGET_FEATURE[capability_id]
    source_key, scope = REAL_INTENSITY_SOURCE_BY_CAPABILITY[capability_id]
    values = [
        float(anchor[source_key][target_feature])
        for anchor in anchors
        if target_feature in anchor.get(source_key, {})
        if math.isfinite(float(anchor[source_key][target_feature]))
    ]
    finite_count = len(values)
    if finite_count < MIN_REAL_FEATURE_COUNT:
        return None, {
            "usable": False,
            "scope": scope,
            "target_feature": target_feature,
            "finite_count": finite_count,
            "minimum_finite_count": MIN_REAL_FEATURE_COUNT,
            "reason_code": "insufficient_finite_real_anchor_features",
        }
    summary = summarize_feature_rows(
        [{target_feature: value} for value in values]
    )[target_feature]
    lower = float(summary["p10"])
    upper = float(summary["p90"])
    if upper <= lower + 1e-12:
        return None, {
            "usable": False,
            "scope": scope,
            "target_feature": target_feature,
            "finite_count": finite_count,
            "minimum_finite_count": MIN_REAL_FEATURE_COUNT,
            "reason_code": "real_anchor_q10_q90_collapsed",
            "summary": summary,
        }
    return summary, {
        "usable": True,
        "scope": scope,
        "target_feature": target_feature,
        "finite_count": finite_count,
        "minimum_finite_count": MIN_REAL_FEATURE_COUNT,
        "reason_code": None,
        "summary": summary,
    }


def parameter_mapping_provenance(
    mappings: Iterable[dict[str, Any]],
    anchor: dict[str, Any],
    *,
    capability_id: str | None = None,
) -> list[dict[str, Any]]:
    support = anchor.get("feature_support", {})
    univariate = {
        name
        for name in anchor.get("features", {})
        if support.get(name, {"usable": True}).get("usable", True)
    }
    scoped_field, scoped_status = {
        "common_factor": (
            "native_multivariate_features",
            "real_native_multivariate",
        ),
        "cross_series_dependence": (
            "native_multivariate_features",
            "real_native_multivariate",
        ),
        "hierarchical_coherence": (
            "hierarchy_children_features",
            "real_hierarchy_children",
        ),
        "covariate_response": (
            "known_future_covariate_features",
            "real_known_future_covariates",
        ),
    }.get(capability_id, ("", ""))
    scoped = {
        name
        for name in anchor.get(scoped_field, {})
        if scoped_field
        if name
        in set(REQUIRED_REAL_FEATURES_BY_CAPABILITY.get(capability_id, ()))
        if support.get(name, {"usable": True}).get("usable", True)
    }
    output: list[dict[str, Any]] = []
    for mapping in mappings:
        row = dict(mapping)
        source = str(row.get("source_feature", ""))
        components = tuple(
            part for part in source.split("/") if part
        )
        if source == "synthetic_protocol_constant":
            status = "protocol_constant"
            fallback_reason = None
        elif components and all(
            component in univariate | scoped
            for component in components
        ):
            status = (
                scoped_status
                if any(component in scoped for component in components)
                else "real_univariate"
            )
            fallback_reason = None
        else:
            status = "protocol_fallback"
            missing = [
                component
                for component in components
                if component not in univariate | scoped
            ]
            fallback_reason = (
                "real_feature_unavailable:"
                + ",".join(missing or [source or "unknown"])
            )
        row["source_status"] = status
        row["fallback_used"] = status == "protocol_fallback"
        row["fallback_reason"] = fallback_reason
        output.append(row)
    return output


def build_conditioning(
    dataset: DatasetSpec,
    *,
    capability_id: str,
    frequency: str,
    season_length: int,
    intensity_lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    parameters: dict[str, float] | None = None,
    target_values: tuple[float, ...] | None = None,
) -> GeneratorConditioning:
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    targets = target_values or levels
    return GeneratorConditioning(
        profile_id=(
            f"{dataset.dataset_id}__{dataset.task_view_id}__"
            f"{capability_id}__L{CONTEXT_LENGTH}_H{HORIZON}"
        ),
        dataset_id=dataset.dataset_id,
        capability_id=capability_id,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        target_dim=TARGET_DIM_BY_CAPABILITY[capability_id],
        season_length=season_length,
        frequency=frequency,
        parameters=dict(parameters or {}),
        intensity_lambdas=tuple(float(value) for value in intensity_lambdas),
        target_percentile_levels=levels,
        target_feature=PRIMARY_TARGET_FEATURE[capability_id],
        target_values=tuple(float(value) for value in targets),
        calibrated_realized_strengths=tuple(float(value) for value in targets),
        calibration_max_normalized_error=0.0,
        intensity_policy_id=REAL_BOUNDED_INTENSITY_POLICY_ID,
        artifact_schema_version="cafe.calibration_bundle.v1",
        artifact_created_at=None,
        calibration_method="cafe.response_curve",
        artifact_generator_version=GENERATOR_VERSION,
    )


def standardize_generated_sample(
    capability_id: str,
    target: np.ndarray,
    covariates: np.ndarray | None,
    *,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    if capability_id == "common_factor":
        target, normalization = standardize_common_factor_counterfactual_member(
            target,
            context_length=CONTEXT_LENGTH,
            metadata=metadata,
        )
        metadata["counterfactual_standardization"] = normalization
    elif capability_id == "cross_series_dependence":
        target, normalization = standardize_cross_series_counterfactual_member(
            target,
            context_length=CONTEXT_LENGTH,
            metadata=metadata,
        )
        metadata["counterfactual_standardization"] = normalization
    elif capability_id == "hierarchical_coherence":
        target = standardize_hierarchy_by_context(target, CONTEXT_LENGTH)
    else:
        target = standardize_by_context(target, CONTEXT_LENGTH)
    if covariates is not None:
        covariates = normalize_covariates(covariates, CONTEXT_LENGTH)
    return np.asarray(target, dtype=float), (
        None if covariates is None else np.asarray(covariates, dtype=float)
    )


def measured_features(
    capability_id: str,
    target: np.ndarray,
    covariates: np.ndarray | None,
    *,
    season_length: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    metadata = metadata or {}
    target_history = np.asarray(target, dtype=float)[:CONTEXT_LENGTH]
    covariate_history = (
        None
        if covariates is None
        else np.asarray(covariates, dtype=float)[:CONTEXT_LENGTH]
    )
    measurement_period = int(season_length)
    if capability_id == "multi_seasonal":
        periods = metadata.get("periods") or []
        if periods:
            measurement_period = int(round(float(periods[0])))
    elif capability_id == "time_varying_seasonality":
        measurement_period = int(
            round(float(metadata.get("primary_period", measurement_period)))
        )
    elif capability_id == "regime_switching":
        measurement_period = int(
            round(float(metadata.get("dwell_length", measurement_period)))
        )
    elif capability_id == "nonlinear_persistence":
        measurement_period = int(
            round(float(metadata.get("seasonal_lag", measurement_period)))
        )
    elif capability_id == "predictable_intermittency":
        measurement_period = int(
            round(float(metadata.get("event_period", measurement_period)))
        )
    elif capability_id == "covariate_response":
        baseline = metadata.get("baseline_process") or {}
        measurement_period = int(
            round(float(baseline.get("period", measurement_period)))
        )
    measurement_period = int(
        np.clip(measurement_period, 1, CONTEXT_LENGTH - 1)
    )
    hierarchy = (
        "additive_first"
        if capability_id == "hierarchical_coherence"
        else None
    )
    values = cafe_feature_vector(
        target_history,
        measurement_period,
        covariates=covariate_history,
        hierarchy=hierarchy,
        include_cross_series_predictability=(
            capability_id == "cross_series_dependence"
        ),
        cross_series_max_lag=(
            24 if capability_id == "cross_series_dependence" else None
        ),
    )
    if capability_id == "nonlinear_persistence":
        nonlinear_strength = float(
            metadata.get("nonlinear_strength", math.nan)
        )
        if math.isfinite(nonlinear_strength):
            values["nonlinear_strength"] = nonlinear_strength
        (
            nonlinear_gain,
            nonlinear_detected_lag,
            nonlinear_candidate_lags,
        ) = cafe_nonlinear_conditional_gain(
            np.mean(target_history, axis=1),
            measurement_period,
        )
        values["nonlinear_conditional_gain"] = nonlinear_gain
        values["nonlinear_conditional_effect_size"] = math.sqrt(
            max(nonlinear_gain, 0.0)
        )
        values["nonlinear_proxy_best_lag"] = float(
            nonlinear_detected_lag
        )
        values["nonlinear_proxy_candidate_lag_count"] = float(
            len(nonlinear_candidate_lags)
        )
        actual_lag = metadata.get("nonlinear_lag")
        if actual_lag is not None:
            values["nonlinear_actual_lag_gain"] = (
                cafe_nonlinear_actual_lag_gain(
                    np.mean(target_history, axis=1),
                    measurement_period,
                    int(actual_lag),
                )
            )
    elif capability_id == "covariate_response":
        effect_share = float(
            metadata.get("covariate_effect_variance_share", math.nan)
        )
        if math.isfinite(effect_share):
            values["covariate_effect_variance_share"] = effect_share
    elif capability_id == "predictable_intermittency":
        effect_share = float(
            metadata.get("event_effect_energy_share", math.nan)
        )
        if math.isfinite(effect_share):
            values["event_effect_energy_share"] = effect_share
    return {
        str(name): float(value)
        for name, value in values.items()
        if math.isfinite(float(value))
        and name != "future_abs_covariate_target_corr"
    }


def generate_calibration_member(
    dataset: DatasetSpec,
    anchor: dict[str, Any],
    *,
    capability_id: str,
    family_role: str,
    lambda_value: float,
    qualification_path_index: int,
    counterfactual_variant: int = 0,
) -> tuple[dict[str, float], dict[str, Any]]:
    parameters, mappings = derive_deterministic_parameters(
        capability_id,
        anchor_summary(
            anchor_feature_values(
                anchor,
                capability_id=capability_id,
            )
        ),
        season_length=int(anchor["feature_period"]),
        context_length=CONTEXT_LENGTH,
    )
    mappings = parameter_mapping_provenance(
        mappings,
        anchor,
        capability_id=capability_id,
    )
    conditioning = build_conditioning(
        dataset,
        capability_id=capability_id,
        frequency=str(anchor["frequency"]),
        season_length=int(anchor["feature_period"]),
        intensity_lambdas=(lambda_value,) * 5,
        parameters=parameters,
    )
    path_seed = stable_seed(
        dataset.dataset_id,
        capability_id,
        qualification_path_index,
        "qualification-path",
        base=QUALIFICATION_PATH_SEED,
    )
    rng = np.random.default_rng(path_seed)
    target, metadata, covariates = generate_deterministic_sample(
        capability_id,
        MASTER_LENGTH,
        CONTEXT_LENGTH,
        conditioning.target_dim,
        conditioning.season_length,
        1,
        rng,
        conditioning=conditioning,
        family_role=family_role,
        counterfactual_variant=counterfactual_variant,
    )
    target, covariates = standardize_generated_sample(
        capability_id,
        target,
        covariates,
        metadata=metadata,
    )
    return (
        measured_features(
            capability_id,
            target,
            covariates,
            season_length=conditioning.season_length,
            metadata=metadata,
        ),
        {
            "parameters": parameters,
            "parameter_mapping": mappings,
            "target": target,
            "covariates": covariates,
            "generation_metadata": metadata,
        },
    )


def structural_calibration_reachability(
    dataset: DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    capability_id: str,
    family_role: str,
    lambda_value: float,
    calibration_seed_count: int,
) -> dict[str, Any]:
    """Check the exact selected I5 dose against formal structural hard gates."""

    if capability_id not in STRUCTURAL_CAPABILITIES:
        raise ValueError(
            f"structural reachability requested for {capability_id!r}"
        )
    path_gates: list[dict[str, Any]] = []
    counterfactual = capability_id in COUNTERFACTUAL_CAPABILITIES
    for seed_index in range(calibration_seed_count):
        anchor = anchor_for_qualification_path(
            anchors,
            dataset_id=dataset.dataset_id,
            capability_id=capability_id,
            seed_index=seed_index,
        )
        _first_features, first = generate_calibration_member(
            dataset,
            anchor,
            capability_id=capability_id,
            family_role=family_role,
            lambda_value=lambda_value,
            qualification_path_index=seed_index,
            counterfactual_variant=0,
        )
        second: dict[str, Any] | None = None
        if counterfactual:
            _second_features, second = generate_calibration_member(
                dataset,
                anchor,
                capability_id=capability_id,
                family_role=family_role,
                lambda_value=lambda_value,
                qualification_path_index=seed_index,
                counterfactual_variant=1,
            )
        selected_dose_gate = structural_calibration_member_gate(
            capability_id,
            np.asarray(first["target"], dtype=float),
            context_length=CONTEXT_LENGTH,
            metadata=first["generation_metadata"],
            second_target=(
                None
                if second is None
                else np.asarray(second["target"], dtype=float)
            ),
            first_covariates=(
                None
                if first["covariates"] is None
                else np.asarray(first["covariates"], dtype=float)
            ),
            second_covariates=(
                None
                if second is None or second["covariates"] is None
                else np.asarray(second["covariates"], dtype=float)
            ),
        )
        if capability_id in {
            "common_factor",
            "cross_series_dependence",
        }:
            _strong_features, strong_first = generate_calibration_member(
                dataset,
                anchor,
                capability_id=capability_id,
                family_role=family_role,
                lambda_value=1.0,
                qualification_path_index=seed_index,
                counterfactual_variant=0,
            )
            _strong_features, strong_second = generate_calibration_member(
                dataset,
                anchor,
                capability_id=capability_id,
                family_role=family_role,
                lambda_value=1.0,
                qualification_path_index=seed_index,
                counterfactual_variant=1,
            )
            strong_dose_gate = structural_calibration_member_gate(
                capability_id,
                np.asarray(strong_first["target"], dtype=float),
                context_length=CONTEXT_LENGTH,
                metadata=strong_first["generation_metadata"],
                second_target=np.asarray(
                    strong_second["target"],
                    dtype=float,
                ),
            )
            if capability_id == "common_factor":
                strong_positive_control_passed = bool(
                    strong_dose_gate.get("counterfactual_passed", False)
                    and strong_dose_gate.get(
                        "joint_observability_passed",
                        False,
                    )
                )
            else:
                strong_positive_control_passed = bool(
                    strong_dose_gate.get(
                        "paired_intervention_source_passed",
                        False,
                    )
                    and strong_dose_gate.get(
                        "incremental_history_holdout_passed",
                        False,
                    )
                    and strong_dose_gate.get(
                        "positive_control_passed",
                        False,
                    )
                )
            selected_dose_gate = {
                **selected_dose_gate,
                "selected_real_calibrated_dose_accepted": bool(
                    selected_dose_gate["accepted"]
                ),
                "strong_dose_positive_control": {
                    "lambda_value": 1.0,
                    "accepted": strong_positive_control_passed,
                    "gate": strong_dose_gate,
                },
                "accepted": bool(
                    selected_dose_gate["accepted"]
                    and strong_positive_control_passed
                ),
            }
        path_gates.append(selected_dose_gate)
    return summarize_structural_calibration_reachability(
        capability_id,
        family_role=family_role,
        lambda_value=lambda_value,
        path_gates=path_gates,
        expected_path_count=calibration_seed_count,
    )


def stable_monotone_support(
    grid: np.ndarray,
    raw_response: np.ndarray,
) -> tuple[int, dict[str, Any]]:
    """Find the largest stable increasing prefix without hiding foldback.

    Small Monte-Carlo wiggles are tolerated.  A family is truncated only after
    two consecutive grid points fall more than 10% of the full response range
    below the best value already reached.  The detected boundary and raw curve
    are persisted so the choice is auditable rather than a family magic number.
    """

    if grid.shape != raw_response.shape or grid.ndim != 1:
        raise ValueError("response grid and values must be aligned vectors")
    if grid.size < 2 or not np.isfinite(raw_response).all():
        raise ValueError("response curve must contain finite grid values")
    response_range = float(np.max(raw_response) - np.min(raw_response))
    tolerance = max(0.10 * response_range, 1e-12)
    best_index = 0
    best_value = float(raw_response[0])
    below_run = 0
    fold_index: int | None = None
    support_index = len(grid) - 1
    for index in range(1, len(grid)):
        value = float(raw_response[index])
        if value >= best_value:
            best_value = value
            best_index = index
            below_run = 0
            continue
        if value < best_value - tolerance:
            below_run += 1
        else:
            below_run = 0
        if below_run >= 2 and best_index >= 2:
            fold_index = index - 1
            support_index = best_index
            break
    return support_index, {
        "policy": "largest_stable_monotone_prefix_v1",
        "mathematical_lambda_support": [
            float(grid[0]),
            float(grid[-1]),
        ],
        "effective_lambda_support": [
            float(grid[0]),
            float(grid[support_index]),
        ],
        "response_range": response_range,
        "foldback_tolerance": tolerance,
        "foldback_detected": fold_index is not None,
        "first_sustained_foldback_index": fold_index,
        "selected_peak_index": int(support_index),
    }


def raw_increasing_response_branch(
    grid: np.ndarray,
    raw_response: np.ndarray,
    *,
    support_index: int,
) -> tuple[int, int, dict[str, Any]]:
    """Select an invertible raw branch without a monotone envelope.

    The stable-support detector may tolerate small local decreases so that
    Monte-Carlo wiggles do not truncate an otherwise useful family.  Those
    decreases still cannot be hidden during inverse calibration: interpolation
    on a cumulative-maximum envelope can select lambdas whose realized
    responses are reversed.  We therefore choose the longest contiguous
    strictly increasing run inside the conservative support.  Response span,
    then the earlier run, break equal-length ties.
    """

    if grid.shape != raw_response.shape or grid.ndim != 1:
        raise ValueError("response grid and values must be aligned vectors")
    if support_index < 0 or support_index >= len(grid):
        raise ValueError("response support index is out of bounds")
    bounded = np.asarray(raw_response[: support_index + 1], dtype=float)
    scale = max(float(np.max(np.abs(bounded))), 1.0)
    tolerance = max(32.0 * np.finfo(float).eps * scale, 1e-15)
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(bounded)):
        if bounded[index] > bounded[index - 1] + tolerance:
            continue
        runs.append((start, index - 1))
        start = index
    runs.append((start, len(bounded) - 1))
    increasing_runs = [
        (left, right)
        for left, right in runs
        if right > left
    ]
    if not increasing_runs:
        return 0, 0, {
            "policy": "longest_contiguous_raw_increasing_branch_v1",
            "strict_increase_tolerance": tolerance,
            "candidate_runs": [
                {
                    "start_index": int(left),
                    "end_index": int(right),
                }
                for left, right in runs
            ],
            "selected_start_index": 0,
            "selected_end_index": 0,
        }
    selected_start, selected_end = max(
        increasing_runs,
        key=lambda item: (
            item[1] - item[0] + 1,
            float(bounded[item[1]] - bounded[item[0]]),
            -item[0],
        ),
    )
    return selected_start, selected_end, {
        "policy": "longest_contiguous_raw_increasing_branch_v1",
        "strict_increase_tolerance": tolerance,
        "candidate_runs": [
            {
                "start_index": int(left),
                "end_index": int(right),
                "point_count": int(right - left + 1),
                "response_span": float(bounded[right] - bounded[left]),
            }
            for left, right in increasing_runs
        ],
        "selected_start_index": int(selected_start),
        "selected_end_index": int(selected_end),
        "selected_point_count": int(selected_end - selected_start + 1),
        "selected_response_span": float(
            bounded[selected_end] - bounded[selected_start]
        ),
    }


def monotone_response_curve(
    dataset: DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    capability_id: str,
    family_role: str,
    calibration_seed_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    full_grid = np.linspace(0.0, 1.0, 21)
    target_feature = PRIMARY_TARGET_FEATURE[capability_id]
    path_responses = np.empty(
        (calibration_seed_count, len(full_grid)),
        dtype=float,
    )
    for seed_index in range(calibration_seed_count):
        anchor = anchor_for_qualification_path(
            anchors,
            dataset_id=dataset.dataset_id,
            capability_id=capability_id,
            seed_index=seed_index,
        )
        for lambda_index, lambda_value in enumerate(full_grid):
            features, _metadata = generate_calibration_member(
                dataset,
                anchor,
                capability_id=capability_id,
                family_role=family_role,
                lambda_value=float(lambda_value),
                qualification_path_index=seed_index,
            )
            path_responses[seed_index, lambda_index] = float(
                features[target_feature]
            )
    raw_response = np.mean(path_responses, axis=0)
    support_index, support = stable_monotone_support(
        full_grid,
        raw_response,
    )
    support["selected_peak_index"] = int(support_index)
    support["stable_effective_lambda_support"] = [
        float(full_grid[0]),
        float(full_grid[support_index]),
    ]
    branch_start, branch_end, branch = raw_increasing_response_branch(
        full_grid,
        raw_response,
        support_index=support_index,
    )
    grid = full_grid[branch_start : branch_end + 1]
    response = raw_response[branch_start : branch_end + 1]
    support["inversion_branch"] = branch
    support["effective_lambda_support"] = [
        float(grid[0]),
        float(grid[-1]),
    ]
    support.update(
        {
            "path_anchor_policy": "independent_qualification_anchor_hash_v1",
            "path_rng_policy": "independent_qualification_path_v1",
            "path_seed_start": 0,
            "raw_lambda_grid": full_grid.tolist(),
            "raw_response_curve": raw_response.tolist(),
            "per_path_raw_response_curves": path_responses.tolist(),
        }
    )
    half = calibration_seed_count // 2
    if half >= 2:
        first_half_response = np.mean(path_responses[:half], axis=0)
        second_half_response = np.mean(path_responses[half:], axis=0)
        first_half_support_index = stable_monotone_support(
            full_grid,
            first_half_response,
        )[0]
        second_half_support_index = stable_monotone_support(
            full_grid,
            second_half_response,
        )[0]
        response_scale = max(
            float(np.max(raw_response) - np.min(raw_response)),
            1e-12,
        )
        support["split_half_diagnostic"] = {
            "policy": "nonblocking_equal_path_blocks_v1",
            "triggers_path_expansion": False,
            "first_half_path_count": int(half),
            "second_half_path_count": int(
                calibration_seed_count - half
            ),
            "first_half_mean_curve_support_lambda": float(
                full_grid[first_half_support_index]
            ),
            "second_half_mean_curve_support_lambda": float(
                full_grid[second_half_support_index]
            ),
            "support_lambda_abs_difference": float(
                abs(
                    full_grid[first_half_support_index]
                    - full_grid[second_half_support_index]
                )
            ),
            "support_difference_gt_two_grid_steps": bool(
                abs(
                    full_grid[first_half_support_index]
                    - full_grid[second_half_support_index]
                )
                > 0.10 + 1e-12
            ),
            "max_abs_mean_response_difference": float(
                np.max(
                    np.abs(first_half_response - second_half_response)
                )
            ),
            "normalized_max_abs_mean_response_difference": float(
                np.max(
                    np.abs(first_half_response - second_half_response)
                )
                / response_scale
            ),
        }
    return grid, response, support


def response_curve_hard_failure_reasons(
    grid: np.ndarray,
    response: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    if grid.ndim != 1 or response.ndim != 1 or grid.shape != response.shape:
        return ["misaligned_response_curve"]
    if grid.size < 2:
        reasons.append("lambda_support_collapsed")
    if not np.isfinite(grid).all() or not np.isfinite(response).all():
        reasons.append("nonfinite_response_curve")
        return reasons
    if grid.size >= 2 and float(grid[-1] - grid[0]) <= 1e-12:
        reasons.append("lambda_support_collapsed")
    if response.size >= 2:
        scale = max(float(np.max(np.abs(response))), 1e-12)
        minimum_span = max(1e-12, 1e-8 * scale)
        if float(response[-1] - response[0]) <= minimum_span:
            reasons.append("realized_response_collapsed")
    return list(dict.fromkeys(reasons))


def inverse_mapping_hard_failure_reasons(
    targets: np.ndarray,
    lambdas: tuple[float, ...],
) -> list[str]:
    selected = np.asarray(lambdas, dtype=float)
    if selected.shape != targets.shape:
        return ["inverse_mapping_shape_mismatch"]
    if not np.isfinite(selected).all():
        return ["inverse_mapping_nonfinite"]
    if np.any(np.diff(selected) <= 1e-10):
        return ["inverse_mapping_not_strictly_increasing"]
    return []


def calibration_path_schedule(
    *,
    initial_path_count: int,
    maximum_path_count: int,
) -> tuple[int, ...]:
    if initial_path_count < 1 or maximum_path_count < initial_path_count:
        raise ValueError("invalid cafe calibration path budget")
    counts = [int(initial_path_count)]
    while counts[-1] < maximum_path_count:
        counts.append(
            min(
                maximum_path_count,
                counts[-1] + CALIBRATION_PATH_EXPANSION_STEP,
            )
        )
    return tuple(counts)


def inverse_response_lambdas(
    grid: np.ndarray,
    response: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, ...]:
    unique_values, unique_indexes = np.unique(response, return_index=True)
    if unique_values.size <= 1:
        return tuple(float(value) for value in np.linspace(0.0, 1.0, 5))
    selected = np.interp(
        targets,
        unique_values,
        grid[unique_indexes],
    )
    return tuple(float(value) for value in selected)


def selected_response_hard_failure_reasons(
    response: Iterable[float],
) -> list[str]:
    values = np.asarray(tuple(response), dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return ["selected_response_invalid"]
    reasons: list[str] = []
    if float(np.max(values) - np.min(values)) <= 1e-9:
        reasons.append("selected_response_span_collapsed")
    if any(
        right < left - 1e-8
        for left, right in zip(values, values[1:], strict=False)
    ):
        reasons.append("selected_response_not_monotone")
    return reasons


def observable_response_separation_gate(
    support_detection: dict[str, Any],
    selected_lambdas: Iterable[float],
    *,
    minimum_adjacent_non_decreasing_path_fraction: float,
    minimum_paired_standardized_separation: float,
) -> dict[str, Any]:
    """Qualify selected doses in observable feature space.

    The raw generator coefficient is not an observable intensity coordinate,
    so a compact lambda interval is not itself a failure.  Instead, selected
    levels must remain ordered on enough independently sampled qualification
    paths and adjacent family-mean levels must be separated relative to the
    between-path feature variability.
    """

    policy = (
        "selected_dose_paired_difference_observable_separation_v2"
    )
    reason_codes: list[str] = []
    validation_errors: list[str] = []
    try:
        raw_grid = np.asarray(
            support_detection["raw_lambda_grid"],
            dtype=float,
        )
        path_curves = np.asarray(
            support_detection["per_path_raw_response_curves"],
            dtype=float,
        )
        selected = np.asarray(tuple(selected_lambdas), dtype=float)
    except (KeyError, TypeError, ValueError):
        raw_grid = np.asarray([], dtype=float)
        path_curves = np.empty((0, 0), dtype=float)
        selected = np.asarray([], dtype=float)
        validation_errors.append("qualification_path_response_data_missing")

    if raw_grid.ndim != 1 or raw_grid.size < 2:
        validation_errors.append("raw_lambda_grid_invalid")
    if (
        path_curves.ndim != 2
        or path_curves.shape[0] < 2
        or path_curves.shape[1] != raw_grid.size
    ):
        validation_errors.append("qualification_path_response_shape_invalid")
    if selected.ndim != 1 or selected.size < 2:
        validation_errors.append("selected_lambda_grid_invalid")
    if (
        not validation_errors
        and (
            not np.isfinite(raw_grid).all()
            or not np.isfinite(path_curves).all()
            or not np.isfinite(selected).all()
        )
    ):
        validation_errors.append("observable_response_inputs_nonfinite")
    if not validation_errors and np.any(np.diff(raw_grid) <= 0.0):
        validation_errors.append("raw_lambda_grid_not_strictly_increasing")
    if not validation_errors and np.any(np.diff(selected) <= 0.0):
        validation_errors.append("selected_lambdas_not_strictly_increasing")
    if (
        not validation_errors
        and (
            float(selected[0]) < float(raw_grid[0]) - 1e-12
            or float(selected[-1]) > float(raw_grid[-1]) + 1e-12
        )
    ):
        validation_errors.append("selected_lambdas_outside_raw_grid")

    base_record = {
        "policy": policy,
        "lambda_span_used_as_hard_gate": False,
        "thresholds": {
            "minimum_adjacent_non_decreasing_path_fraction": float(
                minimum_adjacent_non_decreasing_path_fraction
            ),
            "minimum_paired_standardized_separation": float(
                minimum_paired_standardized_separation
            ),
        },
        "selected_lambdas": selected.tolist(),
        "qualification_path_count": int(
            path_curves.shape[0]
            if path_curves.ndim == 2
            else 0
        ),
    }
    if validation_errors:
        return {
            **base_record,
            "accepted": False,
            "reason_codes": ["observable_response_separation_gate_invalid"],
            "validation_errors": list(dict.fromkeys(validation_errors)),
            "monotone_path_fraction": None,
            "minimum_adjacent_non_decreasing_path_fraction": None,
            "minimum_paired_standardized_separation": None,
            "adjacent_level_diagnostics": [],
        }

    selected_path_responses = np.asarray(
        [
            np.interp(selected, raw_grid, path_response)
            for path_response in path_curves
        ],
        dtype=float,
    )
    response_scale = max(
        float(np.max(np.abs(selected_path_responses))),
        1.0,
    )
    numerical_tolerance = max(
        32.0 * np.finfo(float).eps * response_scale,
        1e-12,
    )
    adjacent_differences = np.diff(selected_path_responses, axis=1)
    monotone_paths = np.all(
        adjacent_differences >= -numerical_tolerance,
        axis=1,
    )
    monotone_path_fraction = float(np.mean(monotone_paths))
    adjacent_diagnostics: list[dict[str, Any]] = []
    paired_standardized_separations: list[float] = []
    adjacent_non_decreasing_fractions: list[float] = []
    for level_index in range(selected.size - 1):
        left = selected_path_responses[:, level_index]
        right = selected_path_responses[:, level_index + 1]
        left_std = float(np.std(left, ddof=1))
        right_std = float(np.std(right, ddof=1))
        pooled_path_std = float(
            math.sqrt(0.5 * (left_std**2 + right_std**2))
        )
        mean_gap = float(np.mean(right) - np.mean(left))
        paired_differences = right - left
        paired_difference_std = float(
            np.std(paired_differences, ddof=1)
        )
        paired_standard_error = float(
            paired_difference_std / math.sqrt(path_curves.shape[0])
        )
        paired_standardized_separation = float(
            mean_gap / max(paired_standard_error, numerical_tolerance)
        )
        pooled_standardized_separation = float(
            mean_gap / max(pooled_path_std, numerical_tolerance)
        )
        non_decreasing_fraction = float(
            np.mean(
                adjacent_differences[:, level_index]
                >= -numerical_tolerance
            )
        )
        paired_standardized_separations.append(
            paired_standardized_separation
        )
        adjacent_non_decreasing_fractions.append(
            non_decreasing_fraction
        )
        adjacent_diagnostics.append(
            {
                "left_level_index": int(level_index),
                "right_level_index": int(level_index + 1),
                "left_lambda": float(selected[level_index]),
                "right_lambda": float(selected[level_index + 1]),
                "left_mean_response": float(np.mean(left)),
                "right_mean_response": float(np.mean(right)),
                "mean_response_gap": mean_gap,
                "left_path_standard_deviation": left_std,
                "right_path_standard_deviation": right_std,
                "pooled_path_standard_deviation": pooled_path_std,
                "pooled_level_standardized_separation_diagnostic": (
                    pooled_standardized_separation
                ),
                "paired_difference_standard_deviation": (
                    paired_difference_std
                ),
                "paired_difference_standard_error": paired_standard_error,
                "paired_standardized_separation": (
                    paired_standardized_separation
                ),
                "non_decreasing_path_fraction": non_decreasing_fraction,
            }
        )
    minimum_paired_separation = float(
        min(paired_standardized_separations)
    )
    minimum_adjacent_fraction = float(
        min(adjacent_non_decreasing_fractions)
    )
    if (
        minimum_adjacent_fraction + 1e-12
        < minimum_adjacent_non_decreasing_path_fraction
    ):
        reason_codes.append(
            "insufficient_adjacent_non_decreasing_path_fraction"
        )
    if (
        minimum_paired_separation + 1e-12
        < minimum_paired_standardized_separation
    ):
        reason_codes.append(
            "insufficient_paired_observable_response_separation"
        )
    return {
        **base_record,
        "accepted": not reason_codes,
        "reason_codes": reason_codes,
        "validation_errors": [],
        "path_direction_policy": (
            "minimum_adjacent_non_decreasing_fraction"
        ),
        "monotonicity_numerical_tolerance": numerical_tolerance,
        "monotone_path_count": int(np.sum(monotone_paths)),
        "monotone_path_fraction": monotone_path_fraction,
        "minimum_adjacent_non_decreasing_path_fraction": (
            minimum_adjacent_fraction
        ),
        "minimum_paired_standardized_separation": (
            minimum_paired_separation
        ),
        "full_path_monotone_fraction_is_diagnostic_only": True,
        "adjacent_level_diagnostics": adjacent_diagnostics,
    }


def calibrate_capabilities(
    dataset: DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    calibration_seed_count: int,
    maximum_calibration_seed_count: int | None = None,
    capability_ids: Iterable[str] = CAPABILITIES,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    feature_summary = summarize_feature_rows(
        anchor["features"] for anchor in anchors
    )
    capabilities: dict[str, Any] = {}
    available_capabilities: list[str] = []
    unavailable_capabilities: dict[str, list[str]] = {}
    for capability_id in capability_ids:
        if capability_id not in CAPABILITIES:
            raise ValueError(f"unsupported capability: {capability_id}")
        real_feature_summary, real_feature_calibration = (
            real_intensity_feature_summary(
                anchors,
                capability_id=capability_id,
            )
        )
        initial_response_seed_count = int(calibration_seed_count)
        maximum_response_seed_count = (
            int(maximum_calibration_seed_count)
            if maximum_calibration_seed_count is not None
            else initial_response_seed_count
        )
        path_schedule = calibration_path_schedule(
            initial_path_count=initial_response_seed_count,
            maximum_path_count=maximum_response_seed_count,
        )
        attempted_path_counts: list[int] = []
        hard_failure_attempts: list[dict[str, Any]] = []
        target_feature = PRIMARY_TARGET_FEATURE[capability_id]
        base_record = {
            "target_feature": target_feature,
            "real_feature_calibration": real_feature_calibration,
            "target_dim": TARGET_DIM_BY_CAPABILITY[capability_id],
            "primary_family": PRIMARY_FAMILY_BY_CAPABILITY[capability_id],
            "secondary_family": SECONDARY_FAMILY_BY_CAPABILITY[capability_id],
        }
        if real_feature_summary is None:
            reason_code = str(
                real_feature_calibration.get(
                    "reason_code",
                    "real_intensity_feature_unavailable",
                )
            )
            capabilities[capability_id] = {
                **base_record,
                "available_for_generation": False,
                "availability_status": "unavailable",
                "unavailable_reason_codes": [reason_code],
                "intensity_calibration_scope": (
                    "dataset_real_calibration_unavailable"
                ),
                "calibration_status": "unavailable",
                "qualification_path_count": 0,
                "qualification_path_policy": {
                    "policy": (
                        "real_feature_precheck_before_generator_qualification_v1"
                    ),
                    "attempted_path_counts": [],
                    "selected_path_count": None,
                    "expanded": False,
                    "hard_failure_attempts": [],
                },
                "real_interval_q10_q90": None,
                "real_alignment_reference": {
                    "policy": (
                        "real_q10_q90_required_no_generator_relative_fallback"
                    ),
                    "real_reference_available": False,
                    "used_for_family_targets": False,
                    "unavailable_reason_codes": [reason_code],
                    "formal_seed_inverse": False,
                    "sample_level_alignment_enforced": False,
                },
                "primary": None,
                "secondary": None,
            }
            unavailable_capabilities[capability_id] = [reason_code]
            continue

        for response_seed_count in path_schedule:
            attempted_path_counts.append(response_seed_count)
            if progress_callback is not None:
                progress_callback(capability_id, response_seed_count)
            (
                primary_grid,
                primary_response,
                primary_support,
            ) = monotone_response_curve(
                dataset,
                anchors,
                capability_id=capability_id,
                family_role="primary",
                calibration_seed_count=response_seed_count,
            )
            (
                secondary_grid,
                secondary_response,
                secondary_support,
            ) = monotone_response_curve(
                dataset,
                anchors,
                capability_id=capability_id,
                family_role="secondary",
                calibration_seed_count=response_seed_count,
            )
            hard_failure_reasons = {
                "primary": response_curve_hard_failure_reasons(
                    primary_grid,
                    primary_response,
                ),
                "secondary": response_curve_hard_failure_reasons(
                    secondary_grid,
                    secondary_response,
                ),
            }
            if not any(hard_failure_reasons.values()):
                break
            hard_failure_attempts.append(
                {
                    "path_count": int(response_seed_count),
                    "reasons": hard_failure_reasons,
                }
            )
        if any(hard_failure_reasons.values()):
            reason_codes = ["generator_response_qualification_failed"]
            capabilities[capability_id] = {
                **base_record,
                "available_for_generation": False,
                "availability_status": "unavailable",
                "unavailable_reason_codes": reason_codes,
                "intensity_calibration_scope": (
                    "dataset_real_calibration_unavailable"
                ),
                "calibration_status": "unavailable",
                "qualification_path_count": response_seed_count,
                "qualification_path_policy": {
                    "policy": (
                        "independent_family_response_qualification_bank_"
                        "fixed_base_hard_failure_only_expansion_v1"
                    ),
                    "initial_path_count": initial_response_seed_count,
                    "maximum_path_count": maximum_response_seed_count,
                    "attempted_path_counts": attempted_path_counts,
                    "selected_path_count": None,
                    "expanded": bool(
                        response_seed_count > initial_response_seed_count
                    ),
                    "hard_failure_attempts": hard_failure_attempts,
                },
                "real_interval_q10_q90": [
                    float(real_feature_summary["p10"]),
                    float(real_feature_summary["p90"]),
                ],
                "real_alignment_reference": {
                    "policy": (
                        "real_q10_q90_required_no_generator_relative_fallback"
                    ),
                    "real_reference_available": True,
                    "used_for_family_targets": False,
                    "unavailable_reason_codes": reason_codes,
                    "formal_seed_inverse": False,
                    "sample_level_alignment_enforced": False,
                },
                "primary": None,
                "secondary": None,
            }
            unavailable_capabilities[capability_id] = reason_codes
            continue

        primary_generator_interval = [
            float(primary_response[0]),
            float(primary_response[-1]),
        ]
        secondary_generator_interval = [
            float(secondary_response[0]),
            float(secondary_response[-1]),
        ]
        real_interval = [
            float(real_feature_summary["p10"]),
            float(real_feature_summary["p90"]),
        ]
        real_span = float(real_interval[1] - real_interval[0])
        overlap_lower = max(
            real_interval[0],
            primary_generator_interval[0],
            secondary_generator_interval[0],
        )
        overlap_upper = min(
            real_interval[1],
            primary_generator_interval[1],
            secondary_generator_interval[1],
        )
        overlap_span = max(0.0, overlap_upper - overlap_lower)
        overlap_fraction = overlap_span / max(real_span, 1e-12)
        real_overlap_interval = (
            [float(overlap_lower), float(overlap_upper)]
            if overlap_span > 0.0
            else None
        )
        unavailable_reasons: list[str] = []
        if overlap_span <= 0.0:
            unavailable_reasons.append(
                "no_real_primary_secondary_generator_support_overlap"
            )
        elif overlap_span <= max(
            MINIMUM_REAL_OVERLAP_FRACTION * real_span,
            1e-9,
        ):
            unavailable_reasons.append(
                "insufficient_real_primary_secondary_generator_support_overlap"
            )

        primary_targets = np.linspace(overlap_lower, overlap_upper, 5)
        primary_lambdas: tuple[float, ...] = ()
        secondary_lambdas: tuple[float, ...] = ()
        primary_selected_response: tuple[float, ...] = ()
        secondary_selected_response: tuple[float, ...] = ()
        primary_mapped_span_fraction: float | None = None
        secondary_mapped_span_fraction: float | None = None
        inverse_failure_reasons: dict[str, list[str]] = {}
        observable_response_gates: dict[str, dict[str, Any]] = {}
        structural_reachability: dict[str, Any] | None = None
        if not unavailable_reasons:
            primary_lambdas = inverse_response_lambdas(
                primary_grid,
                primary_response,
                primary_targets,
            )
            secondary_lambdas = inverse_response_lambdas(
                secondary_grid,
                secondary_response,
                primary_targets,
            )
            primary_mapped_span_fraction = float(
                (primary_lambdas[-1] - primary_lambdas[0])
                / max(float(primary_grid[-1] - primary_grid[0]), 1e-12)
            )
            secondary_mapped_span_fraction = float(
                (secondary_lambdas[-1] - secondary_lambdas[0])
                / max(float(secondary_grid[-1] - secondary_grid[0]), 1e-12)
            )
            observable_gate_policy = (
                OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY.get(
                    capability_id
                )
            )
            if observable_gate_policy is None:
                if (
                    primary_mapped_span_fraction
                    < MINIMUM_MAPPED_LAMBDA_SPAN_FRACTION
                ):
                    unavailable_reasons.append(
                        "real_reference_maps_to_insufficient_primary_lambda_span"
                    )
                if (
                    secondary_mapped_span_fraction
                    < MINIMUM_MAPPED_LAMBDA_SPAN_FRACTION
                ):
                    unavailable_reasons.append(
                        "real_reference_maps_to_insufficient_secondary_lambda_span"
                    )
            inverse_failure_reasons = {
                "primary_inverse": inverse_mapping_hard_failure_reasons(
                    np.asarray(primary_targets, dtype=float),
                    primary_lambdas,
                ),
                "secondary_inverse": inverse_mapping_hard_failure_reasons(
                    np.asarray(primary_targets, dtype=float),
                    secondary_lambdas,
                ),
            }
            primary_selected_response = tuple(
                float(value)
                for value in np.interp(
                    np.asarray(primary_lambdas, dtype=float),
                    primary_grid,
                    primary_response,
                )
            )
            secondary_selected_response = tuple(
                float(value)
                for value in np.interp(
                    np.asarray(secondary_lambdas, dtype=float),
                    secondary_grid,
                    secondary_response,
                )
            )
            inverse_failure_reasons.update(
                {
                    "primary_selected_response": (
                        selected_response_hard_failure_reasons(
                            primary_selected_response
                        )
                    ),
                    "secondary_selected_response": (
                        selected_response_hard_failure_reasons(
                            secondary_selected_response
                        )
                    ),
                }
            )
            for family_name, reasons in inverse_failure_reasons.items():
                unavailable_reasons.extend(
                    f"{family_name}:{reason}" for reason in reasons
                )
            if observable_gate_policy is not None:
                for (
                    family_name,
                    family_support,
                    family_lambdas,
                ) in (
                    (
                        "primary",
                        primary_support,
                        primary_lambdas,
                    ),
                    (
                        "secondary",
                        secondary_support,
                        secondary_lambdas,
                    ),
                ):
                    gate = observable_response_separation_gate(
                        family_support,
                        family_lambdas,
                        minimum_adjacent_non_decreasing_path_fraction=float(
                            observable_gate_policy[
                                "minimum_adjacent_non_decreasing_path_fraction"
                            ]
                        ),
                        minimum_paired_standardized_separation=float(
                            observable_gate_policy[
                                "minimum_paired_standardized_separation"
                            ]
                        ),
                    )
                    observable_response_gates[family_name] = gate
                    unavailable_reasons.extend(
                        f"{family_name}_observable_response_gate:{reason}"
                        for reason in gate["reason_codes"]
                    )
        if (
            not unavailable_reasons
            and capability_id in STRUCTURAL_CAPABILITIES
        ):
            structural_reachability = structural_calibration_reachability(
                dataset,
                anchors,
                capability_id=capability_id,
                family_role="primary",
                lambda_value=float(primary_lambdas[-1]),
                calibration_seed_count=response_seed_count,
            )
            unavailable_reasons.extend(
                str(reason)
                for reason in structural_reachability["reason_codes"]
            )

        qualification_path_policy = {
            "policy": (
                "independent_family_response_qualification_bank_"
                "fixed_base_hard_failure_only_expansion_v1"
            ),
            "path_sampling": {
                "anchor": "independent_qualification_anchor_hash_v1",
                "rng": "independent_qualification_path_v1",
                "seed_start": 0,
            },
            "initial_path_count": initial_response_seed_count,
            "maximum_path_count": maximum_response_seed_count,
            "attempted_path_counts": attempted_path_counts,
            "selected_path_count": response_seed_count,
            "expanded": bool(
                response_seed_count > initial_response_seed_count
            ),
            "hard_failure_attempts": hard_failure_attempts,
            "split_half_diagnostics_trigger_expansion": False,
        }
        real_alignment_reference = {
            "policy": (
                "joint_real_primary_secondary_overlap_required_"
                "no_generator_relative_fallback_v1"
            ),
            "real_reference_available": True,
            "real_interval_q10_q90": real_interval,
            "primary_family_mean_response_interval": (
                primary_generator_interval
            ),
            "secondary_family_mean_response_interval": (
                secondary_generator_interval
            ),
            "overlap_interval": real_overlap_interval,
            "overlap_fraction_of_real_q10_q90": overlap_fraction,
            "used_for_family_targets": not unavailable_reasons,
            "minimum_overlap_fraction_for_use": (
                MINIMUM_REAL_OVERLAP_FRACTION
            ),
            "mapped_lambda_span_fraction": {
                "primary": primary_mapped_span_fraction,
                "secondary": secondary_mapped_span_fraction,
            },
            "mapped_lambda_span_usage": (
                "diagnostic_only_observable_response_gate_replaces_threshold"
                if capability_id
                in OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY
                else "legacy_hard_minimum"
            ),
            "minimum_lambda_span_fraction_for_use": (
                None
                if capability_id
                in OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY
                else MINIMUM_MAPPED_LAMBDA_SPAN_FRACTION
            ),
            "observable_response_separation_policy": (
                OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY.get(
                    capability_id
                )
            ),
            "unavailable_reason_codes": unavailable_reasons,
            "formal_seed_inverse": False,
            "sample_level_alignment_enforced": False,
        }
        primary_record = {
            "lambda_support": primary_support["effective_lambda_support"],
            "support_detection": primary_support,
            "lambda_grid": primary_grid.tolist(),
            "response_curve": primary_response.tolist(),
            "selected_lambdas": list(primary_lambdas),
            "selected_target_values": (
                primary_targets.tolist() if primary_lambdas else []
            ),
            "selected_lambda_mean_response": list(
                primary_selected_response
            ),
            "selected_lambda_response_gate": {
                "policy": "qualification_bank_family_mean_response_v1",
                "accepted": not any(inverse_failure_reasons.values()),
                "failure_reasons": inverse_failure_reasons,
            },
            "selected_lambda_observable_separation_gate": (
                observable_response_gates.get("primary")
            ),
            "structural_gate_reachability": structural_reachability,
        }
        secondary_record = {
            "calibration_status": (
                "family_mean_inverse_on_joint_real_generator_overlap"
                if not unavailable_reasons
                else "unavailable"
            ),
            "lambda_support": secondary_support[
                "effective_lambda_support"
            ],
            "support_detection": secondary_support,
            "lambda_grid": secondary_grid.tolist(),
            "response_curve": secondary_response.tolist(),
            "selected_lambdas": list(secondary_lambdas),
            "selected_target_values": list(secondary_selected_response),
            "matched_primary_target_values": (
                primary_targets.tolist() if secondary_lambdas else []
            ),
            "selected_lambda_observable_separation_gate": (
                observable_response_gates.get("secondary")
            ),
        }
        if unavailable_reasons:
            capabilities[capability_id] = {
                **base_record,
                "available_for_generation": False,
                "availability_status": "unavailable",
                "unavailable_reason_codes": unavailable_reasons,
                "qualification_path_count": response_seed_count,
                "qualification_path_policy": qualification_path_policy,
                "intensity_calibration_scope": (
                    "dataset_real_calibration_unavailable"
                ),
                "calibration_status": "unavailable",
                "real_interval_q10_q90": real_interval,
                "real_alignment_reference": real_alignment_reference,
                "structural_gate_reachability": structural_reachability,
                "primary": primary_record,
                "secondary": secondary_record,
            }
            unavailable_capabilities[capability_id] = unavailable_reasons
            continue

        capabilities[capability_id] = {
            **base_record,
            "available_for_generation": True,
            "availability_status": "available",
            "unavailable_reason_codes": [],
            "qualification_path_count": response_seed_count,
            "qualification_path_policy": qualification_path_policy,
            "intensity_calibration_scope": (
                "dataset_real_generator_overlap_reference"
            ),
            "calibration_status": (
                "family_mean_inverse_on_joint_real_generator_overlap"
            ),
            "real_interval_q10_q90": real_interval,
            "real_alignment_reference": real_alignment_reference,
            "structural_gate_reachability": structural_reachability,
            "primary": primary_record,
            "secondary": secondary_record,
        }
        available_capabilities.append(capability_id)
    return {
        "schema_version": "cafe.capability_calibration.v1",
        "generator_version": GENERATOR_VERSION,
        "intensity_policy": {
            "policy_id": (
                "dataset-real-joint-family-overlap-no-fallback-v1"
            ),
            "real_feature_quantiles": [0.10, 0.90],
            "minimum_real_feature_count": MIN_REAL_FEATURE_COUNT,
            "minimum_overlap_fraction_of_real_span": (
                MINIMUM_REAL_OVERLAP_FRACTION
            ),
            "minimum_mapped_lambda_span_fraction": (
                MINIMUM_MAPPED_LAMBDA_SPAN_FRACTION
            ),
            "mapped_lambda_span_policy": {
                "diagnostic_only_capabilities": sorted(
                    OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY
                ),
                "legacy_hard_gate_capabilities": sorted(
                    set(CAPABILITIES)
                    - set(
                        OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY
                    )
                ),
                "observable_response_separation_by_capability": (
                    OBSERVABLE_RESPONSE_SEPARATION_POLICY_BY_CAPABILITY
                ),
            },
            "generator_relative_fallback_allowed": False,
            "unavailable_cell_generation_policy": "skip",
        },
        "qualification_path_sampling_policy": {
            "anchor": "independent_qualification_anchor_hash_v1",
            "rng": "independent_qualification_path_v1",
            "seed_start": 0,
        },
        "qualification_path_count": calibration_seed_count,
        "maximum_qualification_path_count": (
            int(maximum_calibration_seed_count)
            if maximum_calibration_seed_count is not None
            else int(calibration_seed_count)
        ),
        "feature_summary": feature_summary,
        "available_capabilities": available_capabilities,
        "unavailable_capabilities": unavailable_capabilities,
        "capabilities": capabilities,
    }


def mase_scales(
    target: np.ndarray,
    *,
    season_length: int,
) -> tuple[float, list[float]]:
    policy = mase_scale_policy(target, season_length=season_length)
    return (
        float(policy["scale"]),
        [float(value) for value in policy["scale_by_target"]],
    )


def mase_scale_policy(
    target: np.ndarray,
    *,
    season_length: int,
) -> dict[str, Any]:
    """Resolve finite MASE scales without flooring deterministic sequences.

    An exactly seasonal deterministic history has a zero seasonal-naive
    denominator even though its lag-one variation is well defined.  In that
    degenerate case, use the standard non-seasonal MASE denominator for only
    the affected target and record the effective period explicitly.
    """

    history = np.asarray(target, dtype=float)[:CONTEXT_LENGTH]
    period = int(season_length)
    if not 1 <= period < len(history):
        raise ValueError("cafe MASE period must be defined inside L336")
    differences = np.abs(history[period:] - history[:-period])
    by_target = np.mean(differences, axis=0)
    if not np.isfinite(by_target).all():
        raise ValueError("cafe MASE denominator is non-finite")
    effective_periods = np.full(by_target.shape, period, dtype=int)
    fallback_mask = by_target <= 1e-12
    if np.any(fallback_mask):
        lag_one = np.mean(np.abs(np.diff(history, axis=0)), axis=0)
        if (
            not np.isfinite(lag_one[fallback_mask]).all()
            or np.any(lag_one[fallback_mask] <= 1e-12)
        ):
            raise ValueError(
                "cafe seasonal and lag-one MASE denominators are zero or "
                "non-finite"
            )
        by_target[fallback_mask] = lag_one[fallback_mask]
        effective_periods[fallback_mask] = 1
    if np.any(by_target <= 1e-12):
        raise ValueError("cafe MASE denominator is zero or non-finite")
    return {
        "policy": "seasonal_lag_with_lag1_degeneracy_fallback_v1",
        "requested_period": period,
        "effective_period_by_target": [
            int(value) for value in effective_periods
        ],
        "fallback_target_indices": [
            int(index) for index in np.flatnonzero(fallback_mask)
        ],
        "scale": float(np.mean(by_target)),
        "scale_by_target": [
            float(value) for value in by_target
        ],
    }


def target_and_covariate_sha256(
    target: np.ndarray,
    covariates: np.ndarray | None,
) -> str:
    digest = hashlib.sha256(
        np.asarray(target, dtype="<f8").tobytes()
    )
    if covariates is not None:
        digest.update(np.asarray(covariates, dtype="<f8").tobytes())
    return digest.hexdigest()


def anchor_for_seed(
    anchors: list[dict[str, Any]],
    *,
    dataset_id: str,
    capability_id: str,
    seed_index: int,
) -> dict[str, Any]:
    index = stable_seed(
        dataset_id,
        capability_id,
        seed_index,
        "anchor",
        base=CALIBRATION_SAMPLE_SEED,
    ) % len(anchors)
    return anchors[index]


def anchor_for_qualification_path(
    anchors: list[dict[str, Any]],
    *,
    dataset_id: str,
    capability_id: str,
    seed_index: int,
) -> dict[str, Any]:
    index = stable_seed(
        dataset_id,
        capability_id,
        seed_index,
        "qualification-anchor",
        base=QUALIFICATION_PATH_SEED,
    ) % len(anchors)
    return anchors[index]


def generate_master_sample(
    dataset: DatasetSpec,
    anchor: dict[str, Any],
    capability_calibration: dict[str, Any],
    *,
    capability_id: str,
    family_role: str,
    intensity: int,
    seed_index: int,
    counterfactual_member: int | None,
    evaluation_table: str = "main",
    generation_attempt: int = 0,
) -> dict[str, Any]:
    if generation_attempt < 0:
        raise ValueError("generation_attempt must be non-negative")
    if (
        capability_calibration.get("available_for_generation") is not True
        or capability_calibration.get("intensity_calibration_scope")
        != "dataset_real_generator_overlap_reference"
    ):
        raise ValueError(
            f"{dataset.dataset_id}/{capability_id} has no approved "
            "real-calibrated intensity grid"
        )
    parameters, mappings = derive_deterministic_parameters(
        capability_id,
        anchor_summary(
            anchor_feature_values(
                anchor,
                capability_id=capability_id,
            )
        ),
        season_length=int(anchor["feature_period"]),
        context_length=CONTEXT_LENGTH,
    )
    mappings = parameter_mapping_provenance(
        mappings,
        anchor,
        capability_id=capability_id,
    )
    family_calibration = capability_calibration[family_role]
    lambdas = tuple(
        float(value)
        for value in family_calibration["selected_lambdas"]
    )
    if len(lambdas) != len(INTENSITIES):
        raise ValueError("cafe intensity lambdas must contain five values")
    targets = tuple(
        float(value)
        for value in (
            family_calibration.get("selected_target_values")
            or family_calibration.get("matched_primary_target_values")
        )
    )
    conditioning = build_conditioning(
        dataset,
        capability_id=capability_id,
        frequency=str(anchor["frequency"]),
        season_length=int(anchor["feature_period"]),
        intensity_lambdas=lambdas,
        parameters=parameters,
        target_values=targets,
    )
    path_seed = (
        stable_seed(
            dataset.dataset_id,
            capability_id,
            seed_index,
            "generation-path",
            base=GENERATION_PATH_SEED,
        )
        if generation_attempt == 0
        else stable_seed(
            dataset.dataset_id,
            capability_id,
            seed_index,
            "generation-path-retry",
            generation_attempt,
            base=GENERATION_PATH_SEED,
        )
    )
    target, metadata, covariates = generate_deterministic_sample(
        capability_id,
        MASTER_LENGTH,
        CONTEXT_LENGTH,
        conditioning.target_dim,
        conditioning.season_length,
        intensity,
        np.random.default_rng(path_seed),
        conditioning=conditioning,
        family_role=family_role,
        counterfactual_variant=int(counterfactual_member or 0),
    )
    target, covariates = standardize_generated_sample(
        capability_id,
        target,
        covariates,
        metadata=metadata,
    )
    features = measured_features(
        capability_id,
        target,
        covariates,
        season_length=conditioning.season_length,
        metadata=metadata,
    )
    mase_policy = mase_scale_policy(
        target,
        season_length=int(anchor["mase_period"]),
    )
    mase_scale = float(mase_policy["scale"])
    scale_by_target = list(mase_policy["scale_by_target"])
    dataset_token = safe_id(dataset.dataset_id)
    pair_id = (
        f"cafe__{dataset_token}__{capability_id}__{family_role}__"
        f"i{intensity}__seed{seed_index:06d}"
        if (
            capability_id in COUNTERFACTUAL_CAPABILITIES
            and counterfactual_member is not None
        )
        else None
    )
    table_suffix = (
        ""
        if evaluation_table == "main"
        else f"__{safe_id(evaluation_table)}"
    )
    member_suffix = (
        f"__m{int(counterfactual_member)}"
        if counterfactual_member is not None
        else ""
    )
    sample_id = (
        f"cafe__{dataset_token}__{capability_id}__{family_role}__"
        f"i{intensity}__seed{seed_index:06d}{member_suffix}{table_suffix}"
    )
    target_hash = target_and_covariate_sha256(target, covariates)
    return {
        "schema_version": "cafe.master_sample.v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "master_sample_id": sample_id,
        "paired_group_id": (
            f"cafe__{dataset_token}__{capability_id}__{family_role}__"
            f"seed{seed_index:06d}"
        ),
        "counterfactual_pair_id": pair_id,
        "counterfactual_member": counterfactual_member,
        "dataset_id": dataset.dataset_id,
        "config_id": dataset.config_id,
        "task_id": dataset.task_view_id,
        "task_view_id": dataset.task_view_id,
        "profile_id": conditioning.profile_id,
        "anchor_id": anchor["anchor_id"],
        "anchor_provenance": {
            key: anchor[key]
            for key in (
                "item_id",
                "series_id",
                "channel_id",
                "window_start",
                "observed_fraction",
                "history_sha256",
            )
        },
        "generator_version": GENERATOR_VERSION,
        "generator_family_role": family_role,
        "generator_family_id": metadata["generator_family_id"],
        "capability_id": capability_id,
        "intensity": int(intensity),
        "seed_index": int(seed_index),
        "sample_index": int(seed_index),
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "target_dim": int(target.shape[1]),
        "covariate_dim": (
            0 if covariates is None else int(covariates.shape[1])
        ),
        "covariate_column_names": (
            []
            if covariates is None
            else ["weather_driver", "known_event"][: covariates.shape[1]]
        ),
        "frequency": conditioning.frequency,
        "season_length": int(anchor["calendar_season_length"]),
        "calendar_season_length": int(
            anchor["calendar_season_length"]
        ),
        "calendar_season_feature_observable": bool(
            anchor["calendar_season_feature_observable"]
        ),
        "calendar_cycles_in_calibration_history": float(
            anchor["calendar_cycles_in_calibration_history"]
        ),
        "feature_period": int(anchor["feature_period"]),
        "feature_period_source": str(anchor["feature_period_source"]),
        "hierarchy": (
            "additive_first"
            if capability_id == "hierarchical_coherence"
            else None
        ),
        "target_feature": PRIMARY_TARGET_FEATURE[capability_id],
        "target_feature_value": float(
            features[PRIMARY_TARGET_FEATURE[capability_id]]
        ),
        "intensity_target_feature_value": float(
            targets[intensity - 1]
        ),
        "intensity_lambda": float(lambdas[intensity - 1]),
        "intensity_calibration": {
            "policy": (
                "dataset_real_joint_family_mean_response_inverse_"
                "no_fallback_v1"
            ),
            "scope": capability_calibration[
                "intensity_calibration_scope"
            ],
            "formal_seed_inverse": False,
            "sample_level_target_gate": False,
            "selected_lambdas": list(lambdas),
            "reference_target_values": list(targets),
        },
        "realized_features": features,
        "sampled_generator_parameters": parameters,
        "parameter_mapping": mappings,
        "parameter_sampling": {
            "policy": "direct_real_anchor_feature_row",
            "anchor_id": anchor["anchor_id"],
            "path_seed": path_seed,
            "generation_attempt": int(generation_attempt),
            "attempt_rng_policy": (
                "formal_generation_path_v1"
                if generation_attempt == 0
                else "stable_generation_path_retry_v1"
            ),
        },
        "generation_metadata": metadata,
        "evaluation_table": evaluation_table,
        "input_history_semantics": "clean_latent",
        "scoring_target_semantics": "clean_latent_future",
        "observation_noise_scale": 0.0,
        "future_process_noise_scale": 0.0,
        "mase_period": int(anchor["mase_period"]),
        "mase_period_source": str(anchor["mase_period_source"]),
        "mase_scale": mase_scale,
        "mase_scale_by_target": scale_by_target,
        "mase_scale_effective_period_by_target": list(
            mase_policy["effective_period_by_target"]
        ),
        "mase_scale_fallback_target_indices": list(
            mase_policy["fallback_target_indices"]
        ),
        "mase_scale_policy": str(mase_policy["policy"]),
        "mase_scale_source": (
            "clean_l336_history_seasonal_lag_with_lag1_"
            "degeneracy_fallback"
        ),
        "target_sha256": target_hash,
        "future_sha256": hashlib.sha256(
            np.asarray(target[CONTEXT_LENGTH:], dtype="<f8").tobytes()
        ).hexdigest(),
        "target": target.tolist(),
        "covariates": (
            None if covariates is None else covariates.tolist()
        ),
    }


def _affine_match_history(
    donor: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    donor_values = np.asarray(donor, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    donor_center = float(np.mean(donor_values))
    donor_scale = float(np.std(donor_values))
    reference_center = float(np.mean(reference_values))
    reference_scale = float(np.std(reference_values))
    if donor_scale <= 1e-12:
        return np.full_like(donor_values, reference_center)
    return (
        (donor_values - donor_center)
        * reference_scale
        / donor_scale
        + reference_center
    )


def multivariate_input_ablation_sample(
    clean: dict[str, Any],
    donor: dict[str, Any],
) -> dict[str, Any]:
    """Break cross-channel alignment while preserving marginal scale.

    The future and the focal channel history stay untouched.  Common-factor
    auxiliaries are replaced by an affine-matched donor history.  For the
    delayed SCM only the driver segment that causally covers the forecast is
    replaced.  This yields a diagnostic input corruption, not another scored
    data-generating process.
    """

    capability_id = str(clean["capability_id"])
    if capability_id not in INPUT_ABLATION_CAPABILITIES:
        raise ValueError(
            f"input ablation is unsupported for {capability_id}"
        )
    matching_fields = (
        "dataset_id",
        "capability_id",
        "generator_family_role",
        "intensity",
        "context_length",
        "horizon",
        "target_dim",
    )
    if any(clean[field] != donor[field] for field in matching_fields):
        raise ValueError("input-ablation donor does not match recipient")
    context = int(clean["context_length"])
    target = np.asarray(clean["target"], dtype=float).copy()
    donor_target = np.asarray(donor["target"], dtype=float)
    metadata = clean["generation_metadata"]
    replaced_channels: list[int]
    replaced_slice: list[int]
    if capability_id == "common_factor":
        protected = int(metadata["protected_target_index"])
        replaced_channels = [
            index
            for index in range(int(clean["target_dim"]))
            if index != protected
        ]
        replaced_slice = [0, context]
        for channel in replaced_channels:
            target[:context, channel] = _affine_match_history(
                donor_target[:context, channel],
                target[:context, channel],
            )
        ablation_type = "affine_matched_auxiliary_history_donor"
    else:
        driver = int(metadata["driver_index"])
        driver_slice = metadata.get("counterfactual_driver_slice")
        if (
            not isinstance(driver_slice, list)
            or len(driver_slice) != 2
            or int(driver_slice[1]) != context
        ):
            raise ValueError(
                "cross-series input ablation requires a boundary-aligned "
                "counterfactual driver slice"
            )
        start = int(driver_slice[0])
        width = context - start
        replaced_channels = [driver]
        replaced_slice = [start, context]
        donor_context = donor_target.shape[0] - int(donor["horizon"])
        donor_start = donor_context - width
        donor_reference = donor_target[:donor_start, driver]
        clean_reference = target[:start, driver]
        donor_center = float(np.mean(donor_reference))
        donor_scale = max(float(np.std(donor_reference)), 1e-12)
        clean_center = float(np.mean(clean_reference))
        clean_scale = max(float(np.std(clean_reference)), 1e-12)
        target[start:context, driver] = (
            (
                donor_target[donor_start:donor_context, driver]
                - donor_center
            )
            * clean_scale
            / donor_scale
            + clean_center
        )
        ablation_type = (
            "invariant_prefix_affine_matched_effect_trigger_driver_donor"
        )

    result = json.loads(json.dumps(clean))
    result["schema_version"] = "cafe.input_ablation_sample.v1"
    result["sample_id"] = clean["sample_id"] + "__mv_input_ablation"
    result["master_sample_id"] = result["sample_id"]
    result["clean_master_sample_id"] = clean["sample_id"]
    result["input_ablation_group_id"] = clean["sample_id"]
    result["counterfactual_pair_id"] = None
    result["counterfactual_member"] = None
    result["evaluation_table"] = "multivariate_input_ablation"
    result["input_history_semantics"] = "marginal_matched_cross_channel_ablation"
    result["scoring_target_semantics"] = "original_clean_latent_future"
    result["input_ablation_metadata"] = {
        "ablation_type": ablation_type,
        "donor_sample_id": donor["sample_id"],
        "replaced_channels": replaced_channels,
        "replaced_history_slice": replaced_slice,
        "focal_history_unchanged": True,
        "future_unchanged": True,
        "affine_matched_mean_and_std": (
            capability_id == "common_factor"
        ),
        "affine_reference_scope": (
            "replaced_segment"
            if capability_id == "common_factor"
            else "pair_invariant_driver_prefix"
        ),
    }
    result["target"] = target.tolist()
    result["target_sha256"] = target_and_covariate_sha256(
        target,
        (
            None
            if result["covariates"] is None
            else np.asarray(result["covariates"], dtype=float)
        ),
    )
    return result


def robustness_sample(clean: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(clean["target"], dtype=float)
    observed, noise_metadata = add_observation_noise_to_history(
        target,
        context_length=CONTEXT_LENGTH,
        noise_ratio=ROBUSTNESS_NOISE_RATIO,
        rng=np.random.default_rng(
            stable_seed(
                clean["sample_id"],
                "observation-noise",
                base=ROBUSTNESS_SEED,
            )
        ),
        noise_scale_by_target=np.asarray(
            clean["mase_scale_by_target"],
            dtype=float,
        ),
        noise_scale_source=(
            "clean_l336_mase_denominator_by_target"
        ),
        preserve_additive_hierarchy=(
            clean["capability_id"] == "hierarchical_coherence"
        ),
    )
    result = json.loads(json.dumps(clean))
    result["schema_version"] = "cafe.robustness_master_sample.v1"
    result["sample_id"] = clean["sample_id"] + "__robust15"
    result["master_sample_id"] = result["sample_id"]
    result["clean_master_sample_id"] = clean["sample_id"]
    result["paired_group_id"] = str(clean["paired_group_id"]) + "__robust15"
    if clean.get("counterfactual_pair_id") is not None:
        result["counterfactual_pair_id"] = (
            str(clean["counterfactual_pair_id"]) + "__robust15"
        )
    result["evaluation_table"] = "observation_noise_robustness"
    result["input_history_semantics"] = "noisy_observation"
    result["scoring_target_semantics"] = "clean_latent_future"
    result["observation_noise_scale"] = ROBUSTNESS_NOISE_RATIO
    result["observation_noise_scale_policy"] = (
        "ratio_times_clean_l336_mase_denominator_by_target"
    )
    result["observation_noise_metadata"] = noise_metadata
    result["target"] = observed.tolist()
    result["target_sha256"] = target_and_covariate_sha256(
        observed,
        (
            None
            if result["covariates"] is None
            else np.asarray(result["covariates"], dtype=float)
        ),
    )
    return result


def _shift_generation_metadata(
    metadata: dict[str, Any],
    *,
    capability_id: str,
    context_length: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(metadata))
    offset = CONTEXT_LENGTH - context_length
    if capability_id == "common_factor":
        for name in (
            "final_code_slice",
            "shared_state_evidence_slice",
        ):
            if name in result:
                result[name] = [
                    int(value) - offset for value in result[name]
                ]
    elif capability_id == "cross_series_dependence":
        delay = int(result["cross_lag_steps"])
        result["counterfactual_driver_slice"] = [
            context_length - delay,
            context_length,
        ]
        effect_steps = int(
            result.get("counterfactual_effect_forecast_steps", HORIZON)
        )
        result["counterfactual_effect_future_slice"] = [
            context_length,
            context_length + effect_steps,
        ]
        result["counterfactual_unaffected_future_slice"] = [
            context_length + effect_steps,
            context_length + HORIZON,
        ]
    elif capability_id == "regime_switching":
        result["cut_points"] = [
            int(value) - offset for value in result.get("cut_points", [])
        ]
    elif capability_id == "predictable_intermittency":
        result["pulse_centers"] = [
            int(value) - offset
            for value in result.get("pulse_centers", [])
        ]
    return result


def master_view(
    master: dict[str, Any],
    context_length: int,
) -> dict[str, Any]:
    if context_length not in VIEW_CONTEXT_LENGTHS:
        raise ValueError(f"unsupported cafe context view {context_length}")
    if int(master["context_length"]) != CONTEXT_LENGTH:
        raise ValueError("cafe inference views require an L336 master")
    start = CONTEXT_LENGTH - context_length
    target = np.asarray(master["target"], dtype=float)[start:]
    covariates = (
        None
        if master.get("covariates") is None
        else np.asarray(master["covariates"], dtype=float)[start:]
    )
    result = json.loads(json.dumps(master))
    result["schema_version"] = "cafe.forecast_view.v1"
    result["source_master_sample_id"] = master["sample_id"]
    result["master_sample_id"] = master["sample_id"]
    result["sample_id"] = f"{master['sample_id']}__L{context_length}"
    result["view_id"] = result["sample_id"]
    result["context_length"] = context_length
    result["context_policy_candidates"] = list(VIEW_CONTEXT_LENGTHS)
    result["view_standardization_policy"] = (
        "slice_exact_l336_standardized_master_without_restandardization"
    )
    result["target"] = target.tolist()
    result["covariates"] = (
        None if covariates is None else covariates.tolist()
    )
    result["generation_metadata"] = _shift_generation_metadata(
        master["generation_metadata"],
        capability_id=str(master["capability_id"]),
        context_length=context_length,
    )
    if master.get("counterfactual_pair_id") is not None:
        result["master_counterfactual_pair_id"] = master[
            "counterfactual_pair_id"
        ]
        result["counterfactual_pair_id"] = (
            f"{master['counterfactual_pair_id']}__L{context_length}"
        )
    result["target_sha256"] = target_and_covariate_sha256(
        target,
        covariates,
    )
    result["future_sha256"] = hashlib.sha256(
        np.asarray(target[context_length:], dtype="<f8").tobytes()
    ).hexdigest()
    result["master_future_sha256"] = master["future_sha256"]
    return result


def iter_master_views(
    masters: Iterable[dict[str, Any]],
    *,
    context_lengths: Iterable[int] = VIEW_CONTEXT_LENGTHS,
) -> Iterator[dict[str, Any]]:
    contexts = tuple(int(value) for value in context_lengths)
    for master in masters:
        for context_length in contexts:
            yield master_view(master, context_length)
