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


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_path in (BACKEND_ROOT, REPO_ROOT / "scripts"):
    if str(import_path) not in os.sys.path:
        os.sys.path.insert(0, str(import_path))

from app.services.metric_service import seasonal_period_for_frequency  # noqa: E402
from app.services.synthetic_generation_service import (  # noqa: E402
    _normalize_covariates,
    _realized_features,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    GeneratorConditioning,
    REAL_BOUNDED_INTENSITY_POLICY_ID,
)
from app.services.synthetic_v8_generation import (  # noqa: E402
    GENERATOR_VERSION,
    PRIMARY_FAMILY_BY_CAPABILITY,
    SECONDARY_FAMILY_BY_CAPABILITY,
    add_observation_noise_to_history,
    derive_deterministic_parameters,
    generate_deterministic_sample,
    standardize_common_factor_counterfactual_member,
    standardize_cross_series_counterfactual_member,
)
from paper_v2_transfer_common import impute_observed_window  # noqa: E402
from synthetic_feature_profile import (  # noqa: E402
    adjusted_r2,
    feature_vector,
    file_sha256,
    read_gift_arrow_targets,
    robust_scale,
)


SCHEMA_VERSION = "paper_v8_pipeline.v5"
CONTEXT_LENGTH = 504
HORIZON = 48
MASTER_LENGTH = CONTEXT_LENGTH + HORIZON
VIEW_CONTEXT_LENGTHS = (96, 168, 336, 504)
INTENSITIES = (1, 2, 3, 4, 5)
QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CALIBRATION_SAMPLE_SEED = 2026072401
CALIBRATION_PATH_SEED = 2026072402
GENERATION_PATH_SEED = 2026072403
ROBUSTNESS_SEED = 2026072404
ROBUSTNESS_NOISE_RATIO = 0.15
NONLINEAR_PATH_SUPPORT_QUANTILE = 0.10
DEFAULT_CALIBRATION_PATH_COUNT = 32
MAX_CALIBRATION_PATH_COUNT = 96
CALIBRATION_PATH_EXPANSION_STEP = 32
DEFAULT_NONLINEAR_CALIBRATION_PATH_COUNT = 64
MAX_NONLINEAR_CALIBRATION_PATH_COUNT = 128
NONLINEAR_CALIBRATION_PATH_EXPANSION_STEP = 64
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
INPUT_ABLATION_CAPABILITIES = STRICT_COUNTERFACTUAL_CAPABILITIES
STRUCTURAL_CAPABILITIES = frozenset(
    {
        "common_factor",
        "hierarchical_coherence",
        "cross_series_dependence",
        "covariate_response",
    }
)
SYNTHETIC_DOSE_CAPABILITIES = STRUCTURAL_CAPABILITIES | frozenset(
    {"predictable_intermittency"}
)
CAPABILITIES = tuple(PRIMARY_FAMILY_BY_CAPABILITY)
PRIMARY_TARGET_FEATURE = {
    "trend": "curvature_abs",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "regime_sparse_transition_score",
    "nonlinear_persistence": "nonlinear_conditional_gain",
    # Thresholded spike counts and event-clock R² are both zero-inflated on
    # finite L504 windows.  The generator-known history event-component energy
    # share is continuous and exactly monotone in event prominence; clock
    # recoverability remains a separate observable structural diagnostic.
    "predictable_intermittency": "event_effect_energy_share",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    # Strength is calibrated by the strongest ordered lagged association.
    # Correct edge/lag recovery and incremental prediction remain separate
    # generated-sample gates; folding them into the dose coordinate made the
    # five levels depend on a high-dimensional ridge design.
    "cross_series_dependence": "lead_lag_peak_abs",
    # A current-linear incremental R² is not family-neutral for nonlinear or
    # distributed-lag responses.  The generator-known history effect share is
    # the matched dose; incremental R² remains a descriptive audit feature.
    "covariate_response": "covariate_effect_variance_share",
}
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
            "gift_hierarchical_sales_d",
            "Hierarchical Sales",
            "hierarchical_sales/D",
            "hierarchical_sales/D",
            "Business",
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
            f"unknown v8 dataset {dataset_id!r}; registered="
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
    window_length: int = CONTEXT_LENGTH,
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
            raise ValueError("invalid v8 calibration stratum")
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


def v8_nonlinear_conditional_gain(
    values: np.ndarray,
    season_length: int,
) -> tuple[float, int, tuple[int, ...]]:
    """Measure generic nonlinear persistence over plausible relative lags.

    The pre-v4 proxy fixed the nonlinear lag to ``season_length // 2`` and
    added one sine-squared term.  V8 generators intentionally randomize their
    actual lag and use either shifted-tanh or rational responses, so that proxy
    could react more strongly at I3 than I5 merely because a correlated,
    incorrect lag happened to fit one trajectory.

    Search a small, fixed set of season-relative lags and use quadratic/cubic
    terms.  This remains family-neutral and is also available for real windows,
    while covering the smooth bounded response families used by V8 without
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


def v8_nonlinear_actual_lag_gain(
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

    A sub-daily calendar season may be much longer than L504.  In that case it
    remains provenance, the real-window spectral peak supplies an observable
    feature period, and non-seasonal MASE uses lag one.
    """

    history = np.asarray(standardized_history, dtype=float).reshape(-1)
    calendar_period = int(seasonal_period_for_frequency(frequency))
    provisional = feature_vector(
        history[:, None],
        None,
        context_length=CONTEXT_LENGTH,
    )
    raw_dominant = float(provisional.get("dominant_period", 24.0))
    if not math.isfinite(raw_dominant) or raw_dominant <= 0.0:
        raw_dominant = 24.0
    profile_period = int(
        round(
            np.clip(
                raw_dominant,
                8.0,
                CONTEXT_LENGTH / 3.0,
            )
        )
    )
    calendar_feature_observable = bool(
        calendar_period >= 2
        and 2 * calendar_period <= CONTEXT_LENGTH
    )
    feature_period = (
        calendar_period
        if calendar_feature_observable
        else profile_period
    )
    mase_period = (
        calendar_period
        if 1 <= calendar_period < CONTEXT_LENGTH
        else 1
    )
    return {
        "calendar_season_length": calendar_period,
        "calendar_season_feature_observable": (
            calendar_feature_observable
        ),
        "calendar_cycles_in_l504": float(
            CONTEXT_LENGTH / max(calendar_period, 1)
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
            else "nonseasonal_lag1_calendar_unobservable_in_l504"
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


def build_calibration_anchors(
    dataset: DatasetSpec,
    *,
    gift_eval_dir: Path,
    maximum_anchors: int,
    sample_seed: int = CALIBRATION_SAMPLE_SEED,
    minimum_observed_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    asset_path = gift_eval_dir / dataset.asset_name
    frequency, native_records = read_gift_arrow_targets(asset_path)
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
        raw = np.asarray(values[start : start + CONTEXT_LENGTH], dtype=float)
        imputed, observed_fraction = impute_observed_window(
            raw,
            minimum_observed_fraction=minimum_observed_fraction,
        )
        if imputed is None:
            rejected_missing += 1
            continue
        try:
            standardized, location, scale = standardize_history(imputed)
        except ValueError:
            rejected_uninformative += 1
            continue
        period_policy = calibration_period_policy(
            frequency,
            standardized,
        )
        features = feature_vector(
            standardized[:, None],
            int(period_policy["feature_period"]),
            context_length=CONTEXT_LENGTH,
        )
        (
            nonlinear_gain,
            nonlinear_detected_lag,
            nonlinear_candidate_lags,
        ) = v8_nonlinear_conditional_gain(
            standardized,
            int(period_policy["feature_period"]),
        )
        features["nonlinear_conditional_gain"] = nonlinear_gain
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
        anchor_id = (
            f"{safe_id(dataset.dataset_id)}__{safe_id(item_id)}__"
            f"c{channel_index}__t{start}"
        )
        anchors.append(
            {
                "schema_version": "paper_v8_calibration_anchor.v2",
                "anchor_id": anchor_id,
                "dataset_id": dataset.dataset_id,
                "config_id": dataset.config_id,
                "task_view_id": dataset.task_view_id,
                "profile_id": (
                    f"{dataset.dataset_id}__{dataset.task_view_id}__"
                    f"L{CONTEXT_LENGTH}"
                ),
                "frequency": frequency,
                # ``season_length`` remains the literal calendar period for
                # external provenance. V8 generation and evaluation consume
                # the explicit fields below instead of overloading it.
                "season_length": int(
                    period_policy["calendar_season_length"]
                ),
                **period_policy,
                "item_id": item_id,
                "series_id": series_id,
                "channel_id": channel_index,
                "window_start": start,
                "context_length": CONTEXT_LENGTH,
                "observed_fraction": observed_fraction,
                "history_location": location,
                "history_scale": scale,
                "history_sha256": hashlib.sha256(
                    np.asarray(imputed, dtype="<f8").tobytes()
                ).hexdigest(),
                "features": finite_features,
            }
        )
        if len(anchors) >= target_count:
            break
    if not anchors:
        raise ValueError(f"{dataset.dataset_id} produced no valid v8 anchors")
    arrow_files = sorted(asset_path.glob("data-*.arrow"))
    metadata = {
        "dataset": asdict(dataset),
        "asset_path": str(asset_path),
        "asset_files": [file_record(path) for path in arrow_files],
        "frequency": frequency,
        "season_length": int(
            seasonal_period_for_frequency(frequency)
        ),
        "period_policy": {
            "calendar_season_length": int(
                seasonal_period_for_frequency(frequency)
            ),
            "calendar_feature_observable_rule": (
                "at_least_two_complete_calendar_cycles_in_l504"
            ),
            "profile_period_bounds": [8, CONTEXT_LENGTH // 3],
            "mase_fallback": (
                "lag1_when_calendar_period_is_not_defined_inside_l504"
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
        "window_policy": (
            "full-series non-overlapping capacity strata with deterministic "
            "without-replacement selection and within-stratum jitter"
        ),
    }
    return anchors, metadata


def anchor_summary(features: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        name: {"p50": float(value)}
        for name, value in features.items()
        if math.isfinite(float(value))
    }


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
        artifact_schema_version="paper_v8_calibration_bundle.v1",
        artifact_created_at=None,
        calibration_method="paper_v8_response_curve",
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
        target = _standardize_hierarchy_by_context(target, CONTEXT_LENGTH)
    else:
        target = _standardize_by_context(target, CONTEXT_LENGTH)
    if covariates is not None:
        covariates = _normalize_covariates(covariates, CONTEXT_LENGTH)
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
    values = _realized_features(
        target,
        covariates,
        measurement_period,
        CONTEXT_LENGTH,
    )
    values.update(
        feature_vector(
            target,
            measurement_period,
            covariates=covariates,
            context_length=CONTEXT_LENGTH,
            hierarchy=hierarchy,
            include_cross_series_predictability=(
                capability_id == "cross_series_dependence"
            ),
        )
    )
    if capability_id == "nonlinear_persistence":
        (
            nonlinear_gain,
            nonlinear_detected_lag,
            nonlinear_candidate_lags,
        ) = v8_nonlinear_conditional_gain(
            np.mean(target, axis=1),
            measurement_period,
        )
        values["nonlinear_conditional_gain"] = nonlinear_gain
        values["nonlinear_proxy_best_lag"] = float(
            nonlinear_detected_lag
        )
        values["nonlinear_proxy_candidate_lag_count"] = float(
            len(nonlinear_candidate_lags)
        )
        actual_lag = metadata.get("nonlinear_lag")
        if actual_lag is not None:
            values["nonlinear_actual_lag_gain"] = (
                v8_nonlinear_actual_lag_gain(
                    np.mean(target, axis=1),
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
    calibration_seed_index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    parameters, mappings = derive_deterministic_parameters(
        capability_id,
        anchor_summary(anchor["features"]),
        season_length=int(anchor["feature_period"]),
        context_length=CONTEXT_LENGTH,
    )
    conditioning = build_conditioning(
        dataset,
        capability_id=capability_id,
        frequency=str(anchor["frequency"]),
        season_length=int(anchor["feature_period"]),
        intensity_lambdas=(lambda_value,) * 5,
        parameters=parameters,
    )
    rng = np.random.default_rng(
        stable_seed(
            dataset.dataset_id,
            capability_id,
            calibration_seed_index,
            "response-path",
            base=CALIBRATION_PATH_SEED,
        )
    )
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
        counterfactual_variant=0,
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
        },
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
        anchor = anchors[seed_index % len(anchors)]
        for lambda_index, lambda_value in enumerate(full_grid):
            features, _metadata = generate_calibration_member(
                dataset,
                anchor,
                capability_id=capability_id,
                family_role=family_role,
                lambda_value=float(lambda_value),
                calibration_seed_index=seed_index,
            )
            path_responses[seed_index, lambda_index] = float(
                features[target_feature]
            )
    raw_response = np.mean(path_responses, axis=0)
    support_index, support = stable_monotone_support(
        full_grid,
        raw_response,
    )
    if capability_id == "nonlinear_persistence":
        path_support_indexes = np.asarray(
            [
                stable_monotone_support(full_grid, response)[0]
                for response in path_responses
            ],
            dtype=int,
        )
        path_quantile_index = int(
            np.quantile(
                path_support_indexes,
                NONLINEAR_PATH_SUPPORT_QUANTILE,
                method="lower",
            )
        )
        support_index = min(support_index, path_quantile_index)
        unique, counts = np.unique(
            path_support_indexes,
            return_counts=True,
        )
        support.update(
            {
                "path_support_policy": (
                    "lower_10pct_pathwise_stable_support_intersected_"
                    "with_mean_support"
                ),
                "path_support_quantile": (
                    NONLINEAR_PATH_SUPPORT_QUANTILE
                ),
                "path_support_quantile_index": path_quantile_index,
                "path_support_quantile_lambda": float(
                    full_grid[path_quantile_index]
                ),
                "path_support_index_counts": {
                    str(int(index)): int(count)
                    for index, count in zip(unique, counts, strict=True)
                },
            }
        )
    support["selected_peak_index"] = int(support_index)
    support["effective_lambda_support"] = [
        float(full_grid[0]),
        float(full_grid[support_index]),
    ]
    grid = full_grid[: support_index + 1]
    response = np.maximum.accumulate(
        raw_response[: support_index + 1]
    )
    support.update(
        {
            "raw_lambda_grid": full_grid.tolist(),
            "raw_response_curve": raw_response.tolist(),
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
    capability_id: str,
    *,
    initial_path_count: int,
    maximum_path_count: int,
) -> tuple[int, ...]:
    if initial_path_count < 1 or maximum_path_count < initial_path_count:
        raise ValueError("invalid v8 calibration path budget")
    step = (
        NONLINEAR_CALIBRATION_PATH_EXPANSION_STEP
        if capability_id == "nonlinear_persistence"
        else CALIBRATION_PATH_EXPANSION_STEP
    )
    counts = [int(initial_path_count)]
    while counts[-1] < maximum_path_count:
        counts.append(min(maximum_path_count, counts[-1] + step))
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


def calibrate_capabilities(
    dataset: DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    calibration_seed_count: int,
    maximum_calibration_seed_count: int | None = None,
    nonlinear_calibration_seed_count: int | None = None,
    maximum_nonlinear_calibration_seed_count: int | None = None,
    capability_ids: Iterable[str] = CAPABILITIES,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    feature_summary = summarize_feature_rows(
        anchor["features"] for anchor in anchors
    )
    capabilities: dict[str, Any] = {}
    for capability_id in capability_ids:
        if capability_id not in CAPABILITIES:
            raise ValueError(f"unsupported capability: {capability_id}")
        initial_response_seed_count = (
            int(nonlinear_calibration_seed_count)
            if (
                capability_id == "nonlinear_persistence"
                and nonlinear_calibration_seed_count is not None
            )
            else int(calibration_seed_count)
        )
        if capability_id == "nonlinear_persistence":
            maximum_response_seed_count = (
                int(maximum_nonlinear_calibration_seed_count)
                if maximum_nonlinear_calibration_seed_count is not None
                else initial_response_seed_count
            )
        else:
            maximum_response_seed_count = (
                int(maximum_calibration_seed_count)
                if maximum_calibration_seed_count is not None
                else initial_response_seed_count
            )
        path_schedule = calibration_path_schedule(
            capability_id,
            initial_path_count=initial_response_seed_count,
            maximum_path_count=maximum_response_seed_count,
        )
        attempted_path_counts: list[int] = []
        hard_failure_attempts: list[dict[str, Any]] = []
        target_feature = PRIMARY_TARGET_FEATURE[capability_id]
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
                if capability_id in SYNTHETIC_DOSE_CAPABILITIES:
                    primary_targets = np.linspace(
                        float(primary_response[0]),
                        float(primary_response[-1]),
                        5,
                    )
                    calibration_status = (
                        "fixed_cross_dataset_even_realized_strength_grid"
                        if capability_id in STRUCTURAL_CAPABILITIES
                        else (
                            "generator_relative_continuous_event_dose_grid"
                        )
                    )
                    real_interval = None
                else:
                    reference = feature_summary.get(target_feature)
                    if reference is None:
                        raise ValueError(
                            f"real feature {target_feature} unavailable for "
                            f"{dataset.dataset_id}/{capability_id}"
                        )
                    real_lower = float(reference["p10"])
                    real_upper = float(reference["p90"])
                    lower = max(
                        real_lower,
                        float(primary_response[0]),
                    )
                    upper = min(
                        real_upper,
                        float(primary_response[-1]),
                    )
                    if upper <= lower + 1e-10:
                        lower, upper = (
                            float(primary_response[0]),
                            float(primary_response[-1]),
                        )
                        calibration_status = (
                            "no_real_generator_intersection_"
                            "generator_support_used"
                        )
                    else:
                        calibration_status = (
                            "real_q10_q90_generator_intersection"
                        )
                    primary_targets = np.linspace(lower, upper, 5)
                    real_interval = [real_lower, real_upper]
                primary_lambdas = inverse_response_lambdas(
                    primary_grid,
                    primary_response,
                    primary_targets,
                )
                hard_failure_reasons["primary_inverse"] = (
                    inverse_mapping_hard_failure_reasons(
                        np.asarray(primary_targets, dtype=float),
                        primary_lambdas,
                    )
                )
            if not any(hard_failure_reasons.values()):
                break
            hard_failure_attempts.append(
                {
                    "path_count": int(response_seed_count),
                    "reasons": hard_failure_reasons,
                }
            )
        if any(hard_failure_reasons.values()):
            raise ValueError(
                "v8 response calibration failed after maximum path budget "
                f"for {dataset.dataset_id}/{capability_id}: "
                f"{hard_failure_reasons}"
            )
        secondary_lambdas = inverse_response_lambdas(
            secondary_grid,
            secondary_response,
            np.asarray(primary_targets, dtype=float),
        )
        secondary_status = "matched_primary_target_values"
        nonincreasing_secondary = any(
            right <= left + 1e-8
            for left, right in zip(
                secondary_lambdas,
                secondary_lambdas[1:],
            )
        )
        secondary_support_span = float(
            secondary_grid[-1] - secondary_grid[0]
        )
        nonlinear_audit_span = float(
            secondary_lambdas[4] - secondary_lambdas[2]
        )
        compressed_nonlinear_audit = bool(
            capability_id == "nonlinear_persistence"
            and secondary_support_span > 1e-12
            and nonlinear_audit_span < 0.30 * secondary_support_span
        )
        if nonincreasing_secondary or compressed_nonlinear_audit:
            secondary_lambdas = tuple(
                float(value)
                for value in np.linspace(
                    float(secondary_grid[0]),
                    float(secondary_grid[-1]),
                    5,
                )
            )
            secondary_status = (
                "nonlinear_secondary_compressed_match_"
                "fixed_relative_grid_used"
                if compressed_nonlinear_audit
                else (
                    "primary_targets_outside_secondary_support_"
                    "fixed_relative_grid_used"
                )
            )
        capabilities[capability_id] = {
            "target_feature": target_feature,
            "response_calibration_seed_count": response_seed_count,
            "response_calibration_path_policy": {
                "policy": (
                    "fixed_base_hard_failure_only_expansion_v1"
                ),
                "initial_path_count": initial_response_seed_count,
                "maximum_path_count": maximum_response_seed_count,
                "attempted_path_counts": attempted_path_counts,
                "selected_path_count": response_seed_count,
                "expanded": bool(
                    response_seed_count > initial_response_seed_count
                ),
                "hard_failure_attempts": hard_failure_attempts,
                "split_half_diagnostics_trigger_expansion": False,
            },
            "target_dim": TARGET_DIM_BY_CAPABILITY[capability_id],
            "primary_family": PRIMARY_FAMILY_BY_CAPABILITY[capability_id],
            "secondary_family": SECONDARY_FAMILY_BY_CAPABILITY[capability_id],
            "intensity_calibration_scope": (
                "generator_structural_relative_grid"
                if capability_id in STRUCTURAL_CAPABILITIES
                else (
                    "generator_continuous_event_dose_grid"
                    if capability_id == "predictable_intermittency"
                    else "dataset_real_feature_range"
                )
            ),
            "calibration_status": calibration_status,
            "real_interval_q10_q90": real_interval,
            "primary": {
                "lambda_support": primary_support[
                    "effective_lambda_support"
                ],
                "support_detection": primary_support,
                "lambda_grid": primary_grid.tolist(),
                "response_curve": primary_response.tolist(),
                "selected_lambdas": list(primary_lambdas),
                "selected_target_values": np.asarray(
                    primary_targets, dtype=float
                ).tolist(),
            },
            "secondary": {
                "calibration_status": secondary_status,
                "lambda_support": secondary_support[
                    "effective_lambda_support"
                ],
                "support_detection": secondary_support,
                "lambda_grid": secondary_grid.tolist(),
                "response_curve": secondary_response.tolist(),
                "selected_lambdas": list(secondary_lambdas),
                "matched_primary_target_values": np.asarray(
                    primary_targets, dtype=float
                ).tolist(),
            },
        }
    return {
        "schema_version": "paper_v8_capability_calibration.v2",
        "generator_version": GENERATOR_VERSION,
        "calibration_seed_count": calibration_seed_count,
        "maximum_calibration_seed_count": (
            int(maximum_calibration_seed_count)
            if maximum_calibration_seed_count is not None
            else int(calibration_seed_count)
        ),
        "nonlinear_calibration_seed_count": (
            int(nonlinear_calibration_seed_count)
            if nonlinear_calibration_seed_count is not None
            else int(calibration_seed_count)
        ),
        "maximum_nonlinear_calibration_seed_count": (
            int(maximum_nonlinear_calibration_seed_count)
            if maximum_nonlinear_calibration_seed_count is not None
            else (
                int(nonlinear_calibration_seed_count)
                if nonlinear_calibration_seed_count is not None
                else int(calibration_seed_count)
            )
        ),
        "feature_summary": feature_summary,
        "capabilities": capabilities,
    }


def mase_scales(
    target: np.ndarray,
    *,
    season_length: int,
) -> tuple[float, list[float]]:
    history = np.asarray(target, dtype=float)[:CONTEXT_LENGTH]
    period = int(season_length)
    if not 1 <= period < len(history):
        raise ValueError("v8 MASE period must be defined inside L504")
    differences = np.abs(history[period:] - history[:-period])
    by_target = np.mean(differences, axis=0)
    if not np.isfinite(by_target).all() or np.any(by_target <= 1e-12):
        raise ValueError("v8 MASE denominator is zero or non-finite")
    return float(np.mean(by_target)), [float(value) for value in by_target]


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
) -> dict[str, Any]:
    parameters, mappings = derive_deterministic_parameters(
        capability_id,
        anchor_summary(anchor["features"]),
        season_length=int(anchor["feature_period"]),
        context_length=CONTEXT_LENGTH,
    )
    family_calibration = capability_calibration[family_role]
    lambdas = tuple(
        float(value) for value in family_calibration["selected_lambdas"]
    )
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
    path_seed = stable_seed(
        dataset.dataset_id,
        capability_id,
        seed_index,
        "generation-path",
        base=GENERATION_PATH_SEED,
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
    mase_scale, scale_by_target = mase_scales(
        target,
        season_length=int(anchor["mase_period"]),
    )
    dataset_token = safe_id(dataset.dataset_id)
    pair_id = (
        f"v8__{dataset_token}__{capability_id}__{family_role}__"
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
        f"v8__{dataset_token}__{capability_id}__{family_role}__"
        f"i{intensity}__seed{seed_index:06d}{member_suffix}{table_suffix}"
    )
    target_hash = target_and_covariate_sha256(target, covariates)
    return {
        "schema_version": "paper_v8_master_sample.v2",
        "sample_id": sample_id,
        "master_sample_id": sample_id,
        "paired_group_id": (
            f"v8__{dataset_token}__{capability_id}__{family_role}__"
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
        "calendar_cycles_in_l504": float(
            anchor["calendar_cycles_in_l504"]
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
        "realized_features": features,
        "sampled_generator_parameters": parameters,
        "parameter_mapping": mappings,
        "parameter_sampling": {
            "policy": "direct_real_anchor_feature_row",
            "anchor_id": anchor["anchor_id"],
            "path_seed": path_seed,
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
        "mase_scale_source": "clean_l504_history",
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
        delay = int(metadata["cross_lag_steps"])
        start = context - delay
        replaced_channels = [driver]
        replaced_slice = [start, context]
        target[start:context, driver] = _affine_match_history(
            donor_target[
                donor_target.shape[0] - int(donor["horizon"]) - delay :
                donor_target.shape[0] - int(donor["horizon"]),
                driver,
            ],
            target[start:context, driver],
        )
        ablation_type = "affine_matched_forecast_covering_driver_donor"

    result = json.loads(json.dumps(clean))
    result["schema_version"] = "paper_v8_input_ablation_sample.v1"
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
        "affine_matched_mean_and_std": True,
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
        preserve_additive_hierarchy=(
            clean["capability_id"] == "hierarchical_coherence"
        ),
    )
    result = json.loads(json.dumps(clean))
    result["schema_version"] = "paper_v8_robustness_master_sample.v1"
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
        if "final_code_slice" in result:
            result["final_code_slice"] = [
                int(value) - offset for value in result["final_code_slice"]
            ]
        episodes = []
        for episode in result.get("historical_episodes", []):
            code = [int(value) - offset for value in episode["code_slice"]]
            response = [
                int(value) - offset for value in episode["response_slice"]
            ]
            if code[0] >= 0 and response[1] <= context_length:
                episodes.append(
                    {"code_slice": code, "response_slice": response}
                )
        result["historical_episodes"] = episodes
        result["historical_episode_count_in_view"] = len(episodes)
    elif capability_id == "cross_series_dependence":
        delay = int(result["cross_lag_steps"])
        result["counterfactual_driver_slice"] = [
            context_length - delay,
            context_length,
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
        raise ValueError(f"unsupported v8 context view {context_length}")
    if int(master["context_length"]) != CONTEXT_LENGTH:
        raise ValueError("v8 inference views require an L504 master")
    start = CONTEXT_LENGTH - context_length
    target = np.asarray(master["target"], dtype=float)[start:]
    covariates = (
        None
        if master.get("covariates") is None
        else np.asarray(master["covariates"], dtype=float)[start:]
    )
    result = json.loads(json.dumps(master))
    result["schema_version"] = "paper_v8_forecast_view.v1"
    result["source_master_sample_id"] = master["sample_id"]
    result["master_sample_id"] = master["sample_id"]
    result["sample_id"] = f"{master['sample_id']}__L{context_length}"
    result["view_id"] = result["sample_id"]
    result["context_length"] = context_length
    result["context_policy_candidates"] = list(VIEW_CONTEXT_LENGTHS)
    result["view_standardization_policy"] = (
        "slice_exact_l504_standardized_master_without_restandardization"
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
