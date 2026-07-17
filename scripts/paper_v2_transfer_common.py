#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIFT_EVAL_DIR = Path.home() / "xmy/gift-eval"
PAPER_UNIVARIATE_CAPABILITY_IDS = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)
PAPER_V2_CONTEXT_LENGTH = 504
PAPER_V2_HORIZON = 48
PAPER_V2_SEASON_LENGTH = 24


@dataclass(frozen=True)
class TransferProfileSpec:
    profile_id: str
    dataset_name: str
    family_id: str
    frequency: str
    context_length: int
    horizon: int
    stride: int
    season_length: int
    target_dim: int = 1
    covariate_dim: int = 0
    hierarchy: str | None = None
    synthetic_capabilities: tuple[str, ...] = PAPER_UNIVARIATE_CAPABILITY_IDS

    @property
    def asset_name(self) -> str:
        return self.dataset_name

    @property
    def kind(self) -> str:
        return "gift_univariate"

    @property
    def feature_measurement_horizon(self) -> int:
        """Canonical strength is measured on one primary seasonal period."""

        return int(self.season_length)


@dataclass(frozen=True)
class CanonicalReferenceSpec:
    profile_id: str
    family_id: str
    kind: str
    asset_name: str
    context_length: int = PAPER_V2_CONTEXT_LENGTH
    horizon: int = PAPER_V2_HORIZON
    stride: int = PAPER_V2_HORIZON
    season_length: int = PAPER_V2_SEASON_LENGTH
    target_dim: int = 1
    covariate_dim: int = 0
    hierarchy: str | None = None
    max_series: int = 240
    max_groups: int = 20
    task: int = 1
    frequency: str = "h"
    synthetic_capabilities: tuple[str, ...] = PAPER_UNIVARIATE_CAPABILITY_IDS

    @property
    def feature_measurement_horizon(self) -> int:
        return int(self.season_length)


CANONICAL_REFERENCE_SPECS = (
    CanonicalReferenceSpec(
        "dev_m4_hourly_504ctx_48h",
        "m4_hourly",
        "tsf_univariate",
        "m4_hourly_dataset.zip",
    ),
    CanonicalReferenceSpec(
        "dev_electricity_hourly_504ctx_48h",
        "electricity",
        "tsf_univariate",
        "electricity_hourly_dataset.zip",
    ),
    CanonicalReferenceSpec(
        "dev_traffic_hourly_504ctx_48h",
        "traffic",
        "tsf_univariate",
        "traffic_hourly_dataset.zip",
    ),
    CanonicalReferenceSpec(
        "dev_jena_weather_hourly_504ctx_48h",
        "jena_weather",
        "gift_univariate",
        "jena_weather/H",
    ),
    CanonicalReferenceSpec(
        "dev_bizitobs_l2c_hourly_504ctx_48h",
        "bizitobs_l2c",
        "gift_univariate",
        "bizitobs_l2c/H",
    ),
)


TRANSFER_PROFILE_SPECS = (
    TransferProfileSpec(
        "gift_solar_h_504ctx_48h",
        "solar/H",
        "solar",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_kdd_cup_h_504ctx_48h",
        "kdd_cup_2018_with_missing/H",
        "kdd_cup_2018_with_missing",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_loop_seattle_h_504ctx_48h",
        "LOOP_SEATTLE/H",
        "LOOP_SEATTLE",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_sz_taxi_h_504ctx_48h",
        "SZ_TAXI/H",
        "SZ_TAXI",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_m_dense_h_504ctx_48h",
        "M_DENSE/H",
        "M_DENSE",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_ett1_h_504ctx_48h",
        "ett1/H",
        "ETT",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_ett2_h_504ctx_48h",
        "ett2/H",
        "ETT",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_bitbrains_fast_h_504ctx_48h",
        "bitbrains_fast_storage/H",
        "bitbrains",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
    TransferProfileSpec(
        "gift_bitbrains_rnd_h_504ctx_48h",
        "bitbrains_rnd/H",
        "bitbrains",
        "h",
        PAPER_V2_CONTEXT_LENGTH,
        PAPER_V2_HORIZON,
        PAPER_V2_HORIZON,
        PAPER_V2_SEASON_LENGTH,
    ),
)


def transfer_profile_specs(
    profile_ids: Iterable[str] | None = None,
) -> tuple[TransferProfileSpec, ...]:
    if profile_ids is None:
        return TRANSFER_PROFILE_SPECS
    requested = tuple(str(profile_id) for profile_id in profile_ids)
    by_id = {spec.profile_id: spec for spec in TRANSFER_PROFILE_SPECS}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("unknown paper-v2 transfer profiles: " + ", ".join(unknown))
    return tuple(by_id[profile_id] for profile_id in requested)


def profile_spec_payload(spec: TransferProfileSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["synthetic_capabilities"] = list(spec.synthetic_capabilities)
    return payload


def load_transfer_training_rows(
    spec: TransferProfileSpec,
    *,
    gift_eval_dir: Path = DEFAULT_GIFT_EVAL_DIR,
    max_windows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load leakage-safe GIFT train windows for transfer conditioning.

    The official short-term test tail and the immediately preceding validation
    horizon are both excluded. Missing values are imputed only inside this
    already-observed development prefix.
    """

    from run_synthetic_v2_near_distance_calibration import (  # local to keep import cheap
        make_real_row,
    )
    from synthetic_feature_profile import (
        WindowSpec,
        gift_eval_short_term_test_holdout_steps,
        limit_candidates,
        read_gift_arrow_targets,
        window_starts,
    )

    path = gift_eval_dir / spec.dataset_name
    frequency, records = read_gift_arrow_targets(path)
    holdout_steps = gift_eval_short_term_test_holdout_steps(frequency, records)
    if holdout_steps % spec.horizon:
        raise ValueError(
            f"{spec.profile_id} official holdout {holdout_steps} is not divisible "
            f"by horizon {spec.horizon}"
        )
    official_windows = holdout_steps // spec.horizon
    window_spec = WindowSpec(spec.context_length, spec.horizon, spec.stride)
    candidates: list[tuple[str, str, int, np.ndarray, int, int]] = []
    expanded_series_count = 0
    for item_id, native_values in records:
        channels = native_values if native_values.ndim == 2 else native_values[None, :]
        for channel_index, values in enumerate(channels):
            expanded_series_count += 1
            train_cutoff = int(len(values) - holdout_steps - spec.horizon)
            if train_cutoff < window_spec.length:
                continue
            series_id = (
                str(item_id)
                if native_values.ndim == 1
                else f"{item_id}:dim:{channel_index}"
            )
            candidates.extend(
                (
                    series_id,
                    str(item_id),
                    int(channel_index),
                    np.asarray(values, dtype=float),
                    int(start),
                    train_cutoff,
                )
                for start in window_starts(train_cutoff, window_spec)
            )

    rows: list[dict[str, Any]] = []
    rejected_missing = 0
    rejected_uninformative = 0
    oversampled = limit_candidates(candidates, max(max_windows * 4, max_windows))
    for series_id, base_item_id, channel_index, values, start, train_cutoff in oversampled:
        raw = np.asarray(values[start : start + window_spec.length], dtype=float)
        imputed, observed_fraction = impute_observed_window(raw)
        if imputed is None:
            rejected_missing += 1
            continue
        row = make_real_row(
            imputed[:, None],
            spec,
            group_id=f"gift-item:{base_item_id}",
            window_start=start,
        )
        context = np.asarray(row["target"][: spec.context_length], dtype=float)
        if float(np.std(context)) <= 1e-8:
            rejected_uninformative += 1
            continue
        row.update(
            {
                "series_id": series_id,
                "base_item_id": base_item_id,
                "channel_index": int(channel_index),
                "observed_fraction": float(observed_fraction),
                "source_train_cutoff": int(train_cutoff),
                "source_test_tail_excluded_steps": int(holdout_steps),
                "source_validation_excluded_steps": int(spec.horizon),
            }
        )
        rows.append(row)
        if len(rows) >= max_windows:
            break
    if len(rows) < 60:
        raise ValueError(
            f"{spec.profile_id} produced only {len(rows)} usable train windows "
            f"(candidates={len(candidates)})"
        )
    return rows, {
        "profile_id": spec.profile_id,
        "dataset_name": spec.dataset_name,
        "source_frequency": frequency,
        "expanded_series_count": expanded_series_count,
        "candidate_window_count": len(candidates),
        "selected_window_count": len(rows),
        "rejected_missing_count": rejected_missing,
        "rejected_uninformative_count": rejected_uninformative,
        "minimum_observed_fraction": 0.5,
        "official_prediction_length": int(spec.horizon),
        "official_test_window_count": int(official_windows),
        "official_test_tail_steps": int(holdout_steps),
        "validation_excluded_steps": int(spec.horizon),
        "fit_boundary": "official training prefix only",
        "missing_value_policy": (
            "require at least 50% observed and two finite points; interpolate within "
            "the observed training window and nearest-fill its edges"
        ),
    }


def impute_observed_window(
    values: np.ndarray,
    *,
    minimum_observed_fraction: float = 0.5,
) -> tuple[np.ndarray | None, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(array)
    observed_fraction = float(np.mean(finite)) if array.size else 0.0
    if (
        array.size == 0
        or observed_fraction < minimum_observed_fraction
        or int(np.sum(finite)) < 2
    ):
        return None, observed_fraction
    indexes = np.arange(array.size, dtype=float)
    imputed = np.interp(indexes, indexes[finite], array[finite])
    return np.asarray(imputed, dtype=float), observed_fraction


def primary_feature_intensity_coordinate(
    value: float,
    target_values: Iterable[float],
) -> float:
    targets = np.asarray(tuple(float(item) for item in target_values), dtype=float)
    if targets.shape != (5,) or not np.isfinite(targets).all():
        raise ValueError("canonical target_values must contain five finite values")
    if np.any(np.diff(targets) <= 0.0):
        raise ValueError("canonical target_values must be strictly increasing")
    return float(
        np.interp(
            float(value),
            targets,
            np.arange(1.0, 6.0),
            left=1.0,
            right=5.0,
        )
    )
