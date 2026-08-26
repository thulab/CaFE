from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def default_seasonality(frequency: str) -> int:
    """Compatibility seasonality used when a benchmark has no explicit value."""

    raw = str(frequency)
    if raw.endswith(("H", "h")):
        return 24
    if raw.endswith("D"):
        return 7
    if raw.endswith(("M", "ME")):
        return 12
    if raw.endswith("W") or raw.startswith("W-"):
        return 52
    return 1


def impute_history_only(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Interpolate each channel from history only and preserve its mask."""

    history = np.asarray(values, dtype=float).copy()
    observed = np.isfinite(history)
    fractions: list[float] = []
    empty_channels: list[int] = []
    positions = np.arange(history.shape[0], dtype=float)
    for channel in range(history.shape[1]):
        finite = observed[:, channel]
        fractions.append(float(np.mean(finite)))
        column = history[:, channel]
        if not np.any(finite):
            history[:, channel] = 0.0
            empty_channels.append(channel)
        elif int(np.sum(finite)) == 1:
            history[:, channel] = float(column[finite][0])
        else:
            history[:, channel] = np.interp(
                positions, positions[finite], column[finite]
            )
    return history, observed, {
        "policy": "history_only_linear_interpolation_edge_hold_v1",
        "observed_fraction_by_target": fractions,
        "all_missing_target_indices": empty_channels,
    }


def fill_unobserved_future(
    future: np.ndarray,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill storage placeholders while retaining the official score mask."""

    values = np.asarray(future, dtype=float).copy()
    observed = np.isfinite(values)
    for channel in range(values.shape[1]):
        values[~observed[:, channel], channel] = float(history[-1, channel])
    return values, observed


def impute_dynamic_covariates(
    history: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if history.shape[1] == 0:
        return history.copy(), future.copy(), {"policy": "no_native_covariates"}
    imputed_history, _observed, audit = impute_history_only(history)
    imputed_future = np.asarray(future, dtype=float).copy()
    for channel in range(imputed_future.shape[1]):
        finite = np.isfinite(imputed_future[:, channel])
        imputed_future[~finite, channel] = imputed_history[-1, channel]
    return imputed_history, imputed_future, {
        **audit,
        "policy": "history_only_linear_interpolation_and_future_edge_hold_v1",
    }


@dataclass(frozen=True)
class NativeForecastInstance:
    """Lossless CaFE view of one benchmark-native forecast window.

    Adapters own the benchmark split and visibility rules.  CaFE only
    standardizes array orientation and identity so mechanisms, inference, and
    analysis can operate without knowing how the source benchmark formed the
    window.  History and future arrays are always time-major ``[T, D]``.

    The fields through ``history_imputation`` are kept in the same order as the
    former ``GiftEvalInstance`` contract.  The benchmark-neutral metadata that
    follows is optional so legacy fixtures and frozen GIFT contracts remain
    source-compatible during the adapter migration.
    """

    dataset_id: str
    config_id: str
    item_id: str
    official_instance_id: str
    frequency: str
    term: str
    window_index: int
    window_count: int
    forecast_origin: int | str
    prediction_length: int
    history: np.ndarray
    future: np.ndarray
    future_observed_mask: np.ndarray
    history_covariates: np.ndarray
    future_covariates: np.ndarray
    covariate_column_names: tuple[str, ...]
    covariate_availability: tuple[str, ...]
    future_covariate_visible: tuple[bool, ...]
    target_column_names: tuple[str, ...]
    source_target_length: int
    history_imputation: dict[str, object]

    benchmark_id: str = "gift_eval"
    suite_id: str = ""
    task_id: str = ""
    seasonality: int | None = None
    history_observed_mask: np.ndarray | None = None
    covariate_types: tuple[str, ...] = ()
    static_covariates: Mapping[str, Any] = field(default_factory=dict)
    source_locator: Mapping[str, Any] = field(default_factory=dict)
    native_protocol: Mapping[str, Any] = field(default_factory=dict)
    selected_model_max_contexts: Mapping[str, int] = field(default_factory=dict)

    @property
    def benchmark_instance_id(self) -> str:
        """Benchmark-neutral alias retained beside the legacy field name."""

        return self.official_instance_id

    @property
    def target_dim(self) -> int:
        return int(self.history.shape[1])

    @property
    def context_length(self) -> int:
        return int(self.history.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.prediction_length)

    @property
    def covariate_dim(self) -> int:
        return int(self.history_covariates.shape[1])

    @property
    def has_visible_future_covariates(self) -> bool:
        return any(self.future_covariate_visible)

    @property
    def resolved_task_id(self) -> str:
        return self.task_id or self.dataset_id

    @property
    def resolved_seasonality(self) -> int:
        if self.seasonality is not None:
            return max(1, int(self.seasonality))
        return default_seasonality(self.frequency)


# Compatibility name used by existing public imports and frozen v14 tests.
GiftEvalInstance = NativeForecastInstance
