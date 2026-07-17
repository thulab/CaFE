#!/usr/bin/env python3
"""History-only predictive mechanism probes for paper E4-v3.

The probes in this module do not try to recover a synthetic generator's latent
parameters.  They test a weaker, transferable contract: after fitting only on
the prefix of an observed history, does a capability-specific continuation law
improve untouched pseudo-futures over a nuisance-matched baseline?

Every public entry point consumes a one-dimensional, fully observed history.
The actual benchmark future is deliberately outside the API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np


CAPABILITY_IDS = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)

DEFAULT_FOLD_COUNT = 4
MIN_VALID_FOLDS = 3
PERMUTATION_COUNT = 19
NUMERIC_EPSILON = 1e-9


@dataclass(frozen=True)
class FoldProbeResult:
    origin: int
    horizon: int
    actual: np.ndarray
    baseline: np.ndarray
    probe: np.ndarray
    support: float
    parameter_key: str
    evidence: dict[str, Any]

    @property
    def baseline_mae(self) -> float:
        return float(np.mean(np.abs(self.actual - self.baseline)))

    @property
    def probe_mae(self) -> float:
        return float(np.mean(np.abs(self.actual - self.probe)))

    @property
    def relative_mae_gain(self) -> float:
        return _relative_gain(self.baseline_mae, self.probe_mae)

    @property
    def baseline_mse(self) -> float:
        return float(np.mean((self.actual - self.baseline) ** 2))

    @property
    def probe_mse(self) -> float:
        return float(np.mean((self.actual - self.probe) ** 2))

    @property
    def relative_mse_gain(self) -> float:
        return _relative_gain(self.baseline_mse, self.probe_mse)


@dataclass(frozen=True)
class LinearFit:
    coefficient: np.ndarray
    column_center: np.ndarray
    column_scale: np.ndarray

    def predict(self, design: np.ndarray) -> np.ndarray:
        standardized = _standardize_design(
            design,
            center=self.column_center,
            scale=self.column_scale,
        )
        return standardized @ self.coefficient


def evaluate_capability_gate(
    history: Sequence[float] | np.ndarray,
    *,
    capability_id: str,
    season_length: int,
    pseudo_horizon: int,
    fold_count: int = DEFAULT_FOLD_COUNT,
    permutation_count: int = PERMUTATION_COUNT,
) -> dict[str, Any]:
    """Evaluate one capability without reading the benchmark future."""

    if capability_id not in CAPABILITY_IDS:
        raise ValueError(f"unknown predictive capability: {capability_id}")
    values = _validated_history(history)
    season = max(4, int(season_length))
    horizon = max(4, int(pseudo_horizon))
    origins = pseudo_future_origins(
        len(values),
        season_length=season,
        pseudo_horizon=horizon,
        fold_count=fold_count,
    )
    probe = _PROBES[capability_id]
    folds: list[FoldProbeResult] = []
    failures: list[dict[str, Any]] = []
    for origin in origins:
        fold_horizon = min(horizon, len(values) - origin)
        prefix = values[:origin]
        actual = values[origin : origin + fold_horizon]
        try:
            result = probe(
                prefix,
                actual=actual,
                absolute_origin=origin,
                season_length=season,
            )
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            failures.append(
                {
                    "origin": int(origin),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        if (
            len(result.actual) != fold_horizon
            or not np.isfinite(result.baseline).all()
            or not np.isfinite(result.probe).all()
        ):
            failures.append(
                {
                    "origin": int(origin),
                    "error_type": "invalid_forecast",
                    "message": "probe returned a non-finite or wrong-length forecast",
                }
            )
            continue
        folds.append(result)

    if len(folds) < MIN_VALID_FOLDS:
        return _failed_gate_payload(
            capability_id,
            origins=origins,
            failures=failures,
            valid_fold_count=len(folds),
        )

    mae_gains = np.asarray(
        [fold.relative_mae_gain for fold in folds],
        dtype=float,
    )
    mse_gains = np.asarray(
        [fold.relative_mse_gain for fold in folds],
        dtype=float,
    )
    supports = np.asarray([fold.support for fold in folds], dtype=float)
    gain_mean = float(np.mean(mae_gains))
    gain_se = float(
        np.std(mae_gains, ddof=1) / math.sqrt(len(mae_gains))
        if len(mae_gains) > 1
        else 0.0
    )
    gain_lcb = float(gain_mean - gain_se)
    parameter_stability = _parameter_stability(
        [fold.parameter_key for fold in folds]
    )
    permutation = _phase_permutation_test(
        folds,
        season_length=season,
        permutation_count=permutation_count,
    )
    pooled_actual = np.concatenate([fold.actual for fold in folds])
    pooled_baseline = np.concatenate([fold.baseline for fold in folds])
    pooled_probe = np.concatenate([fold.probe for fold in folds])
    evidence = _aggregate_evidence(folds)
    return {
        "schema_version": "predictive_capability_gate.v1",
        "capability_id": capability_id,
        "history_length": len(values),
        "season_length": season,
        "pseudo_horizon": horizon,
        "fold_origins": [int(fold.origin) for fold in folds],
        "requested_fold_count": len(origins),
        "valid_fold_count": len(folds),
        "failed_fold_count": len(failures),
        "failures": failures,
        "gain_mean": gain_mean,
        "gain_median": float(np.median(mae_gains)),
        "gain_standard_error": gain_se,
        "gain_lcb": gain_lcb,
        "mse_gain_mean": float(np.mean(mse_gains)),
        "positive_fold_fraction": float(np.mean(mae_gains > 0.0)),
        "support_median": float(np.median(supports)),
        "support_min": float(np.min(supports)),
        "parameter_stability": float(parameter_stability),
        "phase_permutation_pvalue": float(permutation["pvalue"]),
        "phase_permutation_null_count": int(permutation["null_count"]),
        "phase_permutation_null_q95": float(permutation["null_q95"]),
        "pooled_baseline_mae": float(
            np.mean(np.abs(pooled_actual - pooled_baseline))
        ),
        "pooled_probe_mae": float(np.mean(np.abs(pooled_actual - pooled_probe))),
        "pooled_relative_mae_gain": _relative_gain(
            float(np.mean(np.abs(pooled_actual - pooled_baseline))),
            float(np.mean(np.abs(pooled_actual - pooled_probe))),
        ),
        "selected_parameter_keys": [
            str(fold.parameter_key) for fold in folds
        ],
        "evidence": evidence,
        "uses_benchmark_future": False,
    }


def evaluate_capability_fingerprint(
    history: Sequence[float] | np.ndarray,
    *,
    season_length: int,
    pseudo_horizon: int,
    fold_count: int = DEFAULT_FOLD_COUNT,
    permutation_count: int = PERMUTATION_COUNT,
) -> dict[str, dict[str, Any]]:
    """Return the six-dimensional predictive fingerprint for one history."""

    return {
        capability_id: evaluate_capability_gate(
            history,
            capability_id=capability_id,
            season_length=season_length,
            pseudo_horizon=pseudo_horizon,
            fold_count=fold_count,
            permutation_count=permutation_count,
        )
        for capability_id in CAPABILITY_IDS
    }


def pseudo_future_origins(
    history_length: int,
    *,
    season_length: int,
    pseudo_horizon: int,
    fold_count: int = DEFAULT_FOLD_COUNT,
) -> tuple[int, ...]:
    """Choose non-overlapping trailing pseudo-futures inside visible history."""

    length = int(history_length)
    season = max(4, int(season_length))
    horizon = max(4, int(pseudo_horizon))
    requested = max(MIN_VALID_FOLDS, int(fold_count))
    minimum_prefix = max(6 * season, 3 * horizon, 96)
    available = max(0, (length - minimum_prefix) // horizon)
    count = min(requested, available)
    if count < MIN_VALID_FOLDS:
        raise ValueError(
            "history is too short for three untouched pseudo-futures: "
            f"length={length}, season={season}, horizon={horizon}"
        )
    first = length - count * horizon
    return tuple(first + index * horizon for index in range(count))


def gate_decision(
    diagnostics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply an already frozen, capability-specific threshold rule."""

    gain_statistic = str(
        thresholds.get("gain_statistic", "gain_lcb")
    )
    if gain_statistic not in {
        "gain_lcb",
        "gain_median",
        "pooled_relative_mae_gain",
    }:
        raise ValueError(f"unsupported gate gain statistic: {gain_statistic}")
    checks = {
        "valid_folds": int(diagnostics.get("valid_fold_count", 0))
        >= int(thresholds.get("minimum_valid_folds", MIN_VALID_FOLDS)),
        "predictive_gain": float(
            diagnostics.get(gain_statistic, -math.inf)
        )
        >= float(thresholds["minimum_predictive_gain"]),
        "positive_fold_fraction": float(
            diagnostics.get("positive_fold_fraction", 0.0)
        )
        >= float(thresholds["minimum_positive_fold_fraction"]),
        "support": float(diagnostics.get("support_median", -math.inf))
        >= float(thresholds["minimum_support"]),
        "parameter_stability": float(
            diagnostics.get("parameter_stability", 0.0)
        )
        >= float(thresholds["minimum_parameter_stability"]),
        "phase_permutation": float(
            diagnostics.get("phase_permutation_pvalue", 1.0)
        )
        <= float(thresholds["maximum_phase_permutation_pvalue"]),
    }
    qualified = bool(all(checks.values()))
    gain_threshold = max(
        float(
            thresholds.get(
                "gain_normalization_scale",
                thresholds["minimum_predictive_gain"],
            )
        ),
        1e-6,
    )
    support_threshold = max(
        float(
            thresholds.get(
                "support_normalization_scale",
                thresholds["minimum_support"],
            )
        ),
        1e-6,
    )
    fingerprint_weight = float(
        max(0.0, float(diagnostics.get(gain_statistic, 0.0)))
        / gain_threshold
        * math.sqrt(
            max(0.0, float(diagnostics.get("support_median", 0.0)))
            / support_threshold
        )
        * float(diagnostics.get("positive_fold_fraction", 0.0))
        * float(diagnostics.get("parameter_stability", 0.0))
    )
    return {
        "qualified": qualified,
        "checks": checks,
        "fingerprint_weight": fingerprint_weight,
        "gain_statistic": gain_statistic,
        "thresholds": thresholds,
    }


def _trend_probe(
    prefix: np.ndarray,
    *,
    actual: np.ndarray,
    absolute_origin: int,
    season_length: int,
) -> FoldProbeResult:
    horizon = len(actual)
    all_times = np.arange(len(prefix) + horizon, dtype=float)
    nuisance_periods = (season_length, 4 * season_length)
    baseline_design = _harmonic_design(
        all_times,
        origin=len(prefix),
        periods=nuisance_periods,
        degree=0,
    )
    baseline_fit = _ridge_fit(baseline_design[: len(prefix)], prefix)
    baseline = baseline_fit.predict(baseline_design[len(prefix) :])
    history_baseline = baseline_fit.predict(baseline_design[: len(prefix)])
    selected = _select_trend_continuation(
        prefix,
        forecast_horizon=horizon,
        season_length=season_length,
    )
    probe_design = _harmonic_design(
        all_times,
        origin=len(prefix),
        periods=nuisance_periods,
        degree=int(selected["degree"]),
    )
    fit_start = max(0, len(prefix) - int(selected["fit_window"]))
    probe_fit = _ridge_fit(
        probe_design[fit_start : len(prefix)],
        prefix[fit_start:],
    )
    forecast = probe_fit.predict(probe_design[len(prefix) :])
    history_probe = probe_fit.predict(probe_design[: len(prefix)])
    support = _history_incremental_r2(
        prefix[fit_start:],
        history_baseline[fit_start:],
        history_probe[fit_start:],
    )
    trend_component_design = _harmonic_design(
        all_times,
        origin=len(prefix),
        periods=(),
        degree=int(selected["degree"]),
    )
    trend_only_fit = _ridge_fit(
        trend_component_design[fit_start : len(prefix)],
        prefix[fit_start:],
    )
    trend_history = trend_only_fit.predict(
        trend_component_design[fit_start : len(prefix)]
    )
    slope = float(trend_history[-1] - trend_history[0])
    return FoldProbeResult(
        origin=absolute_origin,
        horizon=horizon,
        actual=actual,
        baseline=baseline,
        probe=forecast,
        support=support,
        parameter_key="slope:+" if slope >= 0 else "slope:-",
        evidence={
            "history_incremental_r2": support,
            "selected_inner_mae": float(selected["inner_mae"]),
            "selected_degree": int(selected["degree"]),
            "selected_fit_window": int(selected["fit_window"]),
            "fitted_trend_range": slope,
        },
    )


def _select_trend_continuation(
    prefix: np.ndarray,
    *,
    forecast_horizon: int,
    season_length: int,
) -> dict[str, Any]:
    """Select trend degree and memory using an inner untouched suffix."""

    inner_horizon = min(
        int(forecast_horizon),
        max(8, 2 * int(season_length)),
    )
    inner_origin = len(prefix) - inner_horizon
    minimum_train = max(6 * int(season_length), 3 * inner_horizon)
    if inner_origin < minimum_train:
        return {
            "degree": 1,
            "fit_window": len(prefix),
            "inner_mae": math.inf,
        }
    times = np.arange(len(prefix), dtype=float)
    candidates: list[dict[str, Any]] = []
    for degree in (1, 2):
        design = _harmonic_design(
            times,
            origin=inner_origin,
            periods=(season_length, 4 * season_length),
            degree=degree,
        )
        windows = {
            inner_origin,
            min(inner_origin, 8 * int(season_length)),
            min(inner_origin, 4 * int(season_length)),
        }
        for fit_window in sorted(windows, reverse=True):
            fit_start = inner_origin - fit_window
            fit = _ridge_fit(
                design[fit_start:inner_origin],
                prefix[fit_start:inner_origin],
            )
            prediction = fit.predict(design[inner_origin:])
            mae = float(
                np.mean(np.abs(prefix[inner_origin:] - prediction))
            )
            candidates.append(
                {
                    "degree": degree,
                    # Re-express the selected memory relative to the full prefix.
                    "fit_window": min(len(prefix), fit_window + inner_horizon),
                    "inner_mae": mae,
                }
            )
    return min(
        candidates,
        key=lambda item: (
            item["inner_mae"],
            item["degree"],
            -item["fit_window"],
        ),
    )


def _multi_seasonal_probe(
    prefix: np.ndarray,
    *,
    actual: np.ndarray,
    absolute_origin: int,
    season_length: int,
) -> FoldProbeResult:
    horizon = len(actual)
    all_times = np.arange(len(prefix) + horizon, dtype=float)
    baseline_periods = (season_length, 4 * season_length)
    secondary_periods = (
        max(4.0, season_length / 2.0),
        2.0 * season_length,
    )
    baseline_design = _harmonic_design(
        all_times,
        origin=len(prefix),
        periods=baseline_periods,
        degree=1,
    )
    probe_design = _harmonic_design(
        all_times,
        origin=len(prefix),
        periods=(*baseline_periods, *secondary_periods),
        degree=1,
    )
    baseline_fit = _ridge_fit(baseline_design[: len(prefix)], prefix)
    probe_fit = _ridge_fit(probe_design[: len(prefix)], prefix)
    baseline = baseline_fit.predict(baseline_design[len(prefix) :])
    forecast = probe_fit.predict(probe_design[len(prefix) :])
    history_baseline = baseline_fit.predict(baseline_design[: len(prefix)])
    history_probe = probe_fit.predict(probe_design[: len(prefix)])
    support = _history_incremental_r2(prefix, history_baseline, history_probe)
    residual = prefix - history_baseline
    energies = {
        f"{period:.6g}": _period_projection_energy(
            residual,
            period=period,
        )
        for period in secondary_periods
    }
    dominant = max(energies, key=energies.get)
    return FoldProbeResult(
        origin=absolute_origin,
        horizon=horizon,
        actual=actual,
        baseline=baseline,
        probe=forecast,
        support=support,
        parameter_key=f"secondary_period:{dominant}",
        evidence={
            "history_incremental_r2": support,
            "secondary_period_energies": energies,
        },
    )


def _time_varying_seasonality_probe(
    prefix: np.ndarray,
    *,
    actual: np.ndarray,
    absolute_origin: int,
    season_length: int,
) -> FoldProbeResult:
    horizon = len(actual)
    all_times = np.arange(len(prefix) + horizon, dtype=float)
    baseline_periods = (season_length, 2.0 * season_length)
    sideband_periods = (
        max(4.0, 0.8 * season_length),
        4.0 * season_length / 3.0,
    )
    baseline_design = _harmonic_design(
        all_times,
        origin=len(prefix),
        periods=baseline_periods,
        degree=1,
    )
    probe_design = _harmonic_design(
        all_times,
        origin=len(prefix),
        periods=(*baseline_periods, *sideband_periods),
        degree=1,
    )
    baseline_fit = _ridge_fit(baseline_design[: len(prefix)], prefix)
    probe_fit = _ridge_fit(probe_design[: len(prefix)], prefix)
    baseline = baseline_fit.predict(baseline_design[len(prefix) :])
    forecast = probe_fit.predict(probe_design[len(prefix) :])
    history_baseline = baseline_fit.predict(baseline_design[: len(prefix)])
    history_probe = probe_fit.predict(probe_design[: len(prefix)])
    support = _history_incremental_r2(prefix, history_baseline, history_probe)
    residual = prefix - history_baseline
    energies = {
        f"{period:.6g}": _period_projection_energy(
            residual,
            period=period,
        )
        for period in sideband_periods
    }
    dominant = max(energies, key=energies.get)
    return FoldProbeResult(
        origin=absolute_origin,
        horizon=horizon,
        actual=actual,
        baseline=baseline,
        probe=forecast,
        support=support,
        parameter_key=f"sideband:{dominant}",
        evidence={
            "history_incremental_r2": support,
            "sideband_energies": energies,
        },
    )


def _regime_switching_probe(
    prefix: np.ndarray,
    *,
    actual: np.ndarray,
    absolute_origin: int,
    season_length: int,
) -> FoldProbeResult:
    horizon = len(actual)
    total_length = len(prefix) + horizon
    times = np.arange(total_length, dtype=float)
    periods = _unique_integer_periods(
        (
            2 * season_length,
            4 * season_length,
            7 * season_length,
        ),
        maximum=max(8, len(prefix) // 2),
    )
    best: dict[str, Any] | None = None
    for period in periods:
        baseline_design = _harmonic_design(
            times,
            origin=len(prefix),
            periods=(season_length, period),
            degree=1,
        )
        baseline_fit = _ridge_fit(baseline_design[: len(prefix)], prefix)
        baseline_history = baseline_fit.predict(
            baseline_design[: len(prefix)]
        )
        baseline_full = baseline_fit.predict(baseline_design)
        for duty_fraction in (1.0 / 3.0, 0.5, 2.0 / 3.0):
            active = max(2, int(round(period * duty_fraction)))
            phase_step = max(1, period // 16)
            for phase in range(0, period, phase_step):
                state = _periodic_state(
                    total_length,
                    period=period,
                    active_length=active,
                    phase=phase,
                )
                component = _fit_incremental_component(
                    baseline_design,
                    prefix,
                    state,
                    train_length=len(prefix),
                    baseline_prediction=baseline_full,
                )
                history_probe = component["prediction"][: len(prefix)]
                support = _history_incremental_r2(
                    prefix,
                    baseline_history,
                    history_probe,
                )
                historical_switches = int(
                    np.sum(np.diff(state[: len(prefix)]) != 0.0)
                )
                future_switches = int(
                    np.sum(np.diff(state[len(prefix) - 1 :]) != 0.0)
                )
                amplitude_ratio = abs(float(component["coefficient"])) / max(
                    float(np.std(prefix - baseline_history)),
                    1e-9,
                )
                objective = support * min(amplitude_ratio, 3.0) / 3.0
                candidate = {
                    "objective": objective,
                    "support": support,
                    "baseline": baseline_full[len(prefix) :],
                    "probe": component["prediction"][len(prefix) :],
                    "period": period,
                    "phase": phase,
                    "duty_fraction": active / period,
                    "amplitude_ratio": amplitude_ratio,
                    "historical_switches": historical_switches,
                    "future_switches": future_switches,
                }
                if best is None or (
                    objective,
                    -abs(active / period - 0.5),
                    -period,
                    -phase,
                ) > (
                    best["objective"],
                    -abs(best["duty_fraction"] - 0.5),
                    -best["period"],
                    -best["phase"],
                ):
                    best = candidate
    if best is None:
        raise ValueError("no recurring regime candidate has adequate history")
    support = float(best["objective"])
    if best["historical_switches"] < 2 or best["future_switches"] < 1:
        support = 0.0
    return FoldProbeResult(
        origin=absolute_origin,
        horizon=horizon,
        actual=actual,
        baseline=np.asarray(best["baseline"], dtype=float),
        probe=np.asarray(best["probe"], dtype=float),
        support=support,
        parameter_key=(
            f"clock:{best['period']}:duty:{best['duty_fraction']:.3f}"
        ),
        evidence={
            "history_incremental_r2": float(best["support"]),
            "amplitude_ratio": float(best["amplitude_ratio"]),
            "selected_period": int(best["period"]),
            "selected_phase": int(best["phase"]),
            "selected_duty_fraction": float(best["duty_fraction"]),
            "historical_switch_count": int(best["historical_switches"]),
            "future_switch_count": int(best["future_switches"]),
        },
    )


def _nonlinear_persistence_probe(
    prefix: np.ndarray,
    *,
    actual: np.ndarray,
    absolute_origin: int,
    season_length: int,
) -> FoldProbeResult:
    horizon = len(actual)
    total_length = len(prefix) + horizon
    times = np.arange(total_length, dtype=float)
    nuisance_design = _harmonic_design(
        times,
        origin=len(prefix),
        periods=(season_length, 4 * season_length),
        degree=1,
    )
    nuisance_fit = _ridge_fit(
        nuisance_design[: len(prefix)],
        prefix,
    )
    nuisance_prediction = nuisance_fit.predict(nuisance_design)
    residual = prefix - nuisance_prediction[: len(prefix)]
    center = float(np.mean(residual))
    scale = max(float(np.std(residual)), 1e-9)
    standardized = (residual - center) / scale
    baseline_model, baseline_history = _fit_autoregression(
        standardized,
        season_length=season_length,
        nonlinear=False,
    )
    probe_model, probe_history = _fit_autoregression(
        standardized,
        season_length=season_length,
        nonlinear=True,
    )
    baseline_standardized = _recursive_autoregression(
        standardized,
        model=baseline_model,
        horizon=horizon,
        season_length=season_length,
        nonlinear=False,
    )
    probe_standardized = _recursive_autoregression(
        standardized,
        model=probe_model,
        horizon=horizon,
        season_length=season_length,
        nonlinear=True,
    )
    start = int(baseline_model["start"])
    support = _history_incremental_r2(
        standardized[start:],
        baseline_history,
        probe_history,
    )
    nonlinear_coefficient = float(
        np.sum(np.asarray(probe_model["coefficient"])[-3:])
    )
    return FoldProbeResult(
        origin=absolute_origin,
        horizon=horizon,
        actual=actual,
        baseline=(
            nuisance_prediction[len(prefix) :]
            + center
            + scale * baseline_standardized
        ),
        probe=(
            nuisance_prediction[len(prefix) :]
            + center
            + scale * probe_standardized
        ),
        support=support,
        parameter_key=(
            "nonlinear_coefficient:+"
            if nonlinear_coefficient >= 0
            else "nonlinear_coefficient:-"
        ),
        evidence={
            "history_incremental_r2": support,
            "nonlinear_coefficient_sum": nonlinear_coefficient,
        },
    )


def _predictable_intermittency_probe(
    prefix: np.ndarray,
    *,
    actual: np.ndarray,
    absolute_origin: int,
    season_length: int,
) -> FoldProbeResult:
    horizon = len(actual)
    total_length = len(prefix) + horizon
    times = np.arange(total_length, dtype=float)
    periods = _unique_integer_periods(
        (
            max(4, season_length // 2),
            season_length,
            2 * season_length,
        ),
        maximum=max(8, len(prefix) // 2),
    )
    best: dict[str, Any] | None = None
    for period in periods:
        baseline_design = _harmonic_design(
            times,
            origin=len(prefix),
            periods=(season_length, period),
            degree=1,
        )
        baseline_fit = _ridge_fit(baseline_design[: len(prefix)], prefix)
        baseline_history = baseline_fit.predict(
            baseline_design[: len(prefix)]
        )
        baseline_full = baseline_fit.predict(baseline_design)
        widths = sorted(
            {
                max(0.65, period / 40.0),
                max(0.85, period / 20.0),
            }
        )
        for width in widths:
            for phase in range(period):
                pulse = _periodic_gaussian_pulse(
                    total_length,
                    period=period,
                    phase=phase,
                    width=width,
                )
                component = _fit_incremental_component(
                    baseline_design,
                    prefix,
                    pulse,
                    train_length=len(prefix),
                    baseline_prediction=baseline_full,
                )
                history_probe = component["prediction"][: len(prefix)]
                incremental_r2 = _history_incremental_r2(
                    prefix,
                    baseline_history,
                    history_probe,
                )
                amplitude_ratio = abs(float(component["coefficient"])) / max(
                    float(np.std(prefix - baseline_history)),
                    1e-9,
                )
                duty_fraction = float(np.mean(pulse[:period] >= math.exp(-2.0)))
                sparsity = max(0.0, 1.0 - duty_fraction)
                objective = (
                    incremental_r2
                    * min(amplitude_ratio, 3.0)
                    / 3.0
                    * sparsity
                )
                historical_pulses = _pulse_center_count(
                    len(prefix),
                    period=period,
                    phase=phase,
                )
                future_pulses = (
                    _pulse_center_count(
                        total_length,
                        period=period,
                        phase=phase,
                    )
                    - historical_pulses
                )
                candidate = {
                    "objective": objective,
                    "incremental_r2": incremental_r2,
                    "baseline": baseline_full[len(prefix) :],
                    "probe": component["prediction"][len(prefix) :],
                    "period": period,
                    "phase": phase,
                    "width": width,
                    "amplitude_ratio": amplitude_ratio,
                    "duty_fraction": duty_fraction,
                    "historical_pulses": historical_pulses,
                    "future_pulses": future_pulses,
                }
                if best is None or (
                    objective,
                    -duty_fraction,
                    -period,
                    -phase,
                ) > (
                    best["objective"],
                    -best["duty_fraction"],
                    -best["period"],
                    -best["phase"],
                ):
                    best = candidate
    if best is None:
        raise ValueError("no periodic pulse candidate has adequate history")
    support = float(best["objective"])
    if best["historical_pulses"] < 2 or best["future_pulses"] < 1:
        support = 0.0
    return FoldProbeResult(
        origin=absolute_origin,
        horizon=horizon,
        actual=actual,
        baseline=np.asarray(best["baseline"], dtype=float),
        probe=np.asarray(best["probe"], dtype=float),
        support=support,
        parameter_key=f"pulse:{best['period']}:phase:{best['phase']}",
        evidence={
            "history_incremental_r2": float(best["incremental_r2"]),
            "amplitude_ratio": float(best["amplitude_ratio"]),
            "selected_period": int(best["period"]),
            "selected_phase": int(best["phase"]),
            "selected_width": float(best["width"]),
            "duty_fraction": float(best["duty_fraction"]),
            "historical_pulse_count": int(best["historical_pulses"]),
            "future_pulse_count": int(best["future_pulses"]),
        },
    )


_PROBES: dict[
    str,
    Callable[..., FoldProbeResult],
] = {
    "trend": _trend_probe,
    "multi_seasonal": _multi_seasonal_probe,
    "time_varying_seasonality": _time_varying_seasonality_probe,
    "regime_switching": _regime_switching_probe,
    "nonlinear_persistence": _nonlinear_persistence_probe,
    "predictable_intermittency": _predictable_intermittency_probe,
}


def _validated_history(
    history: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("predictive capability gate requires one target channel")
    if len(values) < 96:
        raise ValueError("predictive capability gate history is too short")
    if not np.isfinite(values).all():
        raise ValueError("predictive capability gate does not accept missing history")
    scale = float(np.std(values))
    if scale <= 1e-10:
        raise ValueError("predictive capability gate history is constant")
    center = float(np.median(values))
    robust_scale = float(
        np.median(np.abs(values - center)) * 1.4826
    )
    if robust_scale <= 1e-10:
        robust_scale = scale
    return (values - center) / robust_scale


def _harmonic_design(
    times: np.ndarray,
    *,
    origin: int,
    periods: Iterable[float],
    degree: int,
) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    normalized_time = (values - (origin - 1)) / max(origin - 1, 1)
    columns = [np.ones(len(values), dtype=float)]
    columns.extend(
        normalized_time**power for power in range(1, int(degree) + 1)
    )
    seen: set[float] = set()
    for raw_period in periods:
        period = round(float(raw_period), 8)
        if period < 4.0 or period in seen:
            continue
        seen.add(period)
        angle = 2.0 * np.pi * values / period
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def _ridge_fit(
    design: np.ndarray,
    response: np.ndarray,
    *,
    alpha: float = 1e-4,
) -> LinearFit:
    matrix = np.asarray(design, dtype=float)
    target = np.asarray(response, dtype=float)
    center = np.zeros(matrix.shape[1], dtype=float)
    scale = np.ones(matrix.shape[1], dtype=float)
    if matrix.shape[1] > 1:
        center[1:] = np.mean(matrix[:, 1:], axis=0)
        scale[1:] = np.std(matrix[:, 1:], axis=0)
        scale[1:] = np.where(scale[1:] > 1e-9, scale[1:], 1.0)
    standardized = _standardize_design(
        matrix,
        center=center,
        scale=scale,
    )
    penalty = float(alpha) * np.eye(standardized.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        standardized.T @ standardized + penalty,
        standardized.T @ target,
    )
    return LinearFit(coefficient, center, scale)


def _standardize_design(
    design: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = (np.asarray(design, dtype=float) - center) / scale
    result[:, 0] = 1.0
    return result


def _fit_incremental_component(
    baseline_design: np.ndarray,
    response: np.ndarray,
    component: np.ndarray,
    *,
    train_length: int,
    baseline_prediction: np.ndarray,
) -> dict[str, Any]:
    design = np.asarray(baseline_design, dtype=float)
    extra = np.asarray(component, dtype=float)
    component_fit = _ridge_fit(design[:train_length], extra[:train_length])
    residualized = extra - component_fit.predict(design)
    denominator = float(
        residualized[:train_length] @ residualized[:train_length]
    )
    if denominator <= 1e-9:
        coefficient = 0.0
    else:
        baseline_residual = (
            np.asarray(response, dtype=float)
            - np.asarray(baseline_prediction[:train_length], dtype=float)
        )
        coefficient = float(
            residualized[:train_length] @ baseline_residual / denominator
        )
    prediction = np.asarray(baseline_prediction, dtype=float) + (
        coefficient * residualized
    )
    return {
        "coefficient": coefficient,
        "prediction": prediction,
        "residualized_component": residualized,
    }


def _fit_autoregression(
    values: np.ndarray,
    *,
    season_length: int,
    nonlinear: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    start = max(4, int(season_length))
    rows = []
    responses = []
    for index in range(start, len(values)):
        rows.append(
            _autoregression_features(
                values,
                index=index,
                season_length=season_length,
                nonlinear=nonlinear,
            )
        )
        responses.append(float(values[index]))
    design = np.asarray(rows, dtype=float)
    response = np.asarray(responses, dtype=float)
    fit = _ridge_fit(design, response, alpha=0.05)
    prediction = fit.predict(design)
    return {
        "fit": fit,
        "coefficient": fit.coefficient,
        "start": start,
    }, prediction


def _recursive_autoregression(
    values: np.ndarray,
    *,
    model: dict[str, Any],
    horizon: int,
    season_length: int,
    nonlinear: bool,
) -> np.ndarray:
    output = list(np.asarray(values, dtype=float))
    fit = model["fit"]
    predictions = []
    lower = float(np.quantile(values, 0.005) - 4.0 * np.std(values))
    upper = float(np.quantile(values, 0.995) + 4.0 * np.std(values))
    for _ in range(int(horizon)):
        array = np.asarray(output, dtype=float)
        features = _autoregression_features(
            array,
            index=len(array),
            season_length=season_length,
            nonlinear=nonlinear,
        )
        prediction = float(fit.predict(features[None, :])[0])
        prediction = float(np.clip(prediction, lower, upper))
        output.append(prediction)
        predictions.append(prediction)
    return np.asarray(predictions, dtype=float)


def _autoregression_features(
    values: np.ndarray,
    *,
    index: int,
    season_length: int,
    nonlinear: bool,
) -> np.ndarray:
    period = max(4, int(season_length))
    nonlinear_lag = max(2, period // 2)
    lag1 = float(values[index - 1])
    lag_period = float(values[index - period])
    lag_nonlinear = float(values[index - nonlinear_lag])
    phase = 2.0 * np.pi * index / period
    result = [
        1.0,
        lag1,
        lag_period,
        lag_nonlinear,
        math.sin(phase),
        math.cos(phase),
    ]
    if nonlinear:
        result.extend(
            (
                math.sin(1.1 * lag_nonlinear) ** 2 - 0.25,
                lag_nonlinear**2 - 1.0,
                math.tanh(lag_nonlinear),
            )
        )
    return np.asarray(result, dtype=float)


def _periodic_state(
    length: int,
    *,
    period: int,
    active_length: int,
    phase: int,
) -> np.ndarray:
    position = np.mod(
        np.arange(int(length), dtype=int) - int(phase),
        int(period),
    )
    return np.where(position < int(active_length), 1.0, -1.0)


def _periodic_gaussian_pulse(
    length: int,
    *,
    period: int,
    phase: int,
    width: float,
) -> np.ndarray:
    times = np.arange(int(length), dtype=float)
    distance = np.abs(np.mod(times - float(phase) + period / 2.0, period) - period / 2.0)
    return np.exp(-0.5 * (distance / max(float(width), 1e-6)) ** 2)


def _pulse_center_count(
    length: int,
    *,
    period: int,
    phase: int,
) -> int:
    if length <= 0:
        return 0
    first = int(phase) % int(period)
    if first >= length:
        return 0
    return 1 + (int(length) - 1 - first) // int(period)


def _unique_integer_periods(
    periods: Iterable[int],
    *,
    maximum: int,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(period)
                for period in periods
                if 4 <= int(period) <= int(maximum)
            }
        )
    )


def _history_incremental_r2(
    actual: np.ndarray,
    baseline: np.ndarray,
    probe: np.ndarray,
) -> float:
    baseline_sse = float(np.sum((actual - baseline) ** 2))
    probe_sse = float(np.sum((actual - probe) ** 2))
    return float(max(0.0, (baseline_sse - probe_sse) / max(baseline_sse, 1e-9)))


def _period_projection_energy(
    values: np.ndarray,
    *,
    period: float,
) -> float:
    times = np.arange(len(values), dtype=float)
    design = np.column_stack(
        (
            np.sin(2.0 * np.pi * times / float(period)),
            np.cos(2.0 * np.pi * times / float(period)),
        )
    )
    coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
    projected = design @ coefficient
    return float(np.mean(projected**2))


def _relative_gain(baseline_loss: float, probe_loss: float) -> float:
    return float(
        (float(baseline_loss) - float(probe_loss))
        / max(float(baseline_loss), 1e-9)
    )


def _parameter_stability(keys: Sequence[str]) -> float:
    if not keys:
        return 0.0
    counts: dict[str, int] = {}
    for key in keys:
        counts[str(key)] = counts.get(str(key), 0) + 1
    return float(max(counts.values()) / len(keys))


def _phase_permutation_test(
    folds: Sequence[FoldProbeResult],
    *,
    season_length: int,
    permutation_count: int,
) -> dict[str, Any]:
    observed_baseline = np.concatenate([fold.baseline for fold in folds])
    observed_probe = np.concatenate([fold.probe for fold in folds])
    observed_actual = np.concatenate([fold.actual for fold in folds])
    observed_gain = _relative_gain(
        float(np.mean(np.abs(observed_actual - observed_baseline))),
        float(np.mean(np.abs(observed_actual - observed_probe))),
    )
    minimum_horizon = min(len(fold.actual) for fold in folds)
    if minimum_horizon <= 2 or permutation_count <= 0:
        return {
            "pvalue": 1.0,
            "null_count": 0,
            "null_q95": 0.0,
        }
    candidates = [
        shift
        for shift in range(1, minimum_horizon)
        if shift % max(4, int(season_length)) != 0
    ]
    if len(candidates) > permutation_count:
        indices = np.linspace(
            0,
            len(candidates) - 1,
            num=permutation_count,
            dtype=int,
        )
        shifts = [candidates[int(index)] for index in indices]
    else:
        shifts = candidates
    null_gains: list[float] = []
    for shift in shifts:
        shifted_actual = np.concatenate(
            [np.roll(fold.actual, int(shift)) for fold in folds]
        )
        null_gains.append(
            _relative_gain(
                float(np.mean(np.abs(shifted_actual - observed_baseline))),
                float(np.mean(np.abs(shifted_actual - observed_probe))),
            )
        )
    exceedances = sum(
        value >= observed_gain - 1e-12 for value in null_gains
    )
    return {
        "pvalue": float((1 + exceedances) / (1 + len(null_gains))),
        "null_count": len(null_gains),
        "null_q95": float(
            np.quantile(null_gains, 0.95) if null_gains else 0.0
        ),
    }


def _aggregate_evidence(
    folds: Sequence[FoldProbeResult],
) -> dict[str, Any]:
    numeric_keys = sorted(
        set.intersection(
            *(
                {
                    key
                    for key, value in fold.evidence.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, bool)
                }
                for fold in folds
            )
        )
        if folds
        else set()
    )
    numeric_medians = {
        key: float(
            np.median(
                [float(fold.evidence[key]) for fold in folds]
            )
        )
        for key in numeric_keys
    }
    return {
        "numeric_medians": numeric_medians,
        "folds": [
            {
                "origin": int(fold.origin),
                "relative_mae_gain": fold.relative_mae_gain,
                "relative_mse_gain": fold.relative_mse_gain,
                "support": float(fold.support),
                "parameter_key": str(fold.parameter_key),
                **fold.evidence,
            }
            for fold in folds
        ],
    }


def _failed_gate_payload(
    capability_id: str,
    *,
    origins: Sequence[int],
    failures: Sequence[dict[str, Any]],
    valid_fold_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "predictive_capability_gate.v1",
        "capability_id": capability_id,
        "fold_origins": [int(origin) for origin in origins],
        "requested_fold_count": len(origins),
        "valid_fold_count": int(valid_fold_count),
        "failed_fold_count": len(failures),
        "failures": list(failures),
        "gain_mean": -1.0,
        "gain_median": -1.0,
        "gain_standard_error": 1.0,
        "gain_lcb": -1.0,
        "mse_gain_mean": -1.0,
        "positive_fold_fraction": 0.0,
        "support_median": 0.0,
        "support_min": 0.0,
        "parameter_stability": 0.0,
        "phase_permutation_pvalue": 1.0,
        "phase_permutation_null_count": 0,
        "phase_permutation_null_q95": 0.0,
        "pooled_baseline_mae": 1e12,
        "pooled_probe_mae": 1e12,
        "pooled_relative_mae_gain": -1.0,
        "selected_parameter_keys": [],
        "evidence": {"numeric_medians": {}, "folds": []},
        "uses_benchmark_future": False,
    }
