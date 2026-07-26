from __future__ import annotations

import math
from typing import Any

import numpy as np

from synthetic_feature_profile import (
    feature_vector,
    multitarget_features,
    regime_sparse_transition_score,
    robust_scale,
    seasonal_modulation_features,
    spectral_time_scale_features,
)


FEATURE_SCHEMA_VERSION = "paper_v8_feature_vector.v6"
LOCAL_TREND_WINDOW = 96
LOCAL_TREND_HARMONIC_COUNT = 6
LOCAL_TREND_MAX_REMOVED_PERIOD = 84.0
INTERMITTENCY_LOCAL_BASELINE_WINDOW = 9
LOCAL_TREND_MIN_REMOVED_PERIOD = 4.0
LOCAL_TREND_FREQUENCY_FFT_SIZE = 4096


def _as_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 8:
        raise ValueError("v8 feature history must be a [time, target] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("v8 feature history must be finite")
    return matrix


def _quadratic_residual(values: np.ndarray) -> np.ndarray:
    time = np.linspace(-1.0, 1.0, values.shape[0])
    design = np.column_stack([np.ones(values.shape[0]), time, time**2])
    residual = np.empty_like(values, dtype=float)
    for channel in range(values.shape[1]):
        scaled = robust_scale(values[:, channel])
        try:
            coefficients = np.linalg.lstsq(design, scaled, rcond=None)[0]
            residual[:, channel] = scaled - design @ coefficients
        except np.linalg.LinAlgError:
            residual[:, channel] = scaled - float(np.mean(scaled))
    return residual


def _local_trend_residual(
    values: np.ndarray,
    season_length: int | None,
) -> np.ndarray:
    period = max(2, int(season_length or 2))
    maximum = max(5, values.shape[0] // 3)
    width = min(maximum, max(5, 2 * period + 1))
    if width % 2 == 0:
        width = max(3, width - 1)
    radius = width // 2
    kernel = np.full(width, 1.0 / width)
    residual = np.empty_like(values, dtype=float)
    for channel in range(values.shape[1]):
        scaled = robust_scale(values[:, channel])
        # This residual feeds a cycle-to-cycle amplitude statistic.  Edge
        # replication attenuates the first and last cycles of an otherwise
        # stationary carrier and creates artificial modulation.  Circular
        # padding preserves a complete-cycle window and applies identically to
        # real and synthetic anchors.
        padded = np.pad(scaled, (radius, radius), mode="wrap")
        smooth = np.convolve(padded, kernel, mode="valid")
        residual[:, channel] = scaled - smooth
    return residual


def _robust_scale_denominator(values: np.ndarray) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-9:
        return iqr
    standard_deviation = float(np.std(values))
    return standard_deviation if standard_deviation > 1e-9 else 1.0


def _joinpoint_harmonic_decomposition(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove resolved carrier modes without absorbing the local quadratic.

    Frequency selection uses the full visible history.  A joinpoint trend
    basis (linear history plus a quadratic active only over the final 96
    observations) is included throughout matching, so a genuine local trend is
    protected while stable carriers and the principal AM sidebands are
    removed.  Only periods from 4 to 84 are eligible: slower movement remains
    available to the trend coordinate.
    """

    raw = np.asarray(values, dtype=float)
    observations = raw.size
    evidence_length = min(LOCAL_TREND_WINDOW, observations)
    index = np.arange(observations, dtype=float)
    global_time = np.linspace(-1.0, 1.0, observations)
    join_index = observations - evidence_length
    local_time = np.clip(
        (index - float(join_index)) / max(evidence_length - 1, 1),
        0.0,
        None,
    )
    base = np.column_stack(
        [np.ones(observations), global_time, local_time**2]
    )
    design = base
    scaled = robust_scale(raw)
    selected_frequencies: list[float] = []
    fft_size = max(LOCAL_TREND_FREQUENCY_FFT_SIZE, observations)
    frequencies = np.fft.rfftfreq(fft_size)
    eligible_base = (
        (frequencies >= 1.0 / LOCAL_TREND_MAX_REMOVED_PERIOD)
        & (frequencies <= 1.0 / LOCAL_TREND_MIN_REMOVED_PERIOD)
    )
    reference_energy = max(float(np.sum(scaled**2)), 1.0)

    for _ in range(LOCAL_TREND_HARMONIC_COUNT):
        coefficients = np.linalg.lstsq(design, scaled, rcond=None)[0]
        residual = scaled - design @ coefficients
        if float(np.sum(residual**2)) <= 1e-12 * reference_energy:
            break
        periodogram = np.abs(np.fft.rfft(residual, fft_size)) ** 2
        eligible = eligible_base.copy()
        for selected in selected_frequencies:
            eligible &= (
                np.abs(frequencies - selected)
                >= 0.5 / observations
            )
        candidates = np.flatnonzero(eligible)
        if candidates.size == 0:
            break
        selected_index = int(
            candidates[np.argmax(periodogram[candidates])]
        )
        selected = float(frequencies[selected_index])
        selected_frequencies.append(selected)
        angle = 2.0 * np.pi * selected * index
        design = np.column_stack(
            [design, np.sin(angle), np.cos(angle)]
        )

    coefficients = np.linalg.lstsq(design, scaled, rcond=None)[0]
    harmonic_component = np.zeros(observations, dtype=float)
    if design.shape[1] > base.shape[1]:
        harmonic_component = (
            design[:, base.shape[1] :]
            @ coefficients[base.shape[1] :]
        )
    adjusted = (scaled - harmonic_component)[-evidence_length:]
    unexplained_residual = scaled - design @ coefficients

    # Express the adjusted signal in the same local robust-scale units used by
    # the original coordinate.  Re-scaling the near-zero residual itself would
    # amplify harmless numerical leakage from a fully explained carrier.
    scale_ratio = (
        _robust_scale_denominator(raw)
        / _robust_scale_denominator(raw[-evidence_length:])
    )
    return adjusted * scale_ratio, unexplained_residual, scaled


def _local_trend_features(
    values: np.ndarray,
    decompositions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, float]:
    evidence_length = min(LOCAL_TREND_WINDOW, values.shape[0])
    time = np.linspace(-1.0, 1.0, evidence_length)
    design = np.column_stack(
        [np.ones(evidence_length), time, time**2]
    )
    slopes: list[float] = []
    curvatures: list[float] = []
    quadratic_energy_shares: list[float] = []
    polynomial_energy_shares: list[float] = []
    strengths: list[float] = []
    for adjusted, _, _ in decompositions:
        scaled = adjusted
        try:
            coefficients = np.linalg.lstsq(design, scaled, rcond=None)[0]
            fitted = design @ coefficients
        except np.linalg.LinAlgError:
            coefficients = np.asarray([float(np.mean(scaled)), 0.0, 0.0])
            fitted = np.full_like(scaled, coefficients[0])
        residual_variance = float(np.var(scaled - fitted))
        total_variance = float(np.var(scaled))
        slopes.append(abs(float(coefficients[1])))
        curvatures.append(abs(float(coefficients[2])))
        linear_energy = float(np.var(coefficients[1] * time))
        quadratic_energy = float(
            np.var(coefficients[2] * time**2)
        )
        quadratic_energy_shares.append(
            quadratic_energy
            / max(linear_energy + quadratic_energy, 1e-12)
        )
        cubic_design = np.column_stack(
            [np.ones(evidence_length), time, time**2, time**3]
        )
        try:
            cubic_coefficients = np.linalg.lstsq(
                cubic_design,
                scaled,
                rcond=None,
            )[0]
        except np.linalg.LinAlgError:
            cubic_coefficients = np.asarray(
                [float(np.mean(scaled)), 0.0, 0.0, 0.0]
            )
        cubic_linear_energy = float(
            np.var(cubic_coefficients[1] * time)
        )
        nonlinear_component = (
            cubic_coefficients[2] * time**2
            + cubic_coefficients[3] * time**3
        )
        nonlinear_energy = float(np.var(nonlinear_component))
        polynomial_energy_shares.append(
            nonlinear_energy
            / max(cubic_linear_energy + nonlinear_energy, 1e-12)
        )
        strengths.append(
            0.0
            if total_variance <= 1e-12
            else float(np.clip(1.0 - residual_variance / total_variance, 0.0, 1.0))
        )
    return {
        "trend_strength": float(np.mean(strengths)),
        "slope_abs": float(np.mean(slopes)),
        "curvature_abs": float(np.mean(curvatures)),
        "local_trend_strength_w96": float(np.mean(strengths)),
        "local_slope_abs_w96": float(np.mean(slopes)),
        "local_curvature_abs_w96": float(np.mean(curvatures)),
        "local_quadratic_energy_share_w96": float(
            np.mean(quadratic_energy_shares)
        ),
        "local_polynomial_energy_share_w96": float(
            np.mean(polynomial_energy_shares)
        ),
    }


def _multi_period_energy_share(
    values: np.ndarray,
    season_length: int | None,
) -> float:
    """Measure stationary spectral energy outside the calibrated carrier.

    The legacy coordinate used only the single largest secondary FFT bin.
    That made the response weak when a multi-seasonal intervention distributed
    its dose over two sidebands.  Summing all non-carrier power measures the
    intended mechanism while the preceding quadratic residualization prevents
    smooth trend curvature from being counted as an extra period.
    """

    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or signal.size < 8:
        return 0.0
    centered = signal - float(np.mean(signal))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size <= 1:
        return 0.0
    power[0] = 0.0
    total = float(np.sum(power))
    if not math.isfinite(total) or total <= 1e-12:
        return 0.0

    residual_power = power.copy()
    if season_length is not None and int(season_length) >= 2:
        carrier_bin = int(round(signal.size / float(season_length)))
        if 1 <= carrier_bin < residual_power.size:
            lower = max(1, carrier_bin - 1)
            upper = min(residual_power.size, carrier_bin + 2)
            residual_power[lower:upper] = 0.0
    return float(np.clip(np.sum(residual_power) / total, 0.0, 1.0))


def _event_positive_residual_energy_share(values: np.ndarray) -> float:
    """Measure positive intermittent prominence after smooth-mode removal.

    A fixed centered nine-point moving average removes the local background
    without selecting a data-dependent spectral basis.  The remaining
    positive versus negative energy share is continuous on finite histories
    and needs neither event labels nor generator metadata.  Predictability of
    generated events remains a separate construction gate.
    """

    signal = robust_scale(np.asarray(values, dtype=float).reshape(-1))
    observations = signal.size
    window = INTERMITTENCY_LOCAL_BASELINE_WINDOW
    if observations < window:
        return 0.0
    padding = window // 2
    padded = np.pad(signal, padding, mode="reflect")
    local_background = np.convolve(
        padded,
        np.ones(window, dtype=float) / window,
        mode="valid",
    )
    residual = signal - local_background
    centered = residual - float(np.median(residual))
    positive = np.clip(centered, 0.0, None)
    negative = np.clip(-centered, 0.0, None)
    positive_energy = float(np.sum(positive**2))
    negative_energy = float(np.sum(negative**2))
    return float(
        positive_energy
        / max(positive_energy + negative_energy, 1e-12)
    )


def _mean_finite(rows: list[dict[str, float]], name: str) -> float | None:
    values = [
        float(row[name])
        for row in rows
        if name in row and math.isfinite(float(row[name]))
    ]
    return float(np.mean(values)) if values else None


def v8_feature_vector(
    history: np.ndarray,
    season_length: int | None = None,
    *,
    covariates: np.ndarray | None = None,
    hierarchy: str | None = None,
    include_cross_series_predictability: bool = True,
    cross_series_max_lag: int | None = None,
) -> dict[str, float]:
    """Return the sole Paper-v8 feature vector from visible history only.

    The caller cannot pass a context boundary: every row supplied here is
    treated as observed history.  Trend uses the most recent 96 observations;
    spectral coordinates use a quadratic-detrended signal; amplitude
    modulation uses a local-trend residual; transition sparsity uses the
    material residual left after joinpoint trend and resolved-harmonic
    removal.  These targeted residualizations reduce the known
    cross-sensitivity among trend, spectrum, amplitude non-stationarity, and
    sparse differences.
    """

    values = _as_matrix(history)
    covariate_values = None
    if covariates is not None:
        covariate_values = _as_matrix(covariates)
        if covariate_values.shape[0] != values.shape[0]:
            raise ValueError("v8 history covariates must align with target history")

    output: dict[str, Any] = feature_vector(
        values,
        season_length,
        covariates=covariate_values,
        context_length=values.shape[0],
        hierarchy=hierarchy,
        include_cross_series_predictability=False,
    )
    if include_cross_series_predictability and values.shape[1] > 1:
        output.update(
            multitarget_features(
                values,
                max_lag=(
                    int(cross_series_max_lag)
                    if cross_series_max_lag is not None
                    else min(
                        96,
                        max(48, 2 * int(season_length or 24)),
                    )
                ),
                include_cross_series_predictability=True,
            )
        )
    trend_decompositions = [
        _joinpoint_harmonic_decomposition(values[:, channel])
        for channel in range(values.shape[1])
    ]
    output.update(_local_trend_features(values, trend_decompositions))

    spectral_residual = _quadratic_residual(values)
    spectral_rows = [
        {
            **spectral_time_scale_features(spectral_residual[:, channel]),
            "multi_period_score": _multi_period_energy_share(
                spectral_residual[:, channel],
                season_length,
            ),
        }
        for channel in range(values.shape[1])
    ]
    for name in ("dominant_period", "spectral_concentration", "multi_period_score"):
        value = _mean_finite(spectral_rows, name)
        if value is not None:
            output[name] = value

    local_residual = _local_trend_residual(values, season_length)
    modulation_rows = [
        seasonal_modulation_features(local_residual[:, channel], season_length)
        for channel in range(values.shape[1])
    ]
    for name in ("seasonal_amplitude_modulation", "seasonal_phase_variation"):
        value = _mean_finite(modulation_rows, name)
        if value is not None:
            output[name] = value

    # Regime transitions must not be removed as soon as they become strong
    # enough to appear among the adaptive harmonic modes.  That made the
    # coordinate fold back with increasing intervention strength.  A
    # quadratic residual removes trend while retaining sparse jumps; smooth
    # single- and multi-period carriers still distribute their difference
    # energy rather than concentrating it at a few transition points.
    transition_scores = [
        regime_sparse_transition_score(spectral_residual[:, channel])
        for channel in range(values.shape[1])
    ]
    transition_residuals = [
        spectral_residual[:, channel]
        for channel in range(values.shape[1])
    ]
    residual_mean = np.mean(
        np.column_stack(transition_residuals),
        axis=1,
    )
    residual_diff = np.diff(residual_mean)
    residual_sparsity = float(np.mean(transition_scores))
    output["regime_sparse_transition_score"] = residual_sparsity
    output["residual_diff_sparsity"] = residual_sparsity
    output["diff_spike_rate"] = (
        float(np.mean(np.abs(robust_scale(residual_diff)) > 3.0))
        if residual_diff.size
        else 0.0
    )
    output["event_positive_residual_energy_share"] = float(
        np.mean(
            [
                _event_positive_residual_energy_share(
                    values[:, channel]
                )
                for channel in range(values.shape[1])
            ]
        )
    )
    output["v8_feature_history_length"] = float(values.shape[0])
    return {
        str(name): float(value)
        for name, value in output.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    }
