#!/usr/bin/env python3
"""Render illustrative CaFE capability curves from the formal generators.

The examples deliberately do not claim real-data calibration.  They use one
auditable empirical background profile, the formal cafe primary generator
families, the formal L336/H48 geometry, and a direct I1--I5 lambda grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]

from cafe import protocol as cafe
from cafe.generation.families import (
    derive_deterministic_parameters,
    generate_deterministic_sample,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "figures"
    / "capability-examples"
)
EMPIRICAL_PROFILE_ID = "illustrative_empirical_hourly_profile_v1"
EMPIRICAL_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXAMPLE_INTENSITY = 5
EXAMPLE_BASE_SEED = 2026072507


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titleweight": "regular",
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    title: str
    cluster: str
    domain: str
    action: str


CAPABILITY_PAIRS = (
    (
        CapabilitySpec(
            "trend",
            "Trend Extrapolation",
            "Global Pattern Extrapolation",
            "Intrinsic Temporal Modeling",
            "Extrapolate smooth direction and curvature",
        ),
        CapabilitySpec(
            "multi_seasonal",
            "Multi-seasonal Composition",
            "Global Pattern Extrapolation",
            "Intrinsic Temporal Modeling",
            "Compose multiple periodic components",
        ),
    ),
    (
        CapabilitySpec(
            "time_varying_seasonality",
            "Evolving Seasonality",
            "Nonstationary Dynamics",
            "Intrinsic Temporal Modeling",
            "Track continuous phase and frequency change",
        ),
        CapabilitySpec(
            "regime_switching",
            "Predictable Regime Switching",
            "Nonstationary Dynamics",
            "Intrinsic Temporal Modeling",
            "Anticipate discrete state transitions",
        ),
    ),
    (
        CapabilitySpec(
            "nonlinear_persistence",
            "Nonlinear Persistence",
            "Complex Temporal Mechanisms",
            "Intrinsic Temporal Modeling",
            "Propagate nonlinear lagged recurrence",
        ),
        CapabilitySpec(
            "predictable_intermittency",
            "Predictable Intermittency",
            "Complex Temporal Mechanisms",
            "Intrinsic Temporal Modeling",
            "Anticipate sparse scheduled events",
        ),
    ),
    (
        CapabilitySpec(
            "common_factor",
            "Common Factor Recovery",
            "Joint Structural Modeling",
            "Relational and Conditional Modeling",
            "Recover shared latent dynamics",
        ),
        CapabilitySpec(
            "hierarchical_coherence",
            "Hierarchical Coherence",
            "Joint Structural Modeling",
            "Relational and Conditional Modeling",
            "Preserve aggregation and child contrasts",
        ),
    ),
    (
        CapabilitySpec(
            "cross_series_dependence",
            "Cross-series Dependence",
            "Directed Conditional Forecasting",
            "Relational and Conditional Modeling",
            "Transfer a lagged endogenous driver",
        ),
        CapabilitySpec(
            "covariate_response",
            "Known-future Covariate Response",
            "Directed Conditional Forecasting",
            "Relational and Conditional Modeling",
            "Condition on future exogenous inputs",
        ),
    ),
)
CAPABILITIES = tuple(
    capability for pair in CAPABILITY_PAIRS for capability in pair
)


# Plausible hourly background statistics.  These values control nuisance and
# time scale only; they are not presented as an empirical dataset estimate.
EMPIRICAL_FEATURES = {
    "acf1": 0.84,
    "seasonal_acf": 0.68,
    "dominant_period": 24.0,
    "spectral_concentration": 0.48,
    "trend_strength": 0.24,
    "slope_abs": 0.18,
    "curvature_abs": 0.022,
    "multi_period_score": 0.36,
    "seasonal_amplitude_modulation": 0.34,
    "seasonal_phase_variation": 0.14,
    "change_point_shift_energy": 0.42,
    "level_shift_strength": 0.92,
    "regime_sparse_transition_score": 0.42,
    "spike_rate": 0.045,
    "intermittency_clock_incremental_r2": 0.22,
    "pca_top1_explained": 0.74,
    "effective_factor_rank": 1.65,
    "factor_score_acf1": 0.91,
    "factor_residual_acf1": 0.62,
    "hierarchy_child_heterogeneity": 0.28,
    "hierarchy_aggregate_acf1": 0.89,
    "hierarchy_contrast_acf1": 0.71,
    "hierarchy_aggregate_seasonal_acf": 0.62,
    "hierarchy_contrast_seasonal_acf": 0.46,
    "hierarchy_contrast_to_aggregate_std_ratio": 0.34,
    "hierarchy_aggregate_contrast_abs_corr": 0.20,
    "cross_series_incremental_r2": 0.24,
    "lead_lag_peak_abs": 0.66,
    "lead_lag_peak_lag_abs": 48.0,
    "avg_abs_target_corr": 0.38,
    "covariate_incremental_r2": 0.22,
    "event_lift_abs": 0.86,
    "covariate_residual_acf_abs_mean": 0.41,
}


CLUSTER_COLORS = (
    "#386FA4",
    "#5B8E7D",
    "#8F6E9F",
    "#B07A45",
    "#B4555D",
)
CHANNEL_COLORS = (
    "#1F4E79",
    "#3F7CAC",
    "#6EA8D4",
    "#7A5C99",
    "#C06C84",
)
FUTURE_COLOR = "#D97706"
NEUTRAL = "#687386"
MECHANISM_COLOR = "#C2410C"
RELATION_COLOR = "#7C3AED"
VIEW_START_COMPACT = max(0, cafe.CONTEXT_LENGTH - 144)
VIEW_START_INDIVIDUAL = max(0, cafe.CONTEXT_LENGTH - 168)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render illustrative CaFE capability curves."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--intensity",
        type=int,
        choices=cafe.INTENSITIES,
        default=EXAMPLE_INTENSITY,
    )
    return parser.parse_args()


def stable_example_seed(capability_id: str) -> int:
    payload = (
        f"{EXAMPLE_BASE_SEED}:{capability_id}:primary".encode("utf-8")
    )
    return int.from_bytes(
        hashlib.blake2s(payload, digest_size=8).digest(),
        "big",
    )


def empirical_summary() -> dict[str, dict[str, float]]:
    return {
        name: {"p50": float(value)}
        for name, value in EMPIRICAL_FEATURES.items()
    }


def generate_example(
    dataset: cafe.DatasetSpec,
    spec: CapabilitySpec,
    *,
    intensity: int,
) -> dict[str, Any]:
    parameters, mappings = derive_deterministic_parameters(
        spec.capability_id,
        empirical_summary(),
        season_length=24,
        context_length=cafe.CONTEXT_LENGTH,
    )
    conditioning = cafe.build_conditioning(
        dataset,
        capability_id=spec.capability_id,
        frequency="h",
        season_length=24,
        intensity_lambdas=EMPIRICAL_LAMBDAS,
        parameters=parameters,
        target_values=EMPIRICAL_LAMBDAS,
    )
    seed = stable_example_seed(spec.capability_id)
    target, metadata, covariates = generate_deterministic_sample(
        spec.capability_id,
        cafe.MASTER_LENGTH,
        cafe.CONTEXT_LENGTH,
        conditioning.target_dim,
        conditioning.season_length,
        intensity,
        np.random.default_rng(seed),
        conditioning=conditioning,
        family_role="primary",
        counterfactual_variant=0,
    )
    target, covariates = cafe.standardize_generated_sample(
        spec.capability_id,
        target,
        covariates,
        metadata=metadata,
    )
    features = cafe.measured_features(
        spec.capability_id,
        target,
        covariates,
        season_length=conditioning.season_length,
        metadata=metadata,
    )
    return {
        "spec": asdict(spec),
        "seed": seed,
        "intensity": intensity,
        "lambda": conditioning.lambda_for(intensity),
        "target": target,
        "covariates": covariates,
        "metadata": metadata,
        "realized_features": features,
        "parameters": parameters,
        "parameter_mapping": mappings,
    }


def display_channels(example: dict[str, Any]) -> list[dict[str, Any]]:
    capability_id = example["spec"]["capability_id"]
    target = np.asarray(example["target"], dtype=float)
    if capability_id == "common_factor":
        labels = [f"target {index + 1}" for index in range(target.shape[1])]
        widths = [1.4] * target.shape[1]
    elif capability_id == "hierarchical_coherence":
        labels = ["parent", "child 1", "child 2"]
        widths = [1.8, 1.0, 1.0]
    elif capability_id == "cross_series_dependence":
        labels = ["driver", "responder +", "responder −"]
        widths = [1.8, 1.0, 1.0]
    else:
        labels = ["target"]
        widths = [1.6]
    return [
        {
            "kind": "target",
            "label": labels[index],
            "values": target[:, index],
            "color": CHANNEL_COLORS[index % len(CHANNEL_COLORS)],
            "width": widths[index],
        }
        for index in range(target.shape[1])
    ]


def covariate_channels(example: dict[str, Any]) -> list[dict[str, Any]]:
    covariates = example["covariates"]
    if covariates is None:
        return []
    values = np.asarray(covariates, dtype=float)
    labels = ["weather", "known event"]
    return [
        {
            "kind": "covariate",
            "label": labels[index],
            "values": values[:, index],
            "color": NEUTRAL,
            "width": 0.9,
        }
        for index in range(values.shape[1])
    ]


def analytic_envelope(values: np.ndarray, smoothing_width: int = 11) -> np.ndarray:
    """Return a dependency-free Hilbert envelope with light smoothing."""
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    size = centered.size
    spectrum = np.fft.fft(centered)
    multiplier = np.zeros(size, dtype=float)
    if size % 2 == 0:
        multiplier[0] = 1.0
        multiplier[size // 2] = 1.0
        multiplier[1 : size // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (size + 1) // 2] = 2.0
    envelope = np.abs(np.fft.ifft(spectrum * multiplier))
    kernel = np.ones(smoothing_width, dtype=float) / smoothing_width
    return np.convolve(envelope, kernel, mode="same")


def local_peaks(values: np.ndarray, start: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    peaks = np.flatnonzero(
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
    ) + 1
    return peaks[peaks >= start]


def draw_interval(
    ax: plt.Axes,
    start: float,
    end: float,
    y: float,
    label: str,
    *,
    color: str,
    fontsize: float,
    text_offset: float,
) -> None:
    ax.annotate(
        "",
        xy=(end, y),
        xytext=(start, y),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        clip_on=True,
    )
    ax.text(
        (start + end) / 2,
        y + text_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
        clip_on=True,
    )


def add_mechanism_cue(
    ax: plt.Axes,
    example: dict[str, Any],
    *,
    compact: bool,
    view_start: int,
) -> None:
    """Overlay the causal or structural cue that the benchmark tests."""
    capability_id = example["spec"]["capability_id"]
    metadata = example["metadata"]
    target = np.asarray(example["target"], dtype=float)
    context = cafe.CONTEXT_LENGTH
    horizon = cafe.HORIZON
    x = np.arange(cafe.MASTER_LENGTH)
    y_low, y_high = ax.get_ylim()
    y_span = y_high - y_low
    fontsize = 5.8 if compact else 8.0
    label_box = {
        "boxstyle": "round,pad=0.20",
        "facecolor": "white",
        "edgecolor": MECHANISM_COLOR,
        "linewidth": 0.6,
        "alpha": 0.92,
    }

    if capability_id == "trend":
        series = target[:, 0]
        fit_slice = slice(context - 120, context)
        coefficients = np.polyfit(x[fit_slice], series[fit_slice], 2)
        fit_x = x[context - 96 :]
        ax.plot(
            fit_x,
            np.polyval(coefficients, fit_x),
            color=MECHANISM_COLOR,
            linewidth=1.0 if compact else 1.4,
            linestyle="--",
            alpha=0.9,
            zorder=4,
        )
        ax.text(
            context - 94,
            y_high - 0.12 * y_span,
            "fitted slope + curvature",
            fontsize=fontsize,
            color=MECHANISM_COLOR,
            bbox=label_box,
        )

    elif capability_id == "multi_seasonal":
        periods = sorted(float(value) for value in metadata["periods"])
        base_x = max(view_start + 8, context - 135)
        for index, period in enumerate(periods):
            y = y_high - (0.11 + 0.105 * index) * y_span
            draw_interval(
                ax,
                base_x,
                base_x + period,
                y,
                f"P{index + 1}={period:.0f}",
                color=CHANNEL_COLORS[index + 1],
                fontsize=fontsize,
                text_offset=0.012 * y_span,
            )
            base_x += period + 7

    elif capability_id == "time_varying_seasonality":
        series = target[:, 0]
        envelope = analytic_envelope(series)
        center = float(np.mean(series))
        visible = slice(view_start, cafe.MASTER_LENGTH)
        for sign in (-1.0, 1.0):
            ax.plot(
                x[visible],
                center + sign * envelope[visible],
                color=MECHANISM_COLOR,
                linestyle="--",
                linewidth=0.9 if compact else 1.25,
                alpha=0.9,
                zorder=3,
            )
        peaks = local_peaks(series, max(view_start, context - 105))[-4:]
        for peak in peaks:
            ax.axvline(
                peak,
                color=MECHANISM_COLOR,
                linewidth=0.55,
                linestyle=":",
                alpha=0.65,
            )
        if len(peaks) >= 2:
            pair = peaks[-3:-1] if compact and len(peaks) >= 3 else peaks[-2:]
            left, right = (int(pair[0]), int(pair[1]))
            draw_interval(
                ax,
                left,
                right,
                y_high - 0.10 * y_span,
                f"local period={right - left}",
                color=MECHANISM_COLOR,
                fontsize=fontsize,
                text_offset=0.012 * y_span,
            )
        ax.text(
            view_start + 5,
            y_low + 0.07 * y_span,
            "smooth amplitude / phase envelope",
            fontsize=fontsize,
            color=MECHANISM_COLOR,
            bbox=label_box,
        )

    elif capability_id == "regime_switching":
        cut_points = [
            int(point)
            for point in metadata["cut_points"]
            if int(point) >= view_start
        ]
        boundaries = [view_start] + cut_points + [cafe.MASTER_LENGTH]
        for index in range(len(boundaries) - 1):
            if index % 2 == 0:
                ax.axvspan(
                    boundaries[index],
                    boundaries[index + 1],
                    color=RELATION_COLOR,
                    alpha=0.055,
                    linewidth=0,
                )
        for point in cut_points:
            ax.axvline(
                point,
                color=RELATION_COLOR,
                linewidth=0.8,
                linestyle="--",
                alpha=0.7,
            )
        if cut_points:
            ax.text(
                cut_points[-1] + 2,
                y_high - 0.13 * y_span,
                "hard state\nboundary",
                fontsize=fontsize,
                color=RELATION_COLOR,
                bbox={**label_box, "edgecolor": RELATION_COLOR},
            )

    elif capability_id == "nonlinear_persistence":
        series = target[:, 0]
        lag = int(metadata["nonlinear_lag"])
        destinations = [context - 24, context - 8, context + 12]
        pair_colors = ("#C2410C", "#7C3AED", "#0F766E")
        for index, destination in enumerate(destinations):
            source = destination - lag
            color = pair_colors[index]
            ax.scatter(
                [source, destination],
                [series[source], series[destination]],
                s=18 if compact else 34,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )
            ax.annotate(
                "",
                xy=(destination, series[destination]),
                xytext=(source, series[source]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 1.0 if compact else 1.35,
                    "connectionstyle": "arc3,rad=-0.18",
                },
                zorder=5,
            )
        ax.text(
            context - 95,
            y_high - 0.13 * y_span,
            f"nonlinear recurrence: g(y[t-{lag}]) -> y[t]",
            fontsize=fontsize,
            color=RELATION_COLOR,
            bbox={**label_box, "edgecolor": RELATION_COLOR},
        )

    elif capability_id == "predictable_intermittency":
        width = float(metadata["pulse_width"])
        centers = [
            int(center)
            for center in metadata["pulse_centers"]
            if int(center) >= view_start
        ]
        for center_point in centers:
            ax.axvspan(
                center_point - 1.6 * width,
                center_point + 1.6 * width,
                color=MECHANISM_COLOR,
                alpha=0.10,
                linewidth=0,
            )
        if len(centers) >= 2:
            pair = centers[-3:-1] if compact and len(centers) >= 3 else centers[-2:]
            left, right = pair
            draw_interval(
                ax,
                left,
                right,
                y_high - 0.10 * y_span,
                f"event clock={right - left}",
                color=MECHANISM_COLOR,
                fontsize=fontsize,
                text_offset=0.012 * y_span,
            )
        ax.text(
            view_start + 5,
            y_low + 0.07 * y_span,
            "narrow scheduled event windows",
            fontsize=fontsize,
            color=MECHANISM_COLOR,
            bbox=label_box,
        )

    elif capability_id == "common_factor":
        loadings = np.asarray(
            metadata.get(
                "standardized_response_loadings",
                metadata["response_loadings"],
            ),
            dtype=float,
        )
        factor = target @ loadings / float(loadings @ loadings)
        factor_scale = np.std(target[view_start:]) / max(
            np.std(factor[view_start:]),
            1e-8,
        )
        factor = factor * min(factor_scale, 1.0)
        ax.plot(
            x[view_start:],
            factor[view_start:],
            color="#111827",
            linewidth=1.1 if compact else 1.6,
            linestyle=(0, (4, 2)),
            alpha=0.88,
            zorder=5,
        )
        ax.text(
            view_start + 5,
            y_high - 0.13 * y_span,
            "shared latent factor z(t)",
            fontsize=fontsize,
            color="#111827",
            bbox={**label_box, "edgecolor": "#111827"},
        )

    elif capability_id == "hierarchical_coherence":
        child_sum = target[:, 1] + target[:, 2]
        ax.plot(
            x[context - 72 :],
            child_sum[context - 72 :],
            color=RELATION_COLOR,
            linewidth=1.1 if compact else 1.7,
            linestyle="--",
            alpha=0.95,
            zorder=5,
        )
        ax.text(
            context - 92,
            y_high - 0.13 * y_span,
            "parent == child 1 + child 2",
            fontsize=fontsize,
            color=RELATION_COLOR,
            bbox={**label_box, "edgecolor": RELATION_COLOR},
        )

    elif capability_id == "cross_series_dependence":
        lag = int(metadata["cross_lag_steps"])
        active_steps = min(
            int(
                metadata.get(
                    "counterfactual_effect_forecast_steps",
                    lag,
                )
            ),
            horizon,
        )
        driver = target[:, int(metadata["driver_index"])]
        responder = target[:, int(metadata["responder_indices"][0])]
        source = driver[
            context - lag : context - lag + active_steps
        ]
        response = responder[context : context + active_steps]
        design = np.column_stack([source, np.ones_like(source)])
        scale, offset = np.linalg.lstsq(design, response, rcond=None)[0]
        shifted_driver = scale * source + offset
        ax.axvspan(
            context - lag,
            context,
            color="#2563EB",
            alpha=0.08,
            linewidth=0,
        )
        ax.plot(
            x[context : context + active_steps],
            shifted_driver,
            color=RELATION_COLOR,
            linewidth=1.2 if compact else 1.8,
            linestyle="--",
            alpha=0.95,
            zorder=5,
        )
        draw_interval(
            ax,
            context - lag / 2,
            context + lag / 2,
            y_high - 0.10 * y_span,
            f"shift +{lag}",
            color=RELATION_COLOR,
            fontsize=fontsize,
            text_offset=0.012 * y_span,
        )
        ax.text(
            view_start + 5,
            y_low + 0.07 * y_span,
            "driver history -> responder future",
            fontsize=fontsize,
            color=RELATION_COLOR,
            bbox={**label_box, "edgecolor": RELATION_COLOR},
        )

    elif capability_id == "covariate_response":
        covariates = np.asarray(example["covariates"], dtype=float)
        event_start = int(metadata["future_event_start"])
        event_width = int(metadata["event_width"])
        event_end = min(event_start + event_width, cafe.MASTER_LENGTH - 1)
        ax.axvspan(
            event_start,
            event_end,
            color=RELATION_COLOR,
            alpha=0.16,
            linewidth=0,
        )
        for channel_index, color in enumerate(("#0F766E", RELATION_COLOR)):
            ax.plot(
                x[context:],
                covariates[context:, channel_index],
                color=color,
                linewidth=1.2 if compact else 1.7,
                linestyle="--" if channel_index == 0 else ":",
                alpha=0.95,
                zorder=5,
            )
        ax.annotate(
            "known X_future -> Y_future",
            xy=(event_start, target[event_start, 0]),
            xytext=(context - 84, y_high - 0.13 * y_span),
            fontsize=fontsize,
            color=RELATION_COLOR,
            bbox={**label_box, "edgecolor": RELATION_COLOR},
            arrowprops={
                "arrowstyle": "->",
                "color": RELATION_COLOR,
                "linewidth": 1.0,
            },
        )


def plot_example(
    ax: plt.Axes,
    example: dict[str, Any],
    *,
    compact: bool,
    cluster_color: str,
) -> None:
    context = cafe.CONTEXT_LENGTH
    horizon = cafe.HORIZON
    view_start = VIEW_START_COMPACT if compact else VIEW_START_INDIVIDUAL
    x = np.arange(cafe.MASTER_LENGTH)
    ax.axvspan(
        context,
        context + horizon - 1,
        color=FUTURE_COLOR,
        alpha=0.075,
        linewidth=0,
    )
    ax.axvline(context, color=FUTURE_COLOR, linewidth=0.9, alpha=0.9)
    channels = display_channels(example)
    for channel in channels:
        values = channel["values"]
        ax.plot(
            x[:context],
            values[:context],
            color=channel["color"],
            linewidth=channel["width"] * (0.72 if compact else 0.9),
            alpha=0.76,
        )
        ax.plot(
            x[context - 1 :],
            values[context - 1 :],
            color=channel["color"],
            linewidth=channel["width"] * (1.05 if compact else 1.2),
            alpha=1.0,
        )
    for covariate in covariate_channels(example):
        ax.plot(
            x,
            covariate["values"],
            color=covariate["color"],
            linewidth=covariate["width"],
            linestyle="--" if covariate["label"] == "weather" else ":",
            alpha=0.68,
        )
    ax.set_xlim(view_start, cafe.MASTER_LENGTH - 1)
    all_values = np.column_stack(
        [channel["values"] for channel in channels]
        + [channel["values"] for channel in covariate_channels(example)]
    )
    lower, upper = np.quantile(
        all_values[view_start:],
        [0.01, 0.99],
    )
    padding = max(0.12 * (upper - lower), 0.15)
    ax.set_ylim(lower - padding, upper + padding)
    add_mechanism_cue(
        ax,
        example,
        compact=compact,
        view_start=view_start,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#C9CED6")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", colors="#687386", labelsize=7)
    ax.set_xticks([view_start, context, cafe.MASTER_LENGTH - 1])
    ax.set_xticklabels(
        [
            str(view_start),
            str(context),
            str(cafe.MASTER_LENGTH - 1),
        ]
    )
    ax.grid(axis="x", color="#E3E6EA", linewidth=0.6)
    if compact:
        ax.set_title(
            example["spec"]["title"],
            loc="left",
            fontsize=8.9,
            color="#1E293B",
            pad=18,
        )
        ax.text(
            0.0,
            1.015,
            example["spec"]["action"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#687386",
        )
        ax.add_patch(
            Rectangle(
                (0.0, 1.155),
                1.0,
                0.025,
                transform=ax.transAxes,
                color=cluster_color,
                clip_on=False,
                linewidth=0,
            )
        )
    else:
        handles = [
            Line2D(
                [0],
                [0],
                color=channel["color"],
                linewidth=channel["width"],
                label=channel["label"],
            )
            for channel in channels
        ]
        handles.extend(
            Line2D(
                [0],
                [0],
                color=channel["color"],
                linewidth=channel["width"],
                linestyle="--" if channel["label"] == "weather" else ":",
                label=channel["label"],
            )
            for channel in covariate_channels(example)
        )
        if len(handles) > 1:
            ax.legend(
                handles=handles,
                loc="lower left",
                frameon=False,
                fontsize=8,
                ncol=min(3, len(handles)),
            )
        ax.set_xlabel(
            (
                "Time index (recent context shown; forecast origin = "
                f"{context})"
            ),
            fontsize=9,
        )


def render_overview(
    examples: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(17.5, 7.4),
        constrained_layout=False,
    )
    for column, pair in enumerate(CAPABILITY_PAIRS):
        for row, spec in enumerate(pair):
            plot_example(
                axes[row, column],
                examples[spec.capability_id],
                compact=True,
                cluster_color=CLUSTER_COLORS[column],
            )
        axes[0, column].text(
            0.5,
            1.24,
            pair[0].cluster,
            transform=axes[0, column].transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#27364A",
        )
    fig.subplots_adjust(
        left=0.035,
        right=0.99,
        bottom=0.09,
        top=0.77,
        wspace=0.18,
        hspace=0.48,
    )
    top_positions = [
        axes[0, index].get_position() for index in range(5)
    ]
    intrinsic_left = top_positions[0].x0
    intrinsic_right = top_positions[2].x1
    relational_left = top_positions[3].x0
    relational_right = top_positions[4].x1
    bar_y = 0.935
    bar_height = 0.032
    fig.patches.extend(
        [
            Rectangle(
                (intrinsic_left, bar_y),
                intrinsic_right - intrinsic_left,
                bar_height,
                transform=fig.transFigure,
                facecolor="#386FA4",
                alpha=0.16,
                linewidth=0,
            ),
            Rectangle(
                (relational_left, bar_y),
                relational_right - relational_left,
                bar_height,
                transform=fig.transFigure,
                facecolor="#8F6E9F",
                alpha=0.16,
                linewidth=0,
            ),
        ]
    )
    fig.text(
        (intrinsic_left + intrinsic_right) / 2,
        bar_y + bar_height / 2,
        "INTRINSIC TEMPORAL MODELING",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#244E74",
    )
    fig.text(
        (relational_left + relational_right) / 2,
        bar_y + bar_height / 2,
        "RELATIONAL AND CONDITIONAL MODELING",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#654A73",
    )
    fig.suptitle(
        "CaFE capability examples | mechanism-aware view | primary family | I5",
        y=0.995,
        fontsize=13,
        color="#1E293B",
    )
    fig.text(
        0.5,
        0.025,
        (
            f"Illustrative only: formal cafe L{cafe.CONTEXT_LENGTH}/"
            f"H{cafe.HORIZON} generators with an "
            "experience-based hourly profile; no real-data calibration."
        ),
        ha="center",
        va="center",
        fontsize=8.5,
        color="#687386",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"cafe-capability-overview.{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def render_individual(
    example: dict[str, Any],
    output_dir: Path,
    *,
    cluster_color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    plot_example(
        ax,
        example,
        compact=False,
        cluster_color=cluster_color,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.19, top=0.74)
    fig.text(
        0.055,
        0.925,
        example["spec"]["title"],
        ha="left",
        va="top",
        fontsize=13,
        color="#1E293B",
    )
    fig.text(
        0.055,
        0.845,
        example["spec"]["action"],
        ha="left",
        va="top",
        fontsize=9,
        color="#687386",
    )
    fig.text(
        0.985,
        0.965,
        (
            f"{example['spec']['cluster']} | primary family | "
            f"I{example['intensity']}"
        ),
        ha="right",
        va="top",
        fontsize=8.5,
        color="#687386",
    )
    for suffix in ("png", "pdf"):
        output_path = (
            output_dir
            / f"{example['spec']['capability_id']}.{suffix}"
        )
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def serializable_example(example: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(example["target"], dtype=float)
    covariates = example["covariates"]
    return {
        "spec": example["spec"],
        "seed": int(example["seed"]),
        "intensity": int(example["intensity"]),
        "lambda": float(example["lambda"]),
        "target": np.round(target, 5).tolist(),
        "covariates": (
            None
            if covariates is None
            else np.round(np.asarray(covariates, dtype=float), 5).tolist()
        ),
        "realized_features": {
            name: float(value)
            for name, value in example["realized_features"].items()
        },
        "parameters": {
            name: float(value)
            for name, value in example["parameters"].items()
        },
        "generation_metadata": example["metadata"],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = cafe.DatasetSpec(
        dataset_id="illustrative_empirical_profile",
        logical_name="Illustrative empirical hourly profile",
        config_id=EMPIRICAL_PROFILE_ID,
        asset_name="not-applicable",
        domain="Illustrative",
    )
    examples = {
        spec.capability_id: generate_example(
            dataset,
            spec,
            intensity=args.intensity,
        )
        for spec in CAPABILITIES
    }
    render_overview(examples, output_dir)
    for cluster_index, pair in enumerate(CAPABILITY_PAIRS):
        for spec in pair:
            render_individual(
                examples[spec.capability_id],
                output_dir,
                cluster_color=CLUSTER_COLORS[cluster_index],
            )
    examples_path = output_dir / "examples.json"
    examples_path.write_text(
        json.dumps(
            {
                "schema_version": "cafe.capability_examples.v1",
                "generator_version": cafe.GENERATOR_VERSION,
                "profile_id": EMPIRICAL_PROFILE_ID,
                "calibration_semantics": (
                    "experience-based hourly background profile; "
                    "direct lambda grid; no real-data calibration"
                ),
                "context_length": cafe.CONTEXT_LENGTH,
                "horizon": cafe.HORIZON,
                "intensity_lambdas": list(EMPIRICAL_LAMBDAS),
                "selected_intensity": args.intensity,
                "empirical_features": EMPIRICAL_FEATURES,
                "examples": {
                    capability_id: serializable_example(example)
                    for capability_id, example in examples.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "cafe.capability_example_manifest.v1",
        "generator_version": cafe.GENERATOR_VERSION,
        "profile_id": EMPIRICAL_PROFILE_ID,
        "context_length": cafe.CONTEXT_LENGTH,
        "horizon": cafe.HORIZON,
        "selected_intensity": args.intensity,
        "files": {
            "overview_png": "cafe-capability-overview.png",
            "overview_svg": "cafe-capability-overview.svg",
            "overview_pdf": "cafe-capability-overview.pdf",
            "examples_json": "examples.json",
            "individual_pngs": [
                f"{spec.capability_id}.png" for spec in CAPABILITIES
            ],
            "individual_pdfs": [
                f"{spec.capability_id}.pdf" for spec in CAPABILITIES
            ],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
