from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIXED_CONTEXT_LENGTH = 168
DEFAULT_INTENSITY = 3
CAPABILITY_LABELS = {
    "trend": "Trend",
    "time_varying_seasonality": "Time-varying seasonality",
    "nonlinear_persistence": "Nonlinear persistence",
    "regime_switching": "Regime switching",
    "predictable_intermittency": "Predictable intermittency",
    "multi_seasonal": "Multiple seasonality",
    "cross_series_dependence": "Cross-series dependence",
    "hierarchical_coherence": "Hierarchical coherence",
    "common_factor": "Common factor",
    "covariate_response": "Covariate response",
}
DATASET_LABELS = {
    "gift_ett1_h": "ETT1-H",
    "gift_ett2_h": "ETT2-H",
    "gift_kdd_cup_h": "KDD Cup",
    "gift_m4_hourly": "M4 Hourly",
    "gift_us_births_d": "US Births",
}
CAPABILITY_ORDER = tuple(CAPABILITY_LABELS)
SERIES_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7A5195",
    "#8C564B",
    "#4C78A8",
    "#F58518",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fixed-L168 MASE densities for real anchors and intensity-3 "
            "synthetic capability samples from a CaFE experiment."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="CaFE experiment directory containing gift_* dataset folders.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Dataset to plot. Omit with --rank-candidates.",
    )
    parser.add_argument(
        "--model-id",
        help="Model to plot. Omit with --rank-candidates.",
    )
    parser.add_argument(
        "--intensity",
        type=int,
        default=DEFAULT_INTENSITY,
    )
    parser.add_argument(
        "--display-quantile",
        type=float,
        default=0.98,
        help=(
            "Real-anchor quantile used to choose the visible x range. Values "
            "are not removed from the KDE; only the displayed axis is limited."
        ),
    )
    parser.add_argument(
        "--rank-candidates",
        action="store_true",
        help=(
            "Print dataset/model combinations ranked by the fraction of "
            "capabilities whose synthetic IQR is narrower than the real IQR."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/cafe.figures"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc


def analysis_shards(dataset_dir: Path) -> list[Path]:
    analysis_root = dataset_dir / "04_analysis"
    if not analysis_root.exists():
        return []
    return sorted(
        path
        for path in analysis_root.iterdir()
        if path.is_dir() and (path / "prediction_metrics.jsonl").is_file()
    )


def choose_analysis_shard(dataset_dir: Path) -> Path:
    shards = analysis_shards(dataset_dir)
    if not shards:
        raise FileNotFoundError(
            f"No completed analysis shard under {dataset_dir / '04_analysis'}"
        )
    return shards[-1]


def synthetic_mase_by_capability(
    analysis_dir: Path,
    *,
    model_id: str,
    intensity: int,
) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(analysis_dir / "prediction_metrics.jsonl"):
        if row.get("model_id") != model_id:
            continue
        if row.get("evaluation_table", "main") != "main":
            continue
        if row.get("generator_family_role") != "primary":
            continue
        if int(row.get("context_length", -1)) != FIXED_CONTEXT_LENGTH:
            continue
        if row.get("context_policy") not in (None, "fixed_l168"):
            continue
        if int(row.get("intensity", -1)) != intensity:
            continue
        capability_id = str(row["capability_id"])
        mase = float(row["metrics"]["mase"])
        if math.isfinite(mase) and mase >= 0.0:
            values[capability_id].append(mase)
    return {
        capability_id: np.asarray(capability_values, dtype=float)
        for capability_id, capability_values in values.items()
        if capability_values
    }


def load_real_anchor_mase(
    inference_dir: Path,
    *,
    model_id: str,
) -> np.ndarray:
    views_path = inference_dir / "real_anchor_views.jsonl"
    predictions_path = (
        inference_dir / "real_anchor_predictions" / f"{model_id}.jsonl"
    )
    if not views_path.is_file():
        raise FileNotFoundError(views_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    views = {
        str(row["sample_id"]): row
        for row in read_jsonl(views_path)
        if int(row.get("context_length", -1)) == FIXED_CONTEXT_LENGTH
    }
    values: list[float] = []
    missing_views: list[str] = []
    for prediction in read_jsonl(predictions_path):
        sample_id = str(prediction["sample_id"])
        sample = views.get(sample_id)
        if sample is None:
            missing_views.append(sample_id)
            continue
        target = np.asarray(sample["target"], dtype=float)
        forecast = np.asarray(prediction["forecast"], dtype=float)
        context_length = int(sample["context_length"])
        future = target[context_length:]
        if forecast.shape != future.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: "
                f"forecast={forecast.shape}, future={future.shape}"
            )
        mase_scale = float(sample["mase_scale"])
        mase = float(np.mean(np.abs(future - forecast)) / mase_scale)
        if math.isfinite(mase) and mase >= 0.0:
            values.append(mase)
    if missing_views:
        raise ValueError(
            f"{len(missing_views)} real predictions have no matching view"
        )
    if not values:
        raise ValueError(
            f"No usable real-anchor predictions in {predictions_path}"
        )
    return np.asarray(values, dtype=float)


def inference_dir_for(analysis_dir: Path) -> Path:
    dataset_dir = analysis_dir.parent.parent
    return dataset_dir / "03_inference" / analysis_dir.name


def ordered_capabilities(
    values: dict[str, np.ndarray],
) -> list[str]:
    known = [capability for capability in CAPABILITY_ORDER if capability in values]
    unknown = sorted(set(values) - set(known))
    return known + unknown


def iqr(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def robust_bandwidth(values: np.ndarray) -> float:
    count = len(values)
    if count < 2:
        return max(abs(float(values[0])) * 0.05, 1e-3)
    standard_deviation = float(np.std(values, ddof=1))
    robust_sigma = iqr(values) / 1.349
    positive_scales = [
        scale
        for scale in (standard_deviation, robust_sigma)
        if math.isfinite(scale) and scale > 0.0
    ]
    sigma = min(positive_scales) if positive_scales else 0.0
    if sigma <= 0.0:
        median = float(np.median(values))
        sigma = max(abs(median) * 0.05, 1e-3)
    return max(0.9 * sigma * count ** (-0.2), 1e-4)


def reflected_gaussian_kde(
    values: np.ndarray,
    grid: np.ndarray,
    *,
    bandwidth: float | None = None,
) -> np.ndarray:
    if bandwidth is None:
        bandwidth = robust_bandwidth(values)
    scaled_positive = (
        grid[:, np.newaxis] - values[np.newaxis, :]
    ) / bandwidth
    scaled_reflected = (
        grid[:, np.newaxis] + values[np.newaxis, :]
    ) / bandwidth
    normalization = (
        len(values) * bandwidth * math.sqrt(2.0 * math.pi)
    )
    return (
        np.exp(-0.5 * scaled_positive**2).sum(axis=1)
        + np.exp(-0.5 * scaled_reflected**2).sum(axis=1)
    ) / normalization


def candidate_summary(
    experiment_root: Path,
    *,
    intensity: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset_dir in sorted(experiment_root.glob("gift_*")):
        if not dataset_dir.is_dir():
            continue
        try:
            analysis_dir = choose_analysis_shard(dataset_dir)
        except FileNotFoundError:
            continue
        metric_path = analysis_dir / "prediction_metrics.jsonl"
        model_ids = sorted(
            {
                str(row["model_id"])
                for row in read_jsonl(metric_path)
                if int(row.get("context_length", -1))
                == FIXED_CONTEXT_LENGTH
                and int(row.get("intensity", -1)) == intensity
                and row.get("evaluation_table", "main") == "main"
                and row.get("generator_family_role") == "primary"
            }
        )
        for model_id in model_ids:
            try:
                real = load_real_anchor_mase(
                    inference_dir_for(analysis_dir),
                    model_id=model_id,
                )
            except (FileNotFoundError, ValueError):
                continue
            synthetic = synthetic_mase_by_capability(
                analysis_dir,
                model_id=model_id,
                intensity=intensity,
            )
            if not synthetic or iqr(real) <= 0.0:
                continue
            ratios = {
                capability: iqr(values) / iqr(real)
                for capability, values in synthetic.items()
            }
            summaries.append(
                {
                    "dataset_id": dataset_dir.name,
                    "model_id": model_id,
                    "capability_count": len(ratios),
                    "narrower_fraction": float(
                        np.mean(
                            [ratio < 1.0 for ratio in ratios.values()]
                        )
                    ),
                    "median_iqr_ratio": float(np.median(list(ratios.values()))),
                    "max_iqr_ratio": float(max(ratios.values())),
                    "real_count": len(real),
                    "minimum_synthetic_count": min(
                        len(values) for values in synthetic.values()
                    ),
                }
            )
    return sorted(
        summaries,
        key=lambda row: (
            -row["narrower_fraction"],
            row["median_iqr_ratio"],
            row["max_iqr_ratio"],
            row["dataset_id"],
            row["model_id"],
        ),
    )


def print_candidate_ranking(rows: list[dict[str, Any]]) -> None:
    fields = (
        "dataset_id",
        "model_id",
        "capability_count",
        "narrower_fraction",
        "median_iqr_ratio",
        "max_iqr_ratio",
        "real_count",
        "minimum_synthetic_count",
    )
    print("\t".join(fields))
    for row in rows:
        print(
            "\t".join(
                (
                    str(row["dataset_id"]),
                    str(row["model_id"]),
                    str(row["capability_count"]),
                    f"{row['narrower_fraction']:.3f}",
                    f"{row['median_iqr_ratio']:.3f}",
                    f"{row['max_iqr_ratio']:.3f}",
                    str(row["real_count"]),
                    str(row["minimum_synthetic_count"]),
                )
            )
        )


def plot_distributions(
    *,
    experiment_root: Path,
    dataset_id: str,
    model_id: str,
    intensity: int,
    display_quantile: float,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    if not 0.5 < display_quantile < 1.0:
        raise ValueError("--display-quantile must be between 0.5 and 1")
    dataset_dir = experiment_root / dataset_id
    analysis_dir = choose_analysis_shard(dataset_dir)
    inference_dir = inference_dir_for(analysis_dir)
    real = load_real_anchor_mase(inference_dir, model_id=model_id)
    synthetic = synthetic_mase_by_capability(
        analysis_dir,
        model_id=model_id,
        intensity=intensity,
    )
    capabilities = ordered_capabilities(synthetic)
    if not capabilities:
        raise ValueError(
            f"No intensity-{intensity} fixed-L168 synthetic MASE values for "
            f"{dataset_id}/{model_id}"
        )
    if len(capabilities) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(capabilities)} capabilities exceed the color palette"
        )

    real_limit = float(np.quantile(real, display_quantile))
    synthetic_limit = max(
        float(np.quantile(synthetic[capability], 0.99))
        for capability in capabilities
    )
    x_upper = max(real_limit, synthetic_limit) * 1.04
    if not math.isfinite(x_upper) or x_upper <= 0.0:
        raise ValueError("Could not determine a positive finite x-axis limit")
    grid = np.linspace(0.0, x_upper, 600)
    shared_bandwidth = max(
        robust_bandwidth(real),
        x_upper * 0.015,
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.4, 5.35))
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.83,
        bottom=0.34,
    )
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    real_density = reflected_gaussian_kde(
        real,
        grid,
        bandwidth=shared_bandwidth,
    )
    axis.fill_between(
        grid,
        real_density,
        color="#000000",
        alpha=0.08,
        linewidth=0.0,
        zorder=1,
    )
    axis.plot(
        grid,
        real_density,
        color="#000000",
        linewidth=2.4,
        label="Real anchors",
        zorder=5,
    )

    for index, capability in enumerate(capabilities):
        values = synthetic[capability]
        color = SERIES_COLORS[index]
        density = reflected_gaussian_kde(
            values,
            grid,
            bandwidth=shared_bandwidth,
        )
        label = CAPABILITY_LABELS.get(
            capability,
            capability.replace("_", " ").title(),
        )
        axis.plot(
            grid,
            density,
            color=color,
            linewidth=1.55,
            alpha=0.95,
            label=label,
            zorder=3,
        )

    clipped_real_count = int(np.sum(real > x_upper))
    axis.set_xlim(0.0, x_upper)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Forecast error (MASE)")
    axis.set_ylabel("Probability density")
    dataset_label = DATASET_LABELS.get(
        dataset_id,
        dataset_id.removeprefix("gift_").replace("_", " ").title(),
    )
    figure.text(
        0.105,
        0.96,
        f"MASE distributions on {dataset_label} · {model_id}",
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="normal",
    )
    common_synthetic_count = sorted(
        {len(synthetic[capability]) for capability in capabilities}
    )
    synthetic_count_text = "/".join(
        str(count) for count in common_synthetic_count
    )
    figure.text(
        0.105,
        0.915,
        (
            f"Real anchors vs. intensity-{intensity} synthetic mechanisms "
            f"· fixed context L={FIXED_CONTEXT_LENGTH} · "
            f"n={len(real)} real, n={synthetic_count_text} synthetic"
        ),
        ha="left",
        va="top",
        color="#555555",
        fontsize=8.5,
    )
    if clipped_real_count:
        axis.text(
            0.99,
            0.98,
            (
                f"Display limited at MASE {x_upper:.2f}; "
                f"{clipped_real_count}/{len(real)} real anchors lie to the right"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#555555",
            fontsize=7.8,
        )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        ncol=3 if len(capabilities) >= 6 else 2,
        columnspacing=1.5,
        handlelength=2.1,
        handletextpad=0.55,
        labelspacing=0.75,
        borderaxespad=0.0,
        fontsize=7.6,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"mase-density__{dataset_id}__{model_id}"
        f"__intensity-{intensity}__fixed-l168"
    )
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    metadata_path = output_dir / f"{stem}.json"
    figure.savefig(png_path, dpi=360, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    real_iqr = iqr(real)
    metadata = {
        "schema_version": "cafe.mase_density_figure.v1",
        "experiment_root": str(experiment_root.resolve()),
        "analysis_dir": str(analysis_dir.resolve()),
        "dataset_id": dataset_id,
        "model_id": model_id,
        "context_policy": "fixed_l168",
        "context_length": FIXED_CONTEXT_LENGTH,
        "intensity": intensity,
        "kde": {
            "kernel": "gaussian_with_zero_boundary_reflection",
            "bandwidth": "shared_real_anchor_robust_silverman",
            "bandwidth_value": shared_bandwidth,
            "all_observations_included": True,
        },
        "display": {
            "real_anchor_quantile_requested": display_quantile,
            "x_upper": x_upper,
            "real_anchor_count_beyond_x_upper": clipped_real_count,
            "note": (
                "The x-axis is display-limited only; observations beyond the "
                "limit remain in the KDE denominator and bandwidth calculation."
            ),
        },
        "real_anchors": {
            "count": len(real),
            "median": float(np.median(real)),
            "iqr": real_iqr,
            "maximum": float(np.max(real)),
        },
        "synthetic_capabilities": {
            capability: {
                "count": len(synthetic[capability]),
                "median": float(np.median(synthetic[capability])),
                "iqr": iqr(synthetic[capability]),
                "iqr_over_real_iqr": (
                    iqr(synthetic[capability]) / real_iqr
                    if real_iqr > 0.0
                    else None
                ),
                "maximum": float(np.max(synthetic[capability])),
            }
            for capability in capabilities
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return png_path, pdf_path, metadata_path


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    if args.rank_candidates:
        print_candidate_ranking(
            candidate_summary(
                experiment_root,
                intensity=args.intensity,
            )
        )
        return
    if not args.dataset_id or not args.model_id:
        raise SystemExit(
            "--dataset-id and --model-id are required unless "
            "--rank-candidates is used"
        )
    paths = plot_distributions(
        experiment_root=experiment_root,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        intensity=args.intensity,
        display_quantile=args.display_quantile,
        output_dir=args.output_dir.resolve(),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
