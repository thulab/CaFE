#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cafe import protocol as cafe
from cafe.analysis import runner as analysis
from cafe.analysis import structured


CAPABILITY_ID = "cross_series_dependence"
CONTEXT_LENGTH = 336
INTENSITY = 5
CONTEXT_POLICY = "fixed_l336"
EVALUATION_TABLES = frozenset(
    {"main", "multivariate_input_ablation", "strict_counterfactual_audit"}
)
FOUNDATION_MODELS = (
    "timesfm2.5",
    "Timer-3.5",
    "moirai2",
    "Chronos-2",
    "tirex2",
    "toto2.0",
)
METHODS = (
    ("diagonal_ar", "Diagonal AR", "marginal"),
    ("full_ridge_var", "Full Ridge-VAR", "classical"),
    ("timesfm2.5", "TimesFM 2.5", "univariate"),
    ("Timer-3.5", "Timer 3.5", "univariate"),
    ("moirai2", "Moirai 2", "univariate"),
    ("Chronos-2", "Chronos-2", "native"),
    ("tirex2", "TiRex", "native"),
    ("toto2.0", "Toto", "native"),
)
X_LABELS = {
    "diagonal_ar": "Diag. AR",
    "full_ridge_var": "Full Ridge-VAR",
    "timesfm2.5": "TimesFM",
    "Timer-3.5": "Timer",
    "moirai2": "Moirai",
    "Chronos-2": "Chronos",
    "tirex2": "TiRex",
    "toto2.0": "Toto",
}
ROLE_COLORS = {
    "marginal": "#8D99A6",
    "classical": "#0072B2",
    "univariate": "#B8BDC5",
    "native": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the CaFE Jena Weather cross-series I5/L336 case study."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-id",
        default="gift_jena_weather_h",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    return parser.parse_args()


def selected_samples(task_path: Path) -> list[dict[str, Any]]:
    return [
        sample
        for sample in cafe.iter_jsonl(task_path)
        if sample.get("capability_id") == CAPABILITY_ID
        and sample.get("generator_family_role") == "primary"
        and int(sample.get("intensity", -1)) == INTENSITY
        and int(sample.get("context_length", -1)) == CONTEXT_LENGTH
        and sample.get("evaluation_table", "main") in EVALUATION_TABLES
    ]


def tagged_metric(
    sample: dict[str, Any],
    *,
    model_id: str,
    forecast: np.ndarray,
    adaptation: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        **analysis.metric_row(
            sample,
            model_id=model_id,
            forecast=forecast,
            input_adaptation=adaptation,
        ),
        "context_policy": CONTEXT_POLICY,
    }


def tagged_effect(
    first_sample: dict[str, Any],
    first_forecast: np.ndarray,
    second_sample: dict[str, Any],
    second_forecast: np.ndarray,
    *,
    model_id: str,
) -> dict[str, Any]:
    return {
        **analysis.effect_row(
            first_sample,
            first_forecast,
            second_sample,
            second_forecast,
            model_id=model_id,
        ),
        "context_policy": CONTEXT_POLICY,
    }


def foundation_rows(
    samples: list[dict[str, Any]],
    *,
    inference_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids = {str(sample["sample_id"]) for sample in samples}
    metrics: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for model_id in FOUNDATION_MODELS:
        safe_model_id = analysis.engine.safe_filename(model_id)
        prediction_path = (
            inference_dir
            / "model_shards"
            / safe_model_id
            / "predictions"
            / f"{safe_model_id}.jsonl"
        )
        predictions = {
            str(row["sample_id"]): row
            for row in cafe.iter_jsonl(prediction_path)
            if str(row["sample_id"]) in sample_ids
        }
        if len(predictions) != len(sample_ids):
            raise ValueError(
                f"{model_id}: found {len(predictions)} of "
                f"{len(sample_ids)} required predictions"
            )
        pending: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
        for sample in samples:
            prediction = predictions[str(sample["sample_id"])]
            forecast = np.asarray(prediction["forecast"], dtype=float)
            metrics.append(
                tagged_metric(
                    sample,
                    model_id=model_id,
                    forecast=forecast,
                    adaptation=prediction.get("input_adaptation"),
                )
            )
            pair_id = sample.get("counterfactual_pair_id")
            member = sample.get("counterfactual_member")
            if pair_id is None or member is None:
                continue
            pair_key = str(pair_id)
            if int(member) == 0:
                pending[pair_key] = (sample, forecast)
                continue
            first = pending.pop(pair_key, None)
            if first is not None:
                effects.append(
                    tagged_effect(
                        first[0],
                        first[1],
                        sample,
                        forecast,
                        model_id=model_id,
                    )
                )
    return metrics, effects


def classical_rows(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    pending: dict[
        str,
        dict[str, tuple[dict[str, Any], np.ndarray]],
    ] = {
        "diagonal_ar": {},
        "full_ridge_var": {},
    }
    for sample in samples:
        target = np.asarray(sample["target"], dtype=float)
        context = int(sample["context_length"])
        horizon = int(sample["horizon"])
        pair_id = sample.get("counterfactual_pair_id")
        member = sample.get("counterfactual_member")
        results = {
            "diagonal_ar": structured.forecast(
                sample,
                "diagonal_ar",
            ),
            "full_ridge_var": structured._ar_or_var_forecast(
                target[:context],
                horizon,
                model_id="full_ridge_var",
                diagonal=False,
            ),
        }
        for model_id, result in results.items():
            metrics.append(
                tagged_metric(
                    sample,
                    model_id=model_id,
                    forecast=result.forecast,
                    adaptation={
                        "target_mode": (
                            "local_history_only_classical_baseline"
                        ),
                        "baseline_diagnostics": result.diagnostics,
                    },
                )
            )
            if pair_id is None or member is None:
                continue
            pair_key = str(pair_id)
            if int(member) == 0:
                pending[model_id][pair_key] = (sample, result.forecast)
                continue
            first = pending[model_id].pop(pair_key, None)
            if first is not None:
                effects.append(
                    tagged_effect(
                        first[0],
                        first[1],
                        sample,
                        result.forecast,
                        model_id=model_id,
                    )
                )
    return metrics, effects


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def method_distributions(
    metrics: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
]:
    mase = {
        model_id: finite_array(
            row["metrics"]["mase"]
            for row in metrics
            if row["model_id"] == model_id
            and row["evaluation_table"] == "main"
        )
        for model_id, _label, _role in METHODS
    }
    effect_nrmse = {
        model_id: finite_array(
            row["active_effect_nrmse"]
            for row in effects
            if row["model_id"] == model_id
            and "active_effect_nrmse" in row
        )
        for model_id, _label, _role in METHODS
    }
    comparisons = analysis.matched_comparison_rows(metrics, effects)
    ablation = {
        str(row["model_id"]): (
            100.0 * float(row["accuracy_relative_delta"])
        )
        for row in comparisons
        if row["comparison_id"] == "multivariate_input_ablation"
        and row["capability_id"] == CAPABILITY_ID
        and row["accuracy_relative_delta"] is not None
    }
    ablation["diagonal_ar"] = 0.0
    return mase, effect_nrmse, ablation


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#66717E")
    axis.spines["bottom"].set_color("#66717E")


def distribution_panel(
    axis: plt.Axes,
    distributions: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    reference_line: float | None = None,
) -> None:
    rng = np.random.default_rng(20260727)
    positions = np.arange(1, len(METHODS) + 1)
    for position, (model_id, _label, role) in zip(
        positions,
        METHODS,
        strict=True,
    ):
        values = distributions[model_id]
        box = axis.boxplot(
            [values],
            positions=[position],
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1F2933", "linewidth": 1.3},
            whiskerprops={"color": "#66717E", "linewidth": 1.0},
            capprops={"color": "#66717E", "linewidth": 1.0},
            boxprops={"edgecolor": "#495462", "linewidth": 0.9},
        )
        box["boxes"][0].set_facecolor(ROLE_COLORS[role])
        box["boxes"][0].set_alpha(0.82)
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=10,
            color=ROLE_COLORS[role],
            alpha=0.28,
            linewidths=0,
            zorder=2,
        )
        mean = float(np.mean(values))
        axis.scatter(
            [position],
            [mean],
            marker="D",
            s=25,
            color="#111827",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    if reference_line is not None:
        axis.axhline(
            reference_line,
            color="#B42318",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="No effect recovery",
        )
    axis.set_title(title, loc="left", fontsize=10.8, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [X_LABELS[model_id] for model_id, _label, _role in METHODS],
        rotation=30,
        ha="right",
    )
    style_axis(axis)


def ablation_panel(
    axis: plt.Axes,
    ablation: dict[str, float],
) -> None:
    positions = np.arange(1, len(METHODS) + 1)
    values = np.asarray(
        [ablation.get(model_id, 0.0) for model_id, _label, _role in METHODS],
        dtype=float,
    )
    colors = [ROLE_COLORS[role] for _model, _label, role in METHODS]
    axis.bar(
        positions,
        values,
        width=0.64,
        color=colors,
        edgecolor="#495462",
        linewidth=0.8,
        alpha=0.86,
    )
    axis.axhline(0.0, color="#495462", linewidth=0.9)
    span = max(float(np.max(np.abs(values))), 1.0)
    for position, value in zip(positions, values, strict=True):
        if abs(value) < 0.1:
            continue
        offset = 0.022 * span
        axis.text(
            position,
            value + (offset if value >= 0.0 else -offset),
            f"{value:+.2f}%",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=8.2,
            color="#27313C",
        )
    near_zero = values[np.abs(values) < 0.1]
    if len(near_zero):
        axis.text(
            0.03,
            0.93,
            f"Others: |Δ| < {float(np.max(np.abs(near_zero))):.2f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#5B6571",
        )
    axis.set_title(
        "(c) I5 cross-channel input ablation",
        loc="left",
        fontsize=10.8,
        fontweight="bold",
    )
    axis.set_ylabel("Responder error increase (%)")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [X_LABELS[model_id] for model_id, _label, _role in METHODS],
        rotation=30,
        ha="right",
    )
    style_axis(axis)


def summary_rows(
    mase: dict[str, np.ndarray],
    effect_nrmse: dict[str, np.ndarray],
    ablation: dict[str, float],
    effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    correlations: dict[str, list[float]] = defaultdict(list)
    for row in effects:
        if "active_effect_correlation" in row:
            correlations[str(row["model_id"])].append(
                float(row["active_effect_correlation"])
            )
    rows: list[dict[str, Any]] = []
    for model_id, label, role in METHODS:
        mase_values = mase[model_id]
        effect_values = effect_nrmse[model_id]
        correlation_values = finite_array(correlations[model_id])
        rows.append(
            {
                "model_id": model_id,
                "label": label,
                "role": role,
                "mase_count": len(mase_values),
                "mase_mean": float(np.mean(mase_values)),
                "mase_median": float(np.median(mase_values)),
                "effect_pair_count": len(effect_values),
                "effect_nrmse_median": float(np.median(effect_values)),
                "effect_correlation_median": (
                    float(np.median(correlation_values))
                    if len(correlation_values)
                    else None
                ),
                "input_ablation_relative_error_increase": (
                    float(ablation.get(model_id, 0.0))
                ),
            }
        )
    return rows


def render(
    mase: dict[str, np.ndarray],
    effect_nrmse: dict[str, np.ndarray],
    ablation: dict[str, float],
    *,
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.labelcolor": "#27313C",
            "xtick.color": "#495462",
            "ytick.color": "#495462",
            "text.color": "#1F2933",
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 5.2),
        gridspec_kw={"width_ratios": [1.08, 1.08, 1.0]},
    )
    distribution_panel(
        axes[0],
        mase,
        title="(a) Marginal forecasting accuracy",
        ylabel="MASE (lower is better)",
    )
    distribution_panel(
        axes[1],
        effect_nrmse,
        title="(b) Cross-channel effect recovery",
        ylabel="Active-prefix effect NRMSE (lower is better)",
        reference_line=1.0,
    )
    ablation_panel(axes[2], ablation)
    role_legend = [
        Patch(
            facecolor=ROLE_COLORS[role],
            edgecolor="#495462",
            label=label,
        )
        for role, label in (
            ("marginal", "Marginal baseline"),
            ("classical", "Classical multivariate baseline"),
            ("univariate", "Independent-univariate"),
            ("native", "Native-multivariate"),
        )
    ]
    figure.legend(
        handles=role_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.902),
        ncol=4,
        frameon=False,
        fontsize=8.6,
    )
    figure.suptitle(
        "Marginal accuracy and cross-channel effect recovery can diverge",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.938,
        (
            "GIFT-Eval Jena Weather · cross-series dependence · "
            "real-calibrated I5 · fixed L336"
        ),
        ha="center",
        va="center",
        fontsize=9.8,
        color="#4B5563",
    )
    figure.text(
        0.5,
        0.012,
        (
            "64 ordinary samples; 15 counterfactual pairs. Full Ridge-VAR is "
            "fit independently per member from history only (no shared-pair "
            "fit). Ablation uses responder normalized MAE. Boxes show IQR."
        ),
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#5B6571",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.82,
        bottom=0.225,
        wspace=0.3,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    shard_name = (
        f"seed_{args.seed_start:06d}_"
        f"{args.seed_start + args.seed_count:06d}"
    )
    dataset_dir = args.experiment_root.resolve() / args.dataset_id
    inference_dir = dataset_dir / "03_inference" / shard_name
    task_path = inference_dir / "forecast_views.jsonl"
    if not task_path.is_file():
        raise FileNotFoundError(task_path)
    samples = selected_samples(task_path)
    if not samples:
        raise ValueError("No Jena cross-series I5/L336 samples found")
    foundation_metrics, foundation_effects = foundation_rows(
        samples,
        inference_dir=inference_dir,
    )
    classical_metrics, classical_effects = classical_rows(samples)
    metrics = foundation_metrics + classical_metrics
    effects = foundation_effects + classical_effects
    mase, effect_nrmse, ablation = method_distributions(metrics, effects)
    for model_id, _label, _role in METHODS:
        if len(mase[model_id]) != args.seed_count:
            raise ValueError(
                f"{model_id}: expected {args.seed_count} MASE rows, "
                f"found {len(mase[model_id])}"
            )
        if not len(effect_nrmse[model_id]):
            raise ValueError(f"{model_id}: no strict effect pairs")
    output_dir = args.output_dir or (
        dataset_dir / "04_analysis" / shard_name / "figures"
    )
    output_stem = output_dir / "jena_cross_series_i5_l336_full_var"
    render(
        mase,
        effect_nrmse,
        ablation,
        output_stem=output_stem,
    )
    summary = {
        "schema_version": "cafe.jena_cross_series_case_figure.v1",
        "dataset_id": args.dataset_id,
        "capability_id": CAPABILITY_ID,
        "generator_family_role": "primary",
        "intensity": INTENSITY,
        "context_length": CONTEXT_LENGTH,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "methods": summary_rows(
            mase,
            effect_nrmse,
            ablation,
            effects,
        ),
        "figure_files": [
            str(output_stem.with_suffix(".png")),
            str(output_stem.with_suffix(".pdf")),
        ],
    }
    summary_path = output_stem.with_name(
        f"{output_stem.name}_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "png": str(output_stem.with_suffix(".png")),
                "pdf": str(output_stem.with_suffix(".pdf")),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
