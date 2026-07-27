#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyze_paper_v8 as analysis  # noqa: E402
import paper_v8_pipeline_common as v8  # noqa: E402
import paper_v8_structured_baselines as structured  # noqa: E402


CAPABILITY_ID = "cross_series_dependence"
CONTEXT_LENGTH = 336
HORIZON = 48
INTENSITY = 5
MODEL_IDS = ("Chronos-2", "timesfm2.5", "tirex2", "toto2.0")
COLORS = {
    "truth": "#111827",
    "member_0": "#65717E",
    "member_1": "#7B2CBF",
    "full_ridge_var": "#0072B2",
    "Chronos-2": "#D55E00",
    "timesfm2.5": "#8D99A6",
    "tirex2": "#CC79A7",
    "toto2.0": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a representative Jena Weather cross-series I5/L336 pair."
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="gift_jena_weather_h")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def strict_pairs(task_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    pairs: dict[str, dict[int, dict[str, Any]]] = {}
    for sample in v8.iter_jsonl(task_path):
        if not (
            sample.get("capability_id") == CAPABILITY_ID
            and sample.get("generator_family_role") == "primary"
            and int(sample.get("intensity", -1)) == INTENSITY
            and int(sample.get("context_length", -1)) == CONTEXT_LENGTH
            and sample.get("evaluation_table")
            == "strict_counterfactual_audit"
        ):
            continue
        pair_id = str(sample["counterfactual_pair_id"])
        member = int(sample["counterfactual_member"])
        pairs.setdefault(pair_id, {})[member] = sample
    complete = {
        pair_id: members
        for pair_id, members in pairs.items()
        if set(members) == {0, 1}
    }
    if not complete:
        raise ValueError("No complete strict counterfactual pairs found")
    return complete


def load_predictions(
    inference_dir: Path,
    model_id: str,
) -> dict[str, np.ndarray]:
    safe_model_id = analysis.engine.safe_filename(model_id)
    prediction_path = (
        inference_dir
        / "model_shards"
        / safe_model_id
        / "predictions"
        / f"{safe_model_id}.jsonl"
    )
    return {
        str(row["sample_id"]): np.asarray(row["forecast"], dtype=float)
        for row in v8.iter_jsonl(prediction_path)
    }


def full_var_forecast(
    sample: dict[str, Any],
) -> structured.StructuredForecast:
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    return structured._ar_or_var_forecast(
        target[:context],
        int(sample["horizon"]),
        model_id="full_ridge_var",
        diagonal=False,
    )


def select_pair(
    pairs: dict[str, dict[int, dict[str, Any]]],
) -> tuple[
    str,
    dict[int, dict[str, Any]],
    dict[int, structured.StructuredForecast],
    list[dict[str, Any]],
]:
    candidates: list[
        tuple[
            str,
            dict[int, dict[str, Any]],
            dict[int, structured.StructuredForecast],
            dict[str, Any],
        ]
    ] = []
    for pair_id, members in pairs.items():
        forecasts = {
            member: full_var_forecast(sample)
            for member, sample in members.items()
        }
        effect = analysis.effect_row(
            members[0],
            forecasts[0].forecast,
            members[1],
            forecasts[1].forecast,
            model_id="full_ridge_var",
        )
        candidates.append((pair_id, members, forecasts, effect))
    median_nrmse = float(
        np.median(
            [
                float(candidate[3]["active_effect_nrmse"])
                for candidate in candidates
            ]
        )
    )
    candidates.sort(
        key=lambda candidate: (
            abs(
                float(candidate[3]["active_effect_nrmse"])
                - median_nrmse
            ),
            int(candidate[1][0]["seed_index"]),
        )
    )
    pair_id, members, forecasts, _effect = candidates[0]
    audit = [
        {
            "pair_id": candidate[0],
            "seed_index": int(candidate[1][0]["seed_index"]),
            "active_effect_nrmse": float(
                candidate[3]["active_effect_nrmse"]
            ),
            "distance_from_median": abs(
                float(candidate[3]["active_effect_nrmse"])
                - median_nrmse
            ),
        }
        for candidate in candidates
    ]
    return pair_id, members, forecasts, audit


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#66717E")
    axis.spines["bottom"].set_color("#66717E")


def render(
    members: dict[int, dict[str, Any]],
    forecasts: dict[str, dict[int, np.ndarray]],
    effects: dict[str, dict[str, Any]],
    *,
    responder: int,
    driver: int,
    active_steps: int,
    output_stem: Path,
) -> None:
    member_targets = {
        member: np.asarray(sample["target"], dtype=float)
        for member, sample in members.items()
    }
    responder_scale = max(
        float(np.std(member_targets[0][:CONTEXT_LENGTH, responder])),
        1e-8,
    )
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
        figsize=(14.2, 4.7),
        gridspec_kw={"width_ratios": [0.95, 1.08, 1.12]},
    )

    history_window = 48
    history_x = np.arange(-history_window + 1, 1)
    for member, label in ((0, "Member 0"), (1, "Member 1")):
        axes[0].plot(
            history_x,
            member_targets[member][
                CONTEXT_LENGTH - history_window : CONTEXT_LENGTH,
                driver,
            ],
            color=COLORS[f"member_{member}"],
            linewidth=2.0 if member else 1.7,
            linestyle="-" if member else "--",
            label=label,
        )
    axes[0].axvspan(
        -active_steps + 0.5,
        0.5,
        color="#BDE3F4",
        alpha=0.45,
        linewidth=0,
    )
    axes[0].axvline(0.5, color="#495462", linewidth=1.0)
    axes[0].annotate(
        "paired driver\nperturbation",
        xy=(
            -max(active_steps // 2, 1),
            member_targets[1][
                CONTEXT_LENGTH - max(active_steps // 2, 1),
                driver,
            ],
        ),
        xytext=(-30, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.6,
        color="#4B5563",
        arrowprops={"arrowstyle": "->", "color": "#66717E"},
    )
    axes[0].set_title(
        "(a) Observed driver history",
        loc="left",
        fontsize=10.8,
        fontweight="bold",
    )
    axes[0].set_xlabel("Step relative to forecast origin")
    axes[0].set_ylabel(f"Driver channel {driver}")
    axes[0].legend(frameon=False, fontsize=8.6, loc="upper left")
    style_axis(axes[0])

    horizon_x = np.arange(1, HORIZON + 1)
    branch_steps = min(active_steps + 8, HORIZON)
    axes[1].axvspan(
        0.5,
        active_steps + 0.5,
        color="#BDE3F4",
        alpha=0.45,
        linewidth=0,
    )
    axes[1].axvspan(
        active_steps + 0.5,
        branch_steps + 0.5,
        color="#E5E7EB",
        alpha=0.45,
        linewidth=0,
    )
    for member, label in ((0, "Truth member 0"), (1, "Truth member 1")):
        axes[1].plot(
            horizon_x[:branch_steps],
            member_targets[member][
                CONTEXT_LENGTH : CONTEXT_LENGTH + branch_steps,
                responder,
            ],
            color=COLORS[f"member_{member}"],
            linewidth=2.0,
            linestyle="-" if member else "--",
            label=label,
        )
        axes[1].plot(
            horizon_x[:branch_steps],
            forecasts["full_ridge_var"][member][
                :branch_steps,
                responder,
            ],
            color=COLORS["full_ridge_var"],
            linewidth=1.45,
            linestyle="-" if member else "--",
            alpha=0.9,
            label=f"Full Ridge-VAR m{member}",
        )
    axes[1].axvline(
        active_steps + 0.5,
        color="#6B7280",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1].text(
        (active_steps + branch_steps + 1) / 2,
        0.03,
        "unscored: future driver unobserved",
        transform=axes[1].get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#5B6571",
    )
    axes[1].set_xlim(0.5, branch_steps + 0.5)
    axes[1].set_title(
        f"(b) Responder {responder} future branches",
        loc="left",
        fontsize=10.8,
        fontweight="bold",
    )
    axes[1].set_xlabel("Forecast step")
    axes[1].set_ylabel("Target level")
    axes[1].legend(
        frameon=False,
        fontsize=7.8,
        loc="upper right",
        ncol=2,
    )
    style_axis(axes[1])

    truth_effect = (
        member_targets[1][CONTEXT_LENGTH:, responder]
        - member_targets[0][CONTEXT_LENGTH:, responder]
    ) / responder_scale
    effect_curves = {
        model_id: (
            model_forecasts[1][:, responder]
            - model_forecasts[0][:, responder]
        )
        / responder_scale
        for model_id, model_forecasts in forecasts.items()
    }
    axes[2].axvspan(
        0.5,
        active_steps + 0.5,
        color="#BDE3F4",
        alpha=0.45,
        linewidth=0,
    )
    axes[2].axhline(0.0, color="#65717E", linewidth=0.9)
    axes[2].plot(
        horizon_x[:active_steps],
        truth_effect[:active_steps],
        color=COLORS["truth"],
        linewidth=2.6,
        label="True effect",
        zorder=5,
    )
    for model_id, label, linestyle in (
        ("full_ridge_var", "Full Ridge-VAR", "-"),
        ("Chronos-2", "Chronos-2", "--"),
        ("timesfm2.5", "TimesFM", ":"),
        ("tirex2", "TiRex 2", (0, (5, 1.5))),
        ("toto2.0", "Toto 2.0", "-."),
    ):
        axes[2].plot(
            horizon_x[:active_steps],
            effect_curves[model_id][:active_steps],
            color=COLORS[model_id],
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
        )
    axes[2].text(
        0.98,
        0.97,
        f"active prefix: {active_steps} steps",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#4B5563",
    )
    axes[2].set_xlim(0.5, active_steps + 0.5)
    metric_text = "\n".join(
        [
            (
                f"Full Ridge-VAR: active NRMSE "
                f"{effects['full_ridge_var']['active_effect_nrmse']:.3f}, "
                f"r {effects['full_ridge_var']['active_effect_correlation']:.3f}"
            ),
            (
                f"Chronos-2: active NRMSE "
                f"{effects['Chronos-2']['active_effect_nrmse']:.3f}"
            ),
            (
                f"TimesFM: active NRMSE "
                f"{effects['timesfm2.5']['active_effect_nrmse']:.3f}"
            ),
            (
                f"TiRex 2: active NRMSE "
                f"{effects['tirex2']['active_effect_nrmse']:.3f}"
            ),
            (
                f"Toto 2.0: active NRMSE "
                f"{effects['toto2.0']['active_effect_nrmse']:.3f}"
            ),
        ]
    )
    axes[2].text(
        0.98,
        0.05,
        metric_text,
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CBD2D9",
            "alpha": 0.92,
        },
    )
    axes[2].set_title(
        "(c) History-identified effect recovery",
        loc="left",
        fontsize=10.8,
        fontweight="bold",
    )
    axes[2].set_xlabel("Forecast step")
    axes[2].set_ylabel("Treatment − control effect / history std")
    axes[2].legend(
        frameon=False,
        fontsize=7.7,
        loc="upper left",
        ncol=2,
    )
    style_axis(axes[2])

    seed_index = int(members[0]["seed_index"])
    figure.suptitle(
        "Cross-channel effect recovery differs despite similar forecasts",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.932,
        (
            "GIFT-Eval Jena Weather · cross-series dependence · "
            f"I5 · L336 · seed {seed_index} · median-near exemplar"
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
            "Only steps 1–8 are identified from the observed driver tail and "
            "scored. Exemplar selection: Full Ridge-VAR active NRMSE closest "
            "to the 15-pair median; responder with largest true effect RMS."
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
        bottom=0.19,
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
    pairs = strict_pairs(task_path)
    pair_id, members, full_results, selection_audit = select_pair(pairs)
    prediction_maps = {
        model_id: load_predictions(inference_dir, model_id)
        for model_id in MODEL_IDS
    }
    forecasts: dict[str, dict[int, np.ndarray]] = {
        "full_ridge_var": {
            member: result.forecast
            for member, result in full_results.items()
        }
    }
    for model_id, prediction_map in prediction_maps.items():
        forecasts[model_id] = {
            member: prediction_map[str(sample["sample_id"])]
            for member, sample in members.items()
        }
    effects = {
        model_id: analysis.effect_row(
            members[0],
            model_forecasts[0],
            members[1],
            model_forecasts[1],
            model_id=model_id,
        )
        for model_id, model_forecasts in forecasts.items()
    }
    metadata = members[0]["generation_metadata"]
    driver = int(metadata["driver_index"])
    responders = [int(value) for value in metadata["responder_indices"]]
    active_steps = int(metadata["counterfactual_effect_forecast_steps"])
    member_targets = {
        member: np.asarray(sample["target"], dtype=float)
        for member, sample in members.items()
    }
    truth_effect = (
        member_targets[1][CONTEXT_LENGTH:]
        - member_targets[0][CONTEXT_LENGTH:]
    )
    responder = max(
        responders,
        key=lambda channel: float(
            np.sqrt(np.mean(truth_effect[:active_steps, channel] ** 2))
        ),
    )
    output_dir = args.output_dir or (
        dataset_dir / "04_analysis" / shard_name / "figures"
    )
    output_stem = output_dir / (
        "jena_cross_series_i5_l336_"
        f"seed{int(members[0]['seed_index']):03d}_effect_curve"
    )
    render(
        members,
        forecasts,
        effects,
        responder=responder,
        driver=driver,
        active_steps=active_steps,
        output_stem=output_stem,
    )
    summary = {
        "schema_version": "paper_v8_jena_cross_series_sample_figure.v1",
        "dataset_id": args.dataset_id,
        "capability_id": CAPABILITY_ID,
        "intensity": INTENSITY,
        "context_length": CONTEXT_LENGTH,
        "pair_id": pair_id,
        "seed_index": int(members[0]["seed_index"]),
        "driver_channel": driver,
        "displayed_responder_channel": responder,
        "active_effect_steps": active_steps,
        "selection_policy": (
            "closest Full Ridge-VAR active-effect NRMSE to the 15-pair median; "
            "then responder with largest true active-effect RMS"
        ),
        "selection_audit": selection_audit,
        "model_effect_metrics": {
            model_id: {
                "active_effect_nrmse": float(row["active_effect_nrmse"]),
                "active_effect_correlation": float(
                    row["active_effect_correlation"]
                ),
                "active_effect_amplitude_ratio": float(
                    row["active_effect_amplitude_ratio"]
                ),
                "zero_tail_leakage_nrmse": float(
                    row["zero_tail_leakage_nrmse"]
                ),
                "full_horizon_effect_nrmse": float(
                    row["counterfactual_effect_nrmse"]
                ),
            }
            for model_id, row in effects.items()
        },
        "full_ridge_var_diagnostics": {
            str(member): result.diagnostics
            for member, result in full_results.items()
        },
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
