#!/usr/bin/env python3
"""Derive paper-facing summaries from frozen CaFE main-experiment JSON files.

The script only reads ``source_snapshot`` and writes derived artifacts beside
it.  The production experiment directories on ``timecho92`` are never touched.

Run from the repository root with::

    uv run --with matplotlib python \
      paper_results/work/main_experiments/analyze_main_experiments.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_snapshot"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"

BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20260902

EXPERIMENTS = {
    "gift-v15-short-qualified-feasible-seed2026082701": {
        "suite": "GIFT-Short",
        "benchmark": "GIFT-Eval",
        "horizon_group": "short",
        "order": 0,
    },
    "gift-v15-medium-qualified-feasible-seed2026082701": {
        "suite": "GIFT-Medium",
        "benchmark": "GIFT-Eval",
        "horizon_group": "medium",
        "order": 1,
    },
    "gift-v15-long-qualified-feasible-seed2026082701": {
        "suite": "GIFT-Long",
        "benchmark": "GIFT-Eval",
        "horizon_group": "long",
        "order": 2,
    },
    "fev-mini20-full-v6": {
        "suite": "FEV-Mini20",
        "benchmark": "FEV-Bench",
        "horizon_group": "native",
        "order": 3,
    },
}

MODEL_ORDER = [
    "Chronos-2",
    "Timer-4.0",
    "Timer-3.5",
    "timesfm2.5",
    "tirex2",
    "moirai2",
    "toto2.0",
]

CAPABILITY_ORDER = [
    "trend",
    "regime_switching",
    "common_factor",
    "covariate_impulse_response",
    "time_varying_seasonality",
    "multi_seasonal",
    "predictable_intermittency",
    "cross_series_dependence",
]

CAPABILITY_LABELS = {
    "trend": "Trend",
    "regime_switching": "Regime",
    "common_factor": "Common factor",
    "covariate_impulse_response": "Covariate impulse",
    "time_varying_seasonality": "Time-varying seasonality",
    "multi_seasonal": "Multi-seasonality",
    "predictable_intermittency": "Intermittency",
    "cross_series_dependence": "Cross-series",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / name, index=False, float_format="%.10g")


def oriented_pair_result(row: pd.Series, model_id: str) -> str:
    """Return win/loss/tie for model from a left-minus-right bootstrap row."""

    lower = float(row["paired_task_bootstrap_95_ci_lower"])
    upper = float(row["paired_task_bootstrap_95_ci_upper"])
    if lower <= 0.0 <= upper:
        return "not_significant"
    left_wins = upper < 0.0
    model_wins = (
        row["left_model_id"] == model_id and left_wins
    ) or (
        row["right_model_id"] == model_id and not left_wins
    )
    return "win" if model_wins else "loss"


def load_all() -> dict[str, pd.DataFrame]:
    suite_rows: list[dict[str, Any]] = []
    suite_pairs: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    effect_task_rows: list[dict[str, Any]] = []
    accuracy_task_rows: list[dict[str, Any]] = []
    analysis_manifest_rows: list[dict[str, Any]] = []
    inference_status_rows: list[dict[str, Any]] = []

    for experiment_id, metadata in EXPERIMENTS.items():
        experiment_root = SOURCE / experiment_id
        common = {
            "experiment_id": experiment_id,
            "suite": metadata["suite"],
            "benchmark": metadata["benchmark"],
            "horizon_group": metadata["horizon_group"],
            "suite_order": metadata["order"],
        }
        suite = read_json(
            experiment_root / "04_analysis_suite" / "task_equal_summary.json"
        )
        for row in suite["rows"]:
            suite_rows.append(common | row)
        for row in suite["paired_model_comparisons"]:
            suite_pairs.append(common | row)

        for path in sorted(experiment_root.glob("*/01_generation/manifest.json")):
            manifest = read_json(path)
            config = manifest["config"]
            generation_rows.append(
                common
                | {
                    "task_id": manifest["dataset_id"],
                    "official_instance_count": manifest["official_instance_count"],
                    "treatment_count": manifest["treatment_count"],
                    "input_ablation_count": manifest["input_ablation_count"],
                    "prediction_length": config.get("prediction_length"),
                    "capability_availability": manifest[
                        "available_instance_count_by_capability"
                    ],
                    "unavailable_reasons": manifest[
                        "unavailable_reason_count_by_capability"
                    ],
                }
            )

        for path in sorted(experiment_root.glob("*/04_analysis/manifest.json")):
            manifest = read_json(path)
            analysis_manifest_rows.append(
                common
                | {
                    "task_id": manifest["dataset_id"],
                    "analysis_schema": manifest["schema_version"],
                    "pipeline_schema": manifest["config"][
                        "pipeline_schema_version"
                    ],
                    "accuracy_row_count": manifest["files"]["accuracy_rows"][
                        "row_count"
                    ],
                    "effect_row_count": manifest["files"][
                        "capability_effect_rows"
                    ]["row_count"],
                    "ablation_row_count": manifest["files"][
                        "input_ablation_rows"
                    ]["row_count"],
                }
            )

        for path in sorted(experiment_root.glob("*/03_inference/manifest.json")):
            manifest = read_json(path)
            for status in manifest["model_statuses"]:
                inference_status_rows.append(
                    common
                    | {
                        "task_id": manifest["dataset_id"],
                        "model_id": status["model_id"],
                        "status": status["status"],
                        "prediction_count": status["prediction_count"],
                        "failure_count": status["failure_count"],
                    }
                )

        for path in sorted(
            experiment_root.glob("*/04_analysis/capability_effect_summary.json")
        ):
            task_id = path.parent.parent.name
            for row in read_json(path)["rows"]:
                effect_task_rows.append(common | {"task_id": task_id} | row)

        for path in sorted(
            experiment_root.glob("*/04_analysis/accuracy_summary.json")
        ):
            task_id = path.parent.parent.name
            for row in read_json(path)["rows"]:
                accuracy_task_rows.append(common | {"task_id": task_id} | row)

    return {
        "suite_rows": pd.DataFrame(suite_rows),
        "suite_pairs": pd.DataFrame(suite_pairs),
        "generation": pd.DataFrame(generation_rows),
        "effect_task": pd.DataFrame(effect_task_rows),
        "accuracy_task": pd.DataFrame(accuracy_task_rows),
        "analysis_manifests": pd.DataFrame(analysis_manifest_rows),
        "inference_status": pd.DataFrame(inference_status_rows),
    }


def experiment_inventory(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    generation = data["generation"]
    analysis = data["analysis_manifests"]
    inference = data["inference_status"]
    rows: list[dict[str, Any]] = []
    for suite, group in generation.groupby("suite", sort=False):
        a = analysis[analysis["suite"] == suite]
        inf = inference[inference["suite"] == suite]
        rows.append(
            {
                "suite": suite,
                "benchmark": group["benchmark"].iloc[0],
                "horizon_group": group["horizon_group"].iloc[0],
                "task_count": group["task_id"].nunique(),
                "horizon_min": int(group["prediction_length"].min()),
                "horizon_max": int(group["prediction_length"].max()),
                "official_instance_count": int(
                    group["official_instance_count"].sum()
                ),
                "treatment_count": int(group["treatment_count"].sum()),
                "generated_legacy_ablation_count": int(
                    group["input_ablation_count"].sum()
                ),
                "model_count": inf["model_id"].nunique(),
                "prediction_count": int(inf["prediction_count"].sum()),
                "inference_failure_count": int(inf["failure_count"].sum()),
                "all_inference_status_complete": bool(
                    (inf["status"] == "complete").all()
                ),
                "accuracy_metric_row_count": int(a["accuracy_row_count"].sum()),
                "effect_metric_row_count": int(a["effect_row_count"].sum()),
                "legacy_ablation_metric_row_count": int(
                    a["ablation_row_count"].sum()
                ),
                "pipeline_schema": ";".join(sorted(a["pipeline_schema"].unique())),
                "analysis_schema": ";".join(sorted(a["analysis_schema"].unique())),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        "suite", key=lambda x: x.map({v["suite"]: v["order"] for v in EXPERIMENTS.values()})
    )


def capability_availability(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    generation = data["generation"]
    effects = data["effect_task"]
    rows: list[dict[str, Any]] = []
    for suite, group in generation.groupby("suite", sort=False):
        canonical_model = sorted(
            effects.loc[effects["suite"] == suite, "model_id"].unique()
        )[0]
        model_effects = effects[
            (effects["suite"] == suite)
            & (effects["model_id"] == canonical_model)
        ]
        for capability in CAPABILITY_ORDER:
            available = group["capability_availability"].map(
                lambda value: int(value.get(capability, 0))
            )
            candidates = model_effects[
                model_effects["capability_id"] == capability
            ]
            candidate_count = int(candidates["effect_candidate_count"].sum())
            scored_count = int(candidates["official_instance_count"].sum())
            low_count = int(
                candidates["effect_unavailable_low_signal_count"].sum()
            )
            unobserved_count = int(
                candidates["effect_unavailable_unobserved_count"].sum()
            )
            rows.append(
                {
                    "suite": suite,
                    "capability_id": capability,
                    "suite_task_count": group["task_id"].nunique(),
                    "available_task_count": int((available > 0).sum()),
                    "task_coverage": float((available > 0).mean()),
                    "official_instance_count": int(
                        group["official_instance_count"].sum()
                    ),
                    "available_instance_count": int(available.sum()),
                    "available_instance_share_of_all_official": float(
                        available.sum() / group["official_instance_count"].sum()
                    ),
                    "candidate_treatment_count": candidate_count,
                    "scored_treatment_count": scored_count,
                    "low_signal_treatment_count": low_count,
                    "unobserved_effect_treatment_count": unobserved_count,
                    "effect_scoring_coverage": (
                        scored_count / candidate_count if candidate_count else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def official_accuracy_tables(
    data: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suite_rows = data["suite_rows"]
    accuracy = data["accuracy_task"]
    official_suite = suite_rows[suite_rows["metric"] == "official_mase"].copy()
    official_suite = official_suite.rename(
        columns={
            "task_equal_mean": "official_mase_task_equal_mean",
            "task_bootstrap_95_ci_lower": "official_mase_ci_lower",
            "task_bootstrap_95_ci_upper": "official_mase_ci_upper",
        }
    )
    official_suite["official_mase_rank"] = official_suite.groupby("suite")[
        "official_mase_task_equal_mean"
    ].rank(method="min")

    task = accuracy[accuracy["sample_kind"] == "official_baseline"].copy()
    task = task[
        [
            "suite",
            "benchmark",
            "horizon_group",
            "task_id",
            "model_id",
            "official_instance_count",
            "mase_mean",
            "mae_mean",
        ]
    ].rename(
        columns={"mase_mean": "official_mase", "mae_mean": "official_mae"}
    )

    diagnostics: list[dict[str, Any]] = []
    for (suite, model), group in task.groupby(["suite", "model_id"]):
        ordered = group.sort_values("official_mase", ascending=False)
        without_max = ordered.iloc[1:]
        diagnostics.append(
            {
                "suite": suite,
                "model_id": model,
                "median_task_mase": float(group["official_mase"].median()),
                "maximum_task_mase": float(ordered.iloc[0]["official_mase"]),
                "maximum_mase_task_id": ordered.iloc[0]["task_id"],
                "task_equal_mean_excluding_maximum_task": float(
                    without_max["official_mase"].mean()
                ),
            }
        )
    official_suite = official_suite.merge(
        pd.DataFrame(diagnostics), on=["suite", "model_id"], how="left"
    )
    columns = [
        "suite",
        "benchmark",
        "horizon_group",
        "model_id",
        "official_mase_rank",
        "official_mase_task_equal_mean",
        "official_mase_ci_lower",
        "official_mase_ci_upper",
        "task_count",
        "suite_task_count",
        "task_coverage",
        "median_task_mase",
        "maximum_task_mase",
        "maximum_mase_task_id",
        "task_equal_mean_excluding_maximum_task",
    ]
    return official_suite[columns].sort_values(
        ["suite", "official_mase_rank"]
    ), task.sort_values(["suite", "model_id", "official_mase"], ascending=[True, True, False])


def effect_level_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = data["suite_rows"]
    effect = rows[rows["metric"] == "capability_effect_nrmse"].copy()
    effect = effect.rename(
        columns={
            "task_equal_mean": "effect_nrmse_task_equal",
            "task_bootstrap_95_ci_lower": "effect_nrmse_ci_lower",
            "task_bootstrap_95_ci_upper": "effect_nrmse_ci_upper",
        }
    )
    effect["cell_rank"] = effect.groupby(
        ["suite", "capability_id", "capability_level"]
    )["effect_nrmse_task_equal"].rank(method="min")
    effect["cell_winner"] = effect["cell_rank"] == 1
    return effect[
        [
            "suite",
            "benchmark",
            "horizon_group",
            "model_id",
            "capability_id",
            "capability_level",
            "effect_nrmse_task_equal",
            "effect_nrmse_ci_lower",
            "effect_nrmse_ci_upper",
            "task_count",
            "suite_task_count",
            "task_coverage",
            "cell_rank",
            "cell_winner",
        ]
    ].sort_values(
        ["suite", "capability_id", "capability_level", "cell_rank"]
    )


def task_level_effect_values(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    effect = data["effect_task"].copy()
    effect = effect[effect["effect_nrmse_pooled"].map(finite)]
    return effect[
        [
            "suite",
            "benchmark",
            "horizon_group",
            "task_id",
            "model_id",
            "capability_id",
            "capability_level",
            "effect_nrmse_pooled",
            "effect_scoring_coverage",
            "effect_candidate_count",
            "official_instance_count",
            "effect_unavailable_low_signal_count",
            "effect_unavailable_unobserved_count",
            "effect_correlation_mean",
            "effect_amplitude_ratio_pooled",
        ]
    ]


def bootstrap_effect_summaries(
    data: dict[str, pd.DataFrame],
    level_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build level-averaged capability and descriptive macro summaries.

    The protocol does not define a single aggregate across capabilities.  The
    macro shown here gives equal weight to eight capabilities and five levels.
    Its interval is a capability-stratified paired bootstrap: tasks are
    resampled within each capability, the same draw is applied to all models,
    and the eight capability means are then averaged.
    """

    task = task_level_effect_values(data)
    task_levelavg = (
        task.groupby(
            [
                "suite",
                "benchmark",
                "horizon_group",
                "task_id",
                "model_id",
                "capability_id",
            ],
            as_index=False,
        )
        .agg(
            effect_nrmse_level_average=("effect_nrmse_pooled", "mean"),
            observed_level_count=("capability_level", "nunique"),
        )
    )
    if not (task_levelavg["observed_level_count"] == 5).all():
        raise RuntimeError("a task/model/capability group is missing a level")

    capability_rows: list[dict[str, Any]] = []
    capability_pairs: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    macro_pairs: list[dict[str, Any]] = []

    for suite_index, (suite, suite_group) in enumerate(
        task_levelavg.groupby("suite", sort=False)
    ):
        models = [m for m in MODEL_ORDER if m in set(suite_group["model_id"])]
        capabilities = [
            c for c in CAPABILITY_ORDER if c in set(suite_group["capability_id"])
        ]
        rng = np.random.default_rng(BOOTSTRAP_SEED + suite_index * 10_000)
        macro_boot = np.zeros((BOOTSTRAP_REPETITIONS, len(models)), dtype=float)
        exact_macro = np.zeros(len(models), dtype=float)

        for capability_index, capability in enumerate(capabilities):
            matrix = suite_group[
                suite_group["capability_id"] == capability
            ].pivot(
                index="task_id",
                columns="model_id",
                values="effect_nrmse_level_average",
            )
            matrix = matrix.dropna(subset=models)
            values = matrix[models].to_numpy(dtype=float)
            draw = rng.integers(
                0,
                values.shape[0],
                size=(BOOTSTRAP_REPETITIONS, values.shape[0]),
            )
            boot = values[draw].mean(axis=1)
            exact = values.mean(axis=0)
            exact_macro += exact / len(capabilities)
            macro_boot += boot / len(capabilities)
            coverage_row = level_table[
                (level_table["suite"] == suite)
                & (level_table["capability_id"] == capability)
            ].iloc[0]

            for model_index, model in enumerate(models):
                lower, upper = np.quantile(
                    boot[:, model_index], [0.025, 0.975]
                )
                capability_rows.append(
                    {
                        "suite": suite,
                        "benchmark": suite_group["benchmark"].iloc[0],
                        "horizon_group": suite_group["horizon_group"].iloc[0],
                        "model_id": model,
                        "capability_id": capability,
                        "level_averaged_effect_nrmse": exact[model_index],
                        "task_bootstrap_95_ci_lower": lower,
                        "task_bootstrap_95_ci_upper": upper,
                        "task_count": values.shape[0],
                        "suite_task_count": int(coverage_row["suite_task_count"]),
                        "task_coverage": values.shape[0]
                        / int(coverage_row["suite_task_count"]),
                        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                        "derived_display_metric": True,
                    }
                )

            for left_index, left_model in enumerate(models):
                for right_index in range(left_index + 1, len(models)):
                    right_model = models[right_index]
                    differences = boot[:, left_index] - boot[:, right_index]
                    lower, upper = np.quantile(differences, [0.025, 0.975])
                    capability_pairs.append(
                        {
                            "suite": suite,
                            "capability_id": capability,
                            "left_model_id": left_model,
                            "right_model_id": right_model,
                            "paired_task_count": values.shape[0],
                            "mean_difference_left_minus_right": exact[left_index]
                            - exact[right_index],
                            "paired_task_bootstrap_95_ci_lower": lower,
                            "paired_task_bootstrap_95_ci_upper": upper,
                            "significant_uncorrected": bool(
                                upper < 0.0 or lower > 0.0
                            ),
                            "better_model_id": (
                                left_model
                                if upper < 0.0
                                else right_model if lower > 0.0 else None
                            ),
                        }
                    )

        cell = level_table[level_table["suite"] == suite].copy()
        cell_rank = cell.groupby(
            ["capability_id", "capability_level"]
        )["effect_nrmse_task_equal"].rank(method="average")
        cell = cell.assign(_rank=cell_rank)
        protocol_pairs = data["suite_pairs"]
        protocol_pairs = protocol_pairs[
            (protocol_pairs["suite"] == suite)
            & (protocol_pairs["metric"] == "capability_effect_nrmse")
        ]

        for model_index, model in enumerate(models):
            lower, upper = np.quantile(
                macro_boot[:, model_index], [0.025, 0.975]
            )
            model_cell = cell[cell["model_id"] == model]
            outcomes = [
                oriented_pair_result(row, model)
                for _, row in protocol_pairs[
                    (protocol_pairs["left_model_id"] == model)
                    | (protocol_pairs["right_model_id"] == model)
                ].iterrows()
            ]
            macro_rows.append(
                {
                    "suite": suite,
                    "benchmark": suite_group["benchmark"].iloc[0],
                    "horizon_group": suite_group["horizon_group"].iloc[0],
                    "model_id": model,
                    "derived_macro_effect_nrmse": exact_macro[model_index],
                    "stratified_task_bootstrap_95_ci_lower": lower,
                    "stratified_task_bootstrap_95_ci_upper": upper,
                    "mean_rank_across_40_capability_level_cells": float(
                        model_cell["_rank"].mean()
                    ),
                    "cell_win_count_out_of_40": int(
                        (model_cell["_rank"] == 1).sum()
                    ),
                    "uncorrected_significant_cell_pairwise_wins": outcomes.count(
                        "win"
                    ),
                    "uncorrected_significant_cell_pairwise_losses": outcomes.count(
                        "loss"
                    ),
                    "cell_pairwise_comparison_count": len(outcomes),
                    "capability_count": len(capabilities),
                    "level_count_per_capability": 5,
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                    "derived_display_metric": True,
                    "macro_definition": (
                        "equal mean of 8 capability means; each capability is "
                        "the equal mean of 5 level-specific task-equal NRMSEs"
                    ),
                    "bootstrap_definition": (
                        "capability-stratified paired resampling of eligible "
                        "tasks, followed by equal capability mean"
                    ),
                }
            )

        for left_index, left_model in enumerate(models):
            for right_index in range(left_index + 1, len(models)):
                right_model = models[right_index]
                differences = macro_boot[:, left_index] - macro_boot[:, right_index]
                lower, upper = np.quantile(differences, [0.025, 0.975])
                macro_pairs.append(
                    {
                        "suite": suite,
                        "left_model_id": left_model,
                        "right_model_id": right_model,
                        "mean_difference_left_minus_right": exact_macro[left_index]
                        - exact_macro[right_index],
                        "paired_stratified_task_bootstrap_95_ci_lower": lower,
                        "paired_stratified_task_bootstrap_95_ci_upper": upper,
                        "significant_uncorrected": bool(
                            upper < 0.0 or lower > 0.0
                        ),
                        "better_model_id": (
                            left_model
                            if upper < 0.0
                            else right_model if lower > 0.0 else None
                        ),
                        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                        "derived_display_metric": True,
                    }
                )

    capability_frame = pd.DataFrame(capability_rows)
    capability_frame["capability_rank"] = capability_frame.groupby(
        ["suite", "capability_id"]
    )["level_averaged_effect_nrmse"].rank(method="min")
    macro_frame = pd.DataFrame(macro_rows)
    macro_frame["derived_macro_rank"] = macro_frame.groupby("suite")[
        "derived_macro_effect_nrmse"
    ].rank(method="min")
    capability_frame = capability_frame.sort_values(
        ["suite", "capability_id", "capability_rank"]
    )
    macro_frame = macro_frame.sort_values(["suite", "derived_macro_rank"])
    return (
        capability_frame,
        pd.DataFrame(capability_pairs),
        macro_frame,
        pd.DataFrame(macro_pairs),
    )


def rank_comparison(
    official: pd.DataFrame,
    macro: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = official.merge(
        macro[
            [
                "suite",
                "model_id",
                "derived_macro_effect_nrmse",
                "derived_macro_rank",
                "mean_rank_across_40_capability_level_cells",
                "cell_win_count_out_of_40",
            ]
        ],
        on=["suite", "model_id"],
        how="inner",
    )
    result["capability_rank_minus_official_rank"] = (
        result["derived_macro_rank"] - result["official_mase_rank"]
    )
    correlations: list[dict[str, Any]] = []
    for suite, group in result.groupby("suite", sort=False):
        correlation = spearmanr(
            group["official_mase_task_equal_mean"],
            group["derived_macro_effect_nrmse"],
        )
        correlations.append(
            {
                "suite": suite,
                "model_count": len(group),
                "spearman_rho_official_mase_vs_derived_macro_nrmse": float(
                    correlation.statistic
                ),
                "two_sided_p_value": float(correlation.pvalue),
            }
        )
    return result.sort_values(["suite", "official_mase_rank"]), pd.DataFrame(
        correlations
    )


def treatment_mase_comparison(
    data: dict[str, pd.DataFrame], capability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accuracy = data["accuracy_task"]
    treatment = accuracy[accuracy["sample_kind"] == "capability_treatment"]
    task_equal = (
        treatment.groupby(
            [
                "suite",
                "benchmark",
                "horizon_group",
                "model_id",
                "capability_id",
                "capability_level",
            ],
            as_index=False,
        )
        .agg(
            treatment_mase_task_equal=("mase_mean", "mean"),
            task_count=("task_id", "nunique"),
        )
    )
    levelavg = (
        task_equal.groupby(
            [
                "suite",
                "benchmark",
                "horizon_group",
                "model_id",
                "capability_id",
            ],
            as_index=False,
        )
        .agg(
            level_averaged_treatment_mase=(
                "treatment_mase_task_equal",
                "mean",
            ),
            task_count=("task_count", "min"),
        )
    )
    merged = levelavg.merge(
        capability[
            [
                "suite",
                "model_id",
                "capability_id",
                "level_averaged_effect_nrmse",
                "capability_rank",
            ]
        ],
        on=["suite", "model_id", "capability_id"],
    )
    merged["treatment_mase_rank"] = merged.groupby(
        ["suite", "capability_id"]
    )["level_averaged_treatment_mase"].rank(method="min")
    merged["same_winner"] = (
        (merged["treatment_mase_rank"] == 1)
        & (merged["capability_rank"] == 1)
    )
    concordance: list[dict[str, Any]] = []
    for suite, group in merged.groupby("suite", sort=False):
        mase_winners = (
            group.loc[group["treatment_mase_rank"] == 1]
            .set_index("capability_id")["model_id"]
            .to_dict()
        )
        effect_winners = (
            group.loc[group["capability_rank"] == 1]
            .set_index("capability_id")["model_id"]
            .to_dict()
        )
        matches = [
            capability_id
            for capability_id in CAPABILITY_ORDER
            if mase_winners.get(capability_id)
            == effect_winners.get(capability_id)
        ]
        mismatches = [
            {
                "capability_id": capability_id,
                "treatment_mase_winner": mase_winners.get(capability_id),
                "effect_nrmse_winner": effect_winners.get(capability_id),
            }
            for capability_id in CAPABILITY_ORDER
            if mase_winners.get(capability_id)
            != effect_winners.get(capability_id)
        ]
        concordance.append(
            {
                "suite": suite,
                "matching_winner_count_out_of_8": len(matches),
                "matching_capabilities": ";".join(matches),
                "mismatches_json": json.dumps(mismatches, ensure_ascii=False),
            }
        )
    return merged.sort_values(
        ["suite", "capability_id", "capability_rank"]
    ), pd.DataFrame(concordance)


def protocol_pairwise_summary(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = data["suite_pairs"].copy()
    pairs["significant_uncorrected"] = (
        (pairs["paired_task_bootstrap_95_ci_upper"] < 0.0)
        | (pairs["paired_task_bootstrap_95_ci_lower"] > 0.0)
    )
    pairs["better_model_id"] = np.where(
        pairs["paired_task_bootstrap_95_ci_upper"] < 0.0,
        pairs["left_model_id"],
        np.where(
            pairs["paired_task_bootstrap_95_ci_lower"] > 0.0,
            pairs["right_model_id"],
            None,
        ),
    )
    summaries: list[dict[str, Any]] = []
    for (suite, metric), group in pairs.groupby(["suite", "metric"]):
        models = sorted(set(group["left_model_id"]) | set(group["right_model_id"]))
        for model in models:
            involved = group[
                (group["left_model_id"] == model)
                | (group["right_model_id"] == model)
            ]
            outcomes = [
                oriented_pair_result(row, model) for _, row in involved.iterrows()
            ]
            summaries.append(
                {
                    "suite": suite,
                    "metric": metric,
                    "model_id": model,
                    "pairwise_comparison_count": len(involved),
                    "uncorrected_significant_wins": outcomes.count("win"),
                    "uncorrected_significant_losses": outcomes.count("loss"),
                    "not_significant": outcomes.count("not_significant"),
                    "warning": (
                        "95% paired task-bootstrap intervals; no multiple-"
                        "comparison correction"
                    ),
                }
            )
    return pairs, pd.DataFrame(summaries)


def effect_diagnostics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    task = task_level_effect_values(data)
    return (
        task.groupby(
            [
                "suite",
                "benchmark",
                "horizon_group",
                "model_id",
                "capability_id",
                "capability_level",
            ],
            as_index=False,
        )
        .agg(
            task_equal_effect_nrmse=("effect_nrmse_pooled", "mean"),
            task_equal_effect_correlation=("effect_correlation_mean", "mean"),
            task_equal_effect_amplitude_ratio=(
                "effect_amplitude_ratio_pooled",
                "mean",
            ),
            task_equal_scoring_coverage=("effect_scoring_coverage", "mean"),
            task_count=("task_id", "nunique"),
        )
        .sort_values(
            ["suite", "capability_id", "capability_level", "model_id"]
        )
    )


def build_summary_json(
    inventory: pd.DataFrame,
    availability: pd.DataFrame,
    official: pd.DataFrame,
    capability: pd.DataFrame,
    macro: pd.DataFrame,
    rank: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    concordance: pd.DataFrame,
) -> dict[str, Any]:
    suites: dict[str, Any] = {}
    for suite in inventory["suite"]:
        suite_official = official[official["suite"] == suite]
        suite_macro = macro[macro["suite"] == suite]
        suite_capability = capability[capability["suite"] == suite]
        winners = (
            suite_capability[suite_capability["capability_rank"] == 1]
            .set_index("capability_id")[
                ["model_id", "level_averaged_effect_nrmse"]
            ]
            .to_dict(orient="index")
        )
        suites[suite] = {
            "inventory": inventory[inventory["suite"] == suite].iloc[0].to_dict(),
            "official_mase_winner": suite_official.sort_values(
                "official_mase_rank"
            ).iloc[0][
                [
                    "model_id",
                    "official_mase_task_equal_mean",
                    "official_mase_ci_lower",
                    "official_mase_ci_upper",
                ]
            ].to_dict(),
            "derived_macro_nrmse_winner": suite_macro.sort_values(
                "derived_macro_rank"
            ).iloc[0][
                [
                    "model_id",
                    "derived_macro_effect_nrmse",
                    "stratified_task_bootstrap_95_ci_lower",
                    "stratified_task_bootstrap_95_ci_upper",
                    "cell_win_count_out_of_40",
                    "uncorrected_significant_cell_pairwise_wins",
                    "uncorrected_significant_cell_pairwise_losses",
                ]
            ].to_dict(),
            "level_averaged_capability_winners": winners,
            "minimum_capability_task_coverage": float(
                availability[availability["suite"] == suite]["task_coverage"].min()
            ),
            "official_vs_macro_rank_correlation": rank_correlations[
                rank_correlations["suite"] == suite
            ].iloc[0].to_dict(),
            "treatment_mase_effect_winner_concordance": concordance[
                concordance["suite"] == suite
            ].iloc[0].to_dict(),
            "rank_shifts": rank[rank["suite"] == suite][
                [
                    "model_id",
                    "official_mase_rank",
                    "derived_macro_rank",
                    "capability_rank_minus_official_rank",
                ]
            ].to_dict(orient="records"),
        }
    return {
        "schema_version": "cafe.paper_main_experiment_summary.v1",
        "source_policy": "read-only frozen JSON summaries copied from timecho92",
        "protocol_primary_metric": (
            "task-equal MASE for official accuracy and task-equal pooled effect "
            "NRMSE separately for each capability and level"
        ),
        "derived_macro_warning": (
            "The macro NRMSE is a descriptive display metric, not a protocol-"
            "defined single leaderboard score. It equally averages eight "
            "capabilities and five levels despite unequal capability coverage."
        ),
        "bootstrap_repetitions_for_derived_summaries": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed_for_derived_summaries": BOOTSTRAP_SEED,
        "total_prediction_count": int(inventory["prediction_count"].sum()),
        "total_inference_failure_count": int(
            inventory["inference_failure_count"].sum()
        ),
        "suites": suites,
    }


def validate_derived_results(
    data: dict[str, pd.DataFrame],
    inventory: pd.DataFrame,
    availability: pd.DataFrame,
    effect_level: pd.DataFrame,
    capability: pd.DataFrame,
    macro: pd.DataFrame,
) -> dict[str, Any]:
    generation = data["generation"]
    for _, row in generation.iterrows():
        expected = 5 * sum(int(value) for value in row["capability_availability"].values())
        if expected != int(row["treatment_count"]):
            raise AssertionError(
                f"generation treatment count mismatch for {row['task_id']}"
            )
    if int(inventory["inference_failure_count"].sum()) != 0:
        raise AssertionError("main experiments contain inference failures")
    if not inventory["all_inference_status_complete"].all():
        raise AssertionError("main experiments contain incomplete inference status")

    for suite, group in effect_level.groupby("suite"):
        per_model = group.groupby("model_id").size()
        if not (per_model == 40).all():
            raise AssertionError(f"{suite} does not have 40 cells per model")
        direct_capability = group.groupby(
            ["model_id", "capability_id"]
        )["effect_nrmse_task_equal"].mean()
        derived_capability = capability[capability["suite"] == suite].set_index(
            ["model_id", "capability_id"]
        )["level_averaged_effect_nrmse"]
        if not np.allclose(
            direct_capability.sort_index(),
            derived_capability.sort_index(),
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"{suite} capability aggregation mismatch")
        direct_macro = direct_capability.groupby("model_id").mean()
        derived_macro = macro[macro["suite"] == suite].set_index("model_id")[
            "derived_macro_effect_nrmse"
        ]
        if not np.allclose(
            direct_macro.sort_index(),
            derived_macro.sort_index(),
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"{suite} macro aggregation mismatch")

    return {
        "status": "passed",
        "generation_task_count_checked": len(generation),
        "suite_count_checked": inventory["suite"].nunique(),
        "model_capability_level_cells_checked": len(effect_level),
        "generation_treatment_identity": (
            "treatment_count == 5 * sum(available instances over capabilities)"
        ),
        "inference_failure_count": int(inventory["inference_failure_count"].sum()),
        "inference_status_complete": bool(
            inventory["all_inference_status_complete"].all()
        ),
        "aggregation_checks": [
            "level-averaged capability equals arithmetic mean of 5 frozen suite levels",
            "derived macro equals arithmetic mean of 8 level-averaged capabilities",
        ],
        "effect_scoring_coverage_minimum": float(
            availability["effect_scoring_coverage"].min()
        ),
    }


def make_figures(
    official: pd.DataFrame,
    level: pd.DataFrame,
    capability: pd.DataFrame,
    macro: pd.DataFrame,
    rank: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    suite_order = [value["suite"] for value in EXPERIMENTS.values()]
    colors = dict(zip(MODEL_ORDER, plt.get_cmap("tab10").colors))

    # Official accuracy with protocol task-bootstrap intervals.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for axis, suite in zip(axes.flat, suite_order):
        group = official[official["suite"] == suite].sort_values(
            "official_mase_task_equal_mean", ascending=False
        )
        y = np.arange(len(group))
        means = group["official_mase_task_equal_mean"].to_numpy()
        lower = group["official_mase_ci_lower"].to_numpy()
        upper = group["official_mase_ci_upper"].to_numpy()
        axis.errorbar(
            means,
            y,
            xerr=np.vstack([means - lower, upper - means]),
            fmt="none",
            ecolor="#777777",
            elinewidth=1.2,
            capsize=2.5,
            zorder=1,
        )
        axis.scatter(
            means,
            y,
            c=[colors[m] for m in group["model_id"]],
            s=32,
            zorder=2,
        )
        axis.set_yticks(y, group["model_id"])
        axis.set_xscale("log")
        axis.axvline(1.0, color="#999999", linestyle="--", linewidth=0.8)
        axis.set_title(suite)
        axis.set_xlabel("Official MASE (task-equal mean, log scale; lower is better)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Official benchmark accuracy with 95% task-bootstrap intervals")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig_official_mase.{suffix}", bbox_inches="tight")
    plt.close(fig)

    # Level-averaged capability heatmaps. This is a derived display metric.
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), constrained_layout=True)
    image = None
    for axis, suite in zip(axes.flat, suite_order):
        suite_macro = macro[macro["suite"] == suite].sort_values(
            "derived_macro_rank"
        )
        models = suite_macro["model_id"].tolist()
        matrix = (
            capability[capability["suite"] == suite]
            .pivot(
                index="model_id",
                columns="capability_id",
                values="level_averaged_effect_nrmse",
            )
            .reindex(index=models, columns=CAPABILITY_ORDER)
        )
        image = axis.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="cividis",
            vmin=0.0,
            vmax=1.75,
        )
        axis.set_xticks(
            np.arange(len(CAPABILITY_ORDER)),
            [CAPABILITY_LABELS[c] for c in CAPABILITY_ORDER],
            rotation=42,
            ha="right",
        )
        axis.set_yticks(np.arange(len(models)), models)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value > 0.72 else "black",
                )
        axis.set_title(f"{suite} (models ordered by derived macro)")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    colorbar.set_label("Level-averaged effect NRMSE (lower is better)")
    fig.suptitle(
        "Capability response heatmaps (descriptive 5-level mean; not a single protocol score)"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"fig_capability_heatmaps.{suffix}", bbox_inches="tight"
        )
    plt.close(fig)

    # Per-suite level curves, one panel per capability and one line per model.
    for suite in suite_order:
        suite_level = level[level["suite"] == suite]
        fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.2), constrained_layout=True)
        for axis, capability_id in zip(axes.flat, CAPABILITY_ORDER):
            group = suite_level[suite_level["capability_id"] == capability_id]
            for model in [m for m in MODEL_ORDER if m in set(group["model_id"])]:
                values = group[group["model_id"] == model].sort_values(
                    "capability_level"
                )
                axis.plot(
                    values["capability_level"],
                    values["effect_nrmse_task_equal"],
                    marker="o",
                    markersize=3,
                    linewidth=1.25,
                    label=model,
                    color=colors[model],
                )
            axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
            axis.set_title(CAPABILITY_LABELS[capability_id])
            axis.set_xticks([1, 2, 3, 4, 5])
            axis.set_xlabel("Capability level")
            axis.set_ylabel("Task-equal effect NRMSE")
            axis.grid(alpha=0.18)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
        fig.suptitle(
            f"{suite}: capability response across levels (NRMSE=1 is zero-response reference)"
        )
        slug = suite.lower().replace("-", "_")
        for suffix in ("png", "pdf"):
            fig.savefig(
                FIGURES / f"fig_level_curves_{slug}.{suffix}",
                bbox_inches="tight",
            )
        plt.close(fig)

    # Rank slopes make short/FEV disagreement immediately visible.
    selected = ["GIFT-Short", "FEV-Mini20"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.8), constrained_layout=True)
    for axis, suite in zip(axes, selected):
        group = rank[rank["suite"] == suite]
        model_count = len(group)
        for _, row in group.iterrows():
            model = row["model_id"]
            axis.plot(
                [0, 1],
                [row["official_mase_rank"], row["derived_macro_rank"]],
                marker="o",
                color=colors[model],
                linewidth=1.8,
            )
            axis.text(-0.04, row["official_mase_rank"], model, ha="right", va="center")
            axis.text(1.04, row["derived_macro_rank"], model, ha="left", va="center")
        axis.set_xlim(-0.5, 1.5)
        axis.set_ylim(model_count + 0.6, 0.4)
        axis.set_xticks([0, 1], ["Official MASE rank", "Derived macro NRMSE rank"])
        axis.set_yticks(range(1, model_count + 1))
        axis.set_title(suite)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Accuracy rank and capability-response rank are not interchangeable")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig_rank_divergence.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_all()

    inventory = experiment_inventory(data)
    availability = capability_availability(data)
    official, official_task = official_accuracy_tables(data)
    effect_level = effect_level_table(data)
    (
        capability,
        capability_pairs,
        macro,
        macro_pairs,
    ) = bootstrap_effect_summaries(data, effect_level)
    rank, rank_correlations = rank_comparison(official, macro)
    treatment_vs_effect, concordance = treatment_mase_comparison(data, capability)
    protocol_pairs, protocol_pair_summary = protocol_pairwise_summary(data)
    diagnostics = effect_diagnostics(data)
    task_effect = task_level_effect_values(data)
    treatment_mase_task = data["accuracy_task"].loc[
        data["accuracy_task"]["sample_kind"] == "capability_treatment",
        [
            "suite",
            "benchmark",
            "horizon_group",
            "task_id",
            "model_id",
            "capability_id",
            "capability_level",
            "official_instance_count",
            "mase_mean",
            "mae_mean",
        ],
    ].rename(
        columns={
            "mase_mean": "treatment_mase",
            "mae_mean": "treatment_mae",
        }
    )
    treatment_mase_suite_level = (
        treatment_mase_task.groupby(
            [
                "suite",
                "benchmark",
                "horizon_group",
                "model_id",
                "capability_id",
                "capability_level",
            ],
            as_index=False,
        )
        .agg(
            treatment_mase_task_equal=("treatment_mase", "mean"),
            treatment_mae_task_equal=("treatment_mae", "mean"),
            task_count=("task_id", "nunique"),
        )
        .sort_values(
            ["suite", "capability_id", "capability_level", "model_id"]
        )
    )
    level_profile = (
        effect_level.groupby(
            ["suite", "benchmark", "horizon_group", "capability_id", "capability_level"],
            as_index=False,
        )
        .agg(
            model_mean_effect_nrmse=("effect_nrmse_task_equal", "mean"),
            model_median_effect_nrmse=("effect_nrmse_task_equal", "median"),
            model_count=("model_id", "nunique"),
            task_count=("task_count", "min"),
            task_coverage=("task_coverage", "min"),
        )
        .sort_values(["suite", "capability_id", "capability_level"])
    )
    high_coverage_capabilities = availability.loc[
        availability["task_coverage"] >= 0.8, ["suite", "capability_id"]
    ]
    high_coverage_sensitivity = (
        capability.merge(
            high_coverage_capabilities,
            on=["suite", "capability_id"],
            how="inner",
        )
        .groupby(["suite", "benchmark", "horizon_group", "model_id"], as_index=False)
        .agg(
            high_coverage_macro_effect_nrmse=(
                "level_averaged_effect_nrmse",
                "mean",
            ),
            included_capability_count=("capability_id", "nunique"),
        )
    )
    high_coverage_sensitivity["high_coverage_macro_rank"] = (
        high_coverage_sensitivity.groupby("suite")[
            "high_coverage_macro_effect_nrmse"
        ].rank(method="min")
    )
    high_coverage_sensitivity["selection_rule"] = (
        "capability task coverage >= 0.80; descriptive sensitivity only"
    )
    high_coverage_sensitivity = high_coverage_sensitivity.sort_values(
        ["suite", "high_coverage_macro_rank"]
    )

    write_csv(inventory, "experiment_inventory.csv")
    write_csv(availability, "capability_availability.csv")
    write_csv(official, "official_mase_suite.csv")
    write_csv(official_task, "official_mase_by_task.csv")
    write_csv(effect_level, "effect_nrmse_by_suite_model_capability_level.csv")
    write_csv(task_effect, "effect_nrmse_by_task_model_capability_level.csv")
    write_csv(
        treatment_mase_task,
        "treatment_mase_by_task_model_capability_level.csv",
    )
    write_csv(
        treatment_mase_suite_level,
        "treatment_mase_by_suite_model_capability_level.csv",
    )
    write_csv(capability, "effect_nrmse_level_averaged_capability.csv")
    write_csv(
        capability_pairs,
        "effect_nrmse_level_averaged_capability_pairwise_bootstrap.csv",
    )
    write_csv(macro, "effect_nrmse_derived_macro.csv")
    write_csv(macro_pairs, "effect_nrmse_derived_macro_pairwise_bootstrap.csv")
    write_csv(rank, "official_mase_vs_effect_nrmse_rank.csv")
    write_csv(rank_correlations, "official_mase_vs_effect_rank_correlation.csv")
    write_csv(treatment_vs_effect, "treatment_mase_vs_effect_nrmse.csv")
    write_csv(concordance, "treatment_mase_effect_winner_concordance.csv")
    write_csv(protocol_pairs, "protocol_paired_task_bootstrap_all.csv")
    write_csv(protocol_pair_summary, "protocol_paired_task_bootstrap_counts.csv")
    write_csv(diagnostics, "effect_diagnostics_task_equal.csv")
    write_csv(level_profile, "capability_level_profile_model_mean.csv")
    write_csv(
        high_coverage_sensitivity,
        "effect_nrmse_high_coverage_macro_sensitivity.csv",
    )

    summary = build_summary_json(
        inventory,
        availability,
        official,
        capability,
        macro,
        rank,
        rank_correlations,
        concordance,
    )
    with (HERE / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    validation = validate_derived_results(
        data, inventory, availability, effect_level, capability, macro
    )
    with (HERE / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    make_figures(official, effect_level, capability, macro, rank)


if __name__ == "__main__":
    main()
