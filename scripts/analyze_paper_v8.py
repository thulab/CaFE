#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import paper_v8_pipeline_common as v8
import run_paper_e2_dynamic_stability as engine
import run_paper_v8_model_response as response


DEFAULT_OUTPUT_ROOT = (
    v8.REPO_ROOT / "runtime" / "paper_exp" / "v8_test" / "full_pipeline"
)
PRIMARY_MECHANISM_METRIC = {
    "trend": "trend_slope_relative_abs_error",
    "multi_seasonal": "seasonal_spectral_amplitude_relative_error",
    "time_varying_seasonality": "instantaneous_frequency_nmae",
    "regime_switching": "regime_jump_nmae",
    "nonlinear_persistence": "nonlinear_recurrence_residual_nrmse",
    "predictable_intermittency": "event_window_nmae",
    "common_factor": "common_component_nmae",
    "hierarchical_coherence": "child_contrast_nmae",
    "cross_series_dependence": "responder_normalized_mae",
    "covariate_response": "counterfactual_effect_nrmse",
}
BASELINES = ("last_value", "seasonal_naive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze formal Paper v8 fixed-L504/oracle-context results."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Chronos-2", "toto2.0", "tirex2", "timesfm2.5"],
    )
    return parser.parse_args()


def baseline_forecast(sample: dict[str, Any], model_id: str) -> np.ndarray:
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    history = target[:context]
    horizon = int(sample["horizon"])
    if model_id == "last_value":
        return np.repeat(history[-1:], horizon, axis=0)
    period = min(
        int(sample.get("mase_period", sample.get("season_length", 1))),
        context,
    )
    pattern = history[-period:]
    return np.vstack(
        [pattern[index % period] for index in range(horizon)]
    )


def metric_row(
    sample: dict[str, Any],
    *,
    model_id: str,
    forecast: np.ndarray,
    input_adaptation: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = response.prediction_metrics(sample, forecast)
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    mae = float(np.mean(np.abs(target[context:] - forecast)))
    metrics["mae"] = mae
    metrics["mase"] = mae / float(sample["mase_scale"])
    return {
        "schema_version": "paper_v8_prediction_metrics.v1",
        "model_id": model_id,
        "sample_id": sample["sample_id"],
        "master_sample_id": sample["master_sample_id"],
        "dataset_id": sample["dataset_id"],
        "config_id": sample["config_id"],
        "capability_id": sample["capability_id"],
        "generator_family_role": sample["generator_family_role"],
        "evaluation_table": sample.get("evaluation_table", "main"),
        "intensity": int(sample["intensity"]),
        "seed_index": int(sample["seed_index"]),
        "context_length": context,
        "counterfactual_pair_id": sample.get("counterfactual_pair_id"),
        "master_counterfactual_pair_id": sample.get(
            "master_counterfactual_pair_id"
        ),
        "counterfactual_member": sample.get("counterfactual_member"),
        "clean_master_sample_id": sample.get("clean_master_sample_id"),
        "input_ablation_group_id": sample.get(
            "input_ablation_group_id"
        ),
        "metrics": {
            str(name): float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and math.isfinite(float(value))
        },
        "input_adaptation": input_adaptation,
    }


def effect_channels(sample: dict[str, Any]) -> list[int]:
    capability = str(sample["capability_id"])
    metadata = sample["generation_metadata"]
    if capability == "common_factor":
        return [int(metadata["protected_target_index"])]
    if capability == "cross_series_dependence":
        return [int(value) for value in metadata["responder_indices"]]
    return list(range(int(sample["target_dim"])))


def effect_row(
    first_sample: dict[str, Any],
    first_forecast: np.ndarray,
    second_sample: dict[str, Any],
    second_forecast: np.ndarray,
    *,
    model_id: str,
) -> dict[str, Any]:
    context = int(first_sample["context_length"])
    channels = effect_channels(first_sample)
    first_target = np.asarray(first_sample["target"], dtype=float)
    second_target = np.asarray(second_sample["target"], dtype=float)
    truth_effect = (
        second_target[context:, channels]
        - first_target[context:, channels]
    )
    forecast_effect = (
        second_forecast[:, channels] - first_forecast[:, channels]
    )
    truth_rms = float(np.sqrt(np.mean(truth_effect**2)))
    forecast_rms = float(np.sqrt(np.mean(forecast_effect**2)))
    nrmse = float(
        np.sqrt(np.mean((forecast_effect - truth_effect) ** 2))
        / max(truth_rms, 1e-12)
    )
    return {
        "schema_version": "paper_v8_counterfactual_effect.v1",
        "model_id": model_id,
        "dataset_id": first_sample["dataset_id"],
        "capability_id": first_sample["capability_id"],
        "generator_family_role": first_sample["generator_family_role"],
        "evaluation_table": first_sample.get("evaluation_table", "main"),
        "intensity": int(first_sample["intensity"]),
        "seed_index": int(first_sample["seed_index"]),
        "context_length": context,
        "master_counterfactual_pair_id": first_sample[
            "master_counterfactual_pair_id"
        ],
        "counterfactual_effect_nrmse": nrmse,
        "effect_correlation": response.safe_corr(
            truth_effect,
            forecast_effect,
        ),
        "effect_amplitude_ratio": forecast_rms / max(truth_rms, 1e-12),
        "truth_effect_rms": truth_rms,
        "forecast_effect_rms": forecast_rms,
    }


def analyze_one_model(
    task_path: Path,
    *,
    model_id: str,
    prediction_path: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    predictions = (
        {
            str(row["sample_id"]): row
            for row in v8.iter_jsonl(prediction_path)
        }
        if prediction_path is not None
        else {}
    )
    metrics: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    pending_pairs: dict[
        str, tuple[dict[str, Any], np.ndarray]
    ] = {}
    missing = 0
    for sample in v8.iter_jsonl(task_path):
        if prediction_path is None:
            forecast = baseline_forecast(sample, model_id)
            adaptation = {"target_mode": "local_baseline"}
        else:
            prediction = predictions.get(str(sample["sample_id"]))
            if prediction is None:
                missing += 1
                continue
            forecast = np.asarray(prediction["forecast"], dtype=float)
            adaptation = prediction.get("input_adaptation")
        metrics.append(
            metric_row(
                sample,
                model_id=model_id,
                forecast=forecast,
                input_adaptation=adaptation,
            )
        )
        pair_id = sample.get("counterfactual_pair_id")
        member = sample.get("counterfactual_member")
        if pair_id is None or member is None:
            continue
        key = str(pair_id)
        if int(member) == 0:
            pending_pairs[key] = (sample, forecast)
        else:
            first = pending_pairs.pop(key, None)
            if first is not None:
                effects.append(
                    effect_row(
                        first[0],
                        first[1],
                        sample,
                        forecast,
                        model_id=model_id,
                    )
                )
    return metrics, effects, missing


def selected_context_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    fixed = [
        {**row, "context_policy": "fixed_l504"}
        for row in rows
        if int(row["context_length"]) == v8.CONTEXT_LENGTH
    ]
    unpaired: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    parent_matched: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    paired: dict[
        tuple[str, str], dict[int, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model_id = str(row["model_id"])
        clean_parent = row.get("clean_master_sample_id")
        if clean_parent is not None:
            parent_matched[(model_id, str(clean_parent))].append(row)
            continue
        master_pair = row.get("master_counterfactual_pair_id")
        if master_pair is None:
            unpaired[(model_id, str(row["master_sample_id"]))].append(row)
        else:
            paired[(model_id, str(master_pair))][
                int(row["context_length"])
            ].append(row)
    oracle: list[dict[str, Any]] = []
    pair_context: dict[tuple[str, str], int] = {}
    for candidates in unpaired.values():
        best = min(
            candidates,
            key=lambda row: (
                float(row["metrics"]["mase"]),
                -int(row["context_length"]),
            ),
        )
        oracle.append({**best, "context_policy": "oracle_context"})
    for key, by_context in paired.items():
        complete = {
            context: members
            for context, members in by_context.items()
            if len(members) == 2
        }
        if not complete:
            continue
        selected = min(
            complete,
            key=lambda context: (
                float(
                    np.mean(
                        [
                            member["metrics"]["mase"]
                            for member in complete[context]
                        ]
                    )
                ),
                -context,
            ),
        )
        pair_context[key] = selected
        oracle.extend(
            {
                **member,
                "context_policy": "oracle_context",
            }
            for member in complete[selected]
        )
    selected_parent_context = {
        (str(row["model_id"]), str(row["master_sample_id"])): int(
            row["context_length"]
        )
        for row in oracle
    }
    for key, candidates in parent_matched.items():
        selected = selected_parent_context.get(key)
        if selected is None:
            continue
        matching = [
            row
            for row in candidates
            if int(row["context_length"]) == selected
        ]
        oracle.extend(
            {**row, "context_policy": "oracle_context"}
            for row in matching
        )
    return fixed + oracle, pair_context


def selected_effect_rows(
    effects: list[dict[str, Any]],
    pair_context: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    output = [
        {**row, "context_policy": "fixed_l504"}
        for row in effects
        if int(row["context_length"]) == v8.CONTEXT_LENGTH
    ]
    for row in effects:
        key = (
            str(row["model_id"]),
            str(row["master_counterfactual_pair_id"]),
        )
        if pair_context.get(key) == int(row["context_length"]):
            output.append({**row, "context_policy": "oracle_context"})
    return output


def mean_seed_group_metric(
    rows: Iterable[dict[str, Any]],
    metric_name: str,
) -> float | None:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if metric_name not in row["metrics"]:
            continue
        grouped[(int(row["seed_index"]), int(row["intensity"]))].append(
            float(row["metrics"][metric_name])
        )
    if not grouped:
        return None
    intensity_by_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _intensity), values in grouped.items():
        intensity_by_seed[seed].append(float(np.mean(values)))
    return float(
        np.mean(
            [
                np.mean(values)
                for values in intensity_by_seed.values()
                if values
            ]
        )
    )


def mean_seed_group_mase(
    rows: Iterable[dict[str, Any]],
) -> float | None:
    return mean_seed_group_metric(rows, "mase")


def score_table(
    rows: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    *,
    seed_filter: set[int] | None = None,
    accuracy_metric_by_capability: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if seed_filter is None or int(row["seed_index"]) in seed_filter
    ]
    filtered_effects = [
        row
        for row in effects
        if seed_filter is None or int(row["seed_index"]) in seed_filter
    ]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        key = (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["evaluation_table"]),
            str(row["generator_family_role"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )
        groups[key].append(row)
    effect_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered_effects:
        key = (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["evaluation_table"]),
            str(row["generator_family_role"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )
        effect_groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        capability = key[4]
        accuracy_metric = (
            (accuracy_metric_by_capability or {}).get(
                capability,
                "mase",
            )
        )
        accuracy = mean_seed_group_metric(group, accuracy_metric)
        history_std_normalized_mae = mean_seed_group_metric(
            group,
            "normalized_mae_history_std",
        )
        metric_name = (
            "counterfactual_effect_nrmse"
            if (
                key[2] == "strict_counterfactual_audit"
                and capability
                in {"common_factor", "cross_series_dependence"}
            )
            else PRIMARY_MECHANISM_METRIC[capability]
        )
        if metric_name == "counterfactual_effect_nrmse":
            mechanism_values = [
                float(row[metric_name])
                for row in effect_groups.get(key, [])
                if int(row["intensity"]) == 5
            ]
        else:
            mechanism_values = [
                float(row["metrics"][metric_name])
                for row in group
                if int(row["intensity"]) == 5
                and metric_name in row["metrics"]
            ]
        output.append(
            {
                "dataset_id": key[0],
                "context_policy": key[1],
                "evaluation_table": key[2],
                "generator_family_role": key[3],
                "capability_id": capability,
                "model_id": key[5],
                "accuracy_score": accuracy,
                "accuracy_metric": accuracy_metric,
                "history_std_normalized_mae": (
                    history_std_normalized_mae
                ),
                "mechanism_metric": metric_name,
                "mechanism_score": (
                    float(np.mean(mechanism_values))
                    if mechanism_values
                    else None
                ),
                "seed_count": len(
                    {int(row["seed_index"]) for row in group}
                ),
                "intensities": sorted(
                    {int(row["intensity"]) for row in group}
                ),
                "is_reference_baseline": key[5] in BASELINES,
            }
        )
    add_ranks(output)
    return output


def add_ranks(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset_id"],
                row["context_policy"],
                row["evaluation_table"],
                row["generator_family_role"],
                row["capability_id"],
            )
        ].append(row)
    for group in groups.values():
        foundations = [
            row for row in group if not row["is_reference_baseline"]
        ]
        for score_name, rank_name in (
            ("accuracy_score", "accuracy_rank"),
            ("mechanism_score", "mechanism_rank"),
        ):
            eligible = [
                row for row in foundations if row[score_name] is not None
            ]
            ordered = sorted(
                eligible,
                key=lambda row: (float(row[score_name]), row["model_id"]),
            )
            for index, row in enumerate(ordered, start=1):
                row[rank_name] = index
            for row in group:
                row.setdefault(rank_name, None)


def matched_comparison_rows(
    selected_rows: list[dict[str, Any]],
    selected_effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = (
        (
            "secondary_family",
            lambda row: (
                row["evaluation_table"] == "main"
                and row["generator_family_role"] == "secondary"
            ),
        ),
        (
            "observation_noise_robustness",
            lambda row: (
                row["evaluation_table"] == "observation_noise_robustness"
                and row["generator_family_role"] == "primary"
            ),
        ),
        (
            "multivariate_input_ablation",
            lambda row: (
                row["evaluation_table"] == "multivariate_input_ablation"
                and row["generator_family_role"] == "primary"
            ),
        ),
    )

    def match_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["dataset_id"],
            row["context_policy"],
            row["capability_id"],
            row["model_id"],
            int(row["seed_index"]),
            int(row["intensity"]),
        )

    def score_key(row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )

    output: list[dict[str, Any]] = []
    for comparison_id, is_treatment in comparisons:
        treatment_rows = [
            row for row in selected_rows if is_treatment(row)
        ]
        treatment_keys = {match_key(row) for row in treatment_rows}
        control_rows = [
            row
            for row in selected_rows
            if row["evaluation_table"] == "main"
            and row["generator_family_role"] == "primary"
            and match_key(row) in treatment_keys
        ]
        treatment_effects = [
            row for row in selected_effects if is_treatment(row)
        ]
        control_effects = [
            row
            for row in selected_effects
            if row["evaluation_table"] == "main"
            and row["generator_family_role"] == "primary"
            and match_key(row) in treatment_keys
        ]
        accuracy_override = (
            {
                "common_factor": "protected_target_nmae",
                "cross_series_dependence": "responder_normalized_mae",
            }
            if comparison_id == "multivariate_input_ablation"
            else None
        )
        control_scores = {
            score_key(row): row
            for row in score_table(
                control_rows,
                control_effects,
                accuracy_metric_by_capability=accuracy_override,
            )
        }
        treatment_scores = {
            score_key(row): row
            for row in score_table(
                treatment_rows,
                treatment_effects,
                accuracy_metric_by_capability=accuracy_override,
            )
        }
        for key in sorted(set(control_scores) & set(treatment_scores)):
            control = control_scores[key]
            treatment = treatment_scores[key]
            control_accuracy = float(control["accuracy_score"])
            treatment_accuracy = float(treatment["accuracy_score"])
            control_mechanism = control["mechanism_score"]
            treatment_mechanism = treatment["mechanism_score"]
            output.append(
                {
                    "comparison_id": comparison_id,
                    "dataset_id": key[0],
                    "context_policy": key[1],
                    "capability_id": key[2],
                    "model_id": key[3],
                    "is_reference_baseline": control[
                        "is_reference_baseline"
                    ],
                    "matched_seed_count": treatment["seed_count"],
                    "matched_intensities": treatment["intensities"],
                    "control_accuracy_score": control_accuracy,
                    "treatment_accuracy_score": treatment_accuracy,
                    "accuracy_metric": treatment["accuracy_metric"],
                    "accuracy_delta": (
                        treatment_accuracy - control_accuracy
                    ),
                    "accuracy_relative_delta": (
                        (treatment_accuracy - control_accuracy)
                        / max(abs(control_accuracy), 1e-12)
                    ),
                    "control_accuracy_rank": control["accuracy_rank"],
                    "treatment_accuracy_rank": treatment["accuracy_rank"],
                    "mechanism_metric": treatment["mechanism_metric"],
                    "control_mechanism_score": control_mechanism,
                    "treatment_mechanism_score": treatment_mechanism,
                    "mechanism_delta": (
                        None
                        if control_mechanism is None
                        or treatment_mechanism is None
                        else float(treatment_mechanism)
                        - float(control_mechanism)
                    ),
                    "mechanism_relative_delta": (
                        None
                        if control_mechanism is None
                        or treatment_mechanism is None
                        else (
                            float(treatment_mechanism)
                            - float(control_mechanism)
                        )
                        / max(abs(float(control_mechanism)), 1e-12)
                    ),
                    "control_mechanism_rank": control["mechanism_rank"],
                    "treatment_mechanism_rank": treatment[
                        "mechanism_rank"
                    ],
                }
            )
    return output


def split_bank(
    selected_rows: list[dict[str, Any]],
    selected_effects: list[dict[str, Any]],
    *,
    models: list[str],
    seed_start: int,
    seed_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    maximum = 32
    sizes = []
    while maximum <= seed_count:
        sizes.append(maximum)
        maximum *= 2
    official_rows = [
        row
        for row in selected_rows
        if row["evaluation_table"] == "main"
        and row["generator_family_role"] == "primary"
        and row["model_id"] in models
    ]
    official_effects = [
        row
        for row in selected_effects
        if row["evaluation_table"] == "main"
        and row["generator_family_role"] == "primary"
        and row["model_id"] in models
    ]
    present_capabilities = [
        capability
        for capability in v8.CAPABILITIES
        if any(
            row["capability_id"] == capability
            for row in official_rows
        )
    ]
    for size in sizes:
        batches = []
        for start in range(seed_start, seed_start + seed_count, size):
            stop = start + size
            if stop > seed_start + seed_count:
                break
            scores = score_table(
                official_rows,
                official_effects,
                seed_filter=set(range(start, stop)),
            )
            batches.append(
                {
                    "seed_start": start,
                    "seed_stop": stop,
                    "scores": scores,
                }
            )
        index: dict[
            tuple[int, str, str, str], dict[str, int | None]
        ] = {}
        score_index: dict[
            tuple[int, str, str, str], dict[str, float]
        ] = {}
        for batch_index, batch in enumerate(batches):
            for row in batch["scores"]:
                for kind, rank_name, score_name in (
                    ("accuracy", "accuracy_rank", "accuracy_score"),
                    ("mechanism", "mechanism_rank", "mechanism_score"),
                ):
                    key = (
                        batch_index,
                        row["context_policy"],
                        row["capability_id"],
                        kind,
                    )
                    index.setdefault(key, {})[row["model_id"]] = row[
                        rank_name
                    ]
                    if row[score_name] is not None:
                        score_index.setdefault(key, {})[row["model_id"]] = (
                            float(row[score_name])
                        )
        for policy in ("fixed_l504", "oracle_context"):
            for capability in present_capabilities:
                for kind in ("accuracy", "mechanism"):
                    rank_maps = [
                        index.get((batch_index, policy, capability, kind), {})
                        for batch_index in range(len(batches))
                    ]
                    score_maps = [
                        score_index.get(
                            (batch_index, policy, capability, kind),
                            {},
                        )
                        for batch_index in range(len(batches))
                    ]
                    taus: list[float] = []
                    relative_differences: list[float] = []
                    top3_overlaps: list[float] = []
                    top1: list[str] = []
                    for rank_map in rank_maps:
                        ranked = [
                            (rank, model)
                            for model, rank in rank_map.items()
                            if rank is not None
                        ]
                        if ranked:
                            top1.append(min(ranked)[1])
                    for left_index in range(len(rank_maps)):
                        for right_index in range(left_index + 1, len(rank_maps)):
                            common = sorted(
                                set(rank_maps[left_index])
                                & set(rank_maps[right_index])
                            )
                            if len(common) < 2:
                                continue
                            tau = engine.kendall_tau_b(
                                np.asarray(
                                    [
                                        rank_maps[left_index][name]
                                        for name in common
                                    ],
                                    dtype=float,
                                ),
                                np.asarray(
                                    [
                                        rank_maps[right_index][name]
                                        for name in common
                                    ],
                                    dtype=float,
                                ),
                            )
                            if math.isfinite(float(tau)):
                                taus.append(float(tau))
                            left_top = {
                                model
                                for _rank, model in sorted(
                                    (
                                        (rank, model)
                                        for model, rank in rank_maps[
                                            left_index
                                        ].items()
                                        if rank is not None
                                    )
                                )[:3]
                            }
                            right_top = {
                                model
                                for _rank, model in sorted(
                                    (
                                        (rank, model)
                                        for model, rank in rank_maps[
                                            right_index
                                        ].items()
                                        if rank is not None
                                    )
                                )[:3]
                            }
                            denominator = min(
                                3,
                                len(left_top),
                                len(right_top),
                            )
                            if denominator:
                                top3_overlaps.append(
                                    len(left_top & right_top) / denominator
                                )
                            common_scores = sorted(
                                set(score_maps[left_index])
                                & set(score_maps[right_index])
                            )
                            for model_id in common_scores:
                                left_score = float(
                                    score_maps[left_index][model_id]
                                )
                                right_score = float(
                                    score_maps[right_index][model_id]
                                )
                                relative_differences.append(
                                    abs(left_score - right_score)
                                    / max(
                                        0.5
                                        * (
                                            abs(left_score)
                                            + abs(right_score)
                                        ),
                                        1e-12,
                                    )
                                )
                    output.append(
                        {
                            "batch_size": size,
                            "batch_count": len(batches),
                            "context_policy": policy,
                            "capability_id": capability,
                            "score_kind": kind,
                            "mean_kendall_tau_b": (
                                float(np.mean(taus)) if taus else None
                            ),
                            "mean_pairwise_relative_score_difference": (
                                float(np.mean(relative_differences))
                                if relative_differences
                                else None
                            ),
                            "max_pairwise_relative_score_difference": (
                                float(np.max(relative_differences))
                                if relative_differences
                                else None
                            ),
                            "mean_top3_overlap": (
                                float(np.mean(top3_overlaps))
                                if top3_overlaps
                                else None
                            ),
                            "top1_consistency": (
                                max(
                                    top1.count(model) for model in set(top1)
                                )
                                / len(top1)
                                if len(top1) >= 2
                                else None
                            ),
                            "batch_top1": top1,
                        }
                    )
    return output


def render_report(
    scores: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    *,
    dataset_id: str,
) -> str:
    lines = [
        "# Paper v8 单数据集全链路测试",
        "",
        f"- 数据集：`{dataset_id}`",
        "- 主表：clean primary family；MASE 的四种 context 共用 clean L504 denominator。",
        "- `oracle_context` 是按样本选择的乐观上界；counterfactual pair 共享 context。",
        "",
    ]
    official = [
        row
        for row in scores
        if row["evaluation_table"] == "main"
        and row["generator_family_role"] == "primary"
    ]
    present_capabilities = [
        capability
        for capability in v8.CAPABILITIES
        if any(row["capability_id"] == capability for row in official)
    ]

    def format_score(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    for policy in ("fixed_l504", "oracle_context"):
        lines.extend(
            [
                f"## {policy}",
                "",
                "| capability | model | MASE | history-std NMAE | accuracy rank | mechanism | mechanism rank |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for capability in present_capabilities:
            selected = sorted(
                [
                    row
                    for row in official
                    if row["context_policy"] == policy
                    and row["capability_id"] == capability
                ],
                key=lambda row: (
                    row["is_reference_baseline"],
                    row["accuracy_rank"] or 999,
                    row["model_id"],
                ),
            )
            for row in selected:
                lines.append(
                    f"| {capability} | {row['model_id']} | "
                    f"{format_score(row['accuracy_score'])} | "
                    f"{format_score(row['history_std_normalized_mae'])} | "
                    f"{row['accuracy_rank'] or '-'} | "
                    f"{format_score(row['mechanism_score'])} | "
                    f"{row['mechanism_rank'] or '-'} |"
                )
        lines.append("")
    strict = [
        row
        for row in scores
        if row["evaluation_table"] == "strict_counterfactual_audit"
        and row["generator_family_role"] == "primary"
        and not row["is_reference_baseline"]
    ]
    if strict:
        lines.extend(
            [
                "## I5 strict counterfactual audit",
                "",
                "- `effect NRMSE≈1` 表示模型几乎没有随联合输入恢复反事实效应；该表是诊断审计，不并入主能力分。",
                "",
                "| policy | capability | model | effect NRMSE | rank | seeds |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in strict:
            lines.append(
                f"| {row['context_policy']} | {row['capability_id']} | "
                f"{row['model_id']} | "
                f"{format_score(row['mechanism_score'])} | "
                f"{row['mechanism_rank'] or '-'} | {row['seed_count']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Split-bank",
            "",
            "| N | policy | capability | score | batches | relative Δ | tau-b | Top-1 | Top-3 overlap |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in split_rows:
        relative_difference = (
            "-"
            if row["mean_pairwise_relative_score_difference"] is None
            else f"{row['mean_pairwise_relative_score_difference']:.1%}"
        )
        tau = (
            "-"
            if row["mean_kendall_tau_b"] is None
            else f"{row['mean_kendall_tau_b']:.3f}"
        )
        top = (
            "-"
            if row["top1_consistency"] is None
            else f"{row['top1_consistency']:.1%}"
        )
        top3 = (
            "-"
            if row["mean_top3_overlap"] is None
            else f"{row['mean_top3_overlap']:.1%}"
        )
        lines.append(
            f"| {row['batch_size']} | {row['context_policy']} | "
            f"{row['capability_id']} | {row['score_kind']} | "
            f"{row['batch_count']} | {relative_difference} | {tau} | "
            f"{top} | {top3} |"
        )
    return "\n".join(lines) + "\n"


def render_matched_report(
    comparisons: list[dict[str, Any]],
    *,
    dataset_id: str,
) -> str:
    lines = [
        "# Paper v8 matched sensitivity / robustness 审计",
        "",
        f"- 数据集：`{dataset_id}`",
        "- control 与 treatment 严格匹配 model、capability、seed、intensity 和 context policy。",
        "- `Δ` 为 treatment 相对 clean-primary control 的变化；误差越低越好。",
        "",
    ]

    def score(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    def delta(value: float | None) -> str:
        return "-" if value is None else f"{value:+.1%}"

    official = [
        row for row in comparisons if not row["is_reference_baseline"]
    ]
    for comparison_id in (
        "secondary_family",
        "observation_noise_robustness",
        "multivariate_input_ablation",
    ):
        for policy in ("fixed_l504", "oracle_context"):
            if comparison_id == "multivariate_input_ablation":
                lines.extend(
                    [
                        f"## {comparison_id} / {policy}",
                        "",
                        "- common factor 只评分历史未改的 protected target；cross-series 只评分历史未改的 responders。正 Δ 表示模型使用了被消融的跨变量信息。",
                        "",
                        "| capability | model | focal metric | clean | ablated | Δ |",
                        "|---|---|---|---:|---:|---:|",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"## {comparison_id} / {policy}",
                        "",
                        "| capability | model | clean MASE | treatment MASE | Δ | clean mechanism | treatment mechanism | Δ |",
                        "|---|---|---:|---:|---:|---:|---:|---:|",
                    ]
                )
            for row in official:
                if (
                    row["comparison_id"] != comparison_id
                    or row["context_policy"] != policy
                ):
                    continue
                if comparison_id == "multivariate_input_ablation":
                    lines.append(
                        f"| {row['capability_id']} | {row['model_id']} | "
                        f"{row['accuracy_metric']} | "
                        f"{score(row['control_accuracy_score'])} | "
                        f"{score(row['treatment_accuracy_score'])} | "
                        f"{delta(row['accuracy_relative_delta'])} |"
                    )
                else:
                    lines.append(
                        f"| {row['capability_id']} | {row['model_id']} | "
                        f"{score(row['control_accuracy_score'])} | "
                        f"{score(row['treatment_accuracy_score'])} | "
                        f"{delta(row['accuracy_relative_delta'])} | "
                        f"{score(row['control_mechanism_score'])} | "
                        f"{score(row['treatment_mechanism_score'])} | "
                        f"{delta(row['mechanism_relative_delta'])} |"
                    )
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    dataset = v8.resolve_dataset(args.dataset_id)
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    inference_dir = (
        args.output_root.resolve()
        / dataset.dataset_id
        / "03_inference"
        / shard_name
    )
    inference_manifest = v8.read_json(
        inference_dir / "inference_manifest.json"
    )
    if not inference_manifest["complete"]:
        raise ValueError("inference manifest is incomplete")
    task_path = Path(v8.read_json(inference_dir / "task_manifest.json")[
        "task_file"
    ]["path"])
    all_metrics: list[dict[str, Any]] = []
    all_effects: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for model_id in [*args.models, *BASELINES]:
        prediction_path = (
            inference_dir
            / "model_shards"
            / engine.safe_filename(model_id)
            / "predictions"
            / f"{engine.safe_filename(model_id)}.jsonl"
            if model_id not in BASELINES
            else None
        )
        metrics, effects, missing = analyze_one_model(
            task_path,
            model_id=model_id,
            prediction_path=prediction_path,
        )
        all_metrics.extend(metrics)
        all_effects.extend(effects)
        coverage.append(
            {
                "model_id": model_id,
                "metric_row_count": len(metrics),
                "effect_row_count": len(effects),
                "missing_prediction_count": missing,
            }
        )
        print(
            f"analyzed {model_id}: metrics={len(metrics)}, "
            f"effects={len(effects)}, missing={missing}",
            flush=True,
        )
    selected_metrics, pair_context = selected_context_rows(all_metrics)
    selected_effects = selected_effect_rows(all_effects, pair_context)
    scores = score_table(selected_metrics, selected_effects)
    matched_comparisons = matched_comparison_rows(
        selected_metrics,
        selected_effects,
    )
    split_rows = split_bank(
        selected_metrics,
        selected_effects,
        models=list(args.models),
        seed_start=args.seed_start,
        seed_count=args.seed_count,
    )
    analysis_dir = (
        args.output_root.resolve()
        / dataset.dataset_id
        / "04_analysis"
        / shard_name
    )
    metric_path = analysis_dir / "prediction_metrics.jsonl"
    effect_path = analysis_dir / "counterfactual_effects.jsonl"
    score_path = analysis_dir / "scores.json"
    split_path = analysis_dir / "split_bank.json"
    matched_path = analysis_dir / "matched_comparisons.json"
    v8.write_jsonl(metric_path, selected_metrics)
    v8.write_jsonl(effect_path, selected_effects)
    v8.write_json(score_path, {"scores": scores})
    v8.write_json(split_path, {"split_bank": split_rows})
    v8.write_json(
        matched_path,
        {"matched_comparisons": matched_comparisons},
    )
    report_path = analysis_dir / "REPORT_ZH.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(scores, split_rows, dataset_id=dataset.dataset_id),
        encoding="utf-8",
    )
    matched_report_path = analysis_dir / "MATCHED_AUDITS_ZH.md"
    matched_report_path.write_text(
        render_matched_report(
            matched_comparisons,
            dataset_id=dataset.dataset_id,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "paper_v8_analysis_manifest.v1",
        "created_at": v8.utc_now(),
        "dataset_id": dataset.dataset_id,
        "inference_manifest_sha256": v8.file_sha256(
            inference_dir / "inference_manifest.json"
        ),
        "models": list(args.models),
        "coverage": coverage,
        "context_policies": ["fixed_l504", "oracle_context"],
        "files": {
            "prediction_metrics": v8.file_record(metric_path),
            "counterfactual_effects": v8.file_record(effect_path),
            "scores": v8.file_record(score_path),
            "split_bank": v8.file_record(split_path),
            "matched_comparisons": v8.file_record(matched_path),
            "report": v8.file_record(report_path),
            "matched_report": v8.file_record(matched_report_path),
        },
    }
    v8.write_json(analysis_dir / "analysis_manifest.json", manifest)
    print(
        v8.canonical_json(
            {
                "score_count": len(scores),
                "split_bank_count": len(split_rows),
                "output": str(analysis_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
