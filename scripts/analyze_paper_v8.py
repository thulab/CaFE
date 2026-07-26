#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import paper_v8_structured_baselines as structured
import paper_v8_pipeline_common as v8
import run_paper_e2_dynamic_stability as engine
import run_paper_v8_model_response as response


DEFAULT_OUTPUT_ROOT = v8.REPO_ROOT / "runtime" / "paper_exp" / "v8"
FIXED_CONTEXT_LENGTH = v8.FIXED_CONTEXT_LENGTH
FIXED_CONTEXT_POLICY = f"fixed_l{FIXED_CONTEXT_LENGTH}"
PRIMARY_MECHANISM_METRIC = {
    "trend": "trend_curvature_component_nrmse",
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
SYNTHETIC_EVALUATION_TABLES = frozenset(
    {
        "main",
        "multivariate_input_ablation",
        "observation_noise_robustness",
        "strict_counterfactual_audit",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze formal Paper v8 "
            f"fixed-L{FIXED_CONTEXT_LENGTH}/oracle-context results."
        )
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--aggregate-experiment",
        action="store_true",
        help=(
            "Aggregate completed per-dataset stage-4 scores into separate "
            "fixed-context and oracle-context capability tables."
        ),
    )
    parser.add_argument(
        "--reuse-existing-aggregate",
        action="store_true",
        help=(
            "Reuse an immutable experiment-level analysis summary after "
            "validating all input and output hashes."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Chronos-2",
            "timesfm2.5",
            "tirex2",
            "moirai2",
            "Timer-3.5",
            "toto2.0",
        ],
    )
    return parser.parse_args()


def validated_synthetic_task_path(
    inference_dir: Path,
    inference_manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Validate the exact synthetic task component used for paper analysis."""

    task_manifest_path = inference_dir / "task_manifest.json"
    expected_manifest_sha256 = inference_manifest.get(
        "task_manifest_sha256"
    )
    if (
        not expected_manifest_sha256
        or str(expected_manifest_sha256)
        != v8.file_sha256(task_manifest_path)
    ):
        raise ValueError("analysis task manifest hash mismatch")
    task_manifest = v8.read_json(task_manifest_path)
    if task_manifest.get("schema_version") != (
        "paper_v8_inference_task_manifest.v2"
    ):
        raise ValueError("unsupported Paper-v8 inference task manifest")
    component = task_manifest.get("task_components", {}).get("synthetic")
    if not isinstance(component, dict):
        raise ValueError(
            "analysis requires an explicit synthetic task component"
        )
    task_path = Path(str(component.get("path", "")))
    if not task_path.is_file():
        raise FileNotFoundError(
            f"synthetic analysis task component is missing: {task_path}"
        )
    if (
        component.get("bytes") is None
        or int(component["bytes"]) != task_path.stat().st_size
    ):
        raise ValueError("synthetic analysis task byte-size mismatch")
    if (
        not component.get("sha256")
        or str(component["sha256"]) != v8.file_sha256(task_path)
    ):
        raise ValueError("synthetic analysis task hash mismatch")

    expected_rows = component.get("row_count")
    if expected_rows is None:
        raise ValueError(
            "synthetic analysis task component is missing row_count"
        )
    observed_rows = 0
    sample_ids: set[str] = set()
    for row in v8.iter_jsonl(task_path):
        observed_rows += 1
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(
                f"duplicate synthetic analysis task sample_id: {sample_id}"
            )
        sample_ids.add(sample_id)
        evaluation_table = str(row.get("evaluation_table", "main"))
        if evaluation_table not in SYNTHETIC_EVALUATION_TABLES:
            raise ValueError(
                "non-synthetic evaluation table in synthetic task "
                f"component: {evaluation_table}"
            )
    if observed_rows != int(expected_rows):
        raise ValueError(
            "synthetic analysis task row-count mismatch: "
            f"{observed_rows} != {expected_rows}"
        )
    if observed_rows != int(
        task_manifest.get("synthetic_view_count", -1)
    ):
        raise ValueError(
            "synthetic analysis task count disagrees with task manifest"
        )
    return task_path, task_manifest


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


def cross_effect_prefix_steps(
    sample: dict[str, Any],
) -> tuple[int, str]:
    """Resolve the history-covered cross-series effect window.

    New samples declare the exact counterfactual support. Older v8 artifacts
    may only expose the lag, or no support metadata at all; the latter retain
    the historical full-horizon scoring behavior instead of becoming
    unreadable.
    """

    horizon = int(sample["horizon"])
    metadata = sample.get("generation_metadata") or {}
    for name in (
        "counterfactual_effect_forecast_steps",
        "history_covered_forecast_steps",
        "cross_lag_steps",
    ):
        value = metadata.get(name)
        if value is None:
            continue
        steps = int(value)
        if steps > 0:
            return min(steps, horizon), f"generation_metadata.{name}"
    future_slice = metadata.get("counterfactual_effect_future_slice")
    if (
        isinstance(future_slice, (list, tuple))
        and len(future_slice) == 2
    ):
        steps = int(future_slice[1]) - int(future_slice[0])
        if steps > 0:
            return min(steps, horizon), (
                "generation_metadata.counterfactual_effect_future_slice"
            )
    return horizon, "legacy_full_horizon_fallback"


def _effect_metrics(
    truth_effect: np.ndarray,
    forecast_effect: np.ndarray,
) -> tuple[float, float, float, float, float]:
    truth_rms = float(np.sqrt(np.mean(truth_effect**2)))
    forecast_rms = float(np.sqrt(np.mean(forecast_effect**2)))
    nrmse = float(
        np.sqrt(np.mean((forecast_effect - truth_effect) ** 2))
        / max(truth_rms, 1e-12)
    )
    return (
        nrmse,
        response.safe_corr(truth_effect, forecast_effect),
        forecast_rms / max(truth_rms, 1e-12),
        truth_rms,
        forecast_rms,
    )


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
    (
        nrmse,
        correlation,
        amplitude_ratio,
        truth_rms,
        forecast_rms,
    ) = _effect_metrics(
        truth_effect,
        forecast_effect,
    )
    row = {
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
        "effect_correlation": correlation,
        "effect_amplitude_ratio": amplitude_ratio,
        "truth_effect_rms": truth_rms,
        "forecast_effect_rms": forecast_rms,
    }
    if str(first_sample["capability_id"]) != "cross_series_dependence":
        return row

    active, source = cross_effect_prefix_steps(first_sample)
    (
        active_nrmse,
        active_correlation,
        active_amplitude_ratio,
        active_truth_rms,
        active_forecast_rms,
    ) = _effect_metrics(
        truth_effect[:active],
        forecast_effect[:active],
    )
    tail_forecast = forecast_effect[active:]
    row.update(
        {
            "active_prefix_steps": active,
            "active_prefix_source": source,
            "active_effect_nrmse": active_nrmse,
            "active_effect_correlation": active_correlation,
            "active_effect_amplitude_ratio": active_amplitude_ratio,
            "active_truth_effect_rms": active_truth_rms,
            "active_forecast_effect_rms": active_forecast_rms,
            "zero_tail_leakage_nrmse": (
                float(np.sqrt(np.mean(tail_forecast**2)))
                / max(active_truth_rms, 1e-12)
                if tail_forecast.size
                else 0.0
            ),
        }
    )
    return row


def structured_cross_pair_effect(
    first_sample: dict[str, Any],
    first_forecast: np.ndarray,
    second_sample: dict[str, Any],
    second_forecast: np.ndarray,
    *,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    row = effect_row(
        first_sample,
        first_forecast,
        second_sample,
        second_forecast,
        model_id="ridge_var",
    )
    if row["active_prefix_source"] == "legacy_full_horizon_fallback":
        active = min(
            int(
                diagnostics.get(
                    "counterfactual_active_prefix_steps",
                    first_sample["horizon"],
                )
            ),
            int(first_sample["horizon"]),
        )
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
        (
            active_nrmse,
            active_correlation,
            active_amplitude,
            active_truth_rms,
            active_forecast_rms,
        ) = _effect_metrics(
            truth_effect[:active],
            forecast_effect[:active],
        )
        tail_forecast = forecast_effect[active:]
        row.update(
            {
                "active_prefix_steps": active,
                "active_prefix_source": (
                    "structured_baseline."
                    "counterfactual_active_prefix_steps"
                ),
                "active_effect_nrmse": active_nrmse,
                "active_effect_correlation": active_correlation,
                "active_effect_amplitude_ratio": active_amplitude,
                "active_truth_effect_rms": active_truth_rms,
                "active_forecast_effect_rms": active_forecast_rms,
                "zero_tail_leakage_nrmse": (
                    float(np.sqrt(np.mean(tail_forecast**2)))
                    / max(active_truth_rms, 1e-12)
                    if tail_forecast.size
                    else 0.0
                ),
            }
        )
    return row


def _median(values: Iterable[float]) -> float | None:
    finite = [
        float(value) for value in values if math.isfinite(float(value))
    ]
    return float(np.median(finite)) if finite else None


def _matched_relative_change(
    control: Iterable[dict[str, Any]],
    treatment: Iterable[dict[str, Any]],
    metric_name: str,
) -> tuple[float | None, int]:
    control_by_seed = {
        int(row["seed_index"]): float(row["metrics"][metric_name])
        for row in control
        if metric_name in row["metrics"]
    }
    treatment_by_seed = {
        int(row["seed_index"]): float(row["metrics"][metric_name])
        for row in treatment
        if metric_name in row["metrics"]
    }
    shared = sorted(set(control_by_seed).intersection(treatment_by_seed))
    changes = [
        (
            treatment_by_seed[seed] - control_by_seed[seed]
        )
        / max(abs(control_by_seed[seed]), 1e-12)
        for seed in shared
    ]
    return _median(changes), len(shared)


def _structured_context_curve(
    metrics: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    *,
    dataset_id: str,
) -> list[dict[str, Any]]:
    metric_groups: dict[
        tuple[str, str, int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in metrics:
        metric_groups[
            (
                str(row["capability_id"]),
                str(row["model_id"]),
                int(row["context_length"]),
                str(row["evaluation_table"]),
            )
        ].append(row)
    effect_groups: dict[
        tuple[str, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in effects:
        effect_groups[
            (
                str(row["capability_id"]),
                str(row["model_id"]),
                int(row["context_length"]),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for capability, structure_model in (
        ("common_factor", "dynamic_factor_var"),
        ("cross_series_dependence", "ridge_var"),
    ):
        mechanism_metric = (
            "common_component_nmae"
            if capability == "common_factor"
            else "responder_normalized_mae"
        )
        for context in v8.VIEW_CONTEXT_LENGTHS:
            structure_main = metric_groups[
                (capability, structure_model, context, "main")
            ]
            diagonal_main = metric_groups[
                (capability, "diagonal_ar", context, "main")
            ]
            structure_ablation = metric_groups[
                (
                    capability,
                    structure_model,
                    context,
                    "multivariate_input_ablation",
                )
            ]
            structure_relative_to_diagonal, advantage_count = (
                _matched_relative_change(
                    diagonal_main,
                    structure_main,
                    mechanism_metric,
                )
            )
            diagonal_advantage = (
                None
                if structure_relative_to_diagonal is None
                else -structure_relative_to_diagonal
            )
            ablation_degradation, ablation_count = (
                _matched_relative_change(
                    structure_main,
                    structure_ablation,
                    mechanism_metric,
                )
            )
            structure_effects = effect_groups[
                (capability, structure_model, context)
            ]
            strict_metric_prefix = (
                "active_effect"
                if capability == "cross_series_dependence"
                else "counterfactual_effect"
            )
            strict_nrmse = _median(
                row[f"{strict_metric_prefix}_nrmse"]
                for row in structure_effects
            )
            strict_correlation = _median(
                row[
                    (
                        "active_effect_correlation"
                        if capability == "cross_series_dependence"
                        else "effect_correlation"
                    )
                ]
                for row in structure_effects
            )
            strict_amplitude = _median(
                row[
                    (
                        "active_effect_amplitude_ratio"
                        if capability == "cross_series_dependence"
                        else "effect_amplitude_ratio"
                    )
                ]
                for row in structure_effects
            )
            strict_success_fraction = (
                float(
                    np.mean(
                        [
                            float(row[f"{strict_metric_prefix}_nrmse"]) < 1.0
                            for row in structure_effects
                        ]
                    )
                )
                if structure_effects
                else None
            )
            zero_tail_leakage = (
                _median(
                    row["zero_tail_leakage_nrmse"]
                    for row in structure_effects
                    if "zero_tail_leakage_nrmse" in row
                )
                if capability == "cross_series_dependence"
                else None
            )
            relevant = [
                row
                for key, group in metric_groups.items()
                if key[0] == capability
                and key[1] == structure_model
                and key[2] == context
                for row in group
            ]
            fallback_rate = (
                float(
                    np.mean(
                        [
                            bool(
                                (
                                    row.get("input_adaptation") or {}
                                )
                                .get("structured_baseline", {})
                                .get("fallback_used", False)
                            )
                            for row in relevant
                        ]
                    )
                )
                if relevant
                else None
            )
            episode_counts = [
                int(
                    (row.get("input_adaptation") or {}).get(
                        "historical_teaching_episode_count",
                        0,
                    )
                )
                for row in structure_main
                if capability == "common_factor"
            ]
            factor_correlation = (
                _median(
                    row["metrics"]["factor_trajectory_correlation"]
                    for row in structure_main
                    if "factor_trajectory_correlation" in row["metrics"]
                )
                if capability == "common_factor"
                else None
            )

            failure_codes: list[str] = []
            if advantage_count == 0 or ablation_count == 0:
                failure_codes.append("missing_matched_main_or_ablation_rows")
            if fallback_rate is None or fallback_rate > 0.01:
                failure_codes.append("structured_fit_fallback_rate_above_1pct")
            if (
                diagonal_advantage is None
                or diagonal_advantage < 0.10
            ):
                failure_codes.append("no_10pct_advantage_over_diagonal_ar")
            if (
                ablation_degradation is None
                or ablation_degradation < 0.10
            ):
                failure_codes.append("no_10pct_input_ablation_degradation")

            strict_pair_count = len(structure_effects)
            strict_evaluable = strict_pair_count >= 3
            strict_passed = bool(
                strict_evaluable
                and strict_nrmse is not None
                and strict_nrmse <= 0.70
                and strict_correlation is not None
                and strict_correlation >= 0.60
                and strict_amplitude is not None
                and 0.30 <= strict_amplitude <= 1.70
                and strict_success_fraction is not None
                and strict_success_fraction >= 0.75
                and (
                    capability != "cross_series_dependence"
                    or (
                        zero_tail_leakage is not None
                        and zero_tail_leakage <= 0.10
                    )
                )
            )
            if capability == "common_factor":
                if (
                    factor_correlation is None
                    or factor_correlation < 0.60
                ):
                    failure_codes.append(
                        "shared_factor_trajectory_correlation_below_0_60"
                    )
            if not strict_evaluable:
                failure_codes.append("fewer_than_3_strict_pairs")
            elif not strict_passed:
                failure_codes.append(
                    "strict_counterfactual_recovery_below_threshold"
                )
            hard_passed = not failure_codes

            output.append(
                {
                    "schema_version": (
                        "paper_v8_structured_context_assessment.v1"
                    ),
                    "dataset_id": dataset_id,
                    "capability_id": capability,
                    "context_length": int(context),
                    "structured_model_id": structure_model,
                    "matched_marginal_model_id": "diagonal_ar",
                    "mechanism_metric": mechanism_metric,
                    "structured_main_median": _median(
                        row["metrics"][mechanism_metric]
                        for row in structure_main
                        if mechanism_metric in row["metrics"]
                    ),
                    "diagonal_main_median": _median(
                        row["metrics"][mechanism_metric]
                        for row in diagonal_main
                        if mechanism_metric in row["metrics"]
                    ),
                    "median_relative_advantage_over_diagonal_ar": (
                        diagonal_advantage
                    ),
                    "advantage_matched_seed_count": advantage_count,
                    "median_input_ablation_relative_degradation": (
                        ablation_degradation
                    ),
                    "ablation_matched_seed_count": ablation_count,
                    "factor_trajectory_correlation_median": (
                        factor_correlation
                    ),
                    "strict_pair_count": strict_pair_count,
                    "strict_effect_nrmse_median": strict_nrmse,
                    "strict_effect_correlation_median": (
                        strict_correlation
                    ),
                    "strict_effect_amplitude_ratio_median": (
                        strict_amplitude
                    ),
                    "strict_effect_nrmse_below_1_fraction": (
                        strict_success_fraction
                    ),
                    "strict_metric_scope": (
                        "active_history_covered_prefix"
                        if capability == "cross_series_dependence"
                        else "full_horizon"
                    ),
                    "zero_tail_leakage_nrmse_median": zero_tail_leakage,
                    "strict_effect_evaluable": strict_evaluable,
                    "strict_effect_passed": strict_passed,
                    "strict_effect_assessment": (
                        "evaluated_as_active_prefix_plus_zero_tail"
                        if capability == "cross_series_dependence"
                        else "evaluated_as_blind_shared_fit_hard_gate"
                    ),
                    "structured_fit_fallback_rate": fallback_rate,
                    "historical_teaching_episode_count": (
                        {
                            "minimum": min(episode_counts),
                            "median": float(np.median(episode_counts)),
                            "maximum": max(episode_counts),
                        }
                        if episode_counts
                        else None
                    ),
                    "structured_positive_control_passed": hard_passed,
                    "failure_codes": failure_codes,
                    "interpretation": (
                        "history_only_structure_is_usable"
                        if hard_passed
                        else (
                            "standard_dfm_main_task_failed_or_structure_not_required"
                            if capability == "common_factor"
                            else "lag_structure_not_recovered_or_not_required"
                        )
                    ),
                }
            )
    return output


def analyze_structured_positive_controls(
    task_path: Path,
    *,
    dataset_id: str,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    pending_pairs: dict[
        tuple[str, str], tuple[dict[str, Any], np.ndarray]
    ] = {}
    pending_cross_samples: dict[str, dict[str, Any]] = {}
    pending_common_samples: dict[str, dict[str, Any]] = {}
    coverage: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "forecast_count": 0,
            "fallback_count": 0,
            "effect_count": 0,
        }
    )
    for sample in v8.iter_jsonl(task_path):
        if not structured.is_structured_sample(sample):
            continue
        if int(sample["context_length"]) not in v8.VIEW_CONTEXT_LENGTHS:
            continue
        capability = str(sample["capability_id"])
        for model_id in structured.baseline_ids_for(capability):
            pair_id = sample.get("counterfactual_pair_id")
            member = sample.get("counterfactual_member")
            if (
                capability == "common_factor"
                and model_id == "dynamic_factor_var"
                and sample.get("evaluation_table")
                == "strict_counterfactual_audit"
                and pair_id is not None
                and member is not None
            ):
                key = str(pair_id)
                if int(member) == 0:
                    pending_common_samples[key] = sample
                    continue
                first_sample = pending_common_samples.pop(key, None)
                if first_sample is None:
                    continue
                first_result, second_result = (
                    structured.forecast_common_counterfactual_pair(
                        first_sample,
                        sample,
                    )
                )
                for pair_sample, pair_result in (
                    (first_sample, first_result),
                    (sample, second_result),
                ):
                    pair_adaptation = {
                        "target_mode": (
                            "local_structured_positive_control_shared_pair_fit"
                        ),
                        "structured_baseline": pair_result.diagnostics,
                        "historical_teaching_episode_count": 0,
                    }
                    metrics.append(
                        metric_row(
                            pair_sample,
                            model_id=model_id,
                            forecast=pair_result.forecast,
                            input_adaptation=pair_adaptation,
                        )
                    )
                    coverage[(capability, model_id)][
                        "forecast_count"
                    ] += 1
                    coverage[(capability, model_id)][
                        "fallback_count"
                    ] += int(
                        pair_result.diagnostics["fallback_used"]
                    )
                effects.append(
                    effect_row(
                        first_sample,
                        first_result.forecast,
                        sample,
                        second_result.forecast,
                        model_id=model_id,
                    )
                )
                coverage[(capability, model_id)]["effect_count"] += 1
                continue
            if (
                capability == "cross_series_dependence"
                and model_id == "ridge_var"
                and sample.get("evaluation_table")
                == "strict_counterfactual_audit"
                and pair_id is not None
                and member is not None
            ):
                key = str(pair_id)
                if int(member) == 0:
                    pending_cross_samples[key] = sample
                    continue
                first_sample = pending_cross_samples.pop(key, None)
                if first_sample is None:
                    continue
                first_result, second_result = (
                    structured.forecast_cross_counterfactual_pair(
                        first_sample,
                        sample,
                    )
                )
                for pair_sample, pair_result in (
                    (first_sample, first_result),
                    (sample, second_result),
                ):
                    pair_adaptation = {
                        "target_mode": (
                            "local_structured_positive_control_shared_pair_fit"
                        ),
                        "structured_baseline": pair_result.diagnostics,
                        "historical_teaching_episode_count": 0,
                    }
                    metrics.append(
                        metric_row(
                            pair_sample,
                            model_id=model_id,
                            forecast=pair_result.forecast,
                            input_adaptation=pair_adaptation,
                        )
                    )
                    coverage[(capability, model_id)][
                        "forecast_count"
                    ] += 1
                    coverage[(capability, model_id)][
                        "fallback_count"
                    ] += int(
                        pair_result.diagnostics["fallback_used"]
                    )
                effects.append(
                    structured_cross_pair_effect(
                        first_sample,
                        first_result.forecast,
                        sample,
                        second_result.forecast,
                        diagnostics=first_result.diagnostics,
                    )
                )
                coverage[(capability, model_id)]["effect_count"] += 1
                continue
            result = structured.forecast(sample, model_id)
            metadata = sample.get("generation_metadata", {})
            adaptation = {
                "target_mode": "local_structured_positive_control",
                "structured_baseline": result.diagnostics,
                "historical_teaching_episode_count": int(
                    metadata.get(
                        "historical_episode_count_in_view",
                        metadata.get("historical_episode_count", 0),
                    )
                ),
            }
            metrics.append(
                metric_row(
                    sample,
                    model_id=model_id,
                    forecast=result.forecast,
                    input_adaptation=adaptation,
                )
            )
            coverage[(capability, model_id)]["forecast_count"] += 1
            coverage[(capability, model_id)]["fallback_count"] += int(
                result.diagnostics["fallback_used"]
            )
            if pair_id is None or member is None:
                continue
            key = (model_id, str(pair_id))
            if int(member) == 0:
                pending_pairs[key] = (sample, result.forecast)
            else:
                first = pending_pairs.pop(key, None)
                if first is None:
                    continue
                effects.append(
                    effect_row(
                        first[0],
                        first[1],
                        sample,
                        result.forecast,
                        model_id=model_id,
                    )
                )
                coverage[(capability, model_id)]["effect_count"] += 1
    context_curve = _structured_context_curve(
        metrics,
        effects,
        dataset_id=dataset_id,
    )
    return {
        "schema_version": "paper_v8_structured_positive_controls.v1",
        "dataset_id": dataset_id,
        "scope": {
            "generator_family_role": "primary",
            "intensity": 5,
            "capability_ids": sorted(
                structured.STRUCTURED_CAPABILITIES
            ),
            "evaluation_tables": sorted(
                structured.STRUCTURED_EVALUATION_TABLES
            ),
            "context_lengths": list(v8.VIEW_CONTEXT_LENGTHS),
            "excluded_from_foundation_model_ranking": True,
        },
        "fit_policy": {
            "history_only": True,
            "validation": "final_25pct_chronological_one_step",
            "lag_candidates": (
                "unique([1,4,12,24,horizon]) clipped to context/2"
            ),
            "ridge_alpha_candidates": list(
                structured.RIDGE_ALPHA_CANDIDATES
            ),
            "ordinary_samples_fitted_independently": True,
            "strict_pairs_share_blind_fit": True,
            "paired_members_fitted_independently": False,
            "generator_metadata_used_for_fitting": False,
            "fallback": (
                "last_value_with_explicit_reason; assessment invalid above 1pct"
            ),
        },
        "assessment_thresholds": {
            "minimum_relative_advantage_over_diagonal_ar": 0.10,
            "minimum_input_ablation_relative_degradation": 0.10,
            "maximum_fallback_rate": 0.01,
            "minimum_common_factor_trajectory_correlation": 0.60,
            "minimum_strict_pair_count": 3,
            "maximum_strict_effect_nrmse": 0.70,
            "minimum_strict_effect_correlation": 0.60,
            "strict_effect_amplitude_ratio_range": [0.30, 1.70],
            "minimum_strict_nrmse_below_1_fraction": 0.75,
            "maximum_zero_tail_leakage_nrmse": 0.10,
            "common_strict_effect_is_diagnostic_not_hard_gate": False,
            "common_strict_effect_is_hard_gate": True,
        },
        "coverage": [
            {
                "capability_id": capability,
                "model_id": model_id,
                **counts,
                "fallback_rate": (
                    counts["fallback_count"]
                    / max(counts["forecast_count"], 1)
                ),
            }
            for (capability, model_id), counts in sorted(coverage.items())
        ],
        "context_curve": context_curve,
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
        {**row, "context_policy": FIXED_CONTEXT_POLICY}
        for row in rows
        if int(row["context_length"]) == FIXED_CONTEXT_LENGTH
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
        {**row, "context_policy": FIXED_CONTEXT_POLICY}
        for row in effects
        if int(row["context_length"]) == FIXED_CONTEXT_LENGTH
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
        strict_effect_metric = (
            "active_effect_nrmse"
            if capability == "cross_series_dependence"
            else "counterfactual_effect_nrmse"
        )
        metric_name = (
            strict_effect_metric
            if (
                key[2] == "strict_counterfactual_audit"
                and capability
                in {"common_factor", "cross_series_dependence"}
            )
            else PRIMARY_MECHANISM_METRIC[capability]
        )
        if key[2] == "strict_counterfactual_audit" and capability in {
            "common_factor",
            "cross_series_dependence",
        }:
            mechanism_values = [
                float(row[metric_name])
                for row in effect_groups.get(key, [])
                if int(row["intensity"]) == 5
                and metric_name in row
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


def multivariate_utilization_audit_rows(
    selected_rows: list[dict[str, Any]],
    selected_effects: list[dict[str, Any]],
    matched_comparisons: list[dict[str, Any]],
    *,
    models: list[str],
) -> list[dict[str, Any]]:
    """Build a non-ranking audit of actual cross-channel input use."""

    capabilities = {"common_factor", "cross_series_dependence"}
    main_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in selected_rows:
        if (
            row["evaluation_table"] == "main"
            and row["generator_family_role"] == "primary"
            and row["capability_id"] in capabilities
            and row["model_id"] in models
        ):
            main_groups[
                (
                    str(row["dataset_id"]),
                    str(row["context_policy"]),
                    str(row["capability_id"]),
                    str(row["model_id"]),
                )
            ].append(row)
    ablations = {
        (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        ): row
        for row in matched_comparisons
        if row["comparison_id"] == "multivariate_input_ablation"
        and row["model_id"] in models
        and row["capability_id"] in capabilities
    }
    strict_groups: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in selected_effects:
        if (
            row["evaluation_table"] == "strict_counterfactual_audit"
            and row["generator_family_role"] == "primary"
            and row["capability_id"] in capabilities
            and row["model_id"] in models
            and int(row["intensity"]) == 5
        ):
            strict_groups[
                (
                    str(row["dataset_id"]),
                    str(row["context_policy"]),
                    str(row["capability_id"]),
                    str(row["model_id"]),
                )
            ].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(main_groups.items()):
        target_modes = sorted(
            {
                str((row.get("input_adaptation") or {}).get("target_mode"))
                for row in group
                if (row.get("input_adaptation") or {}).get("target_mode")
            }
        )
        target_mode = (
            target_modes[0] if len(target_modes) == 1 else "mixed_or_unknown"
        )
        is_independent_reference = target_mode == "independent_univariate"
        ablation = ablations.get(key)
        strict = strict_groups.get(key, [])
        strict_nrmse_name = (
            "active_effect_nrmse"
            if key[2] == "cross_series_dependence"
            else "counterfactual_effect_nrmse"
        )
        strict_correlation_name = (
            "active_effect_correlation"
            if key[2] == "cross_series_dependence"
            else "effect_correlation"
        )
        strict_amplitude_name = (
            "active_effect_amplitude_ratio"
            if key[2] == "cross_series_dependence"
            else "effect_amplitude_ratio"
        )
        output.append(
            {
                "schema_version": (
                    "paper_v8_multivariate_utilization_audit.v1"
                ),
                "dataset_id": key[0],
                "context_policy": key[1],
                "capability_id": key[2],
                "model_id": key[3],
                "target_mode": target_mode,
                "audit_role": (
                    "independent_univariate_reference"
                    if is_independent_reference
                    else "multivariate_model"
                ),
                "eligible_for_multivariate_utilization_claim": (
                    not is_independent_reference
                    and target_mode == "native_multivariate"
                ),
                "audit_metrics_excluded_from_existing_main_ranking": True,
                "audit_has_no_ranking": True,
                "input_ablation_metric": (
                    ablation["accuracy_metric"] if ablation else None
                ),
                "input_ablation_relative_degradation": (
                    ablation["accuracy_relative_delta"]
                    if ablation
                    else None
                ),
                "input_ablation_matched_seed_count": (
                    ablation["matched_seed_count"] if ablation else 0
                ),
                "strict_metric_scope": (
                    "active_history_covered_prefix"
                    if key[2] == "cross_series_dependence"
                    else "full_horizon"
                ),
                "strict_effect_nrmse_median": _median(
                    row[strict_nrmse_name]
                    for row in strict
                    if strict_nrmse_name in row
                ),
                "strict_effect_correlation_median": _median(
                    row[strict_correlation_name]
                    for row in strict
                    if strict_correlation_name in row
                ),
                "strict_effect_amplitude_ratio_median": _median(
                    row[strict_amplitude_name]
                    for row in strict
                    if strict_amplitude_name in row
                ),
                "strict_pair_count": len(strict),
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
    present_context_policies = sorted(
        {str(row["context_policy"]) for row in official_rows},
        key=lambda value: (
            value == "oracle_context",
            value,
        ),
    )
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
        for policy in present_context_policies:
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
        "- 主表：clean primary family；各 context 共用 clean master denominator。",
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

    for policy in (FIXED_CONTEXT_POLICY, "oracle_context"):
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
        for policy in (FIXED_CONTEXT_POLICY, "oracle_context"):
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


def render_multivariate_utilization_audit(
    rows: list[dict[str, Any]],
    *,
    dataset_id: str,
) -> str:
    lines = [
        "# Paper v8 多变量利用审计",
        "",
        f"- 数据集：`{dataset_id}`",
        "- 本表不排名，也不改变现有 fixed-L168 / oracle 主能力排名。",
        "- `independent_univariate_reference` 无法读取其他目标通道，仅作为单变量参考。",
        "- 消融正 Δ 表示移除其他通道后焦点预测变差；strict effect NRMSE 越低越好。",
        "",
        "| policy | capability | model | role | input mode | "
        "ablation Δ | strict scope | strict NRMSE | strict corr | pairs |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|",
    ]

    def value(number: float | None) -> str:
        return "-" if number is None else f"{number:.3f}"

    def delta(number: float | None) -> str:
        return "-" if number is None else f"{number:+.1%}"

    for row in sorted(
        rows,
        key=lambda item: (
            item["context_policy"],
            item["capability_id"],
            item["audit_role"],
            item["model_id"],
        ),
    ):
        lines.append(
            f"| {row['context_policy']} | {row['capability_id']} | "
            f"{row['model_id']} | {row['audit_role']} | "
            f"{row['target_mode']} | "
            f"{delta(row['input_ablation_relative_degradation'])} | "
            f"{row['strict_metric_scope']} | "
            f"{value(row['strict_effect_nrmse_median'])} | "
            f"{value(row['strict_effect_correlation_median'])} | "
            f"{row['strict_pair_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def validated_file_record(
    record: dict[str, Any],
    *,
    expected_path: Path | None = None,
) -> Path:
    path = Path(str(record.get("path", "")))
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise ValueError(
            f"analysis file path mismatch: {path} != {expected_path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"analysis file is missing: {path}")
    if (
        record.get("bytes") is None
        or int(record["bytes"]) != path.stat().st_size
    ):
        raise ValueError(f"analysis file byte-size mismatch: {path}")
    if (
        not record.get("sha256")
        or str(record["sha256"]) != v8.file_sha256(path)
    ):
        raise ValueError(f"analysis file hash mismatch: {path}")
    return path


def experiment_capability_rows(
    scores: Iterable[dict[str, Any]],
    *,
    context_policy: str,
    dataset_ids: list[str],
    models: list[str],
    capabilities: list[str],
    capability_dataset_ids: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in scores
        if row.get("context_policy") == context_policy
        and row.get("evaluation_table") == "main"
        and row.get("generator_family_role") == "primary"
        and row.get("dataset_id") in dataset_ids
        and row.get("model_id") in models
        and row.get("capability_id") in capabilities
    ]
    by_key = {
        (
            str(row["dataset_id"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        ): row
        for row in selected
    }
    datasets_by_capability = (
        {
            capability_id: list(dataset_ids)
            for capability_id in capabilities
        }
        if capability_dataset_ids is None
        else {
            capability_id: [
                dataset_id
                for dataset_id in dataset_ids
                if dataset_id
                in set(capability_dataset_ids.get(capability_id, ()))
            ]
            for capability_id in capabilities
        }
    )
    expected_count = sum(
        len(datasets_by_capability[capability_id]) * len(models)
        for capability_id in capabilities
    )
    if len(selected) != expected_count or len(by_key) != expected_count:
        raise ValueError(
            f"{context_policy} main score coverage mismatch: "
            f"rows={len(selected)}, unique={len(by_key)}, "
            f"expected={expected_count}"
        )

    output: list[dict[str, Any]] = []
    for capability_id in capabilities:
        supported_dataset_ids = datasets_by_capability[capability_id]
        if not supported_dataset_ids:
            continue
        capability_rows: list[dict[str, Any]] = []
        for model_id in models:
            rows = [
                by_key[(dataset_id, capability_id, model_id)]
                for dataset_id in supported_dataset_ids
            ]
            accuracy_scores = [
                float(row["accuracy_score"])
                for row in rows
                if row.get("accuracy_score") is not None
            ]
            normalized_maes = [
                float(row["history_std_normalized_mae"])
                for row in rows
                if row.get("history_std_normalized_mae") is not None
            ]
            mechanism_scores = [
                float(row["mechanism_score"])
                for row in rows
                if row.get("mechanism_score") is not None
            ]
            accuracy_ranks = [
                int(row["accuracy_rank"])
                for row in rows
                if row.get("accuracy_rank") is not None
            ]
            mechanism_ranks = [
                int(row["mechanism_rank"])
                for row in rows
                if row.get("mechanism_rank") is not None
            ]
            if (
                len(accuracy_scores) != len(supported_dataset_ids)
                or len(normalized_maes) != len(supported_dataset_ids)
                or len(mechanism_scores) != len(supported_dataset_ids)
                or len(accuracy_ranks) != len(supported_dataset_ids)
                or len(mechanism_ranks) != len(supported_dataset_ids)
            ):
                raise ValueError(
                    f"incomplete aggregate score for {context_policy}/"
                    f"{capability_id}/{model_id}"
                )
            capability_rows.append(
                {
                    "schema_version": (
                        "paper_v8_experiment_capability_score.v1"
                    ),
                    "context_policy": context_policy,
                    "capability_id": capability_id,
                    "model_id": model_id,
                    "dataset_count": len(supported_dataset_ids),
                    "dataset_ids": supported_dataset_ids,
                    "macro_mean_accuracy_score": float(
                        np.mean(accuracy_scores)
                    ),
                    "macro_mean_history_std_normalized_mae": float(
                        np.mean(normalized_maes)
                    ),
                    "mean_dataset_accuracy_rank": float(
                        np.mean(accuracy_ranks)
                    ),
                    "accuracy_dataset_wins": sum(
                        rank == 1 for rank in accuracy_ranks
                    ),
                    "macro_mean_mechanism_score": float(
                        np.mean(mechanism_scores)
                    ),
                    "mean_dataset_mechanism_rank": float(
                        np.mean(mechanism_ranks)
                    ),
                    "mechanism_dataset_wins": sum(
                        rank == 1 for rank in mechanism_ranks
                    ),
                }
            )
        for value_name, rank_name in (
            ("mean_dataset_accuracy_rank", "accuracy_rank"),
            ("mean_dataset_mechanism_rank", "mechanism_rank"),
        ):
            values = {
                str(row["model_id"]): float(row[value_name])
                for row in capability_rows
            }
            for row in capability_rows:
                value = values[str(row["model_id"])]
                row[rank_name] = 1 + sum(
                    other < value for other in values.values()
                )
        output.extend(capability_rows)
    return output


def render_experiment_capability_report(
    rows: list[dict[str, Any]],
    *,
    experiment_id: str,
    context_policy: str,
) -> str:
    lines = [
        f"# Paper v8 {context_policy} 跨数据集能力表",
        "",
        f"- 实验：`{experiment_id}`",
        f"- Context policy：`{context_policy}`",
        "- 每个能力仅在真实校准可用的数据集上等权聚合；"
        "foundation models 内排名，结构化正控不参与。",
        "- 平均 rank 越小越好；Oracle context 仅按逐样本 MASE 选窗。",
        "",
        "| capability | model | mean MASE | mean accuracy rank | "
        "accuracy rank | accuracy wins | mean mechanism | "
        "mean mechanism rank | mechanism rank | mechanism wins |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    capabilities = [
        capability
        for capability in v8.CAPABILITIES
        if any(row["capability_id"] == capability for row in rows)
    ]
    for capability_id in capabilities:
        capability_rows = sorted(
            [
                row
                for row in rows
                if row["capability_id"] == capability_id
            ],
            key=lambda row: (
                int(row["accuracy_rank"]),
                float(row["mean_dataset_accuracy_rank"]),
                str(row["model_id"]),
            ),
        )
        for row in capability_rows:
            lines.append(
                f"| {capability_id} | {row['model_id']} | "
                f"{row['macro_mean_accuracy_score']:.3f} | "
                f"{row['mean_dataset_accuracy_rank']:.3f} | "
                f"{row['accuracy_rank']} | "
                f"{row['accuracy_dataset_wins']} | "
                f"{row['macro_mean_mechanism_score']:.3f} | "
                f"{row['mean_dataset_mechanism_rank']:.3f} | "
                f"{row['mechanism_rank']} | "
                f"{row['mechanism_dataset_wins']} |"
            )
    lines.append("")
    return "\n".join(lines)


def reusable_experiment_analysis_manifest(
    analysis_dir: Path,
    *,
    experiment_manifest_path: Path,
    dataset_ids: list[str],
    models: list[str],
    capabilities: list[str],
) -> bool:
    manifest_path = analysis_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = v8.read_json(manifest_path)
        if manifest.get("schema_version") != (
            "paper_v8_experiment_analysis_manifest.v2"
        ):
            return False
        if str(manifest.get("experiment_manifest_sha256")) != (
            v8.file_sha256(experiment_manifest_path)
        ):
            return False
        if list(manifest.get("datasets") or []) != dataset_ids:
            return False
        if list(manifest.get("models") or []) != models:
            return False
        if list(manifest.get("capabilities") or []) != capabilities:
            return False
        inputs = manifest.get("inputs")
        if not isinstance(inputs, list) or [
            str(row.get("dataset_id"))
            for row in inputs
            if isinstance(row, dict)
        ] != dataset_ids:
            return False
        for row in inputs:
            path = Path(str(row.get("analysis_manifest_path", "")))
            if (
                not path.is_file()
                or str(row.get("analysis_manifest_sha256")) != (
                    v8.file_sha256(path)
                )
            ):
                return False
            dataset_manifest = v8.read_json(path)
            score_record = dataset_manifest.get("files", {}).get("scores")
            if not isinstance(score_record, dict):
                return False
            score_path = validated_file_record(score_record)
            if str(row.get("scores_sha256")) != v8.file_sha256(score_path):
                return False
            generation_path = Path(
                str(row.get("generation_manifest_path", ""))
            )
            if (
                not generation_path.is_file()
                or str(row.get("generation_manifest_sha256"))
                != v8.file_sha256(generation_path)
            ):
                return False
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            return False
        for record in files.values():
            if not isinstance(record, dict):
                return False
            validated_file_record(record)
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def aggregate_experiment_analysis(args: argparse.Namespace) -> int:
    experiment_root = args.output_root.resolve()
    experiment_manifest_path = experiment_root / "experiment_manifest.json"
    experiment_manifest = v8.read_json(experiment_manifest_path)
    protocol = experiment_manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("experiment manifest is missing protocol")
    dataset_ids = [str(value) for value in protocol["dataset_ids"]]
    models = [str(value) for value in protocol["models"]]
    capabilities = [str(value) for value in protocol["capabilities"]]
    if list(args.models) != models:
        raise ValueError(
            "aggregate models must exactly match the experiment protocol"
        )
    if (
        int(protocol["seed_start"]) != args.seed_start
        or int(protocol["seed_count"]) != args.seed_count
    ):
        raise ValueError(
            "aggregate seed shard must exactly match the experiment protocol"
        )
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    analysis_dir = experiment_root / "04_analysis" / shard_name
    if reusable_experiment_analysis_manifest(
        analysis_dir,
        experiment_manifest_path=experiment_manifest_path,
        dataset_ids=dataset_ids,
        models=models,
        capabilities=capabilities,
    ):
        if not args.reuse_existing_aggregate:
            raise FileExistsError(
                "immutable experiment-level analysis already exists; "
                "pass --reuse-existing-aggregate to validate and reuse it"
            )
        print(
            v8.canonical_json(
                {
                    "analysis_status": "already_complete",
                    "output": str(analysis_dir),
                }
            )
        )
        return 0
    if analysis_dir.exists():
        raise ValueError(
            "experiment-level analysis directory exists but is not reusable: "
            f"{analysis_dir}"
        )

    all_scores: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    capability_dataset_ids = {
        capability_id: [] for capability_id in capabilities
    }
    for dataset_id in dataset_ids:
        dataset_analysis_dir = (
            experiment_root / dataset_id / "04_analysis" / shard_name
        )
        dataset_manifest_path = (
            dataset_analysis_dir / "analysis_manifest.json"
        )
        dataset_manifest = v8.read_json(dataset_manifest_path)
        if dataset_manifest.get("schema_version") != (
            "paper_v8_analysis_manifest.v1"
        ):
            raise ValueError(
                f"unsupported dataset analysis manifest: {dataset_id}"
            )
        if str(dataset_manifest.get("dataset_id")) != dataset_id:
            raise ValueError(
                f"dataset analysis manifest binding mismatch: {dataset_id}"
            )
        if list(dataset_manifest.get("models") or []) != models:
            raise ValueError(
                f"dataset analysis model mismatch: {dataset_id}"
            )
        score_record = dataset_manifest.get("files", {}).get("scores")
        if not isinstance(score_record, dict):
            raise ValueError(
                f"dataset analysis is missing scores record: {dataset_id}"
            )
        score_path = validated_file_record(
            score_record,
            expected_path=dataset_analysis_dir / "scores.json",
        )
        score_payload = v8.read_json(score_path)
        scores = score_payload.get("scores")
        if not isinstance(scores, list):
            raise ValueError(f"invalid dataset scores payload: {dataset_id}")
        all_scores.extend(scores)
        generation_manifest_path = (
            experiment_root
            / dataset_id
            / "02_generation"
            / f"manifest__{shard_name}.json"
        )
        generation_manifest = v8.read_json(generation_manifest_path)
        generated_capabilities = [
            str(value)
            for value in generation_manifest.get("config", {}).get(
                "capabilities",
                [],
            )
        ]
        unexpected_capabilities = sorted(
            set(generated_capabilities) - set(capabilities)
        )
        if unexpected_capabilities:
            raise ValueError(
                f"generation manifest has unexpected capabilities for "
                f"{dataset_id}: {unexpected_capabilities}"
            )
        for capability_id in generated_capabilities:
            capability_dataset_ids[capability_id].append(dataset_id)
        input_records.append(
            {
                "dataset_id": dataset_id,
                "analysis_manifest_path": str(dataset_manifest_path),
                "analysis_manifest_sha256": v8.file_sha256(
                    dataset_manifest_path
                ),
                "scores_sha256": v8.file_sha256(score_path),
                "generation_manifest_path": str(
                    generation_manifest_path
                ),
                "generation_manifest_sha256": v8.file_sha256(
                    generation_manifest_path
                ),
                "generated_capabilities": generated_capabilities,
            }
        )

    fixed_rows = experiment_capability_rows(
        all_scores,
        context_policy=FIXED_CONTEXT_POLICY,
        dataset_ids=dataset_ids,
        models=models,
        capabilities=capabilities,
        capability_dataset_ids=capability_dataset_ids,
    )
    oracle_rows = experiment_capability_rows(
        all_scores,
        context_policy="oracle_context",
        dataset_ids=dataset_ids,
        models=models,
        capabilities=capabilities,
        capability_dataset_ids=capability_dataset_ids,
    )
    fixed_path = analysis_dir / "capability_scores_fixed_l168.json"
    oracle_path = analysis_dir / "capability_scores_oracle_context.json"
    fixed_report_path = analysis_dir / "REPORT_FIXED_L168_ZH.md"
    oracle_report_path = analysis_dir / "REPORT_ORACLE_CONTEXT_ZH.md"
    v8.write_json(fixed_path, {"scores": fixed_rows})
    v8.write_json(oracle_path, {"scores": oracle_rows})
    fixed_report_path.write_text(
        render_experiment_capability_report(
            fixed_rows,
            experiment_id=str(experiment_manifest["experiment_id"]),
            context_policy=FIXED_CONTEXT_POLICY,
        ),
        encoding="utf-8",
    )
    oracle_report_path.write_text(
        render_experiment_capability_report(
            oracle_rows,
            experiment_id=str(experiment_manifest["experiment_id"]),
            context_policy="oracle_context",
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "paper_v8_experiment_analysis_manifest.v2",
        "created_at": v8.utc_now(),
        "experiment_id": str(experiment_manifest["experiment_id"]),
        "experiment_manifest_sha256": v8.file_sha256(
            experiment_manifest_path
        ),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "datasets": dataset_ids,
        "models": models,
        "capabilities": capabilities,
        "capability_dataset_ids": capability_dataset_ids,
        "context_policies": [FIXED_CONTEXT_POLICY, "oracle_context"],
        "aggregation_policy": (
            "equal_supported_dataset_macro_mean_with_mean_within_dataset_"
            "model_rank"
        ),
        "oracle_selection_policy": (
            "per_model_master_sample_minimum_mase_over_l96_l168_l336;"
            "counterfactual_pairs_share_context"
        ),
        "inputs": input_records,
        "files": {
            "fixed_scores": v8.file_record(fixed_path),
            "oracle_scores": v8.file_record(oracle_path),
            "fixed_report": v8.file_record(fixed_report_path),
            "oracle_report": v8.file_record(oracle_report_path),
        },
    }
    v8.write_json(analysis_dir / "analysis_manifest.json", manifest)
    print(
        v8.canonical_json(
            {
                "analysis_status": "computed",
                "fixed_score_count": len(fixed_rows),
                "oracle_score_count": len(oracle_rows),
                "output": str(analysis_dir),
            }
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.aggregate_experiment:
        return aggregate_experiment_analysis(args)
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
    task_path, task_manifest = validated_synthetic_task_path(
        inference_dir,
        inference_manifest,
    )
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
    utilization_audit = multivariate_utilization_audit_rows(
        selected_metrics,
        selected_effects,
        matched_comparisons,
        models=list(args.models),
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
    structured_controls = analyze_structured_positive_controls(
        task_path,
        dataset_id=dataset.dataset_id,
    )
    structured_controls["created_at"] = v8.utc_now()
    metric_path = analysis_dir / "prediction_metrics.jsonl"
    effect_path = analysis_dir / "counterfactual_effects.jsonl"
    score_path = analysis_dir / "scores.json"
    split_path = analysis_dir / "split_bank.json"
    matched_path = analysis_dir / "matched_comparisons.json"
    utilization_path = (
        analysis_dir / "multivariate_utilization_audit.json"
    )
    structured_path = analysis_dir / "structured_positive_controls.json"
    v8.write_jsonl(metric_path, selected_metrics)
    v8.write_jsonl(effect_path, selected_effects)
    v8.write_json(score_path, {"scores": scores})
    v8.write_json(split_path, {"split_bank": split_rows})
    v8.write_json(
        matched_path,
        {"matched_comparisons": matched_comparisons},
    )
    v8.write_json(
        utilization_path,
        {
            "schema_version": (
                "paper_v8_multivariate_utilization_audit_bundle.v1"
            ),
            "main_ranking_policy_unchanged": True,
            "rows": utilization_audit,
        },
    )
    v8.write_json(structured_path, structured_controls)
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
    utilization_report_path = (
        analysis_dir / "MULTIVARIATE_UTILIZATION_AUDIT_ZH.md"
    )
    utilization_report_path.write_text(
        render_multivariate_utilization_audit(
            utilization_audit,
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
        "context_policies": [FIXED_CONTEXT_POLICY, "oracle_context"],
        "files": {
            "prediction_metrics": v8.file_record(metric_path),
            "counterfactual_effects": v8.file_record(effect_path),
            "scores": v8.file_record(score_path),
            "split_bank": v8.file_record(split_path),
            "matched_comparisons": v8.file_record(matched_path),
            "multivariate_utilization_audit": v8.file_record(
                utilization_path
            ),
            "structured_positive_controls": v8.file_record(
                structured_path
            ),
            "report": v8.file_record(report_path),
            "matched_report": v8.file_record(matched_report_path),
            "multivariate_utilization_report": v8.file_record(
                utilization_report_path
            ),
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
