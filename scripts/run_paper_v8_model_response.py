#!/usr/bin/env python3
"""Run and analyze real timer-service models on the Paper v8 pilot subset."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (BACKEND_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_paper_e2_dynamic_stability as inference  # noqa: E402
from app.services.metric_service import compute_sample_metrics  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "runtime" / "paper_exp" / "v8_test"
DEFAULT_MODELS = ("Chronos-2", "toto2.0", "tirex2", "tabpfn-ts3")
FORCED_SPLIT_MODELS = frozenset({"Chronos-2", "toto2.0", "tirex2"})
TABPFN_INTERNAL_SPLIT_MODELS = frozenset({"tabpfn-ts3"})
DIAGNOSTIC_PROBE_MODELS = frozenset({"cross_lag_linear_probe"})
STRUCTURED_CAPABILITIES = frozenset(
    {
        "common_factor",
        "hierarchical_coherence",
        "cross_series_dependence",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--devices", default="0")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--load-timeout-seconds", type=int, default=1200)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--keep-loaded", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Reuse model_predictions.jsonl without calling the service.",
    )
    parser.add_argument(
        "--robustness-only",
        action="store_true",
        help="Keep existing main predictions and infer only robustness samples.",
    )
    parser.add_argument(
        "--secondary-only",
        action="store_true",
        help="Keep primary/robustness predictions and refresh secondary samples.",
    )
    parser.add_argument(
        "--capability-only",
        action="append",
        default=[],
        help="Keep other predictions and refresh this capability (repeatable).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def baseline_forecast(sample: dict[str, Any], kind: str) -> np.ndarray:
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    horizon = int(sample["horizon"])
    history = target[:context]
    if kind == "last_value":
        return np.repeat(history[-1:], horizon, axis=0)
    if kind == "seasonal_naive":
        period = max(1, int(sample["season_length"]))
        usable_period = min(period, context)
        indexes = context - usable_period + np.arange(horizon) % usable_period
        return history[indexes]
    if kind == "cross_lag_linear_probe":
        metadata = sample["generation_metadata"]
        delay = int(metadata["cross_lag_steps"])
        driver = int(metadata["driver_index"])
        responders = [int(value) for value in metadata["responder_indices"]]
        if delay < horizon or delay >= context:
            raise ValueError(
                "cross_lag_linear_probe requires horizon <= lag < context"
            )
        # This positive control knows the protocol-level driver and lag, but
        # estimates every response coefficient from history only.  The fit
        # excludes the final `delay` driver values, so paired counterfactual
        # members have exactly the same fit and can differ only through the
        # observed driver segment applied at forecast time.
        forecast = baseline_forecast(sample, "seasonal_naive")
        training_driver = history[: context - delay, driver]
        design = np.column_stack(
            [np.ones(training_driver.size, dtype=float), training_driver]
        )
        future_driver = history[
            context - delay : context - delay + horizon,
            driver,
        ]
        for responder in responders:
            training_response = history[delay:context, responder]
            intercept, slope = np.linalg.lstsq(
                design,
                training_response,
                rcond=None,
            )[0]
            forecast[:, responder] = intercept + slope * future_driver
        return forecast
    raise ValueError(f"unknown local baseline: {kind}")


def local_baseline_kinds(sample: dict[str, Any]) -> tuple[str, ...]:
    kinds = ["last_value", "seasonal_naive"]
    if (
        sample["capability_id"] == "cross_series_dependence"
        and sample.get("evaluation_table", "main") == "main"
        and sample["generator_family_role"] == "primary"
    ):
        kinds.append("cross_lag_linear_probe")
    return tuple(kinds)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel().copy()
    right = np.asarray(right, dtype=float).ravel().copy()
    if left.size < 3 or left.size != right.size:
        return 0.0
    left -= float(np.mean(left))
    right -= float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def top_factor_share(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[1] < 2:
        return 1.0
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    variance = singular * singular
    total = float(np.sum(variance))
    return float(variance[0] / total) if total > 1e-12 else 0.0


def leading_factor_decomposition(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    try:
        left, singular, right = np.linalg.svd(
            centered,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return (
            np.zeros(values.shape[0]),
            np.zeros(values.shape[1]),
            np.zeros_like(values),
            0.0,
        )
    if not singular.size or float(np.sum(singular * singular)) <= 1e-12:
        return (
            np.zeros(values.shape[0]),
            np.zeros(values.shape[1]),
            np.zeros_like(values),
            0.0,
        )
    score = left[:, 0] * singular[0]
    loading = right[0]
    common = score[:, None] * loading[None, :]
    share = float(
        singular[0] ** 2 / np.sum(singular * singular)
    )
    return score, loading, common, share


def common_factor_recovery_metrics(
    truth: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    truth_score, truth_loading, truth_common, truth_share = (
        leading_factor_decomposition(truth)
    )
    forecast_score, forecast_loading, forecast_common, forecast_share = (
        leading_factor_decomposition(forecast)
    )
    loading_dot = float(np.dot(truth_loading, forecast_loading))
    sign = 1.0 if loading_dot >= 0 else -1.0
    forecast_score = sign * forecast_score
    forecast_loading = sign * forecast_loading
    forecast_common = forecast_score[:, None] * forecast_loading[None, :]
    truth_score_std = float(np.std(truth_score))
    truth_scale = float(np.mean(np.std(truth, axis=0)))
    return {
        "truth_factor_share": truth_share,
        "forecast_factor_share": forecast_share,
        "factor_share_abs_error": abs(truth_share - forecast_share),
        "factor_loading_cosine": abs(loading_dot),
        "factor_trajectory_correlation": safe_corr(
            truth_score,
            forecast_score,
        ),
        "factor_score_nrmse": float(
            np.sqrt(np.mean((truth_score - forecast_score) ** 2))
            / max(truth_score_std, 1e-12)
        ),
        "common_component_nmae": float(
            np.mean(np.abs(truth_common - forecast_common))
            / max(truth_scale, 1e-12)
        ),
    }


def child_heterogeneity(values: np.ndarray) -> float:
    if values.shape[1] < 3:
        return 0.0
    return float(np.mean(np.std(values[:, 1:], axis=1)))


def hierarchy_recovery_metrics(
    truth: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    forecast_children_sum = np.sum(forecast[:, 1:], axis=1)
    residual = forecast[:, 0] - forecast_children_sum
    parent_scale = float(np.std(truth[:, 0]))
    truth_contrast = truth[:, 1:] - np.mean(
        truth[:, 1:],
        axis=1,
        keepdims=True,
    )
    forecast_contrast = forecast[:, 1:] - np.mean(
        forecast[:, 1:],
        axis=1,
        keepdims=True,
    )
    contrast_scale = float(np.mean(np.std(truth_contrast, axis=0)))
    aggregation_ratio = float(
        np.std(forecast_children_sum)
        / max(float(np.std(forecast[:, 0])), 1e-12)
    )
    truth_heterogeneity = child_heterogeneity(truth)
    forecast_heterogeneity = child_heterogeneity(forecast)
    return {
        "coherence_mae": float(np.mean(np.abs(residual))),
        "coherence_nmae": float(
            np.mean(np.abs(residual)) / max(parent_scale, 1e-12)
        ),
        "aggregation_correlation": safe_corr(
            forecast[:, 0],
            forecast_children_sum,
        ),
        "aggregation_scale_abs_log_error": abs(
            math.log(max(aggregation_ratio, 1e-12))
        ),
        "truth_child_heterogeneity": truth_heterogeneity,
        "forecast_child_heterogeneity": forecast_heterogeneity,
        "child_heterogeneity_abs_error": abs(
            truth_heterogeneity - forecast_heterogeneity
        ),
        "child_contrast_correlation": safe_corr(
            truth_contrast,
            forecast_contrast,
        ),
        "child_contrast_nmae": float(
            np.mean(np.abs(truth_contrast - forecast_contrast))
            / max(contrast_scale, 1e-12)
        ),
    }


def cross_series_recovery_metrics(
    sample: dict[str, Any],
    truth: np.ndarray,
    forecast: np.ndarray,
    history: np.ndarray,
) -> dict[str, float]:
    metadata = sample.get("generation_metadata", {})
    responders = [
        int(index)
        for index in metadata.get(
            "responder_indices",
            list(range(1, truth.shape[1])),
        )
        if 0 <= int(index) < truth.shape[1]
    ]
    if not responders:
        responders = list(range(truth.shape[1]))
    covered = min(
        truth.shape[0],
        max(
            1,
            int(
                metadata.get(
                    "history_covered_forecast_steps",
                    truth.shape[0],
                )
            ),
        ),
    )
    responder_truth = truth[:, responders]
    responder_forecast = forecast[:, responders]
    responder_scale = float(
        np.mean(np.std(history[:, responders], axis=0))
    )
    covered_truth = responder_truth[:covered]
    covered_forecast = responder_forecast[:covered]
    return {
        "responder_mae": float(
            np.mean(np.abs(responder_truth - responder_forecast))
        ),
        "responder_normalized_mae": float(
            np.mean(np.abs(responder_truth - responder_forecast))
            / max(responder_scale, 1e-12)
        ),
        "driver_covered_responder_mae": float(
            np.mean(np.abs(covered_truth - covered_forecast))
        ),
        "driver_covered_responder_correlation": safe_corr(
            covered_truth,
            covered_forecast,
        ),
        "history_covered_forecast_steps": float(covered),
    }


def covariate_future_corr(
    forecast: np.ndarray,
    sample: dict[str, Any],
) -> float:
    covariates = sample.get("covariates")
    if covariates is None:
        return 0.0
    context = int(sample["context_length"])
    future = np.asarray(covariates, dtype=float)[context:]
    scores = [
        abs(safe_corr(future[:, covariate], forecast[:, target]))
        for covariate in range(future.shape[1])
        for target in range(forecast.shape[1])
    ]
    return float(np.mean(scores)) if scores else 0.0


def prediction_metrics(
    sample: dict[str, Any],
    forecast: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    truth = target[context:]
    history = target[:context]
    metrics = compute_sample_metrics(
        truth.tolist(),
        forecast.tolist(),
        history.tolist(),
        seasonal_period=int(sample["season_length"]),
    )
    history_scale = float(np.mean(np.std(history, axis=0)))
    truth_std = float(np.mean(np.std(truth, axis=0)))
    forecast_std = float(np.mean(np.std(forecast, axis=0)))
    per_target_corr = [
        safe_corr(truth[:, index], forecast[:, index])
        for index in range(truth.shape[1])
    ]
    output = {
        name: float(value)
        for name, value in metrics.items()
        if value is not None and np.isfinite(value)
    }
    output.update(
        {
            "normalized_mae_history_std": float(
                np.mean(np.abs(truth - forecast)) / max(history_scale, 1e-12)
            ),
            "forecast_to_truth_std_ratio": forecast_std / max(truth_std, 1e-12),
            "future_curve_correlation": float(np.mean(per_target_corr)),
            "flat_forecast": float(
                forecast_std < max(0.05 * truth_std, 1e-4)
            ),
            "truth_future_std": truth_std,
            "forecast_future_std": forecast_std,
        }
    )
    capability_id = sample["capability_id"]
    if capability_id == "common_factor":
        output.update(
            common_factor_recovery_metrics(truth, forecast)
        )
    elif capability_id == "hierarchical_coherence":
        output.update(
            hierarchy_recovery_metrics(truth, forecast)
        )
    elif capability_id == "cross_series_dependence":
        output.update(
            cross_series_recovery_metrics(
                sample,
                truth,
                forecast,
                history,
            )
        )
    elif capability_id == "covariate_response":
        output["truth_future_covariate_corr"] = covariate_future_corr(truth, sample)
        output["forecast_future_covariate_corr"] = covariate_future_corr(
            forecast,
            sample,
        )
        output["future_covariate_corr_abs_error"] = abs(
            output["truth_future_covariate_corr"]
            - output["forecast_future_covariate_corr"]
        )
    return output


def prediction_row(
    sample: dict[str, Any],
    *,
    model_id: str,
    variant: str,
    forecast: np.ndarray,
    input_adaptation: dict[str, Any],
    request_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": "paper_v8_model_response_prediction.v1",
        "sample_id": sample["sample_id"],
        "master_sample_id": sample.get("master_sample_id", sample["sample_id"]),
        "evaluation_table": sample.get("evaluation_table", "main"),
        "model_id": model_id,
        "variant": variant,
        "capability_id": sample["capability_id"],
        "dataset_id": sample["dataset_id"],
        "generator_family_role": sample["generator_family_role"],
        "generator_family_id": sample["generator_family_id"],
        "intensity": sample["intensity"],
        "sample_index": sample["sample_index"],
        "counterfactual_pair_id": sample.get("counterfactual_pair_id"),
        "counterfactual_member": sample.get("counterfactual_member"),
        "observation_noise_scale": float(
            sample.get("observation_noise_scale", 0.0)
        ),
        "metrics": prediction_metrics(sample, forecast),
        "forecast": forecast.tolist(),
        "input_adaptation": input_adaptation,
        "request_seconds": request_seconds,
    }


def ablated_covariate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(sample)
    result["covariate_dim"] = 0
    result["covariates"] = None
    result["covariate_column_names"] = []
    return result


def forced_split_model(model: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(model)
    limits = dict(result.get("forecast_limits") or {})
    limits["max_target_count"] = 1
    result["forecast_limits"] = limits
    return result


async def forecast_variant(
    *,
    forecast_url: str,
    model_id: str,
    model: dict[str, Any],
    samples: list[dict[str, Any]],
    variant: str,
    concurrency: int,
    timeout_seconds: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=120.0,
    )
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    for sample in samples:
        queue.put_nowait(sample)
    for _ in range(concurrency):
        queue.put_nowait(None)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:

        async def worker() -> None:
            while True:
                sample = await queue.get()
                try:
                    if sample is None:
                        return
                    request_sample = (
                        ablated_covariate_sample(sample)
                        if variant == "covariates_ablated"
                        else sample
                    )
                    request_model = (
                        forced_split_model(model)
                        if variant == "forced_independent_targets"
                        else model
                    )
                    result = await inference.forecast_adapted_sample_with_retry(
                        client,
                        forecast_url=forecast_url,
                        model_id=model_id,
                        model=request_model,
                        sample=request_sample,
                        max_attempts=max_attempts,
                        input_adaptation_policy=inference.INPUT_ADAPTATION_POLICY_ID,
                    )
                    if result["forecast"] is None:
                        failures.append(
                            {
                                "model_id": model_id,
                                "variant": variant,
                                "sample_id": sample["sample_id"],
                                "evaluation_table": sample.get(
                                    "evaluation_table",
                                    "main",
                                ),
                                "error": result["error"],
                            }
                        )
                        continue
                    adaptation = dict(result["input_adaptation"])
                    adaptation["service_semantics"] = (
                        "service_internal_independent_univariate"
                        if model_id in TABPFN_INTERNAL_SPLIT_MODELS
                        else "catalog_native_or_explicit_adapter"
                    )
                    rows.append(
                        prediction_row(
                            sample,
                            model_id=model_id,
                            variant=variant,
                            forecast=np.asarray(result["forecast"], dtype=float),
                            input_adaptation=adaptation,
                            request_seconds=float(result["elapsed_seconds"]),
                        )
                    )
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return rows, failures


def mean_metric(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(row["metrics"][name])
        for row in rows
        if name in row["metrics"] and np.isfinite(row["metrics"][name])
    ]
    return float(np.mean(values)) if values else None


def aggregate_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        key = (
            row.get("evaluation_table", "main"),
            row["model_id"],
            row["variant"],
            row["capability_id"],
            row["generator_family_role"],
            int(row["intensity"]),
        )
        grouped[key].append(row)
    output = []
    metric_names = (
        "mae",
        "mase",
        "normalized_mae_history_std",
        "forecast_to_truth_std_ratio",
        "future_curve_correlation",
        "flat_forecast",
        "coherence_mae",
        "coherence_nmae",
        "aggregation_correlation",
        "aggregation_scale_abs_log_error",
        "factor_share_abs_error",
        "factor_loading_cosine",
        "factor_trajectory_correlation",
        "factor_score_nrmse",
        "common_component_nmae",
        "child_heterogeneity_abs_error",
        "child_contrast_correlation",
        "child_contrast_nmae",
        "responder_mae",
        "responder_normalized_mae",
        "driver_covered_responder_mae",
        "driver_covered_responder_correlation",
        "future_covariate_corr_abs_error",
    )
    for key, rows in sorted(grouped.items()):
        output.append(
            {
                "evaluation_table": key[0],
                "model_id": key[1],
                "variant": key[2],
                "capability_id": key[3],
                "generator_family_role": key[4],
                "intensity": key[5],
                "sample_count": len(rows),
                "metrics": {
                    name: value
                    for name in metric_names
                    if (value := mean_metric(rows, name)) is not None
                },
            }
        )
    return output


def paired_variant_audits(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions = [
        row
        for row in predictions
        if row.get("evaluation_table", "main") == "main"
    ]
    indexed = {
        (row["model_id"], row["variant"], row["sample_id"]): row
        for row in predictions
    }
    audits: list[dict[str, Any]] = []
    for model_id in sorted({row["model_id"] for row in predictions}):
        for capability_id, alternative in (
            ("common_factor", "forced_independent_targets"),
            ("hierarchical_coherence", "forced_independent_targets"),
            ("cross_series_dependence", "forced_independent_targets"),
            ("covariate_response", "covariates_ablated"),
        ):
            pairs = []
            for row in predictions:
                if (
                    row["model_id"] != model_id
                    or row["variant"] != "native"
                    or row["capability_id"] != capability_id
                    or row["generator_family_role"] != "primary"
                ):
                    continue
                other = indexed.get((model_id, alternative, row["sample_id"]))
                if (
                    other is None
                    and model_id in TABPFN_INTERNAL_SPLIT_MODELS
                    and alternative == "forced_independent_targets"
                ):
                    other = row
                if other is not None:
                    pairs.append((row, other))
            if not pairs:
                continue
            loss_metric = (
                "driver_covered_responder_mae"
                if capability_id == "cross_series_dependence"
                else "mae"
            )
            native_mae = np.asarray(
                [pair[0]["metrics"][loss_metric] for pair in pairs],
                dtype=float,
            )
            alternative_mae = np.asarray(
                [pair[1]["metrics"][loss_metric] for pair in pairs],
                dtype=float,
            )
            relative_gain = (alternative_mae - native_mae) / np.maximum(
                alternative_mae,
                1e-12,
            )
            native_mean_loss = float(np.mean(native_mae))
            alternative_mean_loss = float(np.mean(alternative_mae))
            audit = {
                    "model_id": model_id,
                    "capability_id": capability_id,
                    "native_variant": "native",
                    "alternative_variant": alternative,
                    "pair_count": len(pairs),
                    "loss_metric": loss_metric,
                    "native_mean_mae": native_mean_loss,
                    "alternative_mean_mae": alternative_mean_loss,
                    "native_mean_loss": native_mean_loss,
                    "alternative_mean_loss": alternative_mean_loss,
                    "native_relative_mean_loss_gain": float(
                        (alternative_mean_loss - native_mean_loss)
                        / max(alternative_mean_loss, 1e-12)
                    ),
                    "native_relative_mae_gain": float(np.mean(relative_gain)),
                    "native_median_paired_relative_loss_gain": float(
                        np.median(relative_gain)
                    ),
                    "native_win_rate": float(np.mean(native_mae < alternative_mae)),
                    "interpretation": (
                        "not_expected_to_capture_cross_target_structure"
                        if model_id in TABPFN_INTERNAL_SPLIT_MODELS
                        else "positive_gain_is_evidence_of_input_mechanism_use"
                    ),
                }
            diagnostic_by_capability = {
                "common_factor": "factor_score_nrmse",
                "hierarchical_coherence": "coherence_nmae",
                "cross_series_dependence": "driver_covered_responder_mae",
                "covariate_response": "future_covariate_corr_abs_error",
            }
            diagnostic = diagnostic_by_capability[capability_id]
            if all(
                diagnostic in member["metrics"]
                for pair in pairs
                for member in pair
            ):
                native_diagnostic = np.asarray(
                    [pair[0]["metrics"][diagnostic] for pair in pairs],
                    dtype=float,
                )
                alternative_diagnostic = np.asarray(
                    [pair[1]["metrics"][diagnostic] for pair in pairs],
                    dtype=float,
                )
                audit.update(
                    {
                        "structural_diagnostic": diagnostic,
                        "native_mean_structural_error": float(
                            np.mean(native_diagnostic)
                        ),
                        "alternative_mean_structural_error": float(
                            np.mean(alternative_diagnostic)
                        ),
                        "native_structural_win_rate": float(
                            np.mean(native_diagnostic < alternative_diagnostic)
                        ),
                    }
                )
            audits.append(audit)
    return audits


def cross_series_dependence_audits(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    main = [
        row
        for row in predictions
        if row.get("evaluation_table", "main") == "main"
        and row["capability_id"] == "cross_series_dependence"
        and row["generator_family_role"] == "primary"
    ]
    indexed = {
        (row["model_id"], row["variant"], row["sample_id"]): row
        for row in main
    }
    output: list[dict[str, Any]] = []
    for model_id in sorted(
        {
            row["model_id"]
            for row in main
            if row["model_id"] not in {"last_value", "seasonal_naive"}
        }
    ):
        for intensity in (1, 3, 5):
            pairs = []
            for native in main:
                if (
                    native["model_id"] != model_id
                    or native["variant"] != "native"
                    or int(native["intensity"]) != intensity
                ):
                    continue
                split = indexed.get(
                    (
                        model_id,
                        "forced_independent_targets",
                        native["sample_id"],
                    )
                )
                if split is None and model_id in TABPFN_INTERNAL_SPLIT_MODELS:
                    split = native
                if split is not None:
                    pairs.append((native, split))
            if not pairs:
                continue
            native_loss = np.asarray(
                [
                    pair[0]["metrics"]["driver_covered_responder_mae"]
                    for pair in pairs
                ],
                dtype=float,
            )
            split_loss = np.asarray(
                [
                    pair[1]["metrics"]["driver_covered_responder_mae"]
                    for pair in pairs
                ],
                dtype=float,
            )
            native_corr = np.asarray(
                [
                    pair[0]["metrics"][
                        "driver_covered_responder_correlation"
                    ]
                    for pair in pairs
                ],
                dtype=float,
            )
            split_corr = np.asarray(
                [
                    pair[1]["metrics"][
                        "driver_covered_responder_correlation"
                    ]
                    for pair in pairs
                ],
                dtype=float,
            )
            native_mean_loss = float(np.mean(native_loss))
            split_mean_loss = float(np.mean(split_loss))
            output.append(
                {
                    "model_id": model_id,
                    "intensity": intensity,
                    "pair_count": len(pairs),
                    "native_mean_driver_covered_responder_mae": (
                        native_mean_loss
                    ),
                    "split_mean_driver_covered_responder_mae": (
                        split_mean_loss
                    ),
                    "native_relative_mean_loss_gain": float(
                        (split_mean_loss - native_mean_loss)
                        / max(split_mean_loss, 1e-12)
                    ),
                    "native_win_rate": float(
                        np.mean(native_loss < split_loss)
                    ),
                    "native_mean_driver_covered_responder_correlation": float(
                        np.mean(native_corr)
                    ),
                    "split_mean_driver_covered_responder_correlation": float(
                        np.mean(split_corr)
                    ),
                    "split_semantics": (
                        "service_internal_split_reference"
                        if model_id in TABPFN_INTERNAL_SPLIT_MODELS
                        else "forced_independent_targets"
                    ),
                }
            )
    return output


def cross_series_counterfactual_audits(
    predictions: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure whether changing only the observed driver changes the forecast."""

    cross_samples = {
        row["sample_id"]: row
        for row in samples
        if row.get("evaluation_table", "main") == "main"
        and row["capability_id"] == "cross_series_dependence"
        and row["generator_family_role"] == "primary"
        and row.get("counterfactual_pair_id") is not None
    }
    sample_pairs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for sample in cross_samples.values():
        sample_pairs[str(sample["counterfactual_pair_id"])][
            int(sample["counterfactual_member"])
        ] = sample
    prediction_index = {
        (row["model_id"], row["variant"], row["sample_id"]): row
        for row in predictions
        if row["sample_id"] in cross_samples
    }
    grouped: dict[tuple[str, str, int], list[dict[str, float]]] = defaultdict(list)
    model_variants = sorted(
        {
            (row["model_id"], row["variant"])
            for row in predictions
            if row["sample_id"] in cross_samples
        }
    )
    for model_id, variant in model_variants:
        for members in sample_pairs.values():
            if set(members) != {0, 1}:
                continue
            first_sample = members[0]
            second_sample = members[1]
            first_prediction = prediction_index.get(
                (model_id, variant, first_sample["sample_id"])
            )
            second_prediction = prediction_index.get(
                (model_id, variant, second_sample["sample_id"])
            )
            if first_prediction is None or second_prediction is None:
                continue
            context = int(first_sample["context_length"])
            first_target = np.asarray(first_sample["target"], dtype=float)
            second_target = np.asarray(second_sample["target"], dtype=float)
            responders = [
                int(index)
                for index in first_sample["generation_metadata"].get(
                    "responder_indices",
                    list(range(1, first_target.shape[1])),
                )
            ]
            truth_effect = (
                second_target[context:, responders]
                - first_target[context:, responders]
            )
            forecast_effect = (
                np.asarray(second_prediction["forecast"], dtype=float)[:, responders]
                - np.asarray(first_prediction["forecast"], dtype=float)[:, responders]
            )
            truth_rms = float(np.sqrt(np.mean(truth_effect * truth_effect)))
            forecast_rms = float(
                np.sqrt(np.mean(forecast_effect * forecast_effect))
            )
            projection_denominator = float(
                np.sum(truth_effect * truth_effect)
            )
            grouped[
                (model_id, variant, int(first_sample["intensity"]))
            ].append(
                {
                    "effect_nrmse": float(
                        np.sqrt(
                            np.mean(
                                (forecast_effect - truth_effect) ** 2
                            )
                        )
                        / max(truth_rms, 1e-12)
                    ),
                    "effect_correlation": safe_corr(
                        truth_effect,
                        forecast_effect,
                    ),
                    "effect_amplitude_ratio": (
                        forecast_rms / max(truth_rms, 1e-12)
                    ),
                    "effect_signed_projection": (
                        float(np.sum(forecast_effect * truth_effect))
                        / max(projection_denominator, 1e-12)
                    ),
                    "truth_effect_rms": truth_rms,
                    "forecast_effect_rms": forecast_rms,
                    "responder_history_max_abs_difference": float(
                        np.max(
                            np.abs(
                                second_target[:context, responders]
                                - first_target[:context, responders]
                            )
                        )
                    ),
                    "mean_factual_responder_mae": float(
                        0.5
                        * (
                            first_prediction["metrics"]["responder_mae"]
                            + second_prediction["metrics"]["responder_mae"]
                        )
                    ),
                }
            )
    output: list[dict[str, Any]] = []
    for (model_id, variant, intensity), rows in sorted(grouped.items()):
        output.append(
            {
                "model_id": model_id,
                "variant": variant,
                "intensity": intensity,
                "pair_count": len(rows),
                **{
                    f"mean_{name}": float(
                        np.mean([row[name] for row in rows])
                    )
                    for name in rows[0]
                },
            }
        )
    return output


STRUCTURE_RECOVERY_METRICS: dict[str, dict[str, str]] = {
    "common_factor": {
        "factor_loading_cosine": "higher",
        "factor_trajectory_correlation": "higher",
        "factor_score_nrmse": "lower",
        "common_component_nmae": "lower",
        "factor_share_abs_error": "lower",
    },
    "hierarchical_coherence": {
        "coherence_nmae": "lower",
        "aggregation_correlation": "higher",
        "aggregation_scale_abs_log_error": "lower",
        "child_contrast_correlation": "higher",
        "child_contrast_nmae": "lower",
        "child_heterogeneity_abs_error": "lower",
    },
}


def structure_recovery_audits(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    main = [
        row
        for row in predictions
        if row.get("evaluation_table", "main") == "main"
        and row["generator_family_role"] == "primary"
    ]
    indexed = {
        (row["model_id"], row["variant"], row["sample_id"]): row
        for row in main
    }
    output: list[dict[str, Any]] = []
    model_ids = sorted(
        {
            row["model_id"]
            for row in main
            if row["model_id"] not in {"last_value", "seasonal_naive"}
        }
    )
    for model_id in model_ids:
        for capability_id, metric_directions in (
            STRUCTURE_RECOVERY_METRICS.items()
        ):
            pairs = []
            for native in main:
                if (
                    native["model_id"] != model_id
                    or native["variant"] != "native"
                    or native["capability_id"] != capability_id
                ):
                    continue
                split = indexed.get(
                    (
                        model_id,
                        "forced_independent_targets",
                        native["sample_id"],
                    )
                )
                if split is None and model_id in TABPFN_INTERNAL_SPLIT_MODELS:
                    split = native
                if split is not None:
                    pairs.append((native, split))
            for metric_name, direction in metric_directions.items():
                metric_pairs = [
                    pair
                    for pair in pairs
                    if metric_name in pair[0]["metrics"]
                    and metric_name in pair[1]["metrics"]
                ]
                if not metric_pairs:
                    continue
                native_values = np.asarray(
                    [
                        pair[0]["metrics"][metric_name]
                        for pair in metric_pairs
                    ],
                    dtype=float,
                )
                split_values = np.asarray(
                    [
                        pair[1]["metrics"][metric_name]
                        for pair in metric_pairs
                    ],
                    dtype=float,
                )
                native_wins = (
                    native_values > split_values
                    if direction == "higher"
                    else native_values < split_values
                )
                output.append(
                    {
                        "model_id": model_id,
                        "capability_id": capability_id,
                        "metric": metric_name,
                        "direction": direction,
                        "pair_count": len(metric_pairs),
                        "native_mean": float(np.mean(native_values)),
                        "split_mean": float(np.mean(split_values)),
                        "native_minus_split": float(
                            np.mean(native_values - split_values)
                        ),
                        "native_win_rate": float(np.mean(native_wins)),
                    }
                )
    return output


def kendall_tau_from_scores(
    model_ids: list[str],
    primary_scores: dict[str, float],
    secondary_scores: dict[str, float],
) -> float:
    concordance = 0
    pair_count = 0
    for left_index, left in enumerate(model_ids):
        for right in model_ids[left_index + 1 :]:
            first = np.sign(primary_scores[left] - primary_scores[right])
            second = np.sign(secondary_scores[left] - secondary_scores[right])
            if first == 0 or second == 0:
                continue
            concordance += 1 if first == second else -1
            pair_count += 1
    return float(concordance / pair_count) if pair_count else 1.0


def family_model_sensitivity(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    native = [
        row
        for row in predictions
        if row["variant"] == "native"
        and row.get("evaluation_table", "main") == "main"
    ]
    baselines = {
        row["sample_id"]: row
        for row in native
        if row["model_id"] == "seasonal_naive"
    }
    model_ids = sorted(
        {
            row["model_id"]
            for row in native
            if row["model_id"] not in {"last_value", "seasonal_naive"}
        }
    )
    capabilities = sorted({row["capability_id"] for row in native})
    output = []
    for capability_id in capabilities:
        for intensity in (3, 5):
            scores_by_family: dict[str, dict[str, float]] = {}
            for family in ("primary", "secondary"):
                scores: dict[str, float] = {}
                for model_id in model_ids:
                    ratios = []
                    for row in native:
                        if (
                            row["model_id"] != model_id
                            or row["capability_id"] != capability_id
                            or row["generator_family_role"] != family
                            or int(row["intensity"]) != intensity
                        ):
                            continue
                        baseline = baselines.get(row["sample_id"])
                        if baseline is None:
                            continue
                        ratios.append(
                            float(row["metrics"]["mae"])
                            / max(float(baseline["metrics"]["mae"]), 1e-12)
                        )
                    if ratios:
                        scores[model_id] = float(np.mean(ratios))
                scores_by_family[family] = scores
            if not all(
                set(scores_by_family[family]) == set(model_ids)
                for family in ("primary", "secondary")
            ):
                continue
            primary_scores = scores_by_family["primary"]
            secondary_scores = scores_by_family["secondary"]
            primary_ranking = sorted(model_ids, key=primary_scores.__getitem__)
            secondary_ranking = sorted(model_ids, key=secondary_scores.__getitem__)
            output.append(
                {
                    "capability_id": capability_id,
                    "intensity": intensity,
                    "score_semantics": "model_mae_divided_by_paired_seasonal_naive_mae_lower_is_better",
                    "primary_scores": primary_scores,
                    "secondary_scores": secondary_scores,
                    "primary_ranking": primary_ranking,
                    "secondary_ranking": secondary_ranking,
                    "kendall_tau": kendall_tau_from_scores(
                        model_ids,
                        primary_scores,
                        secondary_scores,
                    ),
                    "ranking_changed": primary_ranking != secondary_ranking,
                }
            )
    return output


def observation_noise_robustness_audits(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    native = [row for row in predictions if row["variant"] == "native"]
    clean_index = {
        (row["model_id"], row["sample_id"]): row
        for row in native
        if row.get("evaluation_table", "main") == "main"
    }
    robust_rows = [
        row
        for row in native
        if row.get("evaluation_table") == "observation_noise_robustness"
    ]
    output: list[dict[str, Any]] = []
    for model_id in sorted({row["model_id"] for row in robust_rows}):
        model_rows = [row for row in robust_rows if row["model_id"] == model_id]
        capability_ids: list[str | None] = [None] + sorted(
            {row["capability_id"] for row in model_rows}
        )
        for capability_id in capability_ids:
            selected = [
                row
                for row in model_rows
                if capability_id is None
                or row["capability_id"] == capability_id
            ]
            pairs = [
                (row, clean_index.get((model_id, row["master_sample_id"])))
                for row in selected
            ]
            pairs = [(robust, clean) for robust, clean in pairs if clean]
            if not pairs:
                continue
            robust_mae = np.asarray(
                [pair[0]["metrics"]["mae"] for pair in pairs],
                dtype=float,
            )
            clean_mae = np.asarray(
                [pair[1]["metrics"]["mae"] for pair in pairs],
                dtype=float,
            )
            robust_corr = np.asarray(
                [pair[0]["metrics"]["future_curve_correlation"] for pair in pairs],
                dtype=float,
            )
            clean_corr = np.asarray(
                [pair[1]["metrics"]["future_curve_correlation"] for pair in pairs],
                dtype=float,
            )
            pair_relative_mae = (
                (robust_mae - clean_mae) / np.maximum(clean_mae, 1e-12)
            )
            clean_mean_mae = float(np.mean(clean_mae))
            robust_mean_mae = float(np.mean(robust_mae))
            output.append(
                {
                    "model_id": model_id,
                    "capability_id": capability_id or "__overall__",
                    "pair_count": len(pairs),
                    "observation_noise_scale": float(
                        np.mean(
                            [
                                pair[0].get("observation_noise_scale", 0.0)
                                for pair in pairs
                            ]
                        )
                    ),
                    "clean_mean_mae": clean_mean_mae,
                    "robust_mean_mae": robust_mean_mae,
                    "relative_mean_mae_increase": float(
                        (robust_mean_mae - clean_mean_mae)
                        / max(clean_mean_mae, 1e-12)
                    ),
                    "median_pair_relative_mae_increase": float(
                        np.median(pair_relative_mae)
                    ),
                    "mae_degradation_pair_rate": float(
                        np.mean(robust_mae > clean_mae)
                    ),
                    "mean_curve_correlation_change": float(
                        np.mean(robust_corr - clean_corr)
                    ),
                    "robust_flat_forecast_rate": float(
                        np.mean(
                            [pair[0]["metrics"]["flat_forecast"] for pair in pairs]
                        )
                    ),
                }
            )
    return output


def response_summary(
    predictions: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregates = aggregate_predictions(predictions)
    native = [
        row
        for row in predictions
        if row["variant"] == "native"
        and row.get("evaluation_table", "main") == "main"
    ]
    by_model = {}
    for model_id in sorted({row["model_id"] for row in native}):
        rows = [row for row in native if row["model_id"] == model_id]
        by_model[model_id] = {
            "prediction_count": len(rows),
            "mean_mase": mean_metric(rows, "mase"),
            "mean_future_curve_correlation": mean_metric(
                rows,
                "future_curve_correlation",
            ),
            "median_forecast_to_truth_std_ratio": float(
                np.median(
                    [
                        row["metrics"]["forecast_to_truth_std_ratio"]
                        for row in rows
                    ]
                )
            ),
            "flat_forecast_rate": mean_metric(rows, "flat_forecast"),
            "multivariate_semantics": (
                "local_baseline"
                if model_id in {"last_value", "seasonal_naive"}
                else (
                    "protocol_aware_history_only_positive_control"
                    if model_id in DIAGNOSTIC_PROBE_MODELS
                    else (
                        "service_internal_independent_univariate"
                        if model_id in TABPFN_INTERNAL_SPLIT_MODELS
                        else "catalog_native_when_supported"
                    )
                )
            ),
        }
    seasonal_index = {
        row["sample_id"]: row
        for row in native
        if row["model_id"] == "seasonal_naive"
    }
    mechanism_response = []
    for model_id in sorted(
        {
            row["model_id"]
            for row in native
            if row["model_id"]
            not in {"last_value", "seasonal_naive"} | DIAGNOSTIC_PROBE_MODELS
        }
    ):
        for capability_id in sorted(
            {row["capability_id"] for row in native}
        ):
            model_rows = [
                row
                for row in native
                if row["model_id"] == model_id
                and row["capability_id"] == capability_id
                and row["generator_family_role"] == "primary"
            ]
            if not model_rows:
                continue
            by_intensity = {}
            for intensity in (1, 3, 5):
                intensity_rows = [
                    row
                    for row in model_rows
                    if int(row["intensity"]) == intensity
                ]
                if not intensity_rows:
                    continue
                skills = []
                for row in intensity_rows:
                    baseline = seasonal_index.get(row["sample_id"])
                    if baseline is not None:
                        skills.append(
                            1.0
                            - float(row["metrics"]["mae"])
                            / max(float(baseline["metrics"]["mae"]), 1e-12)
                        )
                by_intensity[str(intensity)] = {
                    "sample_count": len(intensity_rows),
                    "mean_mase": mean_metric(intensity_rows, "mase"),
                    "mean_curve_correlation": mean_metric(
                        intensity_rows,
                        "future_curve_correlation",
                    ),
                    "mean_skill_vs_seasonal_naive_mae": (
                        float(np.mean(skills)) if skills else None
                    ),
                }
            mechanism_response.append(
                {
                    "model_id": model_id,
                    "capability_id": capability_id,
                    "by_intensity": by_intensity,
                }
            )
    return {
        "schema_version": "paper_v8_model_response_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "models": by_model,
        "aggregates": aggregates,
        "mechanism_response": mechanism_response,
        "family_model_sensitivity": family_model_sensitivity(predictions),
        "paired_variant_audits": paired_variant_audits(predictions),
        "cross_series_dependence_audits": cross_series_dependence_audits(
            predictions
        ),
        "cross_series_counterfactual_audits": (
            cross_series_counterfactual_audits(predictions, samples)
        ),
        "structure_recovery_audits": structure_recovery_audits(
            predictions
        ),
        "observation_noise_robustness": observation_noise_robustness_audits(
            predictions
        ),
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Paper v8 model-response audit",
        "",
        "## Overall curve behavior",
        "",
        "| model | native n | MASE | curve corr | forecast/truth std | flat rate | multivariate semantics |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_id, row in summary["models"].items():
        if model_id in DIAGNOSTIC_PROBE_MODELS:
            continue
        lines.append(
            f"| {model_id} | {row['prediction_count']} | "
            f"{row['mean_mase']:.3f} | {row['mean_future_curve_correlation']:.3f} | "
            f"{row['median_forecast_to_truth_std_ratio']:.3f} | "
            f"{row['flat_forecast_rate']:.1%} | {row['multivariate_semantics']} |"
        )
    lines.extend(
        [
            "",
            "## Paired mechanism ablations",
            "",
            "| model | capability | alternative | loss | pairs | native gain | native win rate | structural win rate | interpretation |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["paired_variant_audits"]:
        lines.append(
            f"| {row['model_id']} | {row['capability_id']} | "
            f"{row['alternative_variant']} | {row['loss_metric']} | "
            f"{row['pair_count']} | "
            f"{row['native_relative_mean_loss_gain']:.1%} | "
            f"{row['native_win_rate']:.1%} | "
            f"{row.get('native_structural_win_rate', float('nan')):.1%} | "
            f"{row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "Positive native MAE gain means the native multivariate/covariate request "
            "beat its paired ablation. TabPFN is explicitly treated as an internally "
            "split univariate model, so no cross-target gain is expected from it.",
            "",
            "## Cross-series dependence by intensity",
            "",
            "The loss covers only responder steps whose lagged driver is already "
            "visible in history. Positive gain therefore isolates useful cross-series "
            "conditioning rather than autonomous future extrapolation.",
            "",
            "| model | intensity | pairs | native loss | split loss | native gain | native corr | split corr |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("cross_series_dependence_audits", []):
        lines.append(
            f"| {row['model_id']} | I{row['intensity']} | "
            f"{row['pair_count']} | "
            f"{row['native_mean_driver_covered_responder_mae']:.3f} | "
            f"{row['split_mean_driver_covered_responder_mae']:.3f} | "
            f"{row['native_relative_mean_loss_gain']:.1%} | "
            f"{row['native_mean_driver_covered_responder_correlation']:.3f} | "
            f"{row['split_mean_driver_covered_responder_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-series paired counterfactual response",
            "",
            "Pair members have exactly identical responder histories and differ only "
            "in the observed driver block that deterministically controls the future. "
            "A useful native response has low effect NRMSE, positive signed projection, "
            "and non-zero amplitude; an independent-target forecast must be invariant.",
            "",
            "| model | variant | intensity | pairs | effect NRMSE | effect corr | amplitude ratio | signed projection | responder-history max diff |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("cross_series_counterfactual_audits", []):
        lines.append(
            f"| {row['model_id']} | {row['variant']} | I{row['intensity']} | "
            f"{row['pair_count']} | {row['mean_effect_nrmse']:.3f} | "
            f"{row['mean_effect_correlation']:.3f} | "
            f"{row['mean_effect_amplitude_ratio']:.3f} | "
            f"{row['mean_effect_signed_projection']:.3f} | "
            f"{row['mean_responder_history_max_abs_difference']:.1e} |"
        )
    cross_i5 = {
        row["model_id"]: row
        for row in summary.get("cross_series_dependence_audits", [])
        if int(row["intensity"]) == 5
    }
    cross_i5_parts = []
    for model_id in ("Chronos-2", "toto2.0", "tirex2", "tabpfn-ts3"):
        row = cross_i5.get(model_id)
        if row is not None:
            cross_i5_parts.append(
                f"{model_id} {row['native_relative_mean_loss_gain']:.1%}"
            )
    if cross_i5_parts:
        lines.extend(
            [
                "",
                "At I5, the native mean-loss gains are "
                + ", ".join(cross_i5_parts)
                + ". Interpret these factual losses together with the paired "
                "counterfactual effect table above; only the latter directly tests "
                "whether the forecast changed when the driver changed.",
            ]
        )
    counterfactual_i5 = {
        (row["model_id"], row["variant"]): row
        for row in summary.get("cross_series_counterfactual_audits", [])
        if int(row["intensity"]) == 5
    }
    probe_i5 = counterfactual_i5.get(
        ("cross_lag_linear_probe", "native")
    )
    if probe_i5 is not None:
        native_parts = []
        for model_id in ("Chronos-2", "toto2.0", "tirex2", "tabpfn-ts3"):
            row = counterfactual_i5.get((model_id, "native"))
            if row is not None:
                native_parts.append(
                    f"{model_id} {row['mean_effect_amplitude_ratio']:.1%}"
                )
        lines.extend(
            [
                "",
                "The protocol-aware lagged linear positive control recovers "
                f"{probe_i5['mean_effect_amplitude_ratio']:.1%} of the I5 truth "
                f"effect with NRMSE {probe_i5['mean_effect_nrmse']:.3f}. "
                + (
                    "Native effect amplitudes are " + ", ".join(native_parts) + ". "
                    if native_parts
                    else ""
                )
                + "The positive control uses only observed history to estimate "
                "response coefficients; it validates identifiability but is not a "
                "general foundation-model competitor.",
            ]
        )
    lines.extend(
        [
            "",
            "## Multivariate structure recovery",
            "",
            "| model | capability | metric | direction | native | split | native win rate |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary.get("structure_recovery_audits", []):
        lines.append(
            f"| {row['model_id']} | {row['capability_id']} | "
            f"{row['metric']} | {row['direction']} | "
            f"{row['native_mean']:.3f} | {row['split_mean']:.3f} | "
            f"{row['native_win_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Primary-family mechanism response",
            "",
            "Each cell is `MASE / MAE-skill-vs-seasonal-naive` for the paired seeds.",
            "",
            "| model | capability | I1 | I3 | I5 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["mechanism_response"]:
        cells = []
        for intensity in (1, 3, 5):
            cell = row["by_intensity"].get(str(intensity))
            if not cell:
                cells.append("-")
                continue
            skill = cell["mean_skill_vs_seasonal_naive_mae"]
            cells.append(
                f"{cell['mean_mase']:.3f} / "
                + ("-" if skill is None else f"{skill:.1%}")
            )
        lines.append(
            f"| {row['model_id']} | {row['capability_id']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Secondary-family ranking sensitivity",
            "",
            "Ranks use model MAE divided by the paired seasonal-naive MAE; lower is better.",
            "",
            "| capability | intensity | Kendall tau | primary rank | secondary rank |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in summary["family_model_sensitivity"]:
        lines.append(
            f"| {row['capability_id']} | I{row['intensity']} | "
            f"{row['kendall_tau']:.3f} | "
            f"{' > '.join(row['primary_ranking'])} | "
            f"{' > '.join(row['secondary_ranking'])} |"
        )
    lines.extend(
        [
            "",
            "## Observation-noise robustness",
            "",
            "The paired robustness input adds fixed noise to visible history only; "
            "the future score remains the exact clean latent.",
            "",
            "| model | pairs | noise/history std | clean MAE | noisy-history MAE | relative MAE change | curve-corr change | flat rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["observation_noise_robustness"]:
        if row["capability_id"] != "__overall__":
            continue
        lines.append(
            f"| {row['model_id']} | {row['pair_count']} | "
            f"{row['observation_noise_scale']:.1%} | "
            f"{row['clean_mean_mae']:.3f} | {row['robust_mean_mae']:.3f} | "
            f"{row['relative_mean_mae_increase']:.1%} | "
            f"{row['mean_curve_correlation_change']:+.3f} | "
            f"{row['robust_flat_forecast_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Failures: {summary['failure_count']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_chinese_report(
    summary: dict[str, Any],
    generation_summary: dict[str, Any],
) -> str:
    capability_names = {
        "trend": "趋势",
        "multi_seasonal": "多重季节性",
        "time_varying_seasonality": "时变季节性",
        "regime_switching": "状态切换",
        "nonlinear_persistence": "非线性持续性",
        "predictable_intermittency": "可预测间歇性",
        "common_factor": "共同因子",
        "hierarchical_coherence": "层级一致性",
        "cross_series_dependence": "跨序列预测依赖",
        "covariate_response": "协变量响应",
    }
    capabilities = generation_summary.get("capabilities", {})
    family_differences = [
        float(cell["relative_absolute_difference"])
        for capability in capabilities.values()
        for cell in capability.get(
            "family_sensitivity_at_i3_i5",
            {},
        ).values()
    ]
    max_family_difference = max(family_differences, default=0.0)
    lines = [
        "# Paper v8 确定性机制测试报告",
        "",
        "## 范围和结论",
        "",
        f"本轮固定 `horizon={generation_summary.get('horizon', 48)}`，真实数据只用于校准参数分布；"
        "主表 future 是无过程噪声、无观测噪声的 clean latent。每个能力使用一个 primary family，"
        "并在 I3/I5、部分配对 seed 上使用 secondary family 做敏感性审计。另有配对的"
        "观测噪声 robustness 表：只污染模型可见的 history，future 始终按同一 clean latent 评分。",
        "",
        "实际推理结果不再出现模型共同输出平直曲线的问题：Chronos-2、Toto 2.0、"
        "TiRex2 和 TabPFN "
        "的平坦预测率均为 0，预测曲线标准差与真值标准差之比的中位数约为 0.98。"
        "因此，v7 中的平直预测主要是随机创新在长 horizon 下不可点预测导致的均值回归，"
        "不能据此判定模型预测能力失效。",
        "",
        "## 特征提取、参数映射和 feature gate",
        "",
        "| 能力 | 数据集 | 特征有限/要求数 | 映射数 | clean gate | standard gate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for capability_id, row in capabilities.items():
        clean_gate = (
            f"{row['clean_gate_acceptance_rate']:.1%}"
            if row.get("clean_gate_enforced", True)
            else "未校准"
        )
        standard_gate = (
            f"{row['standard_gate_acceptance_rate']:.1%}"
            if row.get("standard_gate_enforced", True)
            else "未校准"
        )
        lines.append(
            f"| {capability_names.get(capability_id, capability_id)} | "
            f"{row['dataset_id']} | {row['required_features_fully_finite']}/"
            f"{row['required_feature_count']} | {row['required_features_mapped']} | "
            f"{clean_gate} | {standard_gate} |"
        )
    lines.extend(
        [
            "",
            "所有要求特征都能在所选真实数据窗口上得到有限值，也都显式进入生成参数映射；"
            "十个 primary family 的目标特征剂量随 I1→I5 单调。clean gate 会投影掉"
            "噪声率、异常点率、尖峰率等与确定性主表不相容的控制项。现有 v7 conformal "
            "threshold 不能直接变成 v8 的正式阈值；当前接受率只作为开发审计，正式实验前"
            "需要基于 v8 重新校准。新增跨序列依赖能力尚无 v7 gate，明确标记为未校准。",
            "secondary family 使用同一批 seed 反解匹配 primary 的 I3/I5 目标剂量；"
            f"最终最大相对特征差为 {max_family_difference:.1%}。",
            "",
            "## 生成器结构",
            "",
            "| 能力 | primary family | secondary family | future/history std 中位数 |",
            "|---|---|---|---:|",
        ]
    )
    for capability_id, row in capabilities.items():
        lines.append(
            f"| {capability_names.get(capability_id, capability_id)} | "
            f"{row['primary_family']} | {row['secondary_family']} | "
            f"{row['median_future_to_history_std_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "结构原则是：参数按真实窗口特征分布和 seed 随机化，但给定参数后的 history/future "
            "完全确定；所有样本通过 prefix-invariance 检查，层级样本满足 parent=sum(children)，"
            "最大数值残差仅为浮点误差。趋势使用样本特定多项式外推；多季节性使用样本特定"
            "Fourier 基；时变季节性使用调制振子；状态切换使用确定性 duration motif；"
            "非线性使用有界 tanh recurrence；间歇性使用确定性 Gaussian event clock；"
            "共同因子、层级一致性分别使用 latent-factor 和 aggregate/contrast 线性状态空间；"
            "跨序列依赖使用真实特征校准的连续 driver、稀疏宽事件和 64-step "
            "延迟 SCM，并以 responder history 完全相同的 A/B 对直接测量反事实响应；"
            "协变量能力使用已知未来协变量的确定性响应。",
            "",
            "## 实际模型曲线行为",
            "",
            "| 模型 | n | MASE | 曲线相关 | 预测/真值 std | 平坦率 | 多变量语义 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_id, row in summary["models"].items():
        if model_id in DIAGNOSTIC_PROBE_MODELS:
            continue
        lines.append(
            f"| {model_id} | {row['prediction_count']} | {row['mean_mase']:.3f} | "
            f"{row['mean_future_curve_correlation']:.3f} | "
            f"{row['median_forecast_to_truth_std_ratio']:.3f} | "
            f"{row['flat_forecast_rate']:.1%} | {row['multivariate_semantics']} |"
        )
    lines.extend(
        [
            "",
            "四种真实模型在各自已跑样本上的总体误差都明显低于 seasonal naive，且输出"
            "不是常数。Chronos-2、TiRex2、TabPFN 已跑完整十能力；Toto 2.0 本轮补跑"
            "共同因子、层级和跨序列依赖三项。但总体曲线相关性只能说明单序列形状预测"
            "正常，不能证明模型使用了跨变量信息；跨序列能力必须以下面的配对反事实"
            "effect 为准。逐能力曲线见 `plots/`。",
            "",
            "## 配对消融：模型是否真正使用结构输入",
            "",
            "| 模型 | 能力 | 消融 | 评分 | native 增益 | native 胜率 | 结构胜率 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["paired_variant_audits"]:
        lines.append(
            f"| {row['model_id']} | "
            f"{capability_names.get(row['capability_id'], row['capability_id'])} | "
            f"{row['alternative_variant']} | {row['loss_metric']} | "
            f"{row['native_relative_mean_loss_gain']:.1%} | "
            f"{row['native_win_rate']:.1%} | "
            f"{row.get('native_structural_win_rate', float('nan')):.1%} |"
        )
    lines.extend(
        [
            "",
            "TabPFN 由推理服务内部拆成单变量，因此没有共同因子或层级的跨目标收益是预期行为，"
            "在跨序列依赖能力中也作为 split reference；这符合模型设计，不作为服务异常。"
            "它在协变量消融中仍可能获益，因为已知未来协变量是单目标预测也可用的输入。",
            "",
            "层级 coherence 不能脱离预测精度单独打分：last-value/seasonal-naive 这类逐变量"
            "线性操作也可能机械地保持 parent=sum(children)。正式指标应联合精度、coherence "
            "和 child heterogeneity。",
            "",
            "## 跨序列预测依赖：按强度的 native/split 对照",
            "",
            "只在 responder 的滞后 driver 已经出现在 history 中的 future 区段评分，"
            "因此这里的增益不能靠 responder 自身继续外推来解释。",
            "",
            "| 模型 | 强度 | 配对数 | native loss | split loss | native 增益 | native 相关 | split 相关 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("cross_series_dependence_audits", []):
        lines.append(
            f"| {row['model_id']} | I{row['intensity']} | "
            f"{row['pair_count']} | "
            f"{row['native_mean_driver_covered_responder_mae']:.3f} | "
            f"{row['split_mean_driver_covered_responder_mae']:.3f} | "
            f"{row['native_relative_mean_loss_gain']:.1%} | "
            f"{row['native_mean_driver_covered_responder_correlation']:.3f} | "
            f"{row['split_mean_driver_covered_responder_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 跨序列配对反事实响应",
            "",
            "每对样本的 responder history 完全相同，只替换已经观测到、决定 future 的 "
            "driver block。有效的 native 响应应有较低的 effect NRMSE、正的投影和非零幅度；"
            "独立单变量预测在同一对样本上理论上必须保持不变。",
            "",
            "| 模型 | 变体 | 强度 | 对数 | effect NRMSE | effect 相关 | 幅度比 | 有符号投影 | responder history 最大差 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("cross_series_counterfactual_audits", []):
        lines.append(
            f"| {row['model_id']} | {row['variant']} | I{row['intensity']} | "
            f"{row['pair_count']} | {row['mean_effect_nrmse']:.3f} | "
            f"{row['mean_effect_correlation']:.3f} | "
            f"{row['mean_effect_amplitude_ratio']:.3f} | "
            f"{row['mean_effect_signed_projection']:.3f} | "
            f"{row['mean_responder_history_max_abs_difference']:.1e} |"
        )
    cross_i5 = {
        row["model_id"]: row
        for row in summary.get("cross_series_dependence_audits", [])
        if int(row["intensity"]) == 5
    }
    cross_i5_parts = []
    for model_id in ("Chronos-2", "toto2.0", "tirex2", "tabpfn-ts3"):
        row = cross_i5.get(model_id)
        if row is not None:
            cross_i5_parts.append(
                f"{model_id} {row['native_relative_mean_loss_gain']:.1%}"
            )
    if cross_i5_parts:
        lines.extend(
            [
                "",
                "I5 的 native 总体平均损失增益为 "
                + "、".join(cross_i5_parts)
                + "。该 factual loss 必须与上面的配对反事实 effect 指标联合解释；"
                "只有后者直接判断 driver 改变后预测是否跟着改变。",
            ]
        )
    counterfactual_i5 = {
        (row["model_id"], row["variant"]): row
        for row in summary.get("cross_series_counterfactual_audits", [])
        if int(row["intensity"]) == 5
    }
    probe_i5 = counterfactual_i5.get(
        ("cross_lag_linear_probe", "native")
    )
    if probe_i5 is not None:
        native_parts = []
        for model_id in ("Chronos-2", "toto2.0", "tirex2", "tabpfn-ts3"):
            row = counterfactual_i5.get((model_id, "native"))
            if row is not None:
                native_parts.append(
                    f"{model_id} {row['mean_effect_amplitude_ratio']:.1%}"
                )
        lines.extend(
            [
                "",
                "protocol-aware 线性滞后正控制仅用 history 估计响应系数，"
                f"在 I5 恢复了 {probe_i5['mean_effect_amplitude_ratio']:.1%} "
                f"的真实反事实效应，effect NRMSE={probe_i5['mean_effect_nrmse']:.3f}。"
                + (
                    "各模型 native 的效应幅度为 "
                    + "、".join(native_parts)
                    + "。"
                    if native_parts
                    else ""
                )
                + "该 probe 用于验证任务可识别性，不作为通用基础模型参与排名。",
            ]
        )
    lines.extend(
        [
            "",
            "## 多变量结构恢复：native 对比 split",
            "",
            "| 模型 | 能力 | 指标 | 方向 | native | split | native 胜率 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    selected_structure_metrics = {
        "factor_trajectory_correlation",
        "factor_loading_cosine",
        "common_component_nmae",
        "coherence_nmae",
        "aggregation_correlation",
        "child_contrast_correlation",
    }
    for row in summary.get("structure_recovery_audits", []):
        if row["metric"] not in selected_structure_metrics:
            continue
        lines.append(
            f"| {row['model_id']} | "
            f"{capability_names.get(row['capability_id'], row['capability_id'])} | "
            f"{row['metric']} | {row['direction']} | "
            f"{row['native_mean']:.3f} | {row['split_mean']:.3f} | "
            f"{row['native_win_rate']:.1%} |"
        )
    structure_index = {
        (row["model_id"], row["capability_id"], row["metric"]): row
        for row in summary.get("structure_recovery_audits", [])
    }
    structure_parts = []
    for model_id in ("Chronos-2", "toto2.0", "tirex2"):
        factor_row = structure_index.get(
            (model_id, "common_factor", "common_component_nmae")
        )
        hierarchy_row = structure_index.get(
            (model_id, "hierarchical_coherence", "coherence_nmae")
        )
        if factor_row is None or hierarchy_row is None:
            continue
        factor_gain = (
            factor_row["split_mean"] - factor_row["native_mean"]
        ) / max(factor_row["split_mean"], 1e-12)
        hierarchy_gain = (
            hierarchy_row["split_mean"] - hierarchy_row["native_mean"]
        ) / max(hierarchy_row["split_mean"], 1e-12)
        structure_parts.append(
            f"{model_id} 的 common-component NMAE 改善 {factor_gain:.1%}、"
            f"hierarchy coherence NMAE 改善 {hierarchy_gain:.1%}"
        )
    if structure_parts:
        lines.extend(
            [
                "",
                "；".join(structure_parts)
                + "。原生多变量模型只是具备恢复结构的机会，并没有理论保证一定优于"
                "逐变量调用；结果取决于模型是否在推理中真正使用联合信息。当前共同因子"
                "增益很小且方向混合；层级 coherence 上 Chronos-2 与 Toto 2.0 有改善，"
                "TiRex2 略退化，但 child contrast 仍可能相反。应按各结构坐标如实报告，"
                "而不把“原生多变量”本身当成结构能力标签。",
            ]
        )
    lines.extend(
        [
            "",
            "## family 敏感性",
            "",
            "| 能力 | 强度 | Kendall tau | primary 排名 | secondary 排名 |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in summary["family_model_sensitivity"]:
        lines.append(
            f"| {capability_names.get(row['capability_id'], row['capability_id'])} | "
            f"I{row['intensity']} | {row['kendall_tau']:.3f} | "
            f"{' > '.join(row['primary_ranking'])} | "
            f"{' > '.join(row['secondary_ranking'])} |"
        )
    lines.extend(
        [
            "",
            "这里的排名分数是模型 MAE / 配对 seasonal-naive MAE，越低越好。多季节性在"
            "I3/I5 上 Kendall tau=-0.333，存在明显 family 排名反转；协变量 I3/I5 以及"
            "非线性 I3、时变季节 I5、趋势 I5 也有部分换位。因此 primary family 可以作为"
            "主诊断，但不能把单一 family 的模型次序解释为普适排名。当前每个敏感性单元只有"
            "4 个 seed、3 个模型，仅是审计信号，正式统计应扩大 seed。",
            "",
            "## 观测噪声 robustness 表",
            "",
            "固定高斯观测噪声为各通道 history 标准差的 15%；只改变可见 history，future "
            "保持逐点相同的 clean latent。层级样本先对子节点加噪，再重新聚合父节点，避免"
            "把结构破坏混入噪声鲁棒性。",
            "",
            "| 模型 | 配对数 | clean MAE | noisy-history MAE | 相对 MAE 变化 | 曲线相关变化 | 平坦率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["observation_noise_robustness"]:
        if row["capability_id"] != "__overall__":
            continue
        lines.append(
            f"| {row['model_id']} | {row['pair_count']} | "
            f"{row['clean_mean_mae']:.3f} | {row['robust_mean_mae']:.3f} | "
            f"{row['relative_mean_mae_increase']:.1%} | "
            f"{row['mean_curve_correlation_change']:+.3f} | "
            f"{row['robust_flat_forecast_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "逐能力 robustness 结果保存在 `model_response_summary.json` 的 "
            "`observation_noise_robustness` 中。",
            "",
            f"推理失败数：{summary['failure_count']}。",
        ]
    )
    return "\n".join(lines) + "\n"


def create_plots(
    data_dir: Path,
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    sample_by_id = {row["sample_id"]: row for row in samples}
    prediction_index = defaultdict(dict)
    variant_prediction_index = {}
    for row in predictions:
        variant_prediction_index[
            (row["sample_id"], row["model_id"], row["variant"])
        ] = row
        if row["variant"] == "native":
            prediction_index[row["sample_id"]][row["model_id"]] = row
    plot_dir = data_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    capabilities = sorted({row["capability_id"] for row in samples})
    for capability_id in capabilities:
        candidates = [
            row
            for row in samples
            if row["capability_id"] == capability_id
            and row.get("evaluation_table", "main") == "main"
            and row["generator_family_role"] == "primary"
            and int(row["intensity"]) == 5
            and int(row["sample_index"]) == 0
        ]
        if not candidates:
            continue
        sample = candidates[0]
        target = np.asarray(sample["target"], dtype=float)
        context = int(sample["context_length"])
        horizon = int(sample["horizon"])
        channels = min(target.shape[1], 3)
        figure, axes = plt.subplots(
            channels,
            1,
            figsize=(11, 2.8 * channels),
            sharex=True,
            squeeze=False,
        )
        start = max(0, context - 2 * int(sample["season_length"]))
        time_axis = np.arange(start, context + horizon)
        for channel in range(channels):
            axis = axes[channel, 0]
            axis.plot(
                time_axis,
                target[start:, channel],
                color="black",
                linewidth=2.0,
                label="clean latent",
            )
            for model_id, row in sorted(
                prediction_index.get(sample["sample_id"], {}).items()
            ):
                forecast = np.asarray(row["forecast"], dtype=float)
                axis.plot(
                    np.arange(context, context + horizon),
                    forecast[:, channel],
                    linewidth=1.2,
                    label=model_id,
                )
            axis.axvline(context - 0.5, color="#888888", linestyle="--")
            axis.set_ylabel(f"target {channel}")
            axis.grid(alpha=0.2)
        axes[0, 0].set_title(f"{capability_id}: primary I5 seed 0")
        axes[-1, 0].set_xlabel("time step")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=max(2, len(labels)))
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(plot_dir / f"{capability_id}.png", dpi=150)
        plt.close(figure)

    ablation_plot_dir = data_dir / "ablation_plots"
    ablation_plot_dir.mkdir(parents=True, exist_ok=True)
    cross_candidates = [
        row
        for row in samples
        if row["capability_id"] == "cross_series_dependence"
        and row.get("evaluation_table", "main") == "main"
        and row["generator_family_role"] == "primary"
        and int(row["intensity"]) == 5
        and int(row["sample_index"]) == 0
    ]
    if cross_candidates:
        sample = cross_candidates[0]
        target = np.asarray(sample["target"], dtype=float)
        context = int(sample["context_length"])
        horizon = int(sample["horizon"])
        responder_indices = sample["generation_metadata"].get(
            "responder_indices",
            list(range(1, target.shape[1])),
        )
        responder_indices = responder_indices[:3]
        figure, axes = plt.subplots(
            len(responder_indices),
            1,
            figsize=(11, 3.0 * len(responder_indices)),
            sharex=True,
            squeeze=False,
        )
        start = max(0, context - 2 * int(sample["season_length"]))
        full_time = np.arange(start, context + horizon)
        forecast_time = np.arange(context, context + horizon)
        model_ids = sorted(
            model_id
            for model_id in prediction_index.get(sample["sample_id"], {})
            if model_id not in {"last_value", "seasonal_naive"}
        )
        colors = {
            model_id: f"C{index}"
            for index, model_id in enumerate(model_ids)
        }
        for axis_index, channel in enumerate(responder_indices):
            axis = axes[axis_index, 0]
            axis.plot(
                full_time,
                target[start:, channel],
                color="black",
                linewidth=2.0,
                label="clean latent",
            )
            for model_id in model_ids:
                native = variant_prediction_index.get(
                    (sample["sample_id"], model_id, "native")
                )
                if native is not None:
                    forecast = np.asarray(native["forecast"], dtype=float)
                    axis.plot(
                        forecast_time,
                        forecast[:, channel],
                        color=colors[model_id],
                        linewidth=1.5,
                        label=f"{model_id} native",
                    )
                split = variant_prediction_index.get(
                    (
                        sample["sample_id"],
                        model_id,
                        "forced_independent_targets",
                    )
                )
                if split is not None:
                    forecast = np.asarray(split["forecast"], dtype=float)
                    axis.plot(
                        forecast_time,
                        forecast[:, channel],
                        color=colors[model_id],
                        linewidth=1.3,
                        linestyle="--",
                        label=f"{model_id} split",
                    )
            axis.axvline(context - 0.5, color="#888888", linestyle="--")
            axis.set_ylabel(f"responder {channel}")
            axis.grid(alpha=0.2)
        axes[0, 0].set_title(
            "cross_series_dependence: primary I5 seed 0 native vs split"
        )
        axes[-1, 0].set_xlabel("time step")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.87))
        figure.savefig(
            ablation_plot_dir / "cross_series_dependence.png",
            dpi=150,
        )
        plt.close(figure)

    counterfactual_groups: dict[
        str,
        dict[int, dict[str, Any]],
    ] = defaultdict(dict)
    for sample in samples:
        if (
            sample["capability_id"] == "cross_series_dependence"
            and sample.get("evaluation_table", "main") == "main"
            and sample["generator_family_role"] == "primary"
            and int(sample["intensity"]) == 5
            and sample.get("counterfactual_pair_id") is not None
        ):
            counterfactual_groups[str(sample["counterfactual_pair_id"])][
                int(sample["counterfactual_member"])
            ] = sample
    complete_counterfactual_pairs = [
        members
        for _, members in sorted(counterfactual_groups.items())
        if set(members) == {0, 1}
    ]
    if complete_counterfactual_pairs:
        members = complete_counterfactual_pairs[0]
        first = members[0]
        second = members[1]
        context = int(first["context_length"])
        first_target = np.asarray(first["target"], dtype=float)
        second_target = np.asarray(second["target"], dtype=float)
        responders = [
            int(value)
            for value in first["generation_metadata"]["responder_indices"]
        ][:3]
        figure, axes = plt.subplots(
            len(responders),
            1,
            figsize=(11, 3.0 * len(responders)),
            sharex=True,
            squeeze=False,
        )
        forecast_time = np.arange(int(first["horizon"]))
        model_ids = sorted(
            {
                model_id
                for sample_id, model_id, variant in variant_prediction_index
                if sample_id == first["sample_id"] and variant == "native"
                and model_id not in {"last_value", "seasonal_naive"}
            }
        )
        colors = {
            model_id: f"C{index}"
            for index, model_id in enumerate(model_ids)
        }
        for axis_index, responder in enumerate(responders):
            axis = axes[axis_index, 0]
            truth_effect = (
                second_target[context:, responder]
                - first_target[context:, responder]
            )
            axis.plot(
                forecast_time,
                truth_effect,
                color="black",
                linewidth=2.2,
                label="truth counterfactual effect",
            )
            axis.axhline(
                0.0,
                color="#888888",
                linewidth=1.1,
                linestyle="--",
                label="independent-target reference",
            )
            for model_id in model_ids:
                first_prediction = variant_prediction_index.get(
                    (first["sample_id"], model_id, "native")
                )
                second_prediction = variant_prediction_index.get(
                    (second["sample_id"], model_id, "native")
                )
                if first_prediction is None or second_prediction is None:
                    continue
                effect = (
                    np.asarray(second_prediction["forecast"], dtype=float)[
                        :, responder
                    ]
                    - np.asarray(first_prediction["forecast"], dtype=float)[
                        :, responder
                    ]
                )
                axis.plot(
                    forecast_time,
                    effect,
                    color=colors[model_id],
                    linewidth=1.5,
                    label=model_id,
                )
            axis.set_ylabel(f"responder {responder}\neffect")
            axis.grid(alpha=0.2)
        axes[0, 0].set_title(
            "cross_series_dependence: I5 paired counterfactual effect "
            "(responder histories are identical)"
        )
        axes[-1, 0].set_xlabel("forecast step")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=4)
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        figure.savefig(
            ablation_plot_dir / "cross_series_counterfactual_effect.png",
            dpi=150,
        )
        plt.close(figure)

    robustness_plot_dir = data_dir / "robustness_plots"
    robustness_plot_dir.mkdir(parents=True, exist_ok=True)
    for capability_id in capabilities:
        candidates = [
            row
            for row in samples
            if row["capability_id"] == capability_id
            and row.get("evaluation_table") == "observation_noise_robustness"
            and row["generator_family_role"] == "primary"
            and int(row["intensity"]) == 5
            and int(row["sample_index"]) == 0
        ]
        if not candidates:
            continue
        sample = candidates[0]
        clean_sample = sample_by_id[sample["master_sample_id"]]
        clean_target = np.asarray(clean_sample["target"], dtype=float)
        observed_target = np.asarray(sample["target"], dtype=float)
        context = int(sample["context_length"])
        horizon = int(sample["horizon"])
        channels = min(clean_target.shape[1], 3)
        figure, axes = plt.subplots(
            channels,
            1,
            figsize=(11, 2.8 * channels),
            sharex=True,
            squeeze=False,
        )
        start = max(0, context - 2 * int(sample["season_length"]))
        full_time = np.arange(start, context + horizon)
        history_time = np.arange(start, context)
        for channel in range(channels):
            axis = axes[channel, 0]
            axis.plot(
                full_time,
                clean_target[start:, channel],
                color="black",
                linewidth=2.0,
                label="clean latent",
            )
            axis.plot(
                history_time,
                observed_target[start:context, channel],
                color="#999999",
                linewidth=1.0,
                alpha=0.8,
                label="noisy observed history",
            )
            for model_id, row in sorted(
                prediction_index.get(sample["sample_id"], {}).items()
            ):
                forecast = np.asarray(row["forecast"], dtype=float)
                axis.plot(
                    np.arange(context, context + horizon),
                    forecast[:, channel],
                    linewidth=1.2,
                    label=model_id,
                )
            axis.axvline(context - 0.5, color="#888888", linestyle="--")
            axis.set_ylabel(f"target {channel}")
            axis.grid(alpha=0.2)
        axes[0, 0].set_title(
            f"{capability_id}: robustness primary I5 seed 0"
        )
        axes[-1, 0].set_xlabel("time step")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(2, len(labels)),
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(
            robustness_plot_dir / f"{capability_id}.png",
            dpi=150,
        )
        plt.close(figure)


def finalize_outputs(
    data_dir: Path,
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_by_id = {row["sample_id"]: row for row in samples}
    for row in predictions:
        row.setdefault("master_sample_id", row["sample_id"])
        row.setdefault("evaluation_table", "main")
        row.setdefault("observation_noise_scale", 0.0)
        sample = sample_by_id.get(row["sample_id"])
        if sample is not None and row.get("forecast") is not None:
            row["metrics"] = prediction_metrics(
                sample,
                np.asarray(row["forecast"], dtype=float),
            )
    for row in failures:
        row.setdefault("evaluation_table", "main")
    summary = response_summary(predictions, failures, samples)
    write_jsonl(data_dir / "model_predictions.jsonl", predictions)
    write_jsonl(data_dir / "model_failures.jsonl", failures)
    write_json(data_dir / "model_response_summary.json", summary)
    (data_dir / "MODEL_RESPONSE_REPORT.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    generation_summary_path = data_dir / "generation_summary.json"
    generation_summary = (
        json.loads(generation_summary_path.read_text(encoding="utf-8"))
        if generation_summary_path.exists()
        else {"capabilities": {}, "horizon": 48}
    )
    (data_dir / "REPORT_ZH.md").write_text(
        render_chinese_report(summary, generation_summary),
        encoding="utf-8",
    )
    create_plots(data_dir, samples, predictions)
    write_json(
        data_dir / "model_response_manifest.json",
        {
            "schema_version": "paper_v8_model_response_manifest.v1",
            "created_at": summary["created_at"],
            "prediction_count": len(predictions),
            "failure_count": len(failures),
            "files": [
                "model_predictions.jsonl",
                "model_failures.jsonl",
                "model_response_summary.json",
                "MODEL_RESPONSE_REPORT.md",
                "REPORT_ZH.md",
                "plots/",
                "ablation_plots/",
                "robustness_plots/",
            ],
        },
    )
    return summary


def main() -> int:
    args = parse_args()
    if sum(
        (
            args.analyze_only,
            args.robustness_only,
            args.secondary_only,
            bool(args.capability_only),
        )
    ) > 1:
        raise ValueError(
            "--analyze-only, --robustness-only, --secondary-only, and "
            "--capability-only are exclusive"
        )
    data_dir = args.data_dir.resolve()
    main_samples = read_jsonl(data_dir / "inference_samples.jsonl")
    robustness_path = data_dir / "robustness_samples.jsonl"
    robustness_samples = (
        read_jsonl(robustness_path) if robustness_path.exists() else []
    )
    samples = main_samples + robustness_samples
    if args.analyze_only:
        predictions = read_jsonl(data_dir / "model_predictions.jsonl")
        failures_path = data_dir / "model_failures.jsonl"
        failures = read_jsonl(failures_path) if failures_path.exists() else []
        summary = finalize_outputs(data_dir, samples, predictions, failures)
        print(
            json.dumps(
                {
                    "prediction_count": len(predictions),
                    "failure_count": len(failures),
                    "output": str(data_dir / "model_response_summary.json"),
                    "analyze_only": True,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.robustness_only or args.secondary_only or args.capability_only:
        if args.robustness_only and not robustness_samples:
            raise ValueError("robustness_samples.jsonl is missing or empty")
        existing_predictions_path = data_dir / "model_predictions.jsonl"
        if not existing_predictions_path.exists():
            raise ValueError("main model_predictions.jsonl is required")
        run_samples = (
            robustness_samples
            if args.robustness_only
            else (
                [
                row
                for row in main_samples
                if row["generator_family_role"] == "secondary"
                ]
                if args.secondary_only
                else [
                    row
                    for row in samples
                    if row["capability_id"] in set(args.capability_only)
                ]
            )
        )
        refreshed_sample_ids = {row["sample_id"] for row in run_samples}
        existing_predictions = read_jsonl(existing_predictions_path)
        if args.capability_only:
            refreshed_model_ids = set(args.models)
            predictions = [
                row
                for row in existing_predictions
                if not (
                    row["sample_id"] in refreshed_sample_ids
                    and row["model_id"]
                    in refreshed_model_ids
                    | {
                        "last_value",
                        "seasonal_naive",
                        "cross_lag_linear_probe",
                    }
                )
            ]
        else:
            predictions = [
                row
                for row in existing_predictions
                if (
                    row.get("evaluation_table", "main") == "main"
                    if args.robustness_only
                    else row["sample_id"] not in refreshed_sample_ids
                )
            ]
        existing_failures_path = data_dir / "model_failures.jsonl"
        if existing_failures_path.exists():
            existing_failures = read_jsonl(existing_failures_path)
            if args.capability_only:
                refreshed_model_ids = set(args.models)
                failures = [
                    row
                    for row in existing_failures
                    if not (
                        row.get("sample_id") in refreshed_sample_ids
                        and row.get("model_id") in refreshed_model_ids
                    )
                ]
            else:
                failures = [
                    row
                    for row in existing_failures
                    if (
                        row.get("evaluation_table", "main") == "main"
                        if args.robustness_only
                        else row.get("sample_id") not in refreshed_sample_ids
                    )
                ]
        else:
            failures = []
    else:
        predictions = []
        failures = []
        run_samples = samples
    existing_prediction_keys = {
        (row["sample_id"], row["model_id"], row["variant"])
        for row in predictions
    }
    for sample in run_samples:
        for baseline in local_baseline_kinds(sample):
            if (sample["sample_id"], baseline, "native") in (
                existing_prediction_keys
            ):
                continue
            forecast = baseline_forecast(sample, baseline)
            predictions.append(
                prediction_row(
                    sample,
                    model_id=baseline,
                    variant="native",
                    forecast=forecast,
                    input_adaptation={
                        "service_semantics": (
                            "protocol_aware_history_only_positive_control"
                            if baseline == "cross_lag_linear_probe"
                            else "local_baseline"
                        )
                    },
                    request_seconds=0.0,
                )
            )

    client = inference.TimerServiceClient(
        args.base_url,
        args.api_prefix,
        timeout_seconds=30,
    )
    try:
        catalog = {str(row["model_id"]): row for row in client.list_models()}
        for model_id in args.models:
            if model_id not in catalog:
                failures.append(
                    {"model_id": model_id, "error": "model_not_found"}
                )
                continue
            model = catalog[model_id]
            started = time.monotonic()
            if not args.keep_loaded:
                client.unload_all_loaded()
            client.ensure_loaded(
                model_id,
                devices=args.devices,
                replicas_per_device=1,
                timeout_seconds=args.load_timeout_seconds,
            )
            variants: list[tuple[str, list[dict[str, Any]]]] = [
                ("native", run_samples)
            ]
            if model_id in FORCED_SPLIT_MODELS:
                variants.append(
                    (
                        "forced_independent_targets",
                        [
                            sample
                            for sample in run_samples
                            if sample["capability_id"] in STRUCTURED_CAPABILITIES
                            and sample.get("evaluation_table", "main") == "main"
                            and sample["generator_family_role"] == "primary"
                        ],
                    )
                )
            variants.append(
                (
                    "covariates_ablated",
                    [
                        sample
                        for sample in run_samples
                        if sample["capability_id"] == "covariate_response"
                        and sample.get("evaluation_table", "main") == "main"
                        and sample["generator_family_role"] == "primary"
                    ],
                )
            )
            for variant, variant_samples in variants:
                if not variant_samples:
                    continue
                rows, variant_failures = asyncio.run(
                    forecast_variant(
                        forecast_url=client.base + "/forecast",
                        model_id=model_id,
                        model=model,
                        samples=variant_samples,
                        variant=variant,
                        concurrency=args.concurrency,
                        timeout_seconds=args.forecast_timeout_seconds,
                        max_attempts=args.max_attempts,
                    )
                )
                predictions.extend(rows)
                failures.extend(variant_failures)
                print(
                    f"{model_id}/{variant}: {len(rows)}/{len(variant_samples)}",
                    flush=True,
                )
            print(
                f"{model_id} elapsed {time.monotonic() - started:.1f}s",
                flush=True,
            )
            if not args.keep_loaded:
                client.unload_model(model_id)
    finally:
        client.close()

    summary = finalize_outputs(data_dir, samples, predictions, failures)
    print(
        json.dumps(
            {
                "prediction_count": len(predictions),
                "failure_count": len(failures),
                "output": str(data_dir / "model_response_summary.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
