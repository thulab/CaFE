#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cafe.analysis import structured
from cafe import protocol
from cafe.analysis import metrics
from cafe.analysis import diagnostics


DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"
FIXED_CONTEXT_LENGTH = protocol.FIXED_CONTEXT_LENGTH
FIXED_CONTEXT_POLICY = f"fixed_l{FIXED_CONTEXT_LENGTH}"
HIERARCHY_COHERENCE_PENALTY_WEIGHT = 1.0
PRIMARY_MECHANISM_METRIC = {
    "trend": "trend_curvature_component_nrmse",
    "multi_seasonal": "seasonal_spectral_amplitude_relative_error",
    "time_varying_seasonality": "instantaneous_frequency_nmae",
    "regime_switching": "regime_jump_nmae",
    "nonlinear_persistence": "nonlinear_recurrence_residual_nrmse",
    "predictable_intermittency": "event_window_nmae",
    "common_factor": "counterfactual_effect_nrmse",
    "hierarchical_coherence": "hierarchy_structure_nmae",
    "cross_series_dependence": "counterfactual_effect_nrmse",
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
REAL_ANCHORED_BENCHMARK_TRACK = "real_anchored_counterfactual"
REAL_ANCHORED_COMPONENT_KEYS = (
    "real_anchored_counterfactuals",
    "real_anchored",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze formal CaFE "
            f"fixed-L{FIXED_CONTEXT_LENGTH}/oracle-context results."
        )
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-experiment-root",
        type=Path,
        default=None,
        help=(
            "Read immutable generation/inference inputs from another "
            "experiment while writing a new analysis-only experiment."
        ),
    )
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
        "--analysis-profile",
        choices=("full", "scores_only"),
        default="full",
        help=(
            "scores_only keeps model MASE and mechanism scores but omits "
            "reference baselines, structured controls, split-bank, and "
            "matched/utilization audits."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Timer-4.0",
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
        != protocol.file_sha256(task_manifest_path)
    ):
        raise ValueError("analysis task manifest hash mismatch")
    task_manifest = protocol.read_json(task_manifest_path)
    if task_manifest.get("schema_version") not in {
        "cafe.inference_task_manifest.v1",
        "cafe.inference_task_manifest.v2",
    }:
        raise ValueError("unsupported Paper-cafe inference task manifest")
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
        or str(component["sha256"]) != protocol.file_sha256(task_path)
    ):
        raise ValueError("synthetic analysis task hash mismatch")

    expected_rows = component.get("row_count")
    if expected_rows is None:
        raise ValueError(
            "synthetic analysis task component is missing row_count"
        )
    observed_rows = 0
    sample_ids: set[str] = set()
    for row in protocol.iter_jsonl(task_path):
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


def _real_anchored_component_record(
    container: dict[str, Any],
) -> dict[str, Any] | None:
    for collection_name in (
        "task_components",
        "generation_components",
        "components",
    ):
        collection = container.get(collection_name)
        if not isinstance(collection, dict):
            continue
        for key in REAL_ANCHORED_COMPONENT_KEYS:
            record = collection.get(key)
            if isinstance(record, dict):
                return record
    for key in (
        "real_anchored_generation_component",
        "real_anchored_source",
    ):
        record = container.get(key)
        if isinstance(record, dict):
            return record
    return None


def _real_anchored_generation_record(
    component: dict[str, Any],
    task_manifest: dict[str, Any],
    inference_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    for key in (
        "generation_component",
        "source_generation_component",
        "generation_source",
    ):
        record = component.get(key)
        if isinstance(record, dict):
            return record
    for container in (task_manifest, inference_manifest):
        for key in (
            "real_anchored_generation_component",
            "real_anchored_source",
        ):
            direct_record = container.get(key)
            if isinstance(direct_record, dict):
                return direct_record
        record = _real_anchored_component_record(container)
        if record is not None and record is not component:
            return record
        generation_files = container.get("generation_files")
        if isinstance(generation_files, dict):
            for key in REAL_ANCHORED_COMPONENT_KEYS:
                candidate = generation_files.get(key)
                if isinstance(candidate, dict):
                    return candidate
        if isinstance(generation_files, list):
            for candidate in generation_files:
                if not isinstance(candidate, dict):
                    continue
                identity = " ".join(
                    str(candidate.get(name, ""))
                    for name in ("name", "component", "benchmark_track", "path")
                ).lower()
                if "real_anchored" in identity:
                    return candidate
    return None


def _validated_component_jsonl(
    record: dict[str, Any],
    *,
    label: str,
) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if (
        record.get("bytes") is None
        or int(record["bytes"]) != path.stat().st_size
    ):
        raise ValueError(f"{label} byte-size mismatch")
    if (
        not record.get("sha256")
        or str(record["sha256"]) != protocol.file_sha256(path)
    ):
        raise ValueError(f"{label} hash mismatch")
    expected_rows = record.get("row_count")
    if expected_rows is None:
        raise ValueError(f"{label} is missing row_count")
    observed_rows = sum(1 for _ in protocol.iter_jsonl(path))
    if observed_rows != int(expected_rows):
        raise ValueError(
            f"{label} row-count mismatch: {observed_rows} != {expected_rows}"
        )
    return path


def validated_optional_real_anchored_task_path(
    inference_dir: Path,
    inference_manifest: dict[str, Any],
    task_manifest: dict[str, Any],
) -> Path | None:
    """Resolve the independently ranked real-anchored task when present.

    The component is optional for backwards compatibility. Once a task
    component is declared, however, its immutable generation source must also
    be manifest-bound; silently treating arbitrary forecast rows as a real
    anchored benchmark would defeat the track's provenance guarantee.
    """

    task_components = task_manifest.get("task_components")
    if not isinstance(task_components, dict):
        return None
    component = next(
        (
            task_components[key]
            for key in REAL_ANCHORED_COMPONENT_KEYS
            if isinstance(task_components.get(key), dict)
        ),
        None,
    )
    if component is None:
        return None
    generation_record = _real_anchored_generation_record(
        component,
        task_manifest,
        inference_manifest,
    )
    if generation_record is None:
        raise ValueError(
            "real-anchored task component is not bound to a generation "
            "component"
        )
    _validated_component_jsonl(
        generation_record,
        label="real-anchored generation component",
    )
    task_path = _validated_component_jsonl(
        component,
        label="real-anchored inference task component",
    )
    try:
        task_path.resolve().relative_to(inference_dir.resolve())
    except ValueError as error:
        raise ValueError(
            "real-anchored inference task component is outside inference dir"
        ) from error

    expected_manifest_count = task_manifest.get(
        "real_anchored_view_count"
    )
    if (
        expected_manifest_count is not None
        and int(expected_manifest_count) != int(component["row_count"])
    ):
        raise ValueError(
            "real-anchored task count disagrees with task manifest"
        )

    sample_ids: set[str] = set()
    pair_members: dict[str, dict[int, float]] = defaultdict(dict)
    seed_backgrounds: dict[tuple[str, int], str] = {}
    background_seeds: dict[tuple[str, str], int] = {}
    fixed_context_present = False
    for row in protocol.iter_jsonl(task_path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in sample_ids:
            raise ValueError(
                f"duplicate or empty real-anchored sample_id: {sample_id}"
            )
        sample_ids.add(sample_id)
        if row.get("benchmark_track") != REAL_ANCHORED_BENCHMARK_TRACK:
            raise ValueError(
                "real-anchored task row lost its benchmark_track identity"
            )
        if row.get("evaluation_table") != REAL_ANCHORED_BENCHMARK_TRACK:
            raise ValueError(
                "real-anchored task row has an invalid evaluation_table"
            )
        context = int(row["context_length"])
        fixed_context_present |= context == FIXED_CONTEXT_LENGTH
        target = np.asarray(row["target"], dtype=float)
        if target.ndim != 2 or not np.all(np.isfinite(target)):
            raise ValueError("real-anchored task target must be finite 2D")
        member = int(row.get("counterfactual_member", -1))
        pair_id = row.get("counterfactual_pair_id")
        if pair_id is None or member not in {0, 1}:
            raise ValueError(
                "real-anchored task rows require paired members 0 and 1"
            )
        mase_scale = float(row["mase_scale"])
        if not math.isfinite(mase_scale) or mase_scale <= 0.0:
            raise ValueError("real-anchored task has an invalid MASE scale")
        members = pair_members[str(pair_id)]
        if member in members:
            raise ValueError(
                f"duplicate real-anchored pair member: {pair_id}/{member}"
            )
        members[member] = mase_scale
        capability_id = str(row["capability_id"])
        seed_index = int(row["seed_index"])
        background_id = str(row.get("background_id", ""))
        if not background_id:
            raise ValueError(
                "real-anchored task row is missing background_id"
            )
        seed_key = (capability_id, seed_index)
        prior_background = seed_backgrounds.setdefault(
            seed_key,
            background_id,
        )
        if prior_background != background_id:
            raise ValueError(
                "real-anchored seed maps to multiple backgrounds: "
                f"{capability_id}/{seed_index}"
            )
        background_key = (capability_id, background_id)
        prior_seed = background_seeds.setdefault(
            background_key,
            seed_index,
        )
        if prior_seed != seed_index:
            raise ValueError(
                "real-anchored background was recycled across seeds: "
                f"{capability_id}/{background_id}"
            )
    for pair_id, members in pair_members.items():
        if set(members) != {0, 1}:
            raise ValueError(f"incomplete real-anchored pair: {pair_id}")
        if not math.isclose(
            members[0], members[1], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"real-anchored pair does not share baseline MASE: {pair_id}"
            )
    if sample_ids and not fixed_context_present:
        raise ValueError(
            f"real-anchored task is missing fixed L{FIXED_CONTEXT_LENGTH} views"
        )
    return task_path


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
    metrics = diagnostics.prediction_metrics(sample, forecast)
    if str(sample["capability_id"]) == "hierarchical_coherence":
        hierarchy_score = hierarchy_structure_nmae(metrics)
        if hierarchy_score is not None:
            metrics["hierarchy_structure_nmae"] = hierarchy_score
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    mae = float(np.mean(np.abs(target[context:] - forecast)))
    metrics["mae"] = mae
    metrics["mase"] = mae / float(sample["mase_scale"])
    return {
        "schema_version": "cafe.prediction_metrics.v1",
        "benchmark_track": sample.get("benchmark_track"),
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
        "background_id": sample.get("background_id"),
        "dose_value": sample.get("dose_value"),
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


def hierarchy_structure_nmae(
    metrics: dict[str, Any],
) -> float | None:
    """Combine child allocation recovery with native additivity violation."""

    contrast = metrics.get("child_contrast_nmae")
    coherence = metrics.get("coherence_nmae")
    if contrast is None or coherence is None:
        return None
    values = np.asarray([contrast, coherence], dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return float(
        values[0]
        + HIERARCHY_COHERENCE_PENALTY_WEIGHT * values[1]
    )


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

    New samples declare the exact counterfactual support. Older cafe artifacts
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
        diagnostics.safe_corr(truth_effect, forecast_effect),
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
    first_member = int(first_sample.get("counterfactual_member", 0))
    second_member = int(second_sample.get("counterfactual_member", 1))
    if (first_member, second_member) != (0, 1):
        raise ValueError(
            "counterfactual effect requires baseline member 0 followed by "
            "treatment member 1"
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
        "schema_version": "cafe.counterfactual_effect.v1",
        "benchmark_track": first_sample.get("benchmark_track"),
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
        "background_id": first_sample.get("background_id"),
        "dose_value": second_sample.get("dose_value"),
        "counterfactual_effect_nrmse": nrmse,
        "effect_correlation": correlation,
        "effect_amplitude_ratio": amplitude_ratio,
        "truth_effect_rms": truth_rms,
        "forecast_effect_rms": forecast_rms,
    }
    if (
        first_sample.get("benchmark_track")
        == REAL_ANCHORED_BENCHMARK_TRACK
    ):
        first_background = str(first_sample.get("background_id", ""))
        second_background = str(second_sample.get("background_id", ""))
        if (
            not first_background
            or first_background != second_background
        ):
            raise ValueError(
                "real-anchored counterfactual members must share one "
                "background"
            )
        first_scale = float(first_sample["mase_scale"])
        second_scale = float(second_sample["mase_scale"])
        if not math.isclose(
            first_scale,
            second_scale,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "real-anchored counterfactual members must share the "
                "baseline MASE scale"
            )
        row.update(
            {
                "shared_baseline_mase_scale": first_scale,
                "effect_mae_shared_baseline_mase": float(
                    np.mean(np.abs(forecast_effect - truth_effect))
                    / first_scale
                ),
                "effect_orientation": "treatment_member_1_minus_baseline_member_0",
            }
        )
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
    metadata = first_sample.get("generation_metadata") or {}
    direct_steps = min(
        max(
            int(
                metadata.get(
                    "counterfactual_direct_driver_steps",
                    metadata.get("cross_lag_steps", active),
                )
            ),
            1,
        ),
        int(first_sample["horizon"]),
    )
    (
        direct_nrmse,
        direct_correlation,
        direct_amplitude,
        direct_truth_rms,
        direct_forecast_rms,
    ) = _effect_metrics(
        truth_effect[:direct_steps],
        forecast_effect[:direct_steps],
    )
    persistent_truth = truth_effect[direct_steps:]
    persistent_forecast = forecast_effect[direct_steps:]
    persistent_metrics = (
        _effect_metrics(persistent_truth, persistent_forecast)
        if persistent_truth.size
        else (None, None, None, None, None)
    )
    row.update(
        {
            "direct_driver_prefix_steps": direct_steps,
            "direct_effect_nrmse": direct_nrmse,
            "direct_effect_correlation": direct_correlation,
            "direct_effect_amplitude_ratio": direct_amplitude,
            "direct_truth_effect_rms": direct_truth_rms,
            "direct_forecast_effect_rms": direct_forecast_rms,
            "persistent_tail_steps": int(
                first_sample["horizon"] - direct_steps
            ),
            "persistent_tail_effect_nrmse": persistent_metrics[0],
            "persistent_tail_effect_correlation": persistent_metrics[1],
            "persistent_tail_effect_amplitude_ratio": (
                persistent_metrics[2]
            ),
            "persistent_tail_truth_effect_rms": persistent_metrics[3],
            "persistent_tail_forecast_effect_rms": persistent_metrics[4],
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
        for context in protocol.VIEW_CONTEXT_LENGTHS:
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
            strict_metric_prefix = "counterfactual_effect"
            strict_nrmse = _median(
                row[f"{strict_metric_prefix}_nrmse"]
                for row in structure_effects
            )
            strict_correlation = _median(
                row[
                    (
                        "effect_correlation"
                    )
                ]
                for row in structure_effects
            )
            strict_amplitude = _median(
                row[
                    (
                        "effect_amplitude_ratio"
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
                        "cafe.structured_context_assessment.v1"
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
                    "strict_metric_scope": "full_horizon",
                    "zero_tail_leakage_nrmse_median": zero_tail_leakage,
                    "strict_effect_evaluable": strict_evaluable,
                    "strict_effect_passed": strict_passed,
                    "strict_effect_assessment": (
                        "evaluated_as_full_horizon_shared_fit_hard_gate"
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
    for sample in protocol.iter_jsonl(task_path):
        if not structured.is_structured_sample(sample):
            continue
        if int(sample["context_length"]) not in protocol.VIEW_CONTEXT_LENGTHS:
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
        "schema_version": "cafe.structured_positive_controls.v1",
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
            "context_lengths": list(protocol.VIEW_CONTEXT_LENGTHS),
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
            "legacy_maximum_zero_tail_leakage_nrmse": 0.10,
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
            for row in protocol.iter_jsonl(prediction_path)
        }
        if prediction_path is not None
        else {}
    )
    metrics: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    pending_pairs: dict[
        str, dict[int, tuple[dict[str, Any], np.ndarray]]
    ] = defaultdict(dict)
    missing = 0
    for sample in protocol.iter_jsonl(task_path):
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
        member_index = int(member)
        if member_index not in {0, 1}:
            raise ValueError(
                f"unsupported counterfactual member {member_index}"
            )
        if member_index in pending_pairs[key]:
            raise ValueError(
                f"duplicate counterfactual pair member: {key}/{member_index}"
            )
        pending_pairs[key][member_index] = (sample, forecast)
        if set(pending_pairs[key]) == {0, 1}:
            complete_pair = pending_pairs.pop(key)
            first = complete_pair[0]
            second = complete_pair[1]
            effects.append(
                effect_row(
                    first[0],
                    first[1],
                    second[0],
                    second[1],
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
        strict_effect_metric = "counterfactual_effect_nrmse"
        metric_name = (
            strict_effect_metric
            if (
                key[2] == "strict_counterfactual_audit"
                and capability
                in {"common_factor", "cross_series_dependence"}
            )
            else PRIMARY_MECHANISM_METRIC[capability]
        )
        effect_mechanism_values = [
            float(row[metric_name])
            for row in effect_groups.get(key, [])
            if int(row["intensity"]) == 5 and metric_name in row
        ]
        if effect_mechanism_values:
            mechanism_values = effect_mechanism_values
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


def _real_anchored_background_groups(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    """Validate seed/background bijection and return authentic units."""

    by_background: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seed_to_background: dict[int, str] = {}
    background_to_seed: dict[str, int] = {}
    for row in rows:
        background_id = str(row.get("background_id", ""))
        if not background_id:
            raise ValueError(
                f"real-anchored {label} row is missing background_id"
            )
        seed_index = int(row["seed_index"])
        prior_background = seed_to_background.setdefault(
            seed_index,
            background_id,
        )
        if prior_background != background_id:
            raise ValueError(
                f"real-anchored {label} seed maps to multiple backgrounds: "
                f"{seed_index}"
            )
        prior_seed = background_to_seed.setdefault(
            background_id,
            seed_index,
        )
        if prior_seed != seed_index:
            raise ValueError(
                f"real-anchored {label} background was reused by seeds "
                f"{prior_seed} and {seed_index}: {background_id}"
            )
        by_background[background_id].append(row)
    return dict(by_background)


def _real_anchored_background_mean_metric(
    groups: dict[str, list[dict[str, Any]]],
    metric_name: str,
) -> float | None:
    """Count a repeated pair baseline once within each real background."""

    background_values: list[float] = []
    for background_id, rows in groups.items():
        expected_doses = {int(row["intensity"]) for row in rows}
        seen_members: set[tuple[int, int]] = set()
        baseline_values: list[float] = []
        treatment_by_dose: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            member = int(row["counterfactual_member"])
            dose = int(row["intensity"])
            member_key = (dose, member)
            if member_key in seen_members:
                raise ValueError(
                    "duplicate real-anchored metric member within background: "
                    f"{background_id}/dose={dose}/member={member}"
                )
            seen_members.add(member_key)
            value = row.get("metrics", {}).get(metric_name)
            if value is None or not math.isfinite(float(value)):
                continue
            if member == 0:
                baseline_values.append(float(value))
            elif member == 1:
                treatment_by_dose[dose].append(float(value))
            else:
                raise ValueError(
                    f"invalid real-anchored counterfactual member: {member}"
                )
        if (
            len(baseline_values) != len(expected_doses)
            or set(treatment_by_dose) != expected_doses
            or any(
                len(values) != 1
                for values in treatment_by_dose.values()
            )
        ):
            raise ValueError(
                "real-anchored background metric is incomplete across "
                f"baseline/doses: {background_id}/{metric_name}"
            )
        # The unmodified member is serialized once per dose for pairing, but
        # it is one forecast path and receives one weight, not D weights.
        path_values = [float(np.mean(baseline_values))]
        path_values.extend(
            float(np.mean(treatment_by_dose[dose]))
            for dose in sorted(treatment_by_dose)
        )
        background_values.append(float(np.mean(path_values)))
    return (
        float(np.mean(background_values))
        if background_values
        else None
    )


def real_anchored_score_table(
    rows: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score the real-anchored track without entering synthetic rankings."""

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    effect_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        if row.get("benchmark_track") != REAL_ANCHORED_BENCHMARK_TRACK:
            raise ValueError("foreign metric row in real-anchored scoring")
        if int(row["context_length"]) != FIXED_CONTEXT_LENGTH:
            raise ValueError("real-anchored scores are fixed-L168 only")
        key = (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["evaluation_table"]),
            str(row["generator_family_role"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )
        groups[key].append(row)
    for row in effects:
        if row.get("benchmark_track") != REAL_ANCHORED_BENCHMARK_TRACK:
            raise ValueError("foreign effect row in real-anchored scoring")
        if int(row["context_length"]) != FIXED_CONTEXT_LENGTH:
            raise ValueError("real-anchored effects are fixed-L168 only")
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
        paired_effects = effect_groups.get(key, [])
        metric_backgrounds = _real_anchored_background_groups(
            group,
            label="metric",
        )
        intensity_values = sorted(
            {int(row["intensity"]) for row in group}
        )
        expected_members = {
            (intensity, member)
            for intensity in intensity_values
            for member in (0, 1)
        }
        for background_id, background_rows in metric_backgrounds.items():
            observed_members = {
                (
                    int(row["intensity"]),
                    int(row["counterfactual_member"]),
                )
                for row in background_rows
            }
            if observed_members != expected_members:
                raise ValueError(
                    "real-anchored background has incomplete dose/member "
                    f"coverage: {background_id}"
                )
        effect_backgrounds = _real_anchored_background_groups(
            paired_effects,
            label="effect",
        )
        if set(metric_backgrounds) != set(effect_backgrounds):
            raise ValueError(
                "real-anchored metric/effect background coverage mismatch: "
                + "/".join(key)
            )
        maximum_dose = (
            max(int(row["intensity"]) for row in paired_effects)
            if paired_effects
            else None
        )
        maximum_dose_effects = [
            row
            for row in paired_effects
            if maximum_dose is not None
            and int(row["intensity"]) == maximum_dose
        ]
        maximum_effects_by_background = _real_anchored_background_groups(
            maximum_dose_effects,
            label="maximum-dose effect",
        )
        if set(maximum_effects_by_background) != set(metric_backgrounds):
            raise ValueError(
                "real-anchored maximum-dose effect is incomplete by "
                "background: " + "/".join(key)
            )
        mechanism_values = []
        for background_id, background_effects in (
            maximum_effects_by_background.items()
        ):
            if len(background_effects) != 1:
                raise ValueError(
                    "real-anchored maximum dose has duplicate effects for "
                    f"background {background_id}"
                )
            value = float(
                background_effects[0]["counterfactual_effect_nrmse"]
            )
            if math.isfinite(value):
                mechanism_values.append(value)
        if len(mechanism_values) != len(maximum_effects_by_background):
            raise ValueError(
                "real-anchored maximum-dose mechanism metric is incomplete "
                "by background: " + "/".join(key)
            )
        shared_scales = {
            float(row["shared_baseline_mase_scale"])
            for row in maximum_dose_effects
            if row.get("shared_baseline_mase_scale") is not None
        }
        output.append(
            {
                "schema_version": "cafe.real_anchored_score.v1",
                "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
                "dataset_id": key[0],
                "context_policy": key[1],
                "evaluation_table": key[2],
                "generator_family_role": key[3],
                "capability_id": key[4],
                "model_id": key[5],
                "accuracy_score": _real_anchored_background_mean_metric(
                    metric_backgrounds,
                    "mase",
                ),
                "accuracy_metric": "mase",
                "history_std_normalized_mae": (
                    _real_anchored_background_mean_metric(
                        metric_backgrounds,
                        "normalized_mae_history_std",
                    )
                ),
                "accuracy_statistical_unit": "authentic_real_background",
                "accuracy_path_weighting": (
                    "unmodified_baseline_once_plus_each_treatment_dose_"
                    "once_within_background_then_equal_background_mean"
                ),
                "effective_background_count": len(metric_backgrounds),
                "effective_background_ids_sha256": protocol.json_sha256(
                    sorted(metric_backgrounds)
                ),
                "background_sampling_policy": (
                    "without_replacement_seed_background_bijection"
                ),
                "serialized_metric_row_count": len(group),
                "unique_forecast_path_count": len(metric_backgrounds)
                * (1 + len(intensity_values)),
                "mechanism_statistical_unit": (
                    "authentic_real_background_at_maximum_dose"
                ),
                "mechanism_metric": "counterfactual_effect_nrmse",
                "mechanism_score": (
                    float(np.mean(mechanism_values))
                    if mechanism_values
                    else None
                ),
                "mechanism_intensity": maximum_dose,
                "mechanism_dose_policy": "maximum_available_intervention_dose",
                "mechanism_pair_count": len(maximum_dose_effects),
                "mechanism_background_count": len(
                    maximum_effects_by_background
                ),
                "mechanism_seed_count": len(
                    {
                        int(row["seed_index"])
                        for row in maximum_dose_effects
                    }
                ),
                "mase_scale_policy": (
                    "shared_unmodified_real_background_history_by_pair"
                ),
                "shared_mase_scale_count": len(shared_scales),
                "seed_count": len(
                    {int(row["seed_index"]) for row in group}
                ),
                "intensities": sorted(
                    intensity_values
                ),
                "ranking_scope": (
                    "real_anchored_counterfactual_only_never_synthetic"
                ),
                "is_reference_baseline": key[5] in BASELINES,
            }
        )
    add_ranks(output)
    return output


def analyze_real_anchored_track(
    task_path: Path,
    *,
    model_ids: list[str],
    inference_dir: Path,
) -> dict[str, Any]:
    """Analyze a validated real-anchored task at the fixed L168 policy."""

    all_metrics: list[dict[str, Any]] = []
    all_effects: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for model_id in model_ids:
        prediction_path = (
            inference_dir
            / "model_shards"
            / metrics.safe_filename(model_id)
            / "predictions"
            / f"{metrics.safe_filename(model_id)}.jsonl"
            if model_id not in BASELINES
            else None
        )
        model_metrics, model_effects, missing = analyze_one_model(
            task_path,
            model_id=model_id,
            prediction_path=prediction_path,
        )
        if model_id not in BASELINES and missing:
            raise ValueError(
                "real-anchored model predictions are incomplete: "
                f"{model_id} missing {missing} task(s)"
            )
        all_metrics.extend(model_metrics)
        all_effects.extend(model_effects)
        model_backgrounds_by_capability: dict[str, set[str]] = defaultdict(
            set
        )
        for row in model_metrics:
            background_id = str(row.get("background_id", ""))
            if background_id:
                model_backgrounds_by_capability[
                    str(row["capability_id"])
                ].add(background_id)
        coverage.append(
            {
                "model_id": model_id,
                "metric_row_count": len(model_metrics),
                "effect_row_count": len(model_effects),
                "missing_prediction_count": missing,
                "effective_background_count_by_capability": {
                    capability_id: len(background_ids)
                    for capability_id, background_ids in sorted(
                        model_backgrounds_by_capability.items()
                    )
                },
            }
        )
    selected_metrics = [
        {**row, "context_policy": FIXED_CONTEXT_POLICY}
        for row in all_metrics
        if int(row["context_length"]) == FIXED_CONTEXT_LENGTH
    ]
    selected_effects = [
        {**row, "context_policy": FIXED_CONTEXT_POLICY}
        for row in all_effects
        if int(row["context_length"]) == FIXED_CONTEXT_LENGTH
    ]
    background_ids_by_capability: dict[str, set[str]] = defaultdict(set)
    for row in selected_metrics:
        background_id = str(row.get("background_id", ""))
        if background_id:
            background_ids_by_capability[
                str(row["capability_id"])
            ].add(background_id)
    effective_backgrounds = {
        capability_id: {
            "count": len(background_ids),
            "ids_sha256": protocol.json_sha256(sorted(background_ids)),
            "statistical_unit": "authentic_real_background",
            "sampling_policy": "without_replacement_seed_background_bijection",
        }
        for capability_id, background_ids in sorted(
            background_ids_by_capability.items()
        )
    }
    return {
        "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
        "context_policy": FIXED_CONTEXT_POLICY,
        "prediction_metrics": selected_metrics,
        "counterfactual_effects": selected_effects,
        "effective_backgrounds_by_capability": effective_backgrounds,
        "scores": real_anchored_score_table(
            selected_metrics,
            selected_effects,
        ),
        "coverage": coverage,
    }


def write_real_anchored_analysis(
    analysis_dir: Path,
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metric_path = analysis_dir / "real_anchored_prediction_metrics.jsonl"
    effect_path = (
        analysis_dir / "real_anchored_counterfactual_effects.jsonl"
    )
    score_path = analysis_dir / "real_anchored_scores.json"
    metric_count = protocol.write_jsonl(
        metric_path,
        result["prediction_metrics"],
    )
    effect_count = protocol.write_jsonl(
        effect_path,
        result["counterfactual_effects"],
    )
    protocol.write_json(
        score_path,
        {
            "schema_version": "cafe.real_anchored_scores.v1",
            "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
            "ranking_scope": (
                "independent_from_synthetic_capability_scores_and_ranks"
            ),
            "context_policy": FIXED_CONTEXT_POLICY,
            "statistical_unit": "authentic_real_background",
            "accuracy_path_weighting": (
                "unmodified_baseline_once_plus_each_treatment_dose_once_"
                "within_background_then_equal_background_mean"
            ),
            "effective_backgrounds_by_capability": result.get(
                "effective_backgrounds_by_capability",
                {},
            ),
            "mechanism_metric": "counterfactual_effect_nrmse",
            "mechanism_dose_policy": "maximum_available_intervention_dose",
            "scores": result["scores"],
        },
    )
    return {
        "prediction_metrics": {
            **protocol.file_record(metric_path),
            "row_count": metric_count,
        },
        "counterfactual_effects": {
            **protocol.file_record(effect_path),
            "row_count": effect_count,
        },
        "scores": {
            **protocol.file_record(score_path),
            "row_count": len(result["scores"]),
        },
    }


def complete_effect_level_mechanism_scores(
    scores: list[dict[str, Any]],
    effects: Iterable[dict[str, Any]],
) -> None:
    """Complete legacy missing mechanism scores from manifest-bound effects."""

    grouped_effects: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in effects:
        grouped_effects[
            (
                str(row["dataset_id"]),
                str(row["context_policy"]),
                str(row["evaluation_table"]),
                str(row["generator_family_role"]),
                str(row["capability_id"]),
                str(row["model_id"]),
            )
        ].append(row)
    completed = False
    for row in scores:
        if row.get("mechanism_score") is not None:
            continue
        metric_name = str(row.get("mechanism_metric") or "")
        if not metric_name:
            continue
        key = (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["evaluation_table"]),
            str(row["generator_family_role"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )
        values = [
            float(effect[metric_name])
            for effect in grouped_effects.get(key, [])
            if int(effect["intensity"]) == 5 and metric_name in effect
        ]
        if not values:
            continue
        row["mechanism_score"] = float(np.mean(values))
        completed = True
    if completed:
        add_ranks(scores)


def promote_counterfactual_primary_mechanism_scores(
    scores: list[dict[str, Any]],
) -> None:
    """Bind all-seed I5 pair effects to the factual main-table rows."""

    capabilities = protocol.PRIMARY_MECHANISM_COUNTERFACTUAL_CAPABILITIES
    sources: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in scores:
        if (
            row["evaluation_table"] != "strict_counterfactual_audit"
            or row["generator_family_role"] != "primary"
            or row["capability_id"] not in capabilities
        ):
            continue
        key = (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )
        if key in sources:
            raise ValueError(
                "duplicate primary mechanism counterfactual score: "
                + "/".join(key)
            )
        sources[key] = row

    promoted = False
    for row in scores:
        if (
            row["evaluation_table"] != "main"
            or row["generator_family_role"] != "primary"
            or row["capability_id"] not in capabilities
        ):
            continue
        key = (
            str(row["dataset_id"]),
            str(row["context_policy"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )
        source = sources.get(key)
        if source is None or source.get("mechanism_score") is None:
            raise ValueError(
                "missing all-seed I5 primary mechanism pair score: "
                + "/".join(key)
            )
        accuracy_seed_count = int(row["seed_count"])
        mechanism_seed_count = int(source["seed_count"])
        if mechanism_seed_count != accuracy_seed_count:
            raise ValueError(
                "primary mechanism pair coverage must match factual seed "
                f"coverage for {'/'.join(key)}: "
                f"{mechanism_seed_count} != {accuracy_seed_count}"
            )
        row.update(
            {
                "mechanism_metric": source["mechanism_metric"],
                "mechanism_score": float(source["mechanism_score"]),
                "mechanism_seed_count": mechanism_seed_count,
                "mechanism_intensities": list(source["intensities"]),
                "mechanism_evaluation_table": (
                    "strict_counterfactual_audit"
                ),
                "mechanism_pairing_policy": (
                    "all_formal_seeds_i5_counterfactual_pair"
                ),
            }
        )
        promoted = True
    if promoted:
        add_ranks(scores)


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
        strict_nrmse_name = "counterfactual_effect_nrmse"
        strict_correlation_name = "effect_correlation"
        strict_amplitude_name = "effect_amplitude_ratio"
        output.append(
            {
                "schema_version": (
                    "cafe.multivariate_utilization_audit.v1"
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
                "counterfactual_effect_is_primary_mechanism_score": True,
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
                "strict_metric_scope": "full_horizon",
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
    mechanism_probe_rows = [
        row
        for row in selected_rows
        if row["evaluation_table"] == "strict_counterfactual_audit"
        and row["generator_family_role"] == "primary"
        and row["capability_id"]
        in protocol.PRIMARY_MECHANISM_COUNTERFACTUAL_CAPABILITIES
        and row["model_id"] in models
    ]
    official_effects = [
        row
        for row in selected_effects
        if row["evaluation_table"]
        in {"main", "strict_counterfactual_audit"}
        and row["generator_family_role"] == "primary"
        and row["model_id"] in models
    ]
    present_capabilities = [
        capability
        for capability in protocol.CAPABILITIES
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
                [*official_rows, *mechanism_probe_rows],
                official_effects,
                seed_filter=set(range(start, stop)),
            )
            promote_counterfactual_primary_mechanism_scores(scores)
            scores = [
                row
                for row in scores
                if row["evaluation_table"] == "main"
                and row["generator_family_role"] == "primary"
            ]
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
                            tau = metrics.kendall_tau_b(
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
        "# CaFE 单数据集全链路测试",
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
        for capability in protocol.CAPABILITIES
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
        "# CaFE matched sensitivity / robustness 审计",
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
        "# CaFE 多变量利用审计",
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
        or str(record["sha256"]) != protocol.file_sha256(path)
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
                        "cafe.experiment_capability_score.v1"
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
        f"# CaFE {context_policy} 跨数据集能力表",
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
        for capability in protocol.CAPABILITIES
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


def experiment_real_anchored_capability_rows(
    scores: Iterable[dict[str, Any]],
    *,
    dataset_ids: list[str],
    models: list[str],
    capabilities: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Macro-average the optional real-anchored track independently.

    A capability uses only datasets where that capability is present. Model
    comparisons use the intersection of complete foundation-model rows across
    those datasets, then recompute both dataset and aggregate ranks within that
    common set. Reference baselines remain dataset diagnostics and never enter
    this experiment-level table.
    """

    selected: list[dict[str, Any]] = []
    allowed_datasets = set(dataset_ids)
    allowed_capabilities = set(capabilities)
    allowed_models = set(models)
    for row in scores:
        if row.get("benchmark_track") != REAL_ANCHORED_BENCHMARK_TRACK:
            raise ValueError(
                "foreign score row in experiment real-anchored aggregation"
            )
        if row.get("context_policy") != FIXED_CONTEXT_POLICY:
            raise ValueError(
                "experiment real-anchored aggregation is fixed-L168 only"
            )
        if row.get("evaluation_table") != REAL_ANCHORED_BENCHMARK_TRACK:
            raise ValueError("invalid real-anchored evaluation table")
        dataset_id = str(row.get("dataset_id"))
        capability_id = str(row.get("capability_id"))
        model_id = str(row.get("model_id"))
        if dataset_id not in allowed_datasets:
            raise ValueError(
                f"unexpected real-anchored dataset: {dataset_id}"
            )
        if capability_id not in allowed_capabilities:
            raise ValueError(
                f"unexpected real-anchored capability: {capability_id}"
            )
        if model_id not in allowed_models:
            if bool(row.get("is_reference_baseline")) or model_id in BASELINES:
                continue
            raise ValueError(
                f"unexpected real-anchored model: {model_id}"
            )
        selected.append(row)

    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selected:
        key = (
            str(row["dataset_id"]),
            str(row["capability_id"]),
            str(row["model_id"]),
        )
        if key in by_key:
            raise ValueError(
                "duplicate experiment real-anchored score: "
                + "/".join(key)
            )
        by_key[key] = row

    capability_order = [
        capability_id
        for capability_id in capabilities
        if any(key[1] == capability_id for key in by_key)
    ]
    output: list[dict[str, Any]] = []
    capability_summaries: list[dict[str, Any]] = []
    for capability_id in capability_order:
        supported_datasets = [
            dataset_id
            for dataset_id in dataset_ids
            if any(
                key[0] == dataset_id and key[1] == capability_id
                for key in by_key
            )
        ]
        common_models = [
            model_id
            for model_id in models
            if all(
                (
                    (row := by_key.get(
                        (dataset_id, capability_id, model_id)
                    ))
                    is not None
                    and row.get("accuracy_score") is not None
                    and row.get("mechanism_score") is not None
                    and math.isfinite(float(row["accuracy_score"]))
                    and math.isfinite(float(row["mechanism_score"]))
                )
                for dataset_id in supported_datasets
            )
        ]
        effective_background_count_by_dataset: dict[str, int] = {}
        effective_background_ids_sha256_by_dataset: dict[str, str] = {}
        for dataset_id in supported_datasets:
            coverage_rows = [
                by_key[(dataset_id, capability_id, model_id)]
                for model_id in common_models
            ]
            if not coverage_rows:
                continue
            counts = {
                int(row["effective_background_count"])
                for row in coverage_rows
                if row.get("effective_background_count") is not None
            }
            hashes = {
                str(row["effective_background_ids_sha256"])
                for row in coverage_rows
                if row.get("effective_background_ids_sha256")
            }
            if len(counts) != 1 or len(hashes) != 1:
                raise ValueError(
                    "real-anchored model rows disagree on effective "
                    f"background coverage: {dataset_id}/{capability_id}"
                )
            background_count = next(iter(counts))
            if background_count <= 0:
                raise ValueError(
                    "real-anchored effective background count must be "
                    f"positive: {dataset_id}/{capability_id}"
                )
            for row in coverage_rows:
                if int(row.get("seed_count", -1)) != background_count:
                    raise ValueError(
                        "real-anchored seed count is not the authentic "
                        "background count: "
                        f"{dataset_id}/{capability_id}/{row['model_id']}"
                    )
                if (
                    int(row.get("mechanism_background_count", -1))
                    != background_count
                ):
                    raise ValueError(
                        "real-anchored mechanism coverage is not complete "
                        "by background: "
                        f"{dataset_id}/{capability_id}/{row['model_id']}"
                    )
            effective_background_count_by_dataset[dataset_id] = (
                background_count
            )
            effective_background_ids_sha256_by_dataset[dataset_id] = (
                next(iter(hashes))
            )
        capability_summary = {
            "capability_id": capability_id,
            "dataset_ids": supported_datasets,
            "dataset_count": len(supported_datasets),
            "common_models": common_models,
            "common_model_count": len(common_models),
            "effective_background_count_by_dataset": (
                effective_background_count_by_dataset
            ),
            "effective_background_ids_sha256_by_dataset": (
                effective_background_ids_sha256_by_dataset
            ),
            "total_authentic_background_units": sum(
                effective_background_count_by_dataset.values()
            ),
            "dataset_weighting": "equal_dataset_macro_mean",
            "within_dataset_statistical_unit": (
                "equal_authentic_real_background"
            ),
            "model_set_policy": (
                "intersection_of_complete_requested_model_rows_across_"
                "present_datasets"
            ),
            "status": (
                "aggregated" if common_models else "no_common_complete_models"
            ),
        }
        capability_summaries.append(capability_summary)
        if not common_models:
            continue

        dataset_ranks: dict[
            tuple[str, str], dict[str, int]
        ] = {}
        for dataset_id in supported_datasets:
            rows_by_model = {
                model_id: by_key[(dataset_id, capability_id, model_id)]
                for model_id in common_models
            }
            for score_name, rank_kind in (
                ("accuracy_score", "accuracy"),
                ("mechanism_score", "mechanism"),
            ):
                values = {
                    model_id: float(row[score_name])
                    for model_id, row in rows_by_model.items()
                }
                dataset_ranks[(dataset_id, rank_kind)] = {
                    model_id: 1
                    + sum(other < value for other in values.values())
                    for model_id, value in values.items()
                }

        capability_rows: list[dict[str, Any]] = []
        for model_id in common_models:
            model_rows = [
                by_key[(dataset_id, capability_id, model_id)]
                for dataset_id in supported_datasets
            ]
            accuracy_values = [
                float(row["accuracy_score"]) for row in model_rows
            ]
            mechanism_values = [
                float(row["mechanism_score"]) for row in model_rows
            ]
            normalized_values = [
                float(row["history_std_normalized_mae"])
                for row in model_rows
                if row.get("history_std_normalized_mae") is not None
                and math.isfinite(
                    float(row["history_std_normalized_mae"])
                )
            ]
            accuracy_ranks = [
                dataset_ranks[(dataset_id, "accuracy")][model_id]
                for dataset_id in supported_datasets
            ]
            mechanism_ranks = [
                dataset_ranks[(dataset_id, "mechanism")][model_id]
                for dataset_id in supported_datasets
            ]
            capability_rows.append(
                {
                    "schema_version": (
                        "cafe.experiment_real_anchored_capability_score.v1"
                    ),
                    "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
                    "ranking_scope": (
                        "real_anchored_counterfactual_only_never_synthetic"
                    ),
                    "context_policy": FIXED_CONTEXT_POLICY,
                    "capability_id": capability_id,
                    "model_id": model_id,
                    "dataset_count": len(supported_datasets),
                    "dataset_ids": supported_datasets,
                    "common_model_ids": common_models,
                    "effective_background_count_by_dataset": (
                        effective_background_count_by_dataset
                    ),
                    "effective_background_ids_sha256_by_dataset": (
                        effective_background_ids_sha256_by_dataset
                    ),
                    "total_authentic_background_units": sum(
                        effective_background_count_by_dataset.values()
                    ),
                    "macro_mean_accuracy_score": float(
                        np.mean(accuracy_values)
                    ),
                    "accuracy_metric": "mase",
                    "macro_mean_history_std_normalized_mae": (
                        float(np.mean(normalized_values))
                        if len(normalized_values) == len(model_rows)
                        else None
                    ),
                    "mean_dataset_accuracy_rank": float(
                        np.mean(accuracy_ranks)
                    ),
                    "accuracy_dataset_wins": sum(
                        rank == 1 for rank in accuracy_ranks
                    ),
                    "macro_mean_mechanism_score": float(
                        np.mean(mechanism_values)
                    ),
                    "mechanism_metric": "counterfactual_effect_nrmse",
                    "mean_dataset_mechanism_rank": float(
                        np.mean(mechanism_ranks)
                    ),
                    "mechanism_dataset_wins": sum(
                        rank == 1 for rank in mechanism_ranks
                    ),
                    "aggregation_policy": "equal_dataset_macro_mean",
                    "model_set_policy": (
                        "common_complete_model_intersection"
                    ),
                }
            )
        for score_name, rank_name in (
            ("macro_mean_accuracy_score", "accuracy_rank"),
            ("macro_mean_mechanism_score", "mechanism_rank"),
        ):
            values = {
                str(row["model_id"]): float(row[score_name])
                for row in capability_rows
            }
            for row in capability_rows:
                value = values[str(row["model_id"])]
                row[rank_name] = 1 + sum(
                    other < value for other in values.values()
                )
        output.extend(capability_rows)

    summary = {
        "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
        "status": (
            "aggregated"
            if output
            else (
                "no_common_complete_models"
                if selected
                else "component_absent"
            )
        ),
        "included_in_synthetic_scores_or_ranks": False,
        "context_policy": FIXED_CONTEXT_POLICY,
        "dataset_ids": [
            dataset_id
            for dataset_id in dataset_ids
            if any(row["dataset_id"] == dataset_id for row in selected)
        ],
        "capabilities": capability_summaries,
        "score_count": len(output),
    }
    return output, summary


def reusable_experiment_analysis_manifest(
    analysis_dir: Path,
    *,
    stage_contract_path: Path,
    dataset_ids: list[str],
    models: list[str],
    capabilities: list[str],
    analysis_profile: str,
) -> bool:
    manifest_path = analysis_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = protocol.read_json(manifest_path)
        if manifest.get("schema_version") not in {
            "cafe.experiment_analysis_manifest.v1",
            "cafe.experiment_analysis_manifest.v2",
        }:
            return False
        if str(manifest.get("stage_contract_sha256")) != (
            protocol.file_sha256(stage_contract_path)
        ):
            return False
        if list(manifest.get("datasets") or []) != dataset_ids:
            return False
        if list(manifest.get("models") or []) != models:
            return False
        if list(manifest.get("capabilities") or []) != capabilities:
            return False
        if manifest.get("analysis_profile") != analysis_profile:
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
                    protocol.file_sha256(path)
                )
            ):
                return False
            dataset_manifest = protocol.read_json(path)
            score_record = dataset_manifest.get("files", {}).get("scores")
            if not isinstance(score_record, dict):
                return False
            score_path = validated_file_record(score_record)
            if str(row.get("scores_sha256")) != protocol.file_sha256(score_path):
                return False
            real_score_record = dataset_manifest.get("files", {}).get(
                "real_anchored_scores"
            )
            recorded_real_sha256 = row.get(
                "real_anchored_scores_sha256"
            )
            if real_score_record is None:
                if recorded_real_sha256 is not None:
                    return False
            else:
                if not isinstance(real_score_record, dict):
                    return False
                real_score_path = validated_file_record(real_score_record)
                if (
                    recorded_real_sha256 is None
                    or str(recorded_real_sha256)
                    != protocol.file_sha256(real_score_path)
                ):
                    return False
            generation_path = Path(
                str(row.get("generation_manifest_path", ""))
            )
            if (
                not generation_path.is_file()
                or str(row.get("generation_manifest_sha256"))
                != protocol.file_sha256(generation_path)
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
    source_experiment_root = (
        args.source_experiment_root.resolve()
        if args.source_experiment_root is not None
        else experiment_root
    )
    stage_contract_path = (
        experiment_root / "stage_contracts" / "analysis.json"
    )
    analysis_contract = protocol.read_json(stage_contract_path)
    experiment_record = protocol.read_json(experiment_root / "experiment.json")
    analysis_config = analysis_contract.get("config")
    if not isinstance(analysis_config, dict):
        raise ValueError("analysis stage contract is missing config")
    dataset_ids = [str(value) for value in analysis_config["dataset_ids"]]
    models = [str(value) for value in analysis_config["models"]]
    capabilities = [
        str(value) for value in analysis_config["capabilities"]
    ]
    analysis_profile = str(analysis_config.get("analysis_profile", "full"))
    if args.analysis_profile != analysis_profile:
        raise ValueError(
            "aggregate analysis profile must exactly match the experiment "
            "protocol"
        )
    if list(args.models) != models:
        raise ValueError(
            "aggregate models must exactly match the experiment protocol"
        )
    if (
        int(analysis_config["seed_start"]) != args.seed_start
        or int(analysis_config["seed_count"]) != args.seed_count
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
        stage_contract_path=stage_contract_path,
        dataset_ids=dataset_ids,
        models=models,
        capabilities=capabilities,
        analysis_profile=analysis_profile,
    ):
        if not args.reuse_existing_aggregate:
            raise FileExistsError(
                "immutable experiment-level analysis already exists; "
                "pass --reuse-existing-aggregate to validate and reuse it"
            )
        print(
            protocol.canonical_json(
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
    all_effects: list[dict[str, Any]] = []
    all_real_anchored_scores: list[dict[str, Any]] = []
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
        dataset_manifest = protocol.read_json(dataset_manifest_path)
        if dataset_manifest.get("schema_version") not in {
            "cafe.analysis_manifest.v1",
            "cafe.analysis_manifest.v2",
        }:
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
        if dataset_manifest.get("analysis_profile") != analysis_profile:
            raise ValueError(
                f"dataset analysis profile mismatch: {dataset_id}"
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
        score_payload = protocol.read_json(score_path)
        scores = score_payload.get("scores")
        if not isinstance(scores, list):
            raise ValueError(f"invalid dataset scores payload: {dataset_id}")
        all_scores.extend(scores)
        effect_record = dataset_manifest.get("files", {}).get(
            "counterfactual_effects"
        )
        if not isinstance(effect_record, dict):
            raise ValueError(
                f"dataset analysis is missing effects record: {dataset_id}"
            )
        effect_path = validated_file_record(
            effect_record,
            expected_path=(
                dataset_analysis_dir / "counterfactual_effects.jsonl"
            ),
        )
        all_effects.extend(protocol.iter_jsonl(effect_path))
        real_score_record = dataset_manifest.get("files", {}).get(
            "real_anchored_scores"
        )
        real_score_path: Path | None = None
        dataset_real_scores: list[dict[str, Any]] = []
        if real_score_record is not None:
            if dataset_manifest.get("schema_version") != (
                "cafe.analysis_manifest.v2"
            ):
                raise ValueError(
                    "real-anchored score records require dataset analysis "
                    f"manifest v2: {dataset_id}"
                )
            if not isinstance(real_score_record, dict):
                raise ValueError(
                    f"invalid real-anchored score record: {dataset_id}"
                )
            real_score_path = validated_file_record(
                real_score_record,
                expected_path=(
                    dataset_analysis_dir / "real_anchored_scores.json"
                ),
            )
            real_score_payload = protocol.read_json(real_score_path)
            if real_score_payload.get("schema_version") != (
                "cafe.real_anchored_scores.v1"
            ):
                raise ValueError(
                    f"unsupported real-anchored scores payload: {dataset_id}"
                )
            if real_score_payload.get("benchmark_track") != (
                REAL_ANCHORED_BENCHMARK_TRACK
            ):
                raise ValueError(
                    f"real-anchored score track mismatch: {dataset_id}"
                )
            raw_real_scores = real_score_payload.get("scores")
            if not isinstance(raw_real_scores, list):
                raise ValueError(
                    f"invalid real-anchored scores payload: {dataset_id}"
                )
            dataset_real_scores = [
                dict(row) for row in raw_real_scores
                if isinstance(row, dict)
            ]
            if len(dataset_real_scores) != len(raw_real_scores):
                raise ValueError(
                    f"non-object real-anchored score row: {dataset_id}"
                )
            if (
                real_score_record.get("row_count") is not None
                and int(real_score_record["row_count"])
                != len(dataset_real_scores)
            ):
                raise ValueError(
                    f"real-anchored score row-count mismatch: {dataset_id}"
                )
            for row in dataset_real_scores:
                if str(row.get("dataset_id")) != dataset_id:
                    raise ValueError(
                        "real-anchored dataset score binding mismatch: "
                        f"{dataset_id}"
                    )
            all_real_anchored_scores.extend(dataset_real_scores)
        generation_manifest_path = (
            source_experiment_root
            / dataset_id
            / "02_generation"
            / f"manifest__{shard_name}.json"
        )
        generation_manifest = protocol.read_json(generation_manifest_path)
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
                "analysis_manifest_sha256": protocol.file_sha256(
                    dataset_manifest_path
                ),
                "scores_sha256": protocol.file_sha256(score_path),
                "counterfactual_effects_sha256": protocol.file_sha256(effect_path),
                "real_anchored_scores_path": (
                    None
                    if real_score_path is None
                    else str(real_score_path)
                ),
                "real_anchored_scores_sha256": (
                    None
                    if real_score_path is None
                    else protocol.file_sha256(real_score_path)
                ),
                "real_anchored_score_count": len(dataset_real_scores),
                "generation_manifest_path": str(
                    generation_manifest_path
                ),
                "generation_manifest_sha256": protocol.file_sha256(
                    generation_manifest_path
                ),
                "generated_capabilities": generated_capabilities,
            }
        )

    complete_effect_level_mechanism_scores(all_scores, all_effects)
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
    real_anchored_rows, real_anchored_summary = (
        experiment_real_anchored_capability_rows(
            all_real_anchored_scores,
            dataset_ids=dataset_ids,
            models=models,
            capabilities=capabilities,
        )
    )
    real_anchored_component_dataset_ids = [
        str(row["dataset_id"])
        for row in input_records
        if row.get("real_anchored_scores_path") is not None
    ]
    real_anchored_summary["component_dataset_ids"] = (
        real_anchored_component_dataset_ids
    )
    if (
        real_anchored_component_dataset_ids
        and real_anchored_summary["status"] == "component_absent"
    ):
        real_anchored_summary["status"] = "component_present_no_scores"
    fixed_path = analysis_dir / "capability_scores_fixed_l168.json"
    oracle_path = analysis_dir / "capability_scores_oracle_context.json"
    real_anchored_path = (
        analysis_dir
        / "capability_scores_real_anchored_fixed_l168.json"
    )
    fixed_report_path = analysis_dir / "REPORT_FIXED_L168_ZH.md"
    oracle_report_path = analysis_dir / "REPORT_ORACLE_CONTEXT_ZH.md"
    protocol.write_json(fixed_path, {"scores": fixed_rows})
    protocol.write_json(oracle_path, {"scores": oracle_rows})
    if real_anchored_component_dataset_ids:
        protocol.write_json(
            real_anchored_path,
            {
                "schema_version": (
                    "cafe.experiment_real_anchored_capability_scores.v1"
                ),
                "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
                "context_policy": FIXED_CONTEXT_POLICY,
                "ranking_scope": (
                    "independent_from_synthetic_fixed_and_oracle_rankings"
                ),
                "aggregation_policy": (
                    "equal_dataset_macro_mean_over_capability_present_"
                    "datasets_with_common_complete_model_intersection"
                ),
                "summary": real_anchored_summary,
                "scores": real_anchored_rows,
            },
        )
    fixed_report_path.write_text(
        render_experiment_capability_report(
            fixed_rows,
            experiment_id=str(experiment_record["experiment_id"]),
            context_policy=FIXED_CONTEXT_POLICY,
        ),
        encoding="utf-8",
    )
    oracle_report_path.write_text(
        render_experiment_capability_report(
            oracle_rows,
            experiment_id=str(experiment_record["experiment_id"]),
            context_policy="oracle_context",
        ),
        encoding="utf-8",
    )
    experiment_files = {
        "fixed_scores": protocol.file_record(fixed_path),
        "oracle_scores": protocol.file_record(oracle_path),
        "fixed_report": protocol.file_record(fixed_report_path),
        "oracle_report": protocol.file_record(oracle_report_path),
    }
    if real_anchored_component_dataset_ids:
        experiment_files["real_anchored_fixed_scores"] = (
            protocol.file_record(real_anchored_path)
        )
    manifest = {
        "schema_version": "cafe.experiment_analysis_manifest.v2",
        "created_at": protocol.utc_now(),
        "experiment_id": str(experiment_record["experiment_id"]),
        "stage_contract_sha256": protocol.file_sha256(
            stage_contract_path
        ),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "datasets": dataset_ids,
        "models": models,
        "capabilities": capabilities,
        "analysis_profile": analysis_profile,
        "capability_dataset_ids": capability_dataset_ids,
        "context_policies": [FIXED_CONTEXT_POLICY, "oracle_context"],
        "aggregation_policy": (
            "equal_supported_dataset_macro_mean_with_mean_within_dataset_"
            "model_rank"
        ),
        "mechanism_score_completion_policy": (
            "complete_missing_effect_level_scores_from_manifest_bound_"
            "counterfactual_effects_without_mutating_dataset_analysis"
        ),
        "oracle_selection_policy": (
            "per_model_master_sample_minimum_mase_over_l96_l168_l336;"
            "counterfactual_pairs_share_context"
        ),
        "real_anchored_counterfactual": real_anchored_summary,
        "inputs": input_records,
        "files": experiment_files,
    }
    protocol.write_json(analysis_dir / "analysis_manifest.json", manifest)
    print(
        protocol.canonical_json(
            {
                "analysis_status": "computed",
                "fixed_score_count": len(fixed_rows),
                "oracle_score_count": len(oracle_rows),
                "real_anchored_score_count": len(real_anchored_rows),
                "output": str(analysis_dir),
            }
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.aggregate_experiment:
        return aggregate_experiment_analysis(args)
    dataset = protocol.resolve_dataset(args.dataset_id)
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    experiment_root = args.output_root.resolve()
    source_experiment_root = (
        args.source_experiment_root.resolve()
        if args.source_experiment_root is not None
        else experiment_root
    )
    inference_dir = (
        source_experiment_root
        / dataset.dataset_id
        / "03_inference"
        / shard_name
    )
    inference_manifest = protocol.read_json(
        inference_dir / "inference_manifest.json"
    )
    if not inference_manifest["complete"]:
        raise ValueError("inference manifest is incomplete")
    task_path, task_manifest = validated_synthetic_task_path(
        inference_dir,
        inference_manifest,
    )
    real_anchored_task_path = validated_optional_real_anchored_task_path(
        inference_dir,
        inference_manifest,
        task_manifest,
    )
    all_metrics: list[dict[str, Any]] = []
    all_effects: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    analysis_models = (
        list(args.models)
        if args.analysis_profile == "scores_only"
        else [*args.models, *BASELINES]
    )
    real_anchored_result = (
        analyze_real_anchored_track(
            real_anchored_task_path,
            model_ids=analysis_models,
            inference_dir=inference_dir,
        )
        if real_anchored_task_path is not None
        else None
    )
    for model_id in analysis_models:
        prediction_path = (
            inference_dir
            / "model_shards"
            / metrics.safe_filename(model_id)
            / "predictions"
            / f"{metrics.safe_filename(model_id)}.jsonl"
            if model_id not in BASELINES
            else None
        )
        metric_rows, effects, missing = analyze_one_model(
            task_path,
            model_id=model_id,
            prediction_path=prediction_path,
        )
        all_metrics.extend(metric_rows)
        all_effects.extend(effects)
        coverage.append(
            {
                "model_id": model_id,
                "metric_row_count": len(metric_rows),
                "effect_row_count": len(effects),
                "missing_prediction_count": missing,
            }
        )
        print(
            f"analyzed {model_id}: metrics={len(metric_rows)}, "
            f"effects={len(effects)}, missing={missing}",
            flush=True,
        )
    selected_metrics, pair_context = selected_context_rows(all_metrics)
    selected_effects = selected_effect_rows(all_effects, pair_context)
    scores = score_table(selected_metrics, selected_effects)
    promote_counterfactual_primary_mechanism_scores(scores)
    if args.analysis_profile == "scores_only":
        matched_comparisons = []
        utilization_audit = []
        split_rows = []
    else:
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
        experiment_root
        / dataset.dataset_id
        / "04_analysis"
        / shard_name
    )
    real_anchored_files = (
        write_real_anchored_analysis(analysis_dir, real_anchored_result)
        if real_anchored_result is not None
        else {}
    )
    structured_controls = (
        {
            "schema_version": (
                "cafe.structured_positive_controls_skipped.v1"
            ),
            "dataset_id": dataset.dataset_id,
            "status": "not_requested",
            "analysis_profile": args.analysis_profile,
        }
        if args.analysis_profile == "scores_only"
        else analyze_structured_positive_controls(
            task_path,
            dataset_id=dataset.dataset_id,
        )
    )
    structured_controls["created_at"] = protocol.utc_now()
    metric_path = analysis_dir / "prediction_metrics.jsonl"
    effect_path = analysis_dir / "counterfactual_effects.jsonl"
    score_path = analysis_dir / "scores.json"
    split_path = analysis_dir / "split_bank.json"
    matched_path = analysis_dir / "matched_comparisons.json"
    utilization_path = (
        analysis_dir / "multivariate_utilization_audit.json"
    )
    structured_path = analysis_dir / "structured_positive_controls.json"
    protocol.write_jsonl(metric_path, selected_metrics)
    protocol.write_jsonl(effect_path, selected_effects)
    protocol.write_json(score_path, {"scores": scores})
    protocol.write_json(split_path, {"split_bank": split_rows})
    protocol.write_json(
        matched_path,
        {"matched_comparisons": matched_comparisons},
    )
    protocol.write_json(
        utilization_path,
        {
            "schema_version": (
                "cafe.multivariate_utilization_audit_bundle.v1"
            ),
            "counterfactual_effect_is_primary_mechanism_score": True,
            "rows": utilization_audit,
        },
    )
    protocol.write_json(structured_path, structured_controls)
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
    analysis_files = {
        "prediction_metrics": protocol.file_record(metric_path),
        "counterfactual_effects": protocol.file_record(effect_path),
        "scores": protocol.file_record(score_path),
        "split_bank": protocol.file_record(split_path),
        "matched_comparisons": protocol.file_record(matched_path),
        "multivariate_utilization_audit": protocol.file_record(
            utilization_path
        ),
        "structured_positive_controls": protocol.file_record(
            structured_path
        ),
        "report": protocol.file_record(report_path),
        "matched_report": protocol.file_record(matched_report_path),
        "multivariate_utilization_report": protocol.file_record(
            utilization_report_path
        ),
    }
    if real_anchored_files:
        analysis_files.update(
            {
                "real_anchored_prediction_metrics": (
                    real_anchored_files["prediction_metrics"]
                ),
                "real_anchored_counterfactual_effects": (
                    real_anchored_files["counterfactual_effects"]
                ),
                "real_anchored_scores": real_anchored_files["scores"],
            }
        )
    manifest = {
        "schema_version": "cafe.analysis_manifest.v2",
        "created_at": protocol.utc_now(),
        "dataset_id": dataset.dataset_id,
        "source_experiment_root": str(source_experiment_root),
        "source_inference_manifest_path": str(
            inference_dir / "inference_manifest.json"
        ),
        "inference_manifest_sha256": protocol.file_sha256(
            inference_dir / "inference_manifest.json"
        ),
        "models": list(args.models),
        "analysis_profile": args.analysis_profile,
        "omitted_analyses": (
            [
                "reference_baselines",
                "structured_positive_controls",
                "split_bank",
                "matched_comparisons",
                "multivariate_utilization_audit",
            ]
            if args.analysis_profile == "scores_only"
            else []
        ),
        "coverage": coverage,
        "real_anchored_counterfactual": {
            "status": (
                "analyzed"
                if real_anchored_result is not None
                else "component_absent"
            ),
            "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
            "included_in_synthetic_scores_or_ranks": False,
            "context_policy": FIXED_CONTEXT_POLICY,
            "coverage": (
                []
                if real_anchored_result is None
                else real_anchored_result["coverage"]
            ),
            "effective_backgrounds_by_capability": (
                {}
                if real_anchored_result is None
                else real_anchored_result[
                    "effective_backgrounds_by_capability"
                ]
            ),
            "statistical_unit": "authentic_real_background",
        },
        "context_policies": [FIXED_CONTEXT_POLICY, "oracle_context"],
        "files": analysis_files,
    }
    protocol.write_json(analysis_dir / "analysis_manifest.json", manifest)
    print(
        protocol.canonical_json(
            {
                "score_count": len(scores),
                "real_anchored_score_count": (
                    0
                    if real_anchored_result is None
                    else len(real_anchored_result["scores"])
                ),
                "split_bank_count": len(split_rows),
                "output": str(analysis_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
