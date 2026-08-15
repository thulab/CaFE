from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.gift_eval import (
    GIFT_EVAL_ADAPTER_SCHEMA,
    GIFT_EVAL_SOURCE_REVISION,
    GiftEvalInstance,
    gift_eval_asset_path,
    iter_gift_eval_instances,
)
from cafe.benchmark_extension.mechanisms import (
    CAPABILITY_IDS,
    CapabilityGroup,
    CapabilityTreatment,
    build_capability_group,
)


PIPELINE_SCHEMA = "cafe.pipeline.v6"
GENERATION_SCHEMA = "cafe.benchmark_extension_generation.v1"
SAMPLE_SCHEMA = "cafe.benchmark_extension_sample.v1"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend official GIFT-Eval instances with capability treatments."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument("--term", choices=("short", "medium", "long"), default="short")
    parser.add_argument("--augmentation-seed", type=int, default=2026081601)
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=CAPABILITY_IDS,
        default=list(CAPABILITY_IDS),
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Non-formal source-order prefix for smoke tests.",
    )
    return parser.parse_args()


def _target_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _mase_scale(history: np.ndarray, period: int) -> tuple[float, list[float]]:
    values = np.asarray(history, dtype=float)
    lag = min(max(1, int(period)), max(1, values.shape[0] - 1))
    differences = np.abs(values[lag:] - values[:-lag])
    by_target = np.mean(differences, axis=0) if differences.size else np.ones(values.shape[1])
    fallback = np.mean(np.abs(np.diff(values, axis=0)), axis=0)
    by_target = np.where(np.isfinite(by_target) & (by_target > 1e-8), by_target, fallback)
    by_target = np.where(np.isfinite(by_target) & (by_target > 1e-8), by_target, 1.0)
    return float(np.mean(by_target)), [float(value) for value in by_target]


def _season_length(frequency: str) -> int:
    raw = str(frequency)
    if raw.endswith(("H", "h")):
        return 24
    if raw.endswith("D"):
        return 7
    if raw.endswith(("M", "ME")):
        return 12
    if raw.endswith("W") or raw.startswith("W-"):
        return 52
    return 1


def _baseline_row(instance: GiftEvalInstance) -> dict[str, Any]:
    target = np.vstack((instance.history, instance.future))
    covariates = np.vstack(
        (instance.history_covariates, instance.future_covariates)
    )
    season = _season_length(instance.frequency)
    mase, mase_by_target = _mase_scale(instance.history, season)
    return {
        "schema_version": SAMPLE_SCHEMA,
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "benchmark_track": "gift_eval_capability_extension",
        "evaluation_table": "gift_eval_official_baseline",
        "sample_id": f"{instance.official_instance_id}__baseline",
        "official_instance_id": instance.official_instance_id,
        "baseline_sample_id": None,
        "counterfactual_pair_id": None,
        "counterfactual_member": 0,
        "dataset_id": instance.dataset_id,
        "config_id": instance.config_id,
        "item_id": instance.item_id,
        "window_index": instance.window_index,
        "window_count": instance.window_count,
        "forecast_origin": instance.forecast_origin,
        "source_target_length": instance.source_target_length,
        "capability_id": None,
        "capability_level": 0,
        "augmentation_seed": None,
        "context_length": instance.context_length,
        "horizon": instance.prediction_length,
        "target_dim": instance.target_dim,
        "target_column_names": list(instance.target_column_names),
        "covariate_dim": int(covariates.shape[1]),
        "covariate_column_names": list(instance.covariate_column_names),
        "covariates": covariates.tolist() if covariates.shape[1] else None,
        "frequency": instance.frequency,
        "term": instance.term,
        "season_length": season,
        "target": target.tolist(),
        "future_observed_mask": instance.future_observed_mask.tolist(),
        "history_imputation": instance.history_imputation,
        "mase_scale": mase,
        "mase_scale_by_target": mase_by_target,
        "mase_period": season,
        "target_sha256": _target_sha256(target),
        "history_sha256": _target_sha256(instance.history),
        "future_sha256": _target_sha256(instance.future),
        "scoring_target_semantics": "gift_eval_official_future",
        "input_history_semantics": "gift_eval_official_history_after_history_only_imputation",
        "included_in_capability_ranking": False,
    }


def _treatment_row(
    instance: GiftEvalInstance,
    group: CapabilityGroup,
    treatment: CapabilityTreatment,
    *,
    augmentation_seed: int,
) -> dict[str, Any]:
    baseline_id = f"{instance.official_instance_id}__baseline"
    pair_id = (
        f"{instance.official_instance_id}__{group.capability_id}__"
        f"level{treatment.level}__aug{augmentation_seed}"
    )
    history = instance.history + treatment.history_delta
    future = instance.future + treatment.future_delta
    stored_history_delta = history - instance.history
    stored_future_delta = future - instance.future
    target = np.vstack((history, future))
    covariates = np.vstack(
        (instance.history_covariates, instance.future_covariates)
    )
    season = _season_length(instance.frequency)
    mase, mase_by_target = _mase_scale(instance.history, season)
    parameter_draw = {
        "capability_level": treatment.level,
        "controlled_coordinate": treatment.controlled_coordinate,
        "coordinate_interval": list(treatment.coordinate_interval),
        "sampled_coordinate": treatment.sampled_coordinate,
        "applied_component_gain": treatment.applied_component_gain,
        "augmentation_seed": int(augmentation_seed),
    }
    return {
        "schema_version": SAMPLE_SCHEMA,
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "benchmark_track": "gift_eval_capability_extension",
        "evaluation_table": "gift_eval_capability_treatment",
        "sample_id": f"{pair_id}__treatment",
        "official_instance_id": instance.official_instance_id,
        "baseline_sample_id": baseline_id,
        "counterfactual_pair_id": pair_id,
        "counterfactual_member": 1,
        "dataset_id": instance.dataset_id,
        "config_id": instance.config_id,
        "item_id": instance.item_id,
        "window_index": instance.window_index,
        "window_count": instance.window_count,
        "forecast_origin": instance.forecast_origin,
        "source_target_length": instance.source_target_length,
        "capability_id": group.capability_id,
        "capability_level": treatment.level,
        "augmentation_seed": int(augmentation_seed),
        "controlled_coordinate": treatment.controlled_coordinate,
        "coordinate_interval": list(treatment.coordinate_interval),
        "sampled_coordinate": treatment.sampled_coordinate,
        "applied_component_gain": treatment.applied_component_gain,
        "parameter_draw_sha256": protocol.json_sha256(parameter_draw),
        "context_length": instance.context_length,
        "horizon": instance.prediction_length,
        "target_dim": instance.target_dim,
        "target_column_names": list(instance.target_column_names),
        "affected_target_indices": list(treatment.affected_target_indices),
        "covariate_dim": int(covariates.shape[1]),
        "covariate_column_names": list(instance.covariate_column_names),
        "covariates": covariates.tolist() if covariates.shape[1] else None,
        "frequency": instance.frequency,
        "term": instance.term,
        "season_length": season,
        "target": target.tolist(),
        "future_observed_mask": instance.future_observed_mask.tolist(),
        "history_imputation": instance.history_imputation,
        "mase_scale": mase,
        "mase_scale_by_target": mase_by_target,
        "mase_period": season,
        "target_sha256": _target_sha256(target),
        "history_sha256": _target_sha256(history),
        "future_sha256": _target_sha256(future),
        "source_history_sha256": _target_sha256(instance.history),
        "source_future_sha256": _target_sha256(instance.future),
        "history_delta_sha256": _target_sha256(stored_history_delta),
        "future_delta_sha256": _target_sha256(stored_future_delta),
        "source_distance_gate": treatment.source_distance_gate,
        "anti_copy_gate": {
            "policy": "treatment_to_authentic_source_distance_v1",
            "status": "accepted",
            "treatment_only": True,
        },
        "mechanism_metadata": treatment.metadata,
        "group_metadata": group.group_metadata,
        "scoring_target_semantics": (
            "gift_eval_official_future_plus_history_only_capability_delta"
        ),
        "input_history_semantics": (
            "entire_gift_eval_official_history_plus_capability_treatment"
        ),
        "included_in_capability_ranking": True,
    }


def _availability_row(
    instance: GiftEvalInstance,
    group: CapabilityGroup,
) -> dict[str, Any]:
    return {
        "schema_version": "cafe.instance_capability_availability.v1",
        "dataset_id": instance.dataset_id,
        "official_instance_id": instance.official_instance_id,
        "capability_id": group.capability_id,
        "available": group.available,
        "reason": group.reason,
        "generated_level_count": len(group.treatments),
        "context_length": instance.context_length,
        "horizon": instance.prediction_length,
        "target_dim": instance.target_dim,
        "group_metadata": group.group_metadata,
    }


def _atomic_handles(paths: Iterable[Path]) -> tuple[list[Any], list[Path]]:
    temporary = [path.with_suffix(path.suffix + ".tmp") for path in paths]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return [path.open("w", encoding="utf-8") for path in temporary], temporary


def _write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(protocol.canonical_json(row) + "\n")


def generate_dataset(
    dataset_id: str,
    *,
    gift_eval_dir: Path,
    dataset_root: Path,
    term: str,
    augmentation_seed: int,
    capability_ids: tuple[str, ...],
    max_instances: int | None,
) -> dict[str, Any]:
    generation_dir = dataset_root / "01_generation"
    baseline_path = generation_dir / "official_baselines.jsonl"
    treatment_path = generation_dir / "capability_treatments.jsonl"
    availability_path = generation_dir / "availability.jsonl"
    paths = (baseline_path, treatment_path, availability_path)
    handles, temporary = _atomic_handles(paths)
    baseline_count = treatment_count = availability_count = instance_count = 0
    available_counts = {capability: 0 for capability in capability_ids}
    try:
        baseline_handle, treatment_handle, availability_handle = handles
        for instance in iter_gift_eval_instances(
            dataset_id,
            gift_eval_dir,
            term=term,
            max_instances=max_instances,
        ):
            instance_count += 1
            _write_row(baseline_handle, _baseline_row(instance))
            baseline_count += 1
            for capability_id in capability_ids:
                group = build_capability_group(
                    instance,
                    capability_id,
                    augmentation_seed=augmentation_seed,
                )
                _write_row(availability_handle, _availability_row(instance, group))
                availability_count += 1
                if group.available:
                    available_counts[capability_id] += 1
                for treatment in group.treatments:
                    _write_row(
                        treatment_handle,
                        _treatment_row(
                            instance,
                            group,
                            treatment,
                            augmentation_seed=augmentation_seed,
                        ),
                    )
                    treatment_count += 1
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for temp, final in zip(temporary, paths, strict=True):
            os.replace(temp, final)
    except Exception:
        for handle in handles:
            if not handle.closed:
                handle.close()
        for path in temporary:
            path.unlink(missing_ok=True)
        raise
    source_path = gift_eval_asset_path(dataset_id, gift_eval_dir)
    source_files = [
        {**protocol.file_record(path), "path": str(path.resolve())}
        for path in sorted(source_path.glob("data-*.arrow"))
    ]
    config = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "adapter_schema_version": GIFT_EVAL_ADAPTER_SCHEMA,
        "gift_eval_split_source": GIFT_EVAL_SOURCE_REVISION,
        "dataset_id": dataset_id,
        "term": term,
        "augmentation_seed": int(augmentation_seed),
        "capability_ids": list(capability_ids),
        "max_instances": max_instances,
        "formal": max_instances is None,
        "instance_selection": (
            "all_official_test_instances"
            if max_instances is None
            else "nonformal_source_order_prefix"
        ),
        "native_target_policy": "preserve_gift_eval_target_dimension",
        "treatment_history_scope": "entire_official_input_history",
        "randomness_policy": (
            "counter_based_by_official_instance_capability_level_and_augmentation_seed"
        ),
        "source_distance_policy": (
            "treatment_only_multicontext_source_frozen_normalized_rms_v1"
        ),
    }
    manifest = {
        "schema_version": GENERATION_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": dataset_id,
        "config": config,
        "config_sha256": protocol.json_sha256(config),
        "source_files": source_files,
        "files": {
            "official_baselines": {
                **protocol.file_record(baseline_path),
                "row_count": baseline_count,
            },
            "capability_treatments": {
                **protocol.file_record(treatment_path),
                "row_count": treatment_count,
            },
            "availability": {
                **protocol.file_record(availability_path),
                "row_count": availability_count,
            },
        },
        "official_instance_count": instance_count,
        "available_instance_count_by_capability": available_counts,
        "treatment_count": treatment_count,
    }
    protocol.write_json(generation_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive")
    if len(args.capabilities) != len(set(args.capabilities)):
        raise ValueError("capabilities must be unique")
    dataset_root = args.output_root.resolve() / args.dataset_id
    manifest_path = dataset_root / "01_generation" / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"generation artifact already exists; use a new experiment root: {manifest_path}"
        )
    manifest = generate_dataset(
        args.dataset_id,
        gift_eval_dir=args.gift_eval_dir,
        dataset_root=dataset_root,
        term=args.term,
        augmentation_seed=args.augmentation_seed,
        capability_ids=tuple(args.capabilities),
        max_instances=args.max_instances,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
