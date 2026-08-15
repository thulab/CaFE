from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.generation import (
    GENERATION_SCHEMA,
    PIPELINE_SCHEMA,
    SAMPLE_SCHEMA,
)
from cafe.benchmark_extension.mechanisms import (
    CAPABILITY_LEVELS,
    INTERMITTENCY_GAP_INTERVALS,
    REGIME_RECENCY_INTERVALS,
    STRENGTH_INTERVALS,
    _distance_gate,
)


VALIDATION_SCHEMA = "cafe.benchmark_extension_validation.v1"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate native GIFT-Eval capability treatments."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _validate_file_record(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if protocol.file_sha256(path) != record["sha256"]:
        raise ValueError(f"generation file hash mismatch: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"generation file size mismatch: {path}")
    return path


def _target(row: dict[str, Any]) -> np.ndarray:
    values = np.asarray(row["target"], dtype=float)
    expected = (
        int(row["context_length"]) + int(row["horizon"]),
        int(row["target_dim"]),
    )
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid target shape/content for {row.get('sample_id')}")
    return values


def _covariates(row: dict[str, Any]) -> np.ndarray | None:
    dimension = int(row["covariate_dim"])
    raw = row.get("covariates")
    if dimension == 0:
        if raw is not None:
            raise ValueError("zero-dimensional covariates must be null")
        return None
    values = np.asarray(raw, dtype=float)
    expected = (
        int(row["context_length"]) + int(row["horizon"]),
        dimension,
    )
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError("invalid covariate shape/content")
    if len(row.get("covariate_column_names") or []) != dimension:
        raise ValueError("covariate column names do not match dimension")
    return values


def _float_equal(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def validate_generation(dataset_root: Path) -> dict[str, Any]:
    generation_dir = dataset_root / "01_generation"
    manifest_path = generation_dir / "manifest.json"
    manifest = protocol.read_json(manifest_path)
    failures: list[dict[str, Any]] = []
    if manifest.get("schema_version") != GENERATION_SCHEMA:
        failures.append({"scope": "manifest", "reason": "schema_version"})
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("pipeline_schema_version") != PIPELINE_SCHEMA:
        failures.append({"scope": "manifest", "reason": "pipeline_schema"})
    elif manifest.get("config_sha256") != protocol.json_sha256(config):
        failures.append({"scope": "manifest", "reason": "config_hash"})
    file_paths: dict[str, Path] = {}
    for key in ("official_baselines", "capability_treatments", "availability"):
        try:
            file_paths[key] = _validate_file_record(manifest["files"][key])
        except (KeyError, ValueError, FileNotFoundError) as error:
            failures.append({"scope": "manifest", "reason": f"{key}:{error}"})
    if failures:
        report = {
            "schema_version": VALIDATION_SCHEMA,
            "created_at": protocol.utc_now(),
            "dataset_id": manifest.get("dataset_id"),
            "generation_manifest_sha256": protocol.file_sha256(manifest_path),
            "accepted": False,
            "failures": failures,
        }
        validation_dir = dataset_root / "02_validation"
        protocol.write_json(validation_dir / "report.json", report)
        return report

    baselines: dict[str, dict[str, Any]] = {}
    for row in protocol.iter_jsonl(file_paths["official_baselines"]):
        try:
            if row.get("schema_version") != SAMPLE_SCHEMA:
                raise ValueError("sample_schema")
            if row.get("evaluation_table") != "gift_eval_official_baseline":
                raise ValueError("baseline_table")
            target = _target(row)
            _covariates(row)
            context = int(row["context_length"])
            if protocol.json_sha256(row.get("history_imputation")) is None:
                raise ValueError("history_imputation")
            if row["sample_id"] in baselines:
                raise ValueError("duplicate_baseline")
            if row.get("target_sha256") != _array_sha256(target):
                raise ValueError("baseline_target_hash")
            if row.get("history_sha256") != _array_sha256(target[:context]):
                raise ValueError("baseline_history_hash")
            baselines[str(row["sample_id"])] = row
        except (KeyError, TypeError, ValueError) as error:
            failures.append(
                {"scope": "baseline", "sample_id": row.get("sample_id"), "reason": str(error)}
            )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    treatment_count = 0
    for row in protocol.iter_jsonl(file_paths["capability_treatments"]):
        treatment_count += 1
        try:
            if row.get("schema_version") != SAMPLE_SCHEMA:
                raise ValueError("sample_schema")
            if row.get("evaluation_table") != "gift_eval_capability_treatment":
                raise ValueError("treatment_table")
            if row.get("counterfactual_member") != 1:
                raise ValueError("treatment_only_member")
            baseline = baselines[str(row["baseline_sample_id"])]
            if row["official_instance_id"] != baseline["official_instance_id"]:
                raise ValueError("official_instance_link")
            target = _target(row)
            source = _target(baseline)
            covariates = _covariates(row)
            source_covariates = _covariates(baseline)
            context = int(row["context_length"])
            if target.shape != source.shape:
                raise ValueError("pair_shape")
            if (covariates is None) != (source_covariates is None) or (
                covariates is not None
                and not np.array_equal(covariates, source_covariates)
            ):
                raise ValueError("pair_covariate_path_not_shared")
            if row.get("target_sha256") != _array_sha256(target):
                raise ValueError("treatment_target_hash")
            if row.get("source_history_sha256") != _array_sha256(source[:context]):
                raise ValueError("source_history_hash")
            if row.get("source_future_sha256") != _array_sha256(source[context:]):
                raise ValueError("source_future_hash")
            history_delta = target[:context] - source[:context]
            future_delta = target[context:] - source[context:]
            if row.get("history_delta_sha256") != _array_sha256(history_delta):
                raise ValueError("history_delta_hash")
            if row.get("future_delta_sha256") != _array_sha256(future_delta):
                raise ValueError("future_delta_hash")
            affected = tuple(int(value) for value in row["affected_target_indices"])
            expected_gate = _distance_gate(history_delta, source[:context], affected)
            observed_gate = row.get("source_distance_gate")
            if not isinstance(observed_gate, dict) or not observed_gate.get("accepted"):
                raise ValueError("source_distance_rejected")
            for key in (
                "minimum_observed_macro_distance",
                "maximum_observed_macro_distance",
                "maximum_observed_channel_distance",
            ):
                if not _float_equal(observed_gate[key], expected_gate[key]):
                    raise ValueError(f"source_distance_{key}")
            level = int(row["capability_level"])
            if level not in CAPABILITY_LEVELS:
                raise ValueError("capability_level")
            interval = tuple(float(value) for value in row["coordinate_interval"])
            capability = str(row["capability_id"])
            expected_intervals = (
                REGIME_RECENCY_INTERVALS
                if capability == "regime_switching"
                else (
                    INTERMITTENCY_GAP_INTERVALS
                    if capability == "predictable_intermittency"
                    else STRENGTH_INTERVALS
                )
            )
            if interval != expected_intervals[level - 1]:
                raise ValueError("coordinate_interval")
            sampled = float(row["sampled_coordinate"])
            if not interval[0] <= sampled <= interval[1]:
                raise ValueError("sampled_coordinate")
            parameter_draw = {
                "capability_level": level,
                "controlled_coordinate": row["controlled_coordinate"],
                "coordinate_interval": list(interval),
                "sampled_coordinate": sampled,
                "applied_component_gain": float(row["applied_component_gain"]),
                "augmentation_seed": int(row["augmentation_seed"]),
            }
            if row.get("parameter_draw_sha256") != protocol.json_sha256(
                parameter_draw
            ):
                raise ValueError("parameter_draw_hash")
            groups[(str(row["official_instance_id"]), capability)].append(row)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            failures.append(
                {"scope": "treatment", "sample_id": row.get("sample_id"), "reason": str(error)}
            )

    for (instance_id, capability), rows in groups.items():
        levels = sorted(int(row["capability_level"]) for row in rows)
        if levels != list(CAPABILITY_LEVELS):
            failures.append(
                {
                    "scope": "group",
                    "official_instance_id": instance_id,
                    "capability_id": capability,
                    "reason": "incomplete_five_level_group",
                }
            )
            continue
        ordered = sorted(rows, key=lambda row: int(row["capability_level"]))
        if capability == "regime_switching":
            joins = [int(row["mechanism_metadata"]["change_index"]) for row in ordered]
            if joins != sorted(joins):
                failures.append(
                    {"scope": "group", "official_instance_id": instance_id, "capability_id": capability, "reason": "regime_location_not_ordered"}
                )
        if capability == "predictable_intermittency":
            gaps = [int(row["mechanism_metadata"]["event_gap"]) for row in ordered]
            if gaps != sorted(gaps):
                failures.append(
                    {"scope": "group", "official_instance_id": instance_id, "capability_id": capability, "reason": "event_sparsity_not_ordered"}
                )

    availability_rows = list(protocol.iter_jsonl(file_paths["availability"]))
    declared_available = {
        (str(row["official_instance_id"]), str(row["capability_id"]))
        for row in availability_rows
        if row.get("available")
    }
    if declared_available != set(groups):
        failures.append({"scope": "availability", "reason": "generated_group_set_mismatch"})
    if len(baselines) != int(manifest["files"]["official_baselines"]["row_count"]):
        failures.append({"scope": "manifest", "reason": "baseline_count"})
    if treatment_count != int(manifest["files"]["capability_treatments"]["row_count"]):
        failures.append({"scope": "manifest", "reason": "treatment_count"})
    if len(availability_rows) != int(manifest["files"]["availability"]["row_count"]):
        failures.append({"scope": "manifest", "reason": "availability_count"})
    report = {
        "schema_version": VALIDATION_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": manifest["dataset_id"],
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "generation_manifest_sha256": protocol.file_sha256(manifest_path),
        "accepted": not failures,
        "official_baseline_count": len(baselines),
        "treatment_count": treatment_count,
        "available_group_count": len(groups),
        "failures": failures,
    }
    validation_dir = dataset_root / "02_validation"
    protocol.write_json(validation_dir / "report.json", report)
    return report


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    dataset_root = args.output_root.resolve() / args.dataset_id
    report_path = dataset_root / "02_validation" / "report.json"
    if report_path.exists():
        raise FileExistsError(
            f"validation artifact already exists; use a new experiment root: {report_path}"
        )
    report = validate_generation(dataset_root)
    print(protocol.canonical_json(report))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
