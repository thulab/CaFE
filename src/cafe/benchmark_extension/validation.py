from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from cafe import core as protocol
from cafe.benchmark_extension.generation import (
    CONTRACT_SCHEMA,
    GENERATION_SCHEMA,
    PIPELINE_SCHEMA,
    _compact_record_batch,
    _parallel_work_batches,
    compact_contract_row,
    materialized_samples_for_instance,
)
from cafe.benchmark_extension.gift_eval import iter_gift_eval_instances
from cafe.benchmark_extension.mechanisms import SOURCE_DISTANCE_THRESHOLD
from cafe.benchmark_extension.storage import (
    iter_compact_parquet,
    validate_parquet_record,
)


VALIDATION_SCHEMA = "cafe.benchmark_extension_validation.v4"
VALIDATION_MODES = ("research", "publication")
DEFAULT_VALIDATION_WORKERS = max(1, min(8, os.cpu_count() or 1))
MAX_RECORDED_FAILURES = 100
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate compact GIFT-Eval capability contracts."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument(
        "--mode",
        choices=VALIDATION_MODES,
        default="research",
        help=(
            "research scans every stored treatment distance gate; publication "
            "also verifies hashes and exactly replays every contract"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_VALIDATION_WORKERS,
        help="Process workers used for Parquet row groups or publication replay.",
    )
    return parser.parse_args()


def _next_or_none(iterator: Any) -> dict[str, Any] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _distance_gate_reason(row: dict[str, Any]) -> str | None:
    gate = row.get("source_distance_gate")
    if not isinstance(gate, dict):
        return "source_distance_gate_missing"
    if gate.get("schema_version") != "cafe.treatment_source_distance_gate.v2":
        return "source_distance_gate_schema"
    if gate.get("scope") != "treatment_history_vs_authentic_official_history":
        return "source_distance_gate_scope"
    if gate.get("treatment_only") is not True:
        return "source_distance_gate_not_treatment_only"
    try:
        required = float(gate["minimum_required_distance"])
        observed = float(gate["minimum_observed_macro_distance"])
    except (KeyError, TypeError, ValueError):
        return "source_distance_gate_invalid_distance"
    if not math.isfinite(required) or not math.isclose(
        required,
        SOURCE_DISTANCE_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "source_distance_gate_threshold"
    by_suffix = gate.get("by_suffix")
    if not isinstance(by_suffix, list) or not by_suffix:
        return "source_distance_gate_suffixes_missing"
    suffix_macros: list[float] = []
    for suffix in by_suffix:
        if not isinstance(suffix, dict):
            return "source_distance_gate_suffix_invalid"
        if int(suffix.get("context_length") or 0) <= 0:
            return "source_distance_gate_context_invalid"
        macro = suffix.get("macro_normalized_rms")
        channels = suffix.get("channel_normalized_rms")
        if not _finite_nonnegative(macro):
            return "source_distance_gate_suffix_macro_invalid"
        if not isinstance(channels, list) or not channels:
            return "source_distance_gate_channels_missing"
        if not all(_finite_nonnegative(value) for value in channels):
            return "source_distance_gate_channel_invalid"
        calculated_macro = sum(float(value) for value in channels) / len(channels)
        if not math.isclose(
            float(macro), calculated_macro, rel_tol=1e-9, abs_tol=1e-12
        ):
            return "source_distance_gate_suffix_macro_mismatch"
        suffix_macros.append(float(macro))
    if not math.isclose(
        observed,
        min(suffix_macros),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return "source_distance_gate_observed_mismatch"
    if observed < required - 1e-12:
        return "source_distance_below_minimum"
    if gate.get("accepted") is not True or gate.get("reason") is not None:
        return "source_distance_rejected"
    return None


def _scan_treatment_row_group(
    work: tuple[str, int],
) -> tuple[int, int, list[dict[str, Any]]]:
    path_string, row_group_index = work
    parquet = pq.ParquetFile(path_string)
    table = parquet.read_row_group(row_group_index, columns=("payload_json",))
    payloads = table.column(0).to_pylist()
    failures: list[dict[str, Any]] = []
    failure_count = 0
    for payload in payloads:
        sample_id: Any = None
        try:
            row = json.loads(str(payload))
            if not isinstance(row, dict):
                raise TypeError("payload is not an object")
            sample_id = row.get("sample_id")
            reason = _distance_gate_reason(row)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            reason = f"source_distance_payload:{error}"
        if reason is not None:
            failure_count += 1
            if len(failures) < MAX_RECORDED_FAILURES:
                failures.append(
                    {
                        "scope": "capability_treatments",
                        "sample_id": sample_id,
                        "reason": reason,
                    }
                )
    return len(payloads), failure_count, failures


def _research_validation(
    manifest: dict[str, Any],
    *,
    workers: int,
) -> tuple[dict[str, int], int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    failure_count = 0
    counts = {
        "official_baselines": int(manifest.get("official_instance_count", 0)),
        "capability_treatments": 0,
        "input_ablations": int(manifest.get("input_ablation_count", 0)),
        "availability": int(
            ((manifest.get("files") or {}).get("availability") or {}).get(
                "row_count", 0
            )
        ),
    }
    try:
        treatment_path = Path(
            str(manifest["files"]["capability_treatments"]["path"])
        )
        parquet = pq.ParquetFile(treatment_path)
        work = [
            (str(treatment_path), row_group_index)
            for row_group_index in range(parquet.num_row_groups)
        ]
        if int(workers) > 1 and len(work) > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                results = executor.map(_scan_treatment_row_group, work)
                for count, rejected, rows in results:
                    counts["capability_treatments"] += count
                    failure_count += rejected
                    remaining = MAX_RECORDED_FAILURES - len(failures)
                    failures.extend(rows[: max(0, remaining)])
        else:
            for item in work:
                count, rejected, rows = _scan_treatment_row_group(item)
                counts["capability_treatments"] += count
                failure_count += rejected
                remaining = MAX_RECORDED_FAILURES - len(failures)
                failures.extend(rows[: max(0, remaining)])
    except Exception as error:
        # A malformed or unreadable treatment artifact must yield a rejected
        # report rather than letting inference proceed without a gate audit.
        failure_count += 1
        failures.append(
            {
                "scope": "capability_treatments",
                "sample_id": None,
                "reason": f"source_distance_scan:{error}",
            }
        )
    return counts, failure_count, failures


def _publication_expected_batches(
    manifest: dict[str, Any],
    gift_root: Path,
    workers: int,
) -> Iterator[dict[str, list[dict[str, Any]]]]:
    config = manifest["config"]
    shard_size = int(config.get("generation_execution", {}).get("shard_size", 256))
    if int(workers) <= 1:
        for instance_index, instance in enumerate(
            iter_gift_eval_instances(
                str(config["dataset_id"]),
                gift_root,
                term=str(config["term"]),
                max_instances=config.get("max_instances"),
            )
        ):
            output = {
                "official_baselines": [],
                "capability_treatments": [],
                "input_ablations": [],
                "availability": [],
            }
            for kind, dense_row in materialized_samples_for_instance(
                instance,
                augmentation_seed=int(config["augmentation_seed"]),
                capability_ids=tuple(str(value) for value in config["capability_ids"]),
                source_shard_index=instance_index // max(1, shard_size),
            ):
                output[kind].append(compact_contract_row(dense_row))
            yield output
        return

    batches = iter(
        _parallel_work_batches(
            str(config["dataset_id"]),
            gift_eval_dir=gift_root,
            term=str(config["term"]),
            augmentation_seed=int(config["augmentation_seed"]),
            capability_ids=tuple(str(value) for value in config["capability_ids"]),
            max_instances=config.get("max_instances"),
            shard_size=shard_size,
        )
    )
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        pending: list[Any] = []
        for _ in range(max(1, int(workers) * 2)):
            try:
                pending.append(executor.submit(_compact_record_batch, next(batches)))
            except StopIteration:
                break
        while pending:
            future = pending.pop(0)
            yield future.result()
            try:
                pending.append(executor.submit(_compact_record_batch, next(batches)))
            except StopIteration:
                pass


def _publication_validation(
    manifest: dict[str, Any],
    *,
    gift_root: Path,
    workers: int,
) -> tuple[dict[str, int], int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    config = manifest.get("config")
    if manifest.get("schema_version") != GENERATION_SCHEMA:
        failures.append({"scope": "manifest", "reason": "schema_version"})
    if not isinstance(config, dict) or config.get("pipeline_schema_version") != PIPELINE_SCHEMA:
        failures.append({"scope": "manifest", "reason": "pipeline_schema"})
    elif manifest.get("config_sha256") != protocol.json_sha256(config):
        failures.append({"scope": "manifest", "reason": "config_hash"})
    if isinstance(config, dict):
        storage = config.get("artifact_storage")
        if (
            not isinstance(storage, dict)
            or storage.get("format") != "parquet"
            or storage.get("dense_targets_stored") is not False
            or storage.get("dense_covariates_stored") is not False
        ):
            failures.append({"scope": "manifest", "reason": "dense_storage_policy"})

    artifact_keys = (
        "official_baselines",
        "capability_treatments",
        "input_ablations",
        "availability",
    )
    paths: dict[str, Path] = {}
    for key in artifact_keys:
        try:
            paths[key] = validate_parquet_record(manifest["files"][key])
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            failures.append({"scope": "manifest", "reason": f"{key}:{error}"})
    for record in manifest.get("source_files") or []:
        try:
            source_path = Path(str(record["path"]))
            if protocol.file_sha256(source_path) != record["sha256"]:
                raise ValueError("source_hash")
        except (KeyError, OSError, ValueError) as error:
            failures.append({"scope": "manifest", "reason": f"source:{error}"})

    counts = {key: 0 for key in artifact_keys}
    if not failures and isinstance(config, dict):
        observed = {
            key: iter(iter_compact_parquet(path)) for key, path in paths.items()
        }
        for batch in _publication_expected_batches(manifest, gift_root, workers):
            for kind in artifact_keys:
                for expected in batch[kind]:
                    actual = _next_or_none(observed[kind])
                    counts[kind] += 1
                    if actual is None:
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": expected.get("sample_id"),
                                "reason": "missing_compact_contract",
                            }
                        )
                        continue
                    if actual.get("schema_version") != CONTRACT_SCHEMA:
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": actual.get("sample_id"),
                                "reason": "contract_schema",
                            }
                        )
                    if any(
                        field in actual
                        for field in ("target", "covariates", "future_observed_mask")
                    ):
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": actual.get("sample_id"),
                                "reason": "dense_payload_present",
                            }
                        )
                    if protocol.canonical_json(actual) != protocol.canonical_json(expected):
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": expected.get("sample_id"),
                                "reason": "deterministic_replay_mismatch",
                            }
                        )
                    if kind == "capability_treatments":
                        reason = _distance_gate_reason(expected)
                        if reason is not None:
                            failures.append(
                                {
                                    "scope": kind,
                                    "sample_id": expected.get("sample_id"),
                                    "reason": reason,
                                }
                            )
        for kind, iterator in observed.items():
            extra = _next_or_none(iterator)
            if extra is not None:
                failures.append(
                    {
                        "scope": kind,
                        "sample_id": extra.get("sample_id"),
                        "reason": "unexpected_compact_contract",
                    }
                )
        for kind, count in counts.items():
            declared = int((manifest.get("files", {}).get(kind) or {}).get("row_count", -1))
            if count != declared:
                failures.append(
                    {
                        "scope": "manifest",
                        "reason": f"{kind}_count:{count}!={declared}",
                    }
                )
    return counts, len(failures), failures[:MAX_RECORDED_FAILURES]


def validate_generation(
    dataset_root: Path,
    *,
    gift_eval_dir: Path | None = None,
    mode: str = "research",
    workers: int = DEFAULT_VALIDATION_WORKERS,
) -> dict[str, Any]:
    """Validate every distance gate, with exact replay reserved for publication."""

    if mode not in VALIDATION_MODES:
        raise ValueError(f"unsupported validation mode: {mode}")
    if int(workers) < 1:
        raise ValueError("validation workers must be positive")
    manifest_path = dataset_root / "01_generation" / "manifest.json"
    manifest = protocol.read_json(manifest_path)
    if mode == "publication":
        config = manifest.get("config") or {}
        gift_root = (
            Path(str(config.get("gift_eval_source_root"))).resolve()
            if gift_eval_dir is None and config.get("gift_eval_source_root")
            else (
                protocol.REPO_ROOT / "data" / "gift-eval"
                if gift_eval_dir is None
                else gift_eval_dir.resolve()
            )
        )
        counts, failure_count, failures = _publication_validation(
            manifest,
            gift_root=gift_root,
            workers=int(workers),
        )
        policy = "publication_full_hash_and_exact_source_replay_v1"
    else:
        counts, failure_count, failures = _research_validation(
            manifest,
            workers=int(workers),
        )
        policy = "research_all_treatment_source_distance_gates_v1"

    report = {
        "schema_version": VALIDATION_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": manifest.get("dataset_id"),
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "generation_manifest_sha256": protocol.file_sha256(manifest_path),
        "validation_mode": mode,
        "validation_policy": policy,
        "validation_workers": int(workers),
        "accepted": failure_count == 0,
        "official_baseline_count": counts["official_baselines"],
        "treatment_count": counts["capability_treatments"],
        "input_ablation_count": counts["input_ablations"],
        "availability_count": counts["availability"],
        "source_distance_gate_checked_count": counts["capability_treatments"],
        "failure_count": int(failure_count),
        "failures_truncated": failure_count > len(failures),
        "failures": failures,
    }
    protocol.write_json(dataset_root / "02_validation" / "report.json", report)
    return report


def main() -> int:
    args = parse_args()
    dataset_root = args.output_root.resolve() / args.dataset_id
    report_path = dataset_root / "02_validation" / "report.json"
    if report_path.exists():
        raise FileExistsError(
            f"validation artifact already exists; use a new experiment root: {report_path}"
        )
    report = validate_generation(
        dataset_root,
        gift_eval_dir=args.gift_eval_dir,
        mode=args.mode,
        workers=args.workers,
    )
    print(protocol.canonical_json(report))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
