from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cafe import core as protocol
from cafe.benchmark_extension.generation import (
    CONTRACT_SCHEMA,
    GENERATION_SCHEMA,
    PIPELINE_SCHEMA,
    compact_contract_row,
    materialized_samples_for_instance,
)
from cafe.benchmark_extension.gift_eval import iter_gift_eval_instances
from cafe.benchmark_extension.storage import (
    iter_compact_parquet,
    validate_parquet_record,
)


VALIDATION_SCHEMA = "cafe.benchmark_extension_validation.v3"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate compact GIFT-Eval capability contracts by replay."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    return parser.parse_args()


def _next_or_none(iterator: Any) -> dict[str, Any] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def validate_generation(
    dataset_root: Path,
    *,
    gift_eval_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream all contracts and independently reproduce them from source Arrow."""

    manifest_path = dataset_root / "01_generation" / "manifest.json"
    manifest = protocol.read_json(manifest_path)
    failures: list[dict[str, Any]] = []
    config = manifest.get("config")
    gift_root = (
        Path(str((config or {}).get("gift_eval_source_root"))).resolve()
        if gift_eval_dir is None and (config or {}).get("gift_eval_source_root")
        else (
            protocol.REPO_ROOT / "data" / "gift-eval"
            if gift_eval_dir is None
            else gift_eval_dir.resolve()
        )
    )
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
        shard_size = int(config.get("generation_execution", {}).get("shard_size", 256))
        for instance_index, instance in enumerate(
            iter_gift_eval_instances(
                str(config["dataset_id"]),
                gift_root,
                term=str(config["term"]),
                max_instances=config.get("max_instances"),
            )
        ):
            for kind, dense_row in materialized_samples_for_instance(
                instance,
                augmentation_seed=int(config["augmentation_seed"]),
                capability_ids=tuple(str(value) for value in config["capability_ids"]),
                source_shard_index=instance_index // max(1, shard_size),
            ):
                expected = compact_contract_row(dense_row)
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
                    gate = dense_row.get("source_distance_gate")
                    if not isinstance(gate, dict) or not gate.get("accepted"):
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": expected.get("sample_id"),
                                "reason": "source_distance_rejected",
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
                {"scope": "manifest", "reason": f"{kind}_count:{count}!={declared}"}
            )
    report = {
        "schema_version": VALIDATION_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": manifest.get("dataset_id"),
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "generation_manifest_sha256": protocol.file_sha256(manifest_path),
        "validation_policy": (
            "stream_all_compact_contracts_and_exactly_replay_from_source_arrow_v1"
        ),
        "accepted": not failures,
        "official_baseline_count": counts["official_baselines"],
        "treatment_count": counts["capability_treatments"],
        "input_ablation_count": counts["input_ablations"],
        "availability_count": counts["availability"],
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
    report = validate_generation(dataset_root, gift_eval_dir=args.gift_eval_dir)
    print(protocol.canonical_json(report))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
