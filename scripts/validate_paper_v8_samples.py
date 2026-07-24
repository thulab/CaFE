#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import paper_v8_pipeline_common as v8
from app.services.synthetic_v8_feature_gate import (
    basic_sample_checks,
    validate_sample_collection,
)


DEFAULT_OUTPUT_ROOT = (
    v8.REPO_ROOT / "runtime" / "paper_exp" / "v8_test" / "full_pipeline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a formal Paper v8 generated sample shard."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    return parser.parse_args()


def validate_manifest_file(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if v8.file_sha256(path) != record["sha256"]:
        raise ValueError(f"manifest hash mismatch: {path}")


def robustness_checks(
    clean_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_by_id = {row["sample_id"]: row for row in clean_rows}
    failures: list[dict[str, Any]] = []
    for row in robustness_rows:
        basic = basic_sample_checks(row)
        clean_id = str(row["clean_master_sample_id"])
        clean = clean_by_id.get(clean_id)
        checks = {
            "basic": basic["accepted"],
            "clean_parent_exists": clean is not None,
        }
        if clean is not None:
            context = int(row["context_length"])
            observed = np.asarray(row["target"], dtype=float)
            latent = np.asarray(clean["target"], dtype=float)
            checks.update(
                {
                    "future_exact": bool(
                        np.array_equal(observed[context:], latent[context:])
                    ),
                    "history_changed": bool(
                        not np.array_equal(
                            observed[:context],
                            latent[:context],
                        )
                    ),
                    "mase_scale_reused": bool(
                        float(row["mase_scale"]) == float(clean["mase_scale"])
                    ),
                }
            )
        if not all(checks.values()):
            failures.append(
                {"sample_id": row["sample_id"], "checks": checks}
            )
    return {
        "accepted": not failures,
        "sample_count": len(robustness_rows),
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    dataset = v8.resolve_dataset(args.dataset_id)
    generation_dir = (
        args.output_root.resolve() / dataset.dataset_id / "02_generation"
    )
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    manifest_path = generation_dir / f"manifest__{shard_name}.json"
    manifest = v8.read_json(manifest_path)
    for record in manifest["files"].values():
        validate_manifest_file(record)
    clean_rows = list(
        v8.iter_jsonl(Path(manifest["files"]["clean"]["path"]))
    )
    robustness_rows = list(
        v8.iter_jsonl(Path(manifest["files"]["robustness"]["path"]))
    )
    identifiers = [str(row["sample_id"]) for row in clean_rows]
    duplicate_ids = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    clean_validation = validate_sample_collection(clean_rows)
    robust_validation = robustness_checks(clean_rows, robustness_rows)
    accepted = bool(
        not duplicate_ids
        and clean_validation["accepted"]
        and robust_validation["accepted"]
    )
    report = {
        "schema_version": "paper_v8_generation_validation.v1",
        "created_at": v8.utc_now(),
        "dataset_id": dataset.dataset_id,
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": v8.file_sha256(manifest_path),
        "clean_sample_count": len(clean_rows),
        "robustness_sample_count": len(robustness_rows),
        "duplicate_sample_ids": duplicate_ids,
        "clean_validation": clean_validation,
        "robustness_validation": robust_validation,
        "accepted": accepted,
    }
    report_path = generation_dir / f"validation__{shard_name}.json"
    v8.write_json(report_path, report)
    if not accepted:
        raise ValueError(f"v8 generation validation failed: {report_path}")
    print(
        v8.canonical_json(
            {
                "accepted": True,
                "clean_sample_count": len(clean_rows),
                "robustness_sample_count": len(robustness_rows),
                "report": str(report_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
