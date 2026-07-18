#!/usr/bin/env python3
"""Freeze calibration-source real windows for Paper v5 E2 alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/01_nine_capability_suite"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/02_real_source_window_suite"
)
CONTEXT_LENGTHS = (96, 168, 336, 504)
MAX_CONTEXT_LENGTH = 504
HORIZON = 48
MIN_SUPPORTED_CAPABILITIES = 2
SCHEMA_VERSION = "paper_v5_real_source_window_suite.v1"
SAMPLE_SCHEMA_VERSION = "paper_v5_real_source_master.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze L=504,H=48 real reference windows already used by the "
            "dataset-local Paper v5 calibration."
        )
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def eligible_univariate_datasets(
    support_artifact: dict[str, Any],
) -> dict[str, list[str]]:
    by_dataset: dict[str, set[str]] = {}
    for cell in support_artifact.get("cells", []):
        if (
            cell.get("status") != "supported"
            or str(cell.get("task_id")) != "univariate"
        ):
            continue
        by_dataset.setdefault(str(cell["dataset_id"]), set()).add(
            str(cell["capability_id"])
        )
    return {
        dataset_id: sorted(capabilities)
        for dataset_id, capabilities in sorted(by_dataset.items())
        if len(capabilities) >= MIN_SUPPORTED_CAPABILITIES
    }


def source_rows(
    *,
    near_artifact: dict[str, Any],
    eligible: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for dataset_id, capabilities in eligible.items():
        profile_id = (
            f"{dataset_id}__univariate__L{MAX_CONTEXT_LENGTH}_H{HORIZON}"
        )
        bucket = near_artifact.get("buckets", {}).get(profile_id)
        if not isinstance(bucket, dict):
            support_rows.append(
                {
                    "dataset_id": dataset_id,
                    "task_id": "univariate",
                    "status": "unsupported",
                    "reason_codes": ["missing_l504_near_reference_bucket"],
                    "supported_capabilities": capabilities,
                    "master_sample_count": 0,
                }
            )
            continue
        if (
            int(bucket.get("context_length", -1)) != MAX_CONTEXT_LENGTH
            or int(bucket.get("horizon", -1)) != HORIZON
            or int(bucket.get("target_dim", -1)) != 1
            or int(bucket.get("covariate_dim", 0)) != 0
        ):
            raise ValueError(f"unexpected univariate source bucket: {profile_id}")
        references = np.asarray(bucket.get("reference_raw", []), dtype=float)
        expected_shape = (
            int(bucket.get("reference_count", len(references))),
            MAX_CONTEXT_LENGTH + HORIZON,
        )
        if references.shape != expected_shape:
            raise ValueError(
                f"reference_raw shape mismatch for {profile_id}: "
                f"{references.shape} != {expected_shape}"
            )
        if not np.isfinite(references).all():
            raise ValueError(f"non-finite real source reference: {profile_id}")
        dataset_ids: set[str] = set()
        for source_index, values in enumerate(references):
            target_hash = array_sha256(values[:, None])
            master_sample_id = (
                "real-source-"
                + hashlib.sha256(
                    f"{profile_id}|{source_index}|{target_hash}".encode()
                ).hexdigest()[:24]
            )
            if master_sample_id in observed_ids:
                raise ValueError(
                    f"duplicate real source master id: {master_sample_id}"
                )
            observed_ids.add(master_sample_id)
            dataset_ids.add(master_sample_id)
            rows.append(
                {
                    "schema_version": SAMPLE_SCHEMA_VERSION,
                    "sample_id": master_sample_id,
                    "master_sample_id": master_sample_id,
                    "profile_id": profile_id,
                    "dataset_id": dataset_id,
                    "task_id": "univariate",
                    "context_length": MAX_CONTEXT_LENGTH,
                    "context_lengths": list(CONTEXT_LENGTHS),
                    "horizon": HORIZON,
                    "season_length": int(bucket["season_length"]),
                    "frequency": str(
                        bucket.get("frequency")
                        or near_artifact.get("config", {}).get(
                            "frequency",
                            "h",
                        )
                    ),
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "hierarchy": None,
                    "target": values[:, None].tolist(),
                    "covariates": None,
                    "target_sha256": target_hash,
                    "future_sha256": array_sha256(
                        values[MAX_CONTEXT_LENGTH:, None]
                    ),
                    "source_role": (
                        "dataset-local near-distance reference_raw reused "
                        "for source-window alignment"
                    ),
                    "source_profile_id": profile_id,
                    "source_reference_index": source_index,
                    "source_reference_split": bucket.get("split"),
                    "supported_capabilities": capabilities,
                }
            )
        support_rows.append(
            {
                "dataset_id": dataset_id,
                "task_id": "univariate",
                "status": "supported" if dataset_ids else "unsupported",
                "reason_codes": (
                    [] if dataset_ids else ["empty_source_reference_bucket"]
                ),
                "supported_capabilities": capabilities,
                "supported_capability_count": len(capabilities),
                "master_sample_count": len(dataset_ids),
                "profile_id": profile_id,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            int(row["source_reference_index"]),
        )
    )
    support_rows.sort(key=lambda row: str(row["dataset_id"]))
    return rows, support_rows


def build_suite(calibration_dir: Path, output_dir: Path) -> dict[str, Any]:
    calibration_dir = calibration_dir.resolve()
    output_dir = output_dir.resolve()
    required = {
        "near_distance_artifact.json": (
            calibration_dir / "near_distance_artifact.json"
        ),
        "dataset_capability_support_matrix.json": (
            calibration_dir / "dataset_capability_support_matrix.json"
        ),
        "manifest.json": calibration_dir / "manifest.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing calibration input: {path}")
    manifest = read_json(required["manifest.json"])
    manifest_files = {
        str(row["path"]): str(row["sha256"])
        for row in manifest.get("files", [])
    }
    for name in (
        "near_distance_artifact.json",
        "dataset_capability_support_matrix.json",
    ):
        expected = manifest_files.get(name)
        observed = file_sha256(required[name])
        if expected != observed:
            raise ValueError(
                f"calibration manifest hash mismatch for {name}: "
                f"{observed} != {expected}"
            )
    near_artifact = read_json(required["near_distance_artifact.json"])
    support_artifact = read_json(
        required["dataset_capability_support_matrix.json"]
    )
    eligible = eligible_univariate_datasets(support_artifact)
    rows, support_rows = source_rows(
        near_artifact=near_artifact,
        eligible=eligible,
    )
    if not rows:
        raise ValueError("no real source windows were frozen")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"real source suite output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "calibration_dir": str(calibration_dir.relative_to(REPO_ROOT)),
        "source_artifacts": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in required.items()
        },
        "context_lengths": list(CONTEXT_LENGTHS),
        "master_context_length": MAX_CONTEXT_LENGTH,
        "horizon": HORIZON,
        "minimum_supported_capabilities": MIN_SUPPORTED_CAPABILITIES,
        "dataset_ids": sorted(eligible),
        "source_window_policy": (
            "reuse the exact L504,H48 dataset-local near-distance "
            "reference_raw windows from formal calibration; no test/holdout"
        ),
        "interpretation": (
            "in-sample source-window construct alignment, not held-out "
            "external validity"
        ),
    }
    support = {
        "schema_version": "paper_v5_real_source_support.v1",
        "created_at": created_at,
        "supported_dataset_count": sum(
            row["status"] == "supported" for row in support_rows
        ),
        "unsupported_dataset_count": sum(
            row["status"] != "supported" for row in support_rows
        ),
        "master_sample_count": len(rows),
        "datasets": support_rows,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "dataset_support.json", support)
    write_jsonl(output_dir / "real_source_samples.jsonl", rows)
    output_files = (
        "config.json",
        "dataset_support.json",
        "real_source_samples.jsonl",
    )
    result = {
        "schema_version": "paper_v5_real_source_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "builder_sha256": file_sha256(Path(__file__)),
        "dataset_count": support["supported_dataset_count"],
        "master_sample_count": len(rows),
        "view_count": len(rows) * len(CONTEXT_LENGTHS),
        "files": {
            name: {
                "size_bytes": (output_dir / name).stat().st_size,
                "sha256": file_sha256(output_dir / name),
            }
            for name in output_files
        },
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> int:
    args = parse_args()
    result = build_suite(args.calibration_dir, args.output_dir)
    print(
        f"real source suite: {result['master_sample_count']} masters, "
        f"{result['view_count']} views",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
