#!/usr/bin/env python3
"""Freeze structured calibration-source real windows for Paper v7 E2."""
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
    REPO_ROOT / "runtime/paper_exp/v7/01_nine_capability_suite"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v7/02_real_source_window_suite"
)
CONTEXT_LENGTHS = (96, 168, 336, 504)
MAX_CONTEXT_LENGTH = 504
HORIZON = 48
MIN_SUPPORTED_CAPABILITIES = 2
SCHEMA_VERSION = "paper_v7_real_source_window_suite.v1"
SAMPLE_SCHEMA_VERSION = "paper_v7_real_source_master.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze L=504,H=48 real reference windows already used by the "
            "dataset-local Paper v7 calibration, including structured task "
            "views."
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


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


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


def task_view_id(dataset_id: str, task_id: str) -> str:
    return f"{dataset_id}::{task_id}"


def selected_task_views(
    support_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select v7 real-source task views from the frozen support cells."""

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in support_artifact.get("cells", []):
        if cell.get("status") != "supported":
            continue
        dataset_id = str(cell["dataset_id"])
        task_id = str(cell["task_id"])
        profile_id = str(
            cell.get("generator_profile_id")
            or f"{dataset_id}__{task_id}__L{MAX_CONTEXT_LENGTH}_H{HORIZON}"
        )
        key = (dataset_id, task_id, profile_id)
        record = grouped.setdefault(
            key,
            {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "task_view_id": str(
                    cell.get("task_view_id")
                    or task_view_id(dataset_id, task_id)
                ),
                "profile_id": profile_id,
                "supported_capabilities": set(),
                "supported_cells": [],
                "target_dims": set(),
                "covariate_dims": set(),
                "hierarchies": set(),
            },
        )
        expected_task_view_id = str(
            cell.get("task_view_id")
            or task_view_id(dataset_id, task_id)
        )
        if record["task_view_id"] != expected_task_view_id:
            raise ValueError(f"task-view identity conflict for {key}")
        record["supported_capabilities"].add(
            str(cell["capability_id"])
        )
        record["supported_cells"].append(
            {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "capability_id": str(cell["capability_id"]),
                "profile_id": profile_id,
            }
        )
        structure = cell.get("structure_audit") or {}
        if structure.get("target_dim") is not None:
            record["target_dims"].add(int(structure["target_dim"]))
        if structure.get("covariate_dim") is not None:
            record["covariate_dims"].add(
                int(structure["covariate_dim"])
            )
        if structure.get("hierarchy") is not None:
            record["hierarchies"].add(str(structure["hierarchy"]))

    selected: list[dict[str, Any]] = []
    for record in grouped.values():
        capabilities = sorted(record.pop("supported_capabilities"))
        if (
            record["task_id"] == "univariate"
            and len(capabilities) < MIN_SUPPORTED_CAPABILITIES
        ):
            continue
        for field, output_field in (
            ("target_dims", "target_dim"),
            ("covariate_dims", "covariate_dim"),
            ("hierarchies", "hierarchy"),
        ):
            values = record.pop(field)
            if len(values) > 1:
                raise ValueError(
                    f"support cells disagree on {field}: "
                    f"{record['task_view_id']}={sorted(values)}"
                )
            record[output_field] = (
                next(iter(values)) if values else None
            )
        record["supported_capabilities"] = capabilities
        record["supported_cell_count"] = len(record["supported_cells"])
        selected.append(record)
    selected = sorted(
        selected,
        key=lambda row: (
            str(row["dataset_id"]),
            str(row["task_id"]),
            str(row["profile_id"]),
        ),
    )
    identities = [str(row["task_view_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "one task_view_id maps to multiple L504 source profiles"
        )
    return selected


def _legacy_univariate_task_views(
    eligible: dict[str, list[str]],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": dataset_id,
            "task_id": "univariate",
            "task_view_id": task_view_id(dataset_id, "univariate"),
            "profile_id": (
                f"{dataset_id}__univariate__L{MAX_CONTEXT_LENGTH}_H{HORIZON}"
            ),
            "supported_capabilities": list(capabilities),
            "supported_cells": [
                {
                    "dataset_id": dataset_id,
                    "task_id": "univariate",
                    "capability_id": capability_id,
                }
                for capability_id in capabilities
            ],
            "supported_cell_count": len(capabilities),
            "target_dim": 1,
            "covariate_dim": 0,
            "hierarchy": None,
        }
        for dataset_id, capabilities in sorted(eligible.items())
    ]


def _column_names(
    bucket: dict[str, Any],
    *,
    primary_key: str,
    legacy_key: str,
    count: int,
    fallback_prefix: str,
) -> list[str]:
    configured = bucket.get(primary_key)
    if configured is None:
        configured = bucket.get(legacy_key)
    names = (
        [f"{fallback_prefix}_{index}" for index in range(count)]
        if configured is None
        else [str(value) for value in configured]
    )
    if len(names) != count or len(set(names)) != len(names):
        raise ValueError(
            f"{primary_key} must be unique and match dimension {count}"
        )
    return names


def _reference_targets(
    bucket: dict[str, Any],
    *,
    profile_id: str,
    target_dim: int,
) -> np.ndarray:
    references = np.asarray(bucket.get("reference_raw", []), dtype=float)
    reference_count = int(
        bucket.get("reference_count", len(references))
    )
    steps = MAX_CONTEXT_LENGTH + HORIZON
    if references.shape == (reference_count, steps, target_dim):
        reshaped = references
    elif references.shape == (reference_count, steps * target_dim):
        reshaped = references.reshape(reference_count, steps, target_dim)
    else:
        raise ValueError(
            f"reference_raw shape mismatch for {profile_id}: "
            f"{references.shape} cannot reshape to "
            f"{(reference_count, steps, target_dim)}"
        )
    if not np.isfinite(reshaped).all():
        raise ValueError(f"non-finite real source reference: {profile_id}")
    return reshaped


def _reference_covariates(
    bucket: dict[str, Any],
    *,
    profile_id: str,
    reference_count: int,
    covariate_dim: int,
) -> np.ndarray | None:
    configured = bucket.get("reference_covariates")
    if covariate_dim == 0:
        if configured is None:
            return None
        values = np.asarray(configured, dtype=float)
        if values.size == 0:
            return None
        raise ValueError(
            f"non-covariate bucket unexpectedly stores covariates: {profile_id}"
        )
    if configured is None:
        raise ValueError(
            f"covariate task {profile_id} lacks reference_covariates; "
            "rebuild the v7 near-distance artifact"
        )
    values = np.asarray(configured, dtype=float)
    expected = (
        reference_count,
        MAX_CONTEXT_LENGTH + HORIZON,
        covariate_dim,
    )
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(
            f"reference_covariates shape mismatch for {profile_id}: "
            f"{values.shape} != {expected}"
        )
    return values


def _reference_metadata(
    bucket: dict[str, Any],
    key: str,
    *,
    count: int,
) -> list[Any]:
    configured = bucket.get(key)
    if configured is None:
        return [None] * count
    values = list(configured)
    if len(values) != count:
        raise ValueError(
            f"{key} count mismatch: {len(values)} != {count}"
        )
    return values


def source_rows(
    *,
    near_artifact: dict[str, Any],
    eligible: dict[str, list[str]] | None = None,
    task_views: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if task_views is None:
        if eligible is None:
            raise ValueError("source_rows requires task_views or eligible")
        task_views = _legacy_univariate_task_views(eligible)
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for task_view in task_views:
        dataset_id = str(task_view["dataset_id"])
        task_id = str(task_view["task_id"])
        public_task_view_id = str(task_view["task_view_id"])
        profile_id = str(task_view["profile_id"])
        capabilities = [
            str(value)
            for value in task_view["supported_capabilities"]
        ]
        bucket = near_artifact.get("buckets", {}).get(profile_id)
        if not isinstance(bucket, dict):
            if task_id != "univariate":
                raise ValueError(
                    "supported structured task lacks its L504 near "
                    f"reference bucket: {profile_id}"
                )
            support_rows.append(
                {
                    "dataset_id": dataset_id,
                    "task_id": task_id,
                    "task_view_id": public_task_view_id,
                    "profile_id": profile_id,
                    "status": "unsupported",
                    "reason_codes": ["missing_l504_near_reference_bucket"],
                    "supported_capabilities": capabilities,
                    "supported_cell_count": int(
                        task_view["supported_cell_count"]
                    ),
                    "master_sample_count": 0,
                }
            )
            continue
        if int(bucket.get("context_length", -1)) != MAX_CONTEXT_LENGTH or int(
            bucket.get("horizon", -1)
        ) != HORIZON:
            raise ValueError(f"unexpected source bucket window: {profile_id}")
        target_dim = int(bucket.get("target_dim", -1))
        covariate_dim = int(bucket.get("covariate_dim", 0))
        if target_dim < 1 or covariate_dim < 0:
            raise ValueError(f"invalid source dimensions: {profile_id}")
        supported_target_dim = task_view.get("target_dim")
        supported_covariate_dim = task_view.get("covariate_dim")
        if (
            supported_target_dim is not None
            and int(supported_target_dim) != target_dim
        ) or (
            supported_covariate_dim is not None
            and int(supported_covariate_dim) != covariate_dim
        ):
            raise ValueError(
                f"support/bucket dimension mismatch for {profile_id}"
            )
        references = _reference_targets(
            bucket,
            profile_id=profile_id,
            target_dim=target_dim,
        )
        reference_count = len(references)
        reference_covariates = _reference_covariates(
            bucket,
            profile_id=profile_id,
            reference_count=reference_count,
            covariate_dim=covariate_dim,
        )
        target_column_names = _column_names(
            bucket,
            primary_key="target_column_names",
            legacy_key="target_columns",
            count=target_dim,
            fallback_prefix="target",
        )
        covariate_column_names = _column_names(
            bucket,
            primary_key="covariate_column_names",
            legacy_key="covariate_columns",
            count=covariate_dim,
            fallback_prefix="covariate",
        )
        group_ids = _reference_metadata(
            bucket,
            "reference_group_ids",
            count=reference_count,
        )
        window_starts = _reference_metadata(
            bucket,
            "reference_window_starts",
            count=reference_count,
        )
        window_ids = _reference_metadata(
            bucket,
            "reference_window_ids",
            count=reference_count,
        )
        hierarchy = bucket.get("hierarchy") or task_view.get("hierarchy")
        if task_id in {"hierarchy", "hierarchical"}:
            if hierarchy != "additive_first":
                raise ValueError(
                    f"hierarchy task must preserve additive_first: {profile_id}"
                )
            if target_dim < 3 or not np.allclose(
                references[:, :, 0],
                references[:, :, 1:].sum(axis=2),
                rtol=1e-8,
                # Frozen near artifacts serialize reference_raw to six
                # decimals, so exact additive inputs can retain a residual of
                # one final decimal unit after reconstruction.
                atol=2e-6,
            ):
                raise ValueError(
                    f"additive_first identity is violated: {profile_id}"
                )
        elif hierarchy is not None:
            raise ValueError(
                f"non-hierarchy task declares hierarchy: {profile_id}"
            )
        frequency = str(
            bucket.get("frequency")
            or near_artifact.get("config", {}).get("frequency")
            or "h"
        )
        provenance = bucket.get("provenance")
        known_future_covariates = [
            str(value)
            for value in (
                bucket.get("known_future_covariates")
                or (
                    covariate_column_names if covariate_dim else []
                )
            )
        ]
        task_master_ids: set[str] = set()
        for source_index, values in enumerate(references):
            target_hash = array_sha256(values)
            group_id = group_ids[source_index]
            window_start = window_starts[source_index]
            window_id = window_ids[source_index]
            if window_id is None and (
                group_id is not None or window_start is not None
            ):
                window_id = (
                    f"{group_id if group_id is not None else 'group-unknown'}"
                    f"::start-{window_start}"
                )
            master_sample_id = (
                "real-source-"
                + hashlib.sha256(
                    canonical_json(
                        {
                            "task_view_id": public_task_view_id,
                            "profile_id": profile_id,
                            "source_index": source_index,
                            "group_id": group_id,
                            "window_id": window_id,
                            "target_sha256": target_hash,
                        }
                    ).encode()
                ).hexdigest()[:24]
            )
            if master_sample_id in observed_ids:
                raise ValueError(
                    f"duplicate real source master id: {master_sample_id}"
                )
            observed_ids.add(master_sample_id)
            task_master_ids.add(master_sample_id)
            covariates = (
                None
                if reference_covariates is None
                else reference_covariates[source_index]
            )
            rows.append(
                {
                    "schema_version": SAMPLE_SCHEMA_VERSION,
                    "sample_id": master_sample_id,
                    "master_sample_id": master_sample_id,
                    "profile_id": profile_id,
                    "dataset_id": dataset_id,
                    "task_id": task_id,
                    "task_view_id": public_task_view_id,
                    "context_length": MAX_CONTEXT_LENGTH,
                    "context_lengths": list(CONTEXT_LENGTHS),
                    "horizon": HORIZON,
                    "season_length": int(bucket["season_length"]),
                    "frequency": frequency,
                    "target_dim": target_dim,
                    "covariate_dim": covariate_dim,
                    "hierarchy": hierarchy,
                    "target_column_names": target_column_names,
                    "covariate_column_names": covariate_column_names,
                    "known_future_covariates": known_future_covariates,
                    "covariate_provenance": bucket.get(
                        "covariate_provenance"
                    ),
                    "target": values.tolist(),
                    "covariates": (
                        None if covariates is None else covariates.tolist()
                    ),
                    "target_sha256": target_hash,
                    "future_sha256": array_sha256(
                        values[MAX_CONTEXT_LENGTH:]
                    ),
                    "source_role": (
                        "dataset-local near-distance reference_raw reused "
                        "for source-window alignment"
                    ),
                    "source_profile_id": profile_id,
                    "source_reference_index": source_index,
                    "source_reference_split": bucket.get("split"),
                    "source_group_id": group_id,
                    "source_window_start": window_start,
                    "source_window_id": window_id,
                    "source_provenance": provenance,
                    "supported_capabilities": capabilities,
                    "supported_cells": task_view["supported_cells"],
                }
            )
        support_rows.append(
            {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "task_view_id": public_task_view_id,
                "status": (
                    "supported" if task_master_ids else "unsupported"
                ),
                "reason_codes": (
                    []
                    if task_master_ids
                    else ["empty_source_reference_bucket"]
                ),
                "supported_capabilities": capabilities,
                "supported_capability_count": len(capabilities),
                "supported_cell_count": int(
                    task_view["supported_cell_count"]
                ),
                "master_sample_count": len(task_master_ids),
                "profile_id": profile_id,
                "target_dim": target_dim,
                "covariate_dim": covariate_dim,
                "hierarchy": hierarchy,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            str(row["task_id"]),
            int(row["source_reference_index"]),
        )
    )
    support_rows.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            str(row["task_id"]),
        )
    )
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
    task_views = selected_task_views(support_artifact)
    rows, support_rows = source_rows(
        near_artifact=near_artifact,
        task_views=task_views,
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
        "calibration_dir": display_path(calibration_dir),
        "source_artifacts": {
            name: {
                "path": display_path(path),
                "sha256": file_sha256(path),
            }
            for name, path in required.items()
        },
        "context_lengths": list(CONTEXT_LENGTHS),
        "master_context_length": MAX_CONTEXT_LENGTH,
        "horizon": HORIZON,
        "minimum_supported_capabilities": MIN_SUPPORTED_CAPABILITIES,
        "dataset_ids": sorted(
            {str(row["dataset_id"]) for row in task_views}
        ),
        "task_view_ids": [
            str(row["task_view_id"]) for row in task_views
        ],
        "task_view_count": len(task_views),
        "selected_supported_cell_count": sum(
            int(row["supported_cell_count"]) for row in task_views
        ),
        "structured_selection_policy": (
            "include every supported structured task cell; group cells by "
            "dataset/task/profile without duplicating reference windows"
        ),
        "source_window_policy": (
            "reuse the exact L504,H48 dataset-local near-distance "
            "reference_raw windows and reconstructable structured metadata "
            "from formal calibration; no test/holdout"
        ),
        "interpretation": (
            "in-sample source-window construct alignment, not held-out "
            "external validity"
        ),
    }
    support = {
        "schema_version": "paper_v7_real_source_support.v1",
        "created_at": created_at,
        "supported_dataset_count": len(
            {
                str(row["dataset_id"])
                for row in support_rows
                if row["status"] == "supported"
            }
        ),
        "unsupported_dataset_count": len(
            {
                str(row["dataset_id"])
                for row in support_rows
                if row["status"] != "supported"
                and not any(
                    other["status"] == "supported"
                    and str(other["dataset_id"])
                    == str(row["dataset_id"])
                    for other in support_rows
                )
            }
        ),
        "supported_task_view_count": sum(
            row["status"] == "supported" for row in support_rows
        ),
        "unsupported_task_view_count": sum(
            row["status"] != "supported" for row in support_rows
        ),
        "master_sample_count": len(rows),
        "task_views": support_rows,
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
        "schema_version": "paper_v7_real_source_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "builder_sha256": file_sha256(Path(__file__)),
        "dataset_count": support["supported_dataset_count"],
        "task_view_count": support["supported_task_view_count"],
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
