#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_paper_v4_profile_suite import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_GIFT_EVAL_DIR,
    DatasetSpec,
    dataset_asset_path,
    nested_view,
    selected_dataset_specs,
    sha256_path,
)
from paper_v2_transfer_common import impute_observed_window  # noqa: E402
from run_synthetic_v2_near_distance_calibration import (  # noqa: E402
    is_informative_target,
)
from synthetic_feature_profile import (  # noqa: E402
    gift_eval_short_term_test_holdout_steps,
    limit_candidates,
    read_gift_arrow_targets,
    read_tsf_series_records,
)


SCHEMA_VERSION = "paper_v4_real_evaluation_suite.v1"
SAMPLE_SCHEMA_VERSION = "paper_v4_real_evaluation_sample.v1"
SUPPORT_SCHEMA_VERSION = "paper_v4_real_evaluation_support.v1"
MANIFEST_SCHEMA_VERSION = "paper_v4_real_evaluation_manifest.v1"
CONTEXT_LENGTHS = (96, 168, 336, 504)
MAX_CONTEXT_LENGTH = max(CONTEXT_LENGTHS)
HORIZON = 48
SEASON_LENGTH = 24
TASK_ID = "univariate"
DEFAULT_MAX_SAMPLES_PER_DATASET = 240
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v4/02_real_evaluation_suite"
)


@dataclass(frozen=True)
class CandidateRef:
    native_record_index: int
    item_id: str
    series_id: str
    channel_index: int
    values: np.ndarray
    origin_index: int
    source_origin: int
    evaluation_split_start: int
    development_read_end_exclusive: int
    origin_role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-isolated, dataset-local real evaluation samples "
            "for H=48 and L in {96, 168, 336, 504}."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gift-eval-dir", type=Path, default=DEFAULT_GIFT_EVAL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_DATASET,
        help="Maximum paired master samples before expansion to four lookbacks.",
    )
    return parser.parse_args()


def canonical_frequency(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"h", "1h", "hour", "hourly"}:
        return "h"
    return normalized


def evaluation_origins(
    *,
    kind: str,
    series_length: int,
    official_test_tail_steps: int,
) -> list[tuple[int, int, int, int, str]]:
    """Return origin tuples without exposing evaluation targets to development.

    Each tuple contains ``(origin_index, source_origin, evaluation_split_start,
    development_read_end_exclusive, origin_role)``.
    """

    if kind == "gift_univariate":
        if official_test_tail_steps <= 0 or official_test_tail_steps % HORIZON:
            raise ValueError(
                "GIFT official test tail must be a positive multiple of H=48"
            )
        evaluation_start = int(series_length - official_test_tail_steps)
        development_end = int(evaluation_start - HORIZON)
        return [
            (
                origin_index,
                int(evaluation_start + origin_index * HORIZON),
                evaluation_start,
                development_end,
                "gift_official_short_term_test_tail",
            )
            for origin_index in range(official_test_tail_steps // HORIZON)
        ]
    if kind == "tsf_univariate":
        evaluation_start = int(series_length - HORIZON)
        return [
            (
                0,
                evaluation_start,
                evaluation_start,
                evaluation_start,
                "tsf_final_internal_validation_horizon",
            )
        ]
    raise ValueError(f"unsupported dataset kind: {kind}")


def candidate_refs(
    *,
    dataset: DatasetSpec,
    records: list[tuple[str, np.ndarray]],
    official_test_tail_steps: int,
) -> tuple[list[CandidateRef], dict[str, Any]]:
    candidates: list[CandidateRef] = []
    expanded_series_count = 0
    insufficient_history_count = 0
    for native_record_index, (item_id, native_values) in enumerate(records):
        native = np.asarray(native_values, dtype=float)
        channels = native if native.ndim == 2 else native.reshape(1, -1)
        for channel_index, values in enumerate(channels):
            expanded_series_count += 1
            series_id = (
                str(item_id)
                if native.ndim == 1
                else f"{item_id}:dim:{channel_index}"
            )
            for (
                origin_index,
                source_origin,
                evaluation_split_start,
                development_end,
                origin_role,
            ) in evaluation_origins(
                kind=dataset.kind,
                series_length=len(values),
                official_test_tail_steps=official_test_tail_steps,
            ):
                if (
                    source_origin < MAX_CONTEXT_LENGTH
                    or source_origin + HORIZON > len(values)
                ):
                    insufficient_history_count += 1
                    continue
                if source_origin < evaluation_split_start:
                    raise AssertionError(
                        "evaluation target begins before its isolated split"
                    )
                candidates.append(
                    CandidateRef(
                        native_record_index=native_record_index,
                        item_id=str(item_id),
                        series_id=series_id,
                        channel_index=channel_index,
                        values=np.asarray(values, dtype=float),
                        origin_index=origin_index,
                        source_origin=source_origin,
                        evaluation_split_start=evaluation_split_start,
                        development_read_end_exclusive=development_end,
                        origin_role=origin_role,
                    )
                )
    return candidates, {
        "native_record_count": len(records),
        "expanded_series_count": expanded_series_count,
        "raw_candidate_count": len(candidates),
        "insufficient_history_count": insufficient_history_count,
    }


def balanced_candidate_subset(
    candidates: list[CandidateRef],
    max_items: int,
) -> list[CandidateRef]:
    """Balance the processing pool across official origins and series."""

    if max_items <= 0 or not candidates:
        return []
    by_origin: dict[int, list[CandidateRef]] = defaultdict(list)
    for candidate in sorted(
        candidates,
        key=lambda row: (
            row.origin_index,
            row.series_id,
            row.native_record_index,
            row.channel_index,
        ),
    ):
        by_origin[candidate.origin_index].append(candidate)
    origin_indexes = sorted(by_origin)
    allocations = {origin_index: 0 for origin_index in origin_indexes}
    slots = min(max_items, len(candidates))
    while slots:
        progressed = False
        for origin_index in origin_indexes:
            if allocations[origin_index] >= len(by_origin[origin_index]):
                continue
            allocations[origin_index] += 1
            slots -= 1
            progressed = True
            if not slots:
                break
        if not progressed:
            break
    selected_by_origin = {
        origin_index: limit_candidates(
            (
                by_origin[origin_index][
                    origin_index % len(by_origin[origin_index]) :
                ]
                + by_origin[origin_index][
                    : origin_index % len(by_origin[origin_index])
                ]
            ),
            allocations[origin_index],
        )
        for origin_index in origin_indexes
    }
    selected: list[CandidateRef] = []
    row_index = 0
    while len(selected) < min(max_items, len(candidates)):
        active = False
        for origin_index in origin_indexes:
            rows = selected_by_origin[origin_index]
            if row_index < len(rows):
                selected.append(rows[row_index])
                active = True
        if not active:
            break
        row_index += 1
    return selected[:max_items]


def rows_from_candidate(
    dataset: DatasetSpec,
    candidate: CandidateRef,
    *,
    frequency: str,
    official_test_tail_steps: int,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    origin = candidate.source_origin
    raw_future = np.asarray(
        candidate.values[origin : origin + HORIZON],
        dtype=float,
    )
    if raw_future.shape != (HORIZON,) or not np.isfinite(raw_future).all():
        return None, "future_missing"

    histories: dict[int, np.ndarray] = {}
    observed_fractions: dict[int, float] = {}
    raw_master = np.asarray(
        candidate.values[
            origin - MAX_CONTEXT_LENGTH : origin + HORIZON
        ],
        dtype=float,
    )
    if raw_master.shape != (MAX_CONTEXT_LENGTH + HORIZON,):
        return None, "insufficient_history"
    for lookback in CONTEXT_LENGTHS:
        raw_view = nested_view(raw_master, lookback)
        raw_history = raw_view[:lookback]
        history, observed_fraction = impute_observed_window(raw_history)
        if history is None:
            return None, "context_missing"
        if not is_informative_target(history[:, None], lookback):
            return None, "context_uninformative"
        histories[lookback] = np.asarray(history, dtype=float)
        observed_fractions[lookback] = float(observed_fraction)

    master_raw_id = (
        f"{dataset.dataset_id}|{candidate.series_id}|{origin}|"
        f"{MAX_CONTEXT_LENGTH}|{HORIZON}"
    )
    master_sample_id = "real-master-" + hashlib.sha256(
        master_raw_id.encode("utf-8")
    ).hexdigest()[:24]
    future_sha256 = hashlib.sha256(
        np.asarray(raw_future, dtype="<f8").tobytes()
    ).hexdigest()
    rows: list[dict[str, Any]] = []
    for lookback in CONTEXT_LENGTHS:
        sample_id = "real-eval-" + hashlib.sha256(
            f"{master_sample_id}|L{lookback}".encode("utf-8")
        ).hexdigest()[:24]
        rows.append(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "sample_id": sample_id,
                "master_sample_id": master_sample_id,
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "domain": dataset.domain,
                "task_id": TASK_ID,
                "lookback": lookback,
                "horizon": HORIZON,
                "season_length": SEASON_LENGTH,
                "frequency": canonical_frequency(frequency),
                "target_dim": 1,
                "target_history": histories[lookback].astype(float).tolist(),
                "target_future": raw_future.astype(float).tolist(),
                "future_sha256": future_sha256,
                "series_id": candidate.series_id,
                "item_id": candidate.item_id,
                "native_record_index": candidate.native_record_index,
                "channel_index": candidate.channel_index,
                "origin_index": candidate.origin_index,
                "source_origin": origin,
                "origin_role": candidate.origin_role,
                "evaluation_split_start": candidate.evaluation_split_start,
                "development_read_end_exclusive": (
                    candidate.development_read_end_exclusive
                ),
                "official_test_tail_steps": official_test_tail_steps,
                "context_observed_fraction": observed_fractions[lookback],
            }
        )
    return rows, None


def build_rows_from_records(
    dataset: DatasetSpec,
    *,
    records: list[tuple[str, np.ndarray]],
    frequency: str,
    official_test_tail_steps: int,
    max_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, candidate_summary = candidate_refs(
        dataset=dataset,
        records=records,
        official_test_tail_steps=official_test_tail_steps,
    )
    processing_limit = min(
        len(candidates),
        max(max_samples, max_samples * 8),
    )
    processing_pool = balanced_candidate_subset(candidates, processing_limit)
    rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    selected_master_count = 0
    used_series: set[str] = set()
    selected_per_origin: dict[int, int] = defaultdict(int)
    for candidate in processing_pool:
        candidate_rows, rejection_reason = rows_from_candidate(
            dataset,
            candidate,
            frequency=frequency,
            official_test_tail_steps=official_test_tail_steps,
        )
        if candidate_rows is None:
            rejection_counts[str(rejection_reason)] += 1
            continue
        rows.extend(candidate_rows)
        selected_master_count += 1
        used_series.add(candidate.series_id)
        selected_per_origin[candidate.origin_index] += 1
        if selected_master_count >= max_samples:
            break

    support = {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.dataset_name,
        "domain": dataset.domain,
        "kind": dataset.kind,
        "task_id": TASK_ID,
        "status": "supported" if selected_master_count else "unsupported",
        "reason_codes": (
            []
            if selected_master_count
            else ["no_eligible_evaluation_samples"]
        ),
        "frequency": canonical_frequency(frequency),
        "context_lengths": list(CONTEXT_LENGTHS),
        "horizon": HORIZON,
        "season_length": SEASON_LENGTH,
        "official_test_tail_steps": int(official_test_tail_steps),
        **candidate_summary,
        "processed_candidate_count": len(processing_pool),
        "selected_master_sample_count": selected_master_count,
        "output_row_count": len(rows),
        "used_series_count": len(used_series),
        "selected_per_origin": {
            str(key): int(value)
            for key, value in sorted(selected_per_origin.items())
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return rows, support


def build_dataset(
    dataset: DatasetSpec,
    *,
    gift_eval_dir: Path,
    data_dir: Path,
    max_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = dataset_asset_path(
        dataset,
        gift_eval_dir=gift_eval_dir,
        data_dir=data_dir,
    )
    if dataset.kind == "gift_univariate":
        frequency, records = read_gift_arrow_targets(path)
        official_test_tail_steps = gift_eval_short_term_test_holdout_steps(
            frequency,
            records,
        )
    elif dataset.kind == "tsf_univariate":
        metadata, tsf_records = read_tsf_series_records(path)
        declared_horizon = int(metadata.get("horizon", HORIZON))
        if declared_horizon != HORIZON:
            raise ValueError(
                f"{dataset.dataset_id} declares horizon={declared_horizon}, "
                f"expected {HORIZON}"
            )
        frequency = str(metadata.get("frequency", dataset.frequency))
        records = [
            (record.series_id, np.asarray(record.values, dtype=float))
            for record in tsf_records
        ]
        official_test_tail_steps = 0
    else:
        raise ValueError(f"unsupported dataset kind: {dataset.kind}")

    rows, support = build_rows_from_records(
        dataset,
        records=records,
        frequency=frequency,
        official_test_tail_steps=official_test_tail_steps,
        max_samples=max_samples,
    )
    return rows, {
        **support,
        "asset_path": str(path),
        "asset_sha256": sha256_path(path),
    }


def build_suite(
    datasets: tuple[DatasetSpec, ...],
    *,
    gift_eval_dir: Path,
    data_dir: Path,
    max_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    created_at = datetime.now(timezone.utc).isoformat()
    all_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(datasets):
        print(
            f"[{dataset_index + 1}/{len(datasets)}] real eval "
            f"{dataset.dataset_id}",
            flush=True,
        )
        try:
            rows, support = build_dataset(
                dataset,
                gift_eval_dir=gift_eval_dir,
                data_dir=data_dir,
                max_samples=max_samples,
            )
        except Exception as error:
            rows = []
            support = {
                **asdict(dataset),
                "task_id": TASK_ID,
                "status": "unsupported",
                "reason_codes": ["dataset_build_failed"],
                "detail": str(error),
                "selected_master_sample_count": 0,
                "output_row_count": 0,
            }
        all_rows.extend(rows)
        support_rows.append(support)

    all_rows.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            int(row["origin_index"]),
            str(row["series_id"]),
            int(row["lookback"]),
        )
    )
    support_payload = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "created_at": created_at,
        "supported_dataset_count": sum(
            row["status"] == "supported" for row in support_rows
        ),
        "unsupported_dataset_count": sum(
            row["status"] == "unsupported" for row in support_rows
        ),
        "datasets": support_rows,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "dataset_ids": [dataset.dataset_id for dataset in datasets],
        "task_id": TASK_ID,
        "context_lengths": list(CONTEXT_LENGTHS),
        "horizon": HORIZON,
        "season_length": SEASON_LENGTH,
        "max_samples_per_dataset": max_samples,
        "pairing_policy": (
            "one raw 504+48 evaluation master expands to four suffix histories "
            "with one identical raw future"
        ),
        "isolation_policy": {
            "gift_univariate": (
                "targets come only from the official short-term test tail; "
                "profile/gate development excluded the tail and preceding H=48"
            ),
            "tsf_univariate": (
                "targets are the final H=48 internal validation horizon that "
                "profile/gate development excluded"
            ),
        },
    }
    return all_rows, support_payload, config


def write_outputs(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    support: dict[str, Any],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"real evaluation suite is already sealed: {output_dir}"
        )
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "dataset_support.json", support)
    write_jsonl(output_dir / "real_samples.jsonl", rows)
    output_files = (
        "config.json",
        "dataset_support.json",
        "real_samples.jsonl",
    )
    dependencies = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/build_paper_v4_profile_suite.py",
        REPO_ROOT / "scripts/synthetic_feature_profile.py",
        REPO_ROOT / "scripts/paper_v2_transfer_common.py",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "dependencies": {
            str(path.relative_to(REPO_ROOT)): sha256_path(path)
            for path in dependencies
        },
        "files": {
            name: {
                "sha256": sha256_path(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in output_files
        },
    }
    write_json(output_dir / "manifest.json", manifest)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
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


def main() -> int:
    args = parse_args()
    if args.max_samples_per_dataset < 1:
        raise ValueError("--max-samples-per-dataset must be positive")
    datasets = selected_dataset_specs(args.datasets)
    rows, support, config = build_suite(
        datasets,
        gift_eval_dir=args.gift_eval_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        max_samples=int(args.max_samples_per_dataset),
    )
    write_outputs(
        args.output_dir.resolve(),
        rows=rows,
        support=support,
        config=config,
    )
    print(
        f"real evaluation suite: {args.output_dir.resolve()} "
        f"({len(rows)} rows)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
