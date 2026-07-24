#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import paper_v8_pipeline_common as v8


DEFAULT_OUTPUT_ROOT = v8.REPO_ROOT / "runtime" / "paper_exp" / "v8"
DEFAULT_GIFT_EVAL_DIR = Path("/root/xmy/gift-eval")
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the formal Paper v8 real-data calibration bundle."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=DEFAULT_GIFT_EVAL_DIR,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument(
        "--calibration-seeds",
        type=int,
        default=v8.DEFAULT_CALIBRATION_PATH_COUNT,
    )
    parser.add_argument(
        "--max-calibration-seeds",
        type=int,
        default=v8.MAX_CALIBRATION_PATH_COUNT,
        help=(
            "Only used when the base response paths produce no usable "
            "support or primary inverse."
        ),
    )
    parser.add_argument(
        "--nonlinear-calibration-seeds",
        type=int,
        default=v8.DEFAULT_NONLINEAR_CALIBRATION_PATH_COUNT,
        help=(
            "Path budget for conservative nonlinear response support; other "
            "capabilities use --calibration-seeds."
        ),
    )
    parser.add_argument(
        "--max-nonlinear-calibration-seeds",
        type=int,
        default=v8.MAX_NONLINEAR_CALIBRATION_PATH_COUNT,
        help=(
            "Only used when the base nonlinear response paths produce no "
            "usable support or primary inverse."
        ),
    )
    parser.add_argument("--minimum-observed-fraction", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Independent capability calibration processes. Use 1 for the "
            "serial reference implementation."
        ),
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=v8.CAPABILITIES,
        default=list(v8.CAPABILITIES),
    )
    return parser.parse_args()


def calibrate_one_capability(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    capability_id: str,
    calibration_seed_count: int,
    maximum_calibration_seed_count: int,
    nonlinear_calibration_seed_count: int,
    maximum_nonlinear_calibration_seed_count: int,
) -> dict[str, Any]:
    return v8.calibrate_capabilities(
        dataset,
        anchors,
        calibration_seed_count=calibration_seed_count,
        maximum_calibration_seed_count=maximum_calibration_seed_count,
        nonlinear_calibration_seed_count=nonlinear_calibration_seed_count,
        maximum_nonlinear_calibration_seed_count=(
            maximum_nonlinear_calibration_seed_count
        ),
        capability_ids=(capability_id,),
    )


def merge_capability_calibrations(
    results: dict[str, dict[str, Any]],
    capability_ids: tuple[str, ...],
) -> dict[str, Any]:
    first = results[capability_ids[0]]
    merged = {
        key: value
        for key, value in first.items()
        if key != "capabilities"
    }
    merged["capabilities"] = {
        capability_id: results[capability_id]["capabilities"][capability_id]
        for capability_id in capability_ids
    }
    return merged


def calibrate_capabilities(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    capability_ids: tuple[str, ...],
    workers: int,
    calibration_seed_count: int,
    maximum_calibration_seed_count: int,
    nonlinear_calibration_seed_count: int,
    maximum_nonlinear_calibration_seed_count: int,
) -> dict[str, Any]:
    keyword_arguments = {
        "calibration_seed_count": calibration_seed_count,
        "maximum_calibration_seed_count": maximum_calibration_seed_count,
        "nonlinear_calibration_seed_count": nonlinear_calibration_seed_count,
        "maximum_nonlinear_calibration_seed_count": (
            maximum_nonlinear_calibration_seed_count
        ),
    }
    if workers == 1 or len(capability_ids) == 1:
        return v8.calibrate_capabilities(
            dataset,
            anchors,
            capability_ids=capability_ids,
            progress_callback=lambda capability_id, path_count: print(
                v8.canonical_json(
                    {
                        "dataset_id": dataset.dataset_id,
                        "calibrating_capability": capability_id,
                        "response_path_count": path_count,
                    }
                ),
                flush=True,
            ),
            **keyword_arguments,
        )

    results: dict[str, dict[str, Any]] = {}
    maximum_workers = min(workers, len(capability_ids))
    with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
        future_capabilities = {
            executor.submit(
                calibrate_one_capability,
                dataset,
                anchors,
                capability_id=capability_id,
                **keyword_arguments,
            ): capability_id
            for capability_id in capability_ids
        }
        for future in as_completed(future_capabilities):
            capability_id = future_capabilities[future]
            result = future.result()
            results[capability_id] = result
            calibration = result["capabilities"][capability_id]
            print(
                v8.canonical_json(
                    {
                        "calibrated_capability": capability_id,
                        "dataset_id": dataset.dataset_id,
                        "response_path_count": calibration[
                            "response_calibration_seed_count"
                        ],
                    }
                ),
                flush=True,
            )
    return merge_capability_calibrations(results, capability_ids)


def main() -> int:
    args = parse_args()
    if (
        args.max_anchors < 1
        or args.calibration_seeds < 1
        or args.max_calibration_seeds < args.calibration_seeds
        or args.nonlinear_calibration_seeds < 1
        or (
            args.max_nonlinear_calibration_seeds
            < args.nonlinear_calibration_seeds
        )
        or args.workers < 1
    ):
        raise ValueError(
            "anchor, calibration path budgets, and workers must be positive "
            "and maximums must not be smaller than base counts"
        )
    if not 0.0 < args.minimum_observed_fraction <= 1.0:
        raise ValueError("minimum observed fraction must be in (0, 1]")
    dataset = v8.resolve_dataset(args.dataset_id)
    output_dir = args.output_root.resolve() / dataset.dataset_id / "01_calibration"
    anchors, source_metadata = v8.build_calibration_anchors(
        dataset,
        gift_eval_dir=args.gift_eval_dir.resolve(),
        maximum_anchors=args.max_anchors,
        minimum_observed_fraction=args.minimum_observed_fraction,
    )
    anchor_path = output_dir / "anchors.jsonl"
    v8.write_jsonl(anchor_path, anchors)
    capability_ids = tuple(args.capabilities)
    capability_calibration = calibrate_capabilities(
        dataset,
        anchors,
        capability_ids=capability_ids,
        workers=args.workers,
        calibration_seed_count=args.calibration_seeds,
        maximum_calibration_seed_count=args.max_calibration_seeds,
        nonlinear_calibration_seed_count=args.nonlinear_calibration_seeds,
        maximum_nonlinear_calibration_seed_count=(
            args.max_nonlinear_calibration_seeds
        ),
    )
    capability_path = output_dir / "capability_calibration.json"
    v8.write_json(capability_path, capability_calibration)
    bundle = {
        "schema_version": "paper_v8_calibration_bundle.v4",
        "created_at": v8.utc_now(),
        "pipeline_schema_version": v8.SCHEMA_VERSION,
        "generator_version": v8.GENERATOR_VERSION,
        "dataset": source_metadata["dataset"],
        "source": source_metadata,
        "anchor_count": len(anchors),
        "capabilities": list(args.capabilities),
        "execution": {
            "capability_workers": min(args.workers, len(capability_ids)),
            "blas_threads_per_process": 1,
        },
        "response_calibration_path_budget": {
            "policy": (
                "formal_generation_seed_bank_"
                "fixed_base_hard_failure_only_expansion_v2"
            ),
            "path_sampling": {
                "anchor": "formal_logical_seed_hash_v1",
                "rng": "formal_generation_path_v1",
                "seed_start": 0,
            },
            "default": {
                "base": int(args.calibration_seeds),
                "maximum": int(args.max_calibration_seeds),
            },
            "nonlinear_persistence": {
                "base": int(args.nonlinear_calibration_seeds),
                "maximum": int(
                    args.max_nonlinear_calibration_seeds
                ),
            },
            "split_half_diagnostic": "record_only_nonblocking",
        },
        "feature_contract": {
            "background_features": (
                "direct finite feature row from one real L504 anchor"
            ),
            "univariate_primary_strength": (
                "dataset q10-q90 intersected with generator response support; "
                "predictable intermittency uses continuous generator-known "
                "event energy dose because thresholded real spike counts are "
                "zero-inflated"
            ),
            "structural_primary_strength": (
                "fixed cross-dataset evenly spaced realized-strength grid"
            ),
            "structural_identifiability": "measured on generated samples only",
            "removed_features": ["future_abs_covariate_target_corr"],
            "time_scale_semantics": {
                "calendar_season_length": (
                    "frequency-derived provenance"
                ),
                "feature_period": (
                    "calendar season when two cycles fit L504, otherwise "
                    "observable profile dominant period"
                ),
                "generator_period": (
                    "capability-specific observable clipping of the direct "
                    "anchor profile dominant period"
                ),
                "mase_period": (
                    "calendar season when defined inside L504, otherwise "
                    "non-seasonal lag 1"
                ),
            },
        },
        "files": {
            "anchors": v8.file_record(anchor_path),
            "capability_calibration": v8.file_record(capability_path),
        },
    }
    bundle["bundle_content_sha256"] = v8.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    v8.write_json(output_dir / "calibration_bundle.json", bundle)
    print(
        v8.canonical_json(
            {
                "dataset_id": dataset.dataset_id,
                "anchor_count": len(anchors),
                "output": str(output_dir),
                "bundle_content_sha256": bundle["bundle_content_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
