#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from cafe import protocol


DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"
DEFAULT_GIFT_EVAL_DIR = Path("/root/xmy/gift-eval")
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the formal CaFE real-data calibration bundle."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=DEFAULT_GIFT_EVAL_DIR,
        help=(
            "Backward-compatible default source root for gift_arrow datasets."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Root consumed by the selected dataset's registered real-data "
            "adapter. Required for non-GIFT sources such as M5."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument(
        "--calibration-seeds",
        type=int,
        default=protocol.DEFAULT_CALIBRATION_PATH_COUNT,
        help=(
            "Independent qualification paths used only to estimate each "
            "family-level response curve; these are not formal sample seeds."
        ),
    )
    parser.add_argument(
        "--max-calibration-seeds",
        type=int,
        default=protocol.MAX_CALIBRATION_PATH_COUNT,
        help=(
            "Only used when the base qualification paths produce no usable "
            "family-level support."
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
        choices=protocol.CAPABILITIES,
        default=list(protocol.CAPABILITIES),
    )
    return parser.parse_args()


def calibrate_one_capability(
    dataset: protocol.DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    capability_id: str,
    calibration_seed_count: int,
    maximum_calibration_seed_count: int,
) -> dict[str, Any]:
    return protocol.calibrate_capabilities(
        dataset,
        anchors,
        calibration_seed_count=calibration_seed_count,
        maximum_calibration_seed_count=maximum_calibration_seed_count,
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
    merged["available_capabilities"] = [
        capability_id
        for capability_id in capability_ids
        if merged["capabilities"][capability_id][
            "available_for_generation"
        ]
    ]
    merged["unavailable_capabilities"] = {
        capability_id: list(
            merged["capabilities"][capability_id].get(
                "unavailable_reason_codes",
                [],
            )
        )
        for capability_id in capability_ids
        if not merged["capabilities"][capability_id][
            "available_for_generation"
        ]
    }
    return merged


def calibrate_capabilities(
    dataset: protocol.DatasetSpec,
    anchors: list[dict[str, Any]],
    *,
    capability_ids: tuple[str, ...],
    workers: int,
    calibration_seed_count: int,
    maximum_calibration_seed_count: int,
) -> dict[str, Any]:
    keyword_arguments = {
        "calibration_seed_count": calibration_seed_count,
        "maximum_calibration_seed_count": maximum_calibration_seed_count,
    }
    if workers == 1 or len(capability_ids) == 1:
        return protocol.calibrate_capabilities(
            dataset,
            anchors,
            capability_ids=capability_ids,
            progress_callback=lambda capability_id, path_count: print(
                protocol.canonical_json(
                    {
                        "dataset_id": dataset.dataset_id,
                        "qualifying_capability": capability_id,
                        "qualification_path_count": path_count,
                    }
                ),
                flush=True,
            ),
            **keyword_arguments,
        )

    results: dict[str, dict[str, Any]] = {}
    maximum_workers = min(workers, len(capability_ids))
    submission_order = protocol.preparation_capability_order(capability_ids)
    with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
        future_capabilities = {
            executor.submit(
                calibrate_one_capability,
                dataset,
                anchors,
                capability_id=capability_id,
                **keyword_arguments,
            ): capability_id
            for capability_id in submission_order
        }
        for future in as_completed(future_capabilities):
            capability_id = future_capabilities[future]
            result = future.result()
            results[capability_id] = result
            calibration = result["capabilities"][capability_id]
            print(
                protocol.canonical_json(
                    {
                        "calibrated_capability": capability_id,
                        "dataset_id": dataset.dataset_id,
                        "available_for_generation": calibration[
                            "available_for_generation"
                        ],
                        "qualification_path_count": calibration.get(
                            "qualification_path_count"
                        ),
                        "unavailable_reason_codes": calibration.get(
                            "unavailable_reason_codes",
                            [],
                        ),
                    }
                ),
                flush=True,
            )
    return merge_capability_calibrations(results, capability_ids)


def main() -> int:
    run_started = time.perf_counter()
    args = parse_args()
    if (
        args.max_anchors < 1
        or args.calibration_seeds < 1
        or args.max_calibration_seeds < args.calibration_seeds
        or args.workers < 1
    ):
        raise ValueError(
            "anchor, calibration path budgets, and workers must be positive "
            "and maximums must not be smaller than base counts"
        )
    if not 0.0 < args.minimum_observed_fraction <= 1.0:
        raise ValueError("minimum observed fraction must be in (0, 1]")
    dataset = protocol.resolve_dataset(args.dataset_id)
    if (
        dataset.real_data_adapter
        not in protocol.GIFT_EVAL_REAL_DATA_ADAPTERS
        and args.source_root is None
    ):
        raise ValueError(
            f"{dataset.dataset_id} uses adapter "
            f"{dataset.real_data_adapter!r}; pass --source-root"
        )
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else args.gift_eval_dir.resolve()
    )
    output_dir = args.output_root.resolve() / dataset.dataset_id / "01_calibration"
    anchor_started = time.perf_counter()
    anchors, source_metadata = protocol.build_calibration_anchors(
        dataset,
        source_root=source_root,
        maximum_anchors=args.max_anchors,
        minimum_observed_fraction=args.minimum_observed_fraction,
    )
    anchor_extraction_seconds = time.perf_counter() - anchor_started
    anchor_artifact_started = time.perf_counter()
    real_forecast_masters = [
        dict(anchor.pop("real_forecast_master")) for anchor in anchors
    ]
    anchor_path = output_dir / "anchors.jsonl"
    protocol.write_jsonl(anchor_path, anchors)
    real_forecast_path = output_dir / "real_anchor_masters.jsonl"
    protocol.write_jsonl(real_forecast_path, real_forecast_masters)
    anchor_artifact_seconds = (
        time.perf_counter() - anchor_artifact_started
    )
    capability_ids = tuple(args.capabilities)
    capability_calibration_started = time.perf_counter()
    capability_calibration = calibrate_capabilities(
        dataset,
        anchors,
        capability_ids=capability_ids,
        workers=args.workers,
        calibration_seed_count=args.calibration_seeds,
        maximum_calibration_seed_count=args.max_calibration_seeds,
    )
    capability_calibration_seconds = (
        time.perf_counter() - capability_calibration_started
    )
    calibration_artifact_started = time.perf_counter()
    capability_path = output_dir / "capability_calibration.json"
    protocol.write_json(capability_path, capability_calibration)
    calibration_artifact_seconds = (
        time.perf_counter() - calibration_artifact_started
    )
    elapsed_before_bundle_write = time.perf_counter() - run_started
    bundle = {
        "schema_version": "cafe.calibration_bundle.v1",
        "created_at": protocol.utc_now(),
        "pipeline_schema_version": protocol.SCHEMA_VERSION,
        "generator_version": protocol.GENERATOR_VERSION,
        "dataset": source_metadata["dataset"],
        "source": source_metadata,
        "anchor_count": len(anchors),
        "real_forecast_anchor_count": len(real_forecast_masters),
        "requested_capabilities": list(args.capabilities),
        "capabilities": list(
            capability_calibration["available_capabilities"]
        ),
        "unavailable_capabilities": capability_calibration[
            "unavailable_capabilities"
        ],
        "execution": {
            "capability_workers": min(args.workers, len(capability_ids)),
            "blas_threads_per_process": 1,
            "timing_seconds": {
                "anchor_extraction": anchor_extraction_seconds,
                "anchor_artifact_write": anchor_artifact_seconds,
                "capability_family_response_qualification": (
                    capability_calibration_seconds
                ),
                "capability_calibration_artifact_write": (
                    calibration_artifact_seconds
                ),
                "elapsed_before_bundle_write": (
                    elapsed_before_bundle_write
                ),
            },
        },
        "qualification_path_budget": {
            "policy": (
                "independent_family_response_qualification_bank_"
                "fixed_base_hard_failure_only_expansion_v1"
            ),
            "path_sampling": {
                "anchor": "independent_qualification_anchor_hash_v1",
                "rng": "independent_qualification_path_v1",
                "seed_start": 0,
            },
            "default": {
                "base": int(args.calibration_seeds),
                "maximum": int(args.max_calibration_seeds),
            },
            "split_half_diagnostic": "record_only_nonblocking",
        },
        "feature_contract": {
            "background_features": (
                "direct finite feature row from one forecastable real L168 anchor"
            ),
            "feature_schema_version": protocol.FEATURE_SCHEMA_VERSION,
            "feature_measurement": (
                "single history-only cafe feature vector; local trend evidence "
                "uses the trailing 96 observations"
            ),
            "univariate_primary_strength": (
                "six targets require the usable joint overlap between "
                "dataset q10-q90 and both families' mean response support; "
                "formal seeds share this family-level lambda scale; a missing "
                "or unsupported real coordinate makes the dataset-capability "
                "cell unavailable"
            ),
            "structural_primary_strength": (
                "common factor and cross-series use a native multivariate "
                "q10-q90 range; hierarchy constructs the parent from two "
                "synchronized real children and uses their q10-q90 range; "
                "covariate response requires a semantically matched "
                "known-future-covariate primary coordinate; missing structural "
                "inputs make the cell unavailable"
            ),
            "parameter_feature_provenance": (
                "per-parameter real_univariate, real_native_multivariate, "
                "real_hierarchy_children, real_known_future_covariates, "
                "protocol_constant, or explicit protocol_fallback"
            ),
            "structural_identifiability": (
                "selected-I5 primary-family reachability uses the "
                "real-aligned observable and paired construction on "
                "independent paths; common-factor and cross-series also "
                "require a separate lambda-1 blind positive control; "
                "near-distance is excluded from calibration"
            ),
            "response_inverse": (
                "21-point primary and secondary family-mean responses over "
                "independent qualification paths; both families must support "
                "the same real-derived targets; formal seeds are never "
                "individually inverted"
            ),
            "realized_target_alignment": (
                "diagnostic only; no per-sample target-error rejection"
            ),
            "real_feature_support_audit": (
                "raw anchor min-max expanded around its midpoint to 1.2 "
                "times the observed span; diagnostic only"
            ),
            "hard_generation_acceptance": (
                "finite valid samples, optional near-copy DCR/NNDR, "
                "batch intensity ordering, and mechanism structure"
            ),
            "removed_features": ["future_abs_covariate_target_corr"],
            "time_scale_semantics": {
                "calendar_season_length": (
                    "frequency-derived provenance"
                ),
                "feature_period": (
                    "calendar season when two cycles fit L168, otherwise "
                    "observable profile dominant period"
                ),
                "generator_period": (
                    "capability-specific observable clipping of the direct "
                    "anchor profile dominant period"
                ),
                "mase_period": (
                    "calendar season when defined inside L168, otherwise "
                    "non-seasonal lag 1"
                ),
            },
            "real_anchor_forecast": (
                "independent auxiliary table; every collected anchor stores "
                "L168 history plus a held-out H48 future and never enters "
                "synthetic mechanism ranking"
            ),
        },
        "files": {
            "anchors": protocol.file_record(anchor_path),
            "real_anchor_masters": protocol.file_record(real_forecast_path),
            "capability_calibration": protocol.file_record(capability_path),
        },
    }
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(output_dir / "calibration_bundle.json", bundle)
    total_seconds = time.perf_counter() - run_started
    print(
        protocol.canonical_json(
            {
                "dataset_id": dataset.dataset_id,
                "anchor_count": len(anchors),
                "real_forecast_anchor_count": len(real_forecast_masters),
                "output": str(output_dir),
                "bundle_content_sha256": bundle["bundle_content_sha256"],
                "timing_seconds": {
                    **bundle["execution"]["timing_seconds"],
                    "total": total_seconds,
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
