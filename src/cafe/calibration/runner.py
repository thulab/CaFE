#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from cafe.data.real import load_real_dataset
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_GENERATOR_VERSION,
    REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS,
    build_availability,
    fit_background_capability_contracts,
    public_background,
)
from cafe.generation.real_anchored_policy import (
    REAL_ANCHORED_FORMAL_CAPABILITIES,
    REAL_ANCHORED_QUALIFICATION_ONLY_CAPABILITIES,
    protocol_decisions as real_anchored_protocol_decisions,
)
from cafe.generation.reference_bank import (
    build_combined_real_anchored_bank_split_audit,
    freeze_real_anchored_qualification_policy,
    split_real_anchored_background_banks,
    unavailable_real_anchored_qualification_policy,
    validate_evaluation_qualification_policy,
)
from cafe.generation.structural_real_counterfactuals import (
    STRUCTURAL_CAPABILITIES,
    build_structural_donor_commitment_manifest,
    build_structural_real_anchored_backgrounds,
    fit_structural_capability_contracts,
    public_structural_background,
)


DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"
DEFAULT_GIFT_EVAL_DIR = protocol.REPO_ROOT / "data" / "gift-eval"
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
    record_limit = (
        max(64, min(256, int(math.ceil(args.max_anchors / 4))))
        if dataset.real_data_adapter == "m5_csv"
        else None
    )
    real_bundle = load_real_dataset(
        dataset.real_data_adapter,
        source_root / dataset.asset_name,
        record_limit=record_limit,
    )
    anchor_started = time.perf_counter()
    anchors, source_metadata = protocol.build_calibration_anchors(
        dataset,
        source_root=source_root,
        maximum_anchors=args.max_anchors,
        minimum_observed_fraction=args.minimum_observed_fraction,
        real_bundle=real_bundle,
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
    real_anchored_started = time.perf_counter()
    univariate_capabilities = tuple(
        capability_id
        for capability_id in args.capabilities
        if capability_id not in STRUCTURAL_CAPABILITIES
    )
    structural_capabilities = tuple(
        capability_id
        for capability_id in args.capabilities
        if capability_id in STRUCTURAL_CAPABILITIES
    )
    candidate_limit = max(2, 2 * int(args.max_anchors))
    candidate_backgrounds, real_anchored_source = (
        protocol.build_real_anchored_backgrounds(
            dataset,
            source_root=source_root,
            maximum_backgrounds=candidate_limit,
            minimum_observed_fraction=args.minimum_observed_fraction,
            real_bundle=real_bundle,
        )
    )
    structural_candidates, structural_source = (
        build_structural_real_anchored_backgrounds(
            dataset,
            source_root=source_root,
            maximum_backgrounds=candidate_limit,
            minimum_observed_fraction=args.minimum_observed_fraction,
            real_bundle=real_bundle,
        )
    )
    (
        combined_evaluation_backgrounds,
        combined_reference_backgrounds,
        combined_split_base_audit,
    ) = split_real_anchored_background_banks(
        [*candidate_backgrounds, *structural_candidates],
        maximum_evaluation_backgrounds=2 * args.max_anchors,
        maximum_reference_backgrounds=2 * args.max_anchors,
        source_window_length=protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
    )
    is_structural = lambda row: str(row.get("schema_version", "")).startswith(
        "cafe.structural_real_background."
    )
    private_backgrounds = [
        row for row in combined_evaluation_backgrounds
        if not is_structural(row)
    ][: args.max_anchors]
    structural_private_backgrounds = [
        row for row in combined_evaluation_backgrounds
        if is_structural(row)
    ][: args.max_anchors]
    reference_private_backgrounds = [
        row for row in combined_reference_backgrounds
        if not is_structural(row)
    ][: args.max_anchors]
    structural_reference_private_backgrounds = [
        row for row in combined_reference_backgrounds
        if is_structural(row)
    ][: args.max_anchors]
    reference_univariate_contracts, _reference_univariate_availability = (
        fit_background_capability_contracts(
            reference_private_backgrounds,
            capability_ids=univariate_capabilities,
        )
    )
    reference_structural_contracts, _reference_structural_availability = (
        fit_structural_capability_contracts(
            structural_reference_private_backgrounds,
            capability_ids=structural_capabilities,
        )
    )
    reference_contracts = [
        *reference_univariate_contracts,
        *reference_structural_contracts,
    ]
    reference_background_ids = [
        str(background["background_id"])
        for background in (
            *reference_private_backgrounds,
            *structural_reference_private_backgrounds,
        )
    ]
    combined_bank_split_audit = (
        build_combined_real_anchored_bank_split_audit(
            [*private_backgrounds, *structural_private_backgrounds],
            [
                *reference_private_backgrounds,
                *structural_reference_private_backgrounds,
            ],
            base_split_audit=combined_split_base_audit,
        )
    )
    reference_capability_ids = {
        str(row["capability_id"]) for row in reference_contracts
    }
    qualified_univariate_capabilities = tuple(
        capability_id
        for capability_id in univariate_capabilities
        if capability_id in reference_capability_ids
    )
    qualified_structural_capabilities = tuple(
        capability_id
        for capability_id in structural_capabilities
        if capability_id in reference_capability_ids
    )
    if reference_contracts:
        qualification_policy = freeze_real_anchored_qualification_policy(
            reference_contracts,
            reference_background_ids=reference_background_ids,
            bank_split_audit=combined_bank_split_audit,
        )
        real_anchored_contracts, real_anchored_availability = (
            fit_background_capability_contracts(
                private_backgrounds,
                capability_ids=qualified_univariate_capabilities,
                qualification_policy=qualification_policy,
            )
        )
        structural_contracts, qualified_structural_availability = (
            fit_structural_capability_contracts(
                structural_private_backgrounds,
                capability_ids=qualified_structural_capabilities,
                frozen_qualification_policy=qualification_policy,
            )
        )
    else:
        qualification_policy = unavailable_real_anchored_qualification_policy(
            reference_background_ids=reference_background_ids,
            bank_split_audit=combined_bank_split_audit,
        )
        real_anchored_contracts = []
        structural_contracts = []
        _unused_rows, qualified_structural_availability = (
            fit_structural_capability_contracts((), capability_ids=())
        )
        del _unused_rows
    real_anchored_availability = build_availability(
        real_anchored_contracts,
        requested_capability_ids=univariate_capabilities,
        minimum_eligible_backgrounds=(
            REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS
        ),
    )
    missing_univariate_reference = sorted(
        set(univariate_capabilities) - reference_capability_ids
    )
    if missing_univariate_reference:
        real_anchored_availability["qualification_block_reason"] = (
            "independent_reference_bank_unavailable"
        )
        real_anchored_availability[
            "qualification_blocked_capabilities"
        ] = missing_univariate_reference
        for cell in real_anchored_availability["cells"]:
            if cell["capability_id"] in missing_univariate_reference:
                cell["reason_codes"] = sorted(
                    {
                        *cell["reason_codes"],
                        "independent_reference_bank_unavailable",
                    }
                )
    _unused_rows, structural_availability = fit_structural_capability_contracts(
        (),
        capability_ids=structural_capabilities,
    )
    del _unused_rows
    if qualified_structural_capabilities:
        qualified_cells = {
            str(cell["capability_id"]): cell
            for cell in qualified_structural_availability["cells"]
        }
        structural_availability[
            "formal_background_count_by_capability"
        ].update(
            qualified_structural_availability[
                "formal_background_count_by_capability"
            ]
        )
        structural_availability[
            "sensitivity_background_count_by_capability"
        ].update(
            qualified_structural_availability[
                "sensitivity_background_count_by_capability"
            ]
        )
        structural_availability[
            "qualification_background_count_by_capability"
        ].update(
            qualified_structural_availability[
                "qualification_background_count_by_capability"
            ]
        )
        structural_availability["cells"] = [
            qualified_cells.get(str(cell["capability_id"]), cell)
            for cell in structural_availability["cells"]
        ]
        structural_availability["unavailable_reason_counts"] = (
            qualified_structural_availability[
                "unavailable_reason_counts"
            ]
        )
    structural_availability["frozen_qualification_policy_sha256"] = (
        qualification_policy["qualification_policy_sha256"]
    )
    missing_structural_reference = sorted(
        set(structural_capabilities) - reference_capability_ids
    )
    if missing_structural_reference:
        structural_availability["qualification_block_reason"] = (
            "independent_reference_bank_unavailable"
        )
        structural_availability[
            "qualification_blocked_capabilities"
        ] = missing_structural_reference
        for cell in structural_availability["cells"]:
            if cell["capability_id"] in missing_structural_reference:
                cell["reason_codes"] = sorted(
                    {
                        *cell["reason_codes"],
                        "independent_reference_bank_unavailable",
                    }
                )
    validate_evaluation_qualification_policy(
        [*real_anchored_contracts, *structural_contracts],
        qualification_policy,
    )
    real_anchored_availability["dataset_id"] = dataset.dataset_id
    for cell in real_anchored_availability["cells"]:
        cell["dataset_id"] = dataset.dataset_id
    real_anchored_backgrounds = [
        public_background(background) for background in private_backgrounds
    ]
    structural_backgrounds = [
        public_structural_background(background)
        for background in structural_private_backgrounds
    ]
    reference_backgrounds = [
        public_background(background)
        for background in reference_private_backgrounds
    ]
    structural_reference_backgrounds = [
        public_structural_background(background)
        for background in structural_reference_private_backgrounds
    ]
    structural_donor_commitments = (
        build_structural_donor_commitment_manifest(
            structural_backgrounds,
            structural_contracts,
            dataset_id=dataset.dataset_id,
        )
    )
    real_anchored_background_path = (
        output_dir / "real_anchored_backgrounds.jsonl"
    )
    protocol.write_jsonl(
        real_anchored_background_path,
        real_anchored_backgrounds,
    )
    real_anchored_contract_path = (
        output_dir / "real_anchored_contracts.jsonl"
    )
    protocol.write_jsonl(
        real_anchored_contract_path,
        real_anchored_contracts,
    )
    real_anchored_availability_path = (
        output_dir / "real_anchored_availability.json"
    )
    protocol.write_json(
        real_anchored_availability_path,
        real_anchored_availability,
    )
    structural_background_path = (
        output_dir / "structural_real_anchored_backgrounds.jsonl"
    )
    protocol.write_jsonl(
        structural_background_path,
        structural_backgrounds,
    )
    structural_contract_path = (
        output_dir / "structural_real_anchored_contracts.jsonl"
    )
    protocol.write_jsonl(structural_contract_path, structural_contracts)
    structural_donor_commitment_path = (
        output_dir / "structural_real_anchored_donor_commitments.json"
    )
    protocol.write_json(
        structural_donor_commitment_path,
        structural_donor_commitments,
    )
    structural_availability_path = (
        output_dir / "structural_real_anchored_availability.json"
    )
    protocol.write_json(structural_availability_path, structural_availability)
    reference_background_path = (
        output_dir / "real_anchored_reference_backgrounds.jsonl"
    )
    protocol.write_jsonl(
        reference_background_path,
        reference_backgrounds,
    )
    structural_reference_background_path = (
        output_dir / "structural_real_anchored_reference_backgrounds.jsonl"
    )
    protocol.write_jsonl(
        structural_reference_background_path,
        structural_reference_backgrounds,
    )
    reference_contract_path = (
        output_dir / "real_anchored_reference_contracts.jsonl"
    )
    protocol.write_jsonl(reference_contract_path, reference_contracts)
    bank_split_path = output_dir / "real_anchored_bank_split_audit.json"
    protocol.write_json(bank_split_path, combined_bank_split_audit)
    qualification_policy_path = (
        output_dir / "real_anchored_qualification_policy.json"
    )
    protocol.write_json(qualification_policy_path, qualification_policy)
    hierarchy_qualification_path = (
        output_dir / "structural_hierarchy_qualification.jsonl"
    )
    hierarchy_qualification_count = protocol.write_jsonl(
        hierarchy_qualification_path,
        (
            row
            for row in structural_contracts
            if row["capability_id"] == "hierarchical_coherence"
        ),
    )
    real_anchored_seconds = time.perf_counter() - real_anchored_started
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
        "schema_version": "cafe.calibration_bundle.v3",
        "created_at": protocol.utc_now(),
        "pipeline_schema_version": protocol.SCHEMA_VERSION,
        "generator_version": protocol.GENERATOR_VERSION,
        "dataset": source_metadata["dataset"],
        "source": source_metadata,
        "anchor_count": len(anchors),
        "real_forecast_anchor_count": len(real_forecast_masters),
        "real_anchored_background_count": len(real_anchored_backgrounds),
        "real_anchored_contract_count": len(real_anchored_contracts),
        "real_anchored_reference_background_count": len(
            reference_backgrounds
        ),
        "real_anchored_reference_contract_count": len(reference_contracts),
        "structural_real_anchored_background_count": len(
            structural_backgrounds
        ),
        "structural_real_anchored_contract_count": len(structural_contracts),
        "real_anchored_generator_version": REAL_ANCHORED_GENERATOR_VERSION,
        "real_anchored_source": real_anchored_source,
        "real_anchored_availability": real_anchored_availability,
        "structural_real_anchored_availability": structural_availability,
        "real_anchored_protocol": real_anchored_protocol_decisions(),
        "real_anchored_qualification_policy_sha256": qualification_policy[
            "qualification_policy_sha256"
        ],
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
                "real_anchored_background_and_contracts": (
                    real_anchored_seconds
                ),
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
            "real_anchored_counterfactual": {
                "benchmark_track": "real_anchored_counterfactual",
                "fit_scope": "history_only_l504",
                "model_visible_baseline": "suffix_l336_plus_real_h48",
                "normalization": "shared_unmodified_real_l336_history",
                "multi_seasonal_law": (
                    "x_alpha=x+(alpha-1)*secondary_harmonic_sum; "
                    "carrier fixed"
                ),
                "trend_law": (
                    "x_alpha=x+(alpha-1)*local_trend_nonlinearity; "
                    "level and linear trend fixed"
                ),
                "time_varying_seasonality_law": (
                    "x_alpha=x+(alpha-1)*phase_locked_symmetric_"
                    "constrained_am_component; carrier phase fixed"
                ),
                "regime_switching_law": (
                    "x_alpha=x+(alpha-1)*history_joinpoint_level_shift; "
                    "constant post-join extension"
                ),
                "nonlinear_persistence_law": (
                    "history_parameter_intervention_plus_zero_future_"
                    "innovation_recursive_rollout_delta"
                ),
                "predictable_intermittency_law": (
                    "x_alpha=x+(alpha-1)*history_fitted_sparse_clock_"
                    "pulse_template"
                ),
                "structural_laws": (
                    "authentic_synchronized_panel_or_known_future_"
                    "covariate_components_with_d_ge_3_formal_gate"
                ),
                "hierarchy_policy": (
                    "qualification_only_never_generated_or_ranked"
                ),
                "input_ablation_policy": (
                    "mandatory_for_common_and_cross_reported_separately_"
                    "not_score_weighted"
                ),
                "qualification_thresholds": (
                    "frozen_on_source_time_disjoint_reference_bank; final_"
                    "evaluation_origins_forbidden_for_tuning"
                ),
                "future_semantics": (
                    "observed real nuisance plus deterministic intervention"
                ),
                "ranking_separation": (
                    "never included in deterministic synthetic scores"
                ),
            },
        },
        "files": {
            "anchors": protocol.file_record(anchor_path),
            "real_anchor_masters": protocol.file_record(real_forecast_path),
            "real_anchored_backgrounds": protocol.file_record(
                real_anchored_background_path
            ),
            "real_anchored_contracts": protocol.file_record(
                real_anchored_contract_path
            ),
            "real_anchored_availability": protocol.file_record(
                real_anchored_availability_path
            ),
            "structural_real_anchored_backgrounds": protocol.file_record(
                structural_background_path
            ),
            "structural_real_anchored_contracts": protocol.file_record(
                structural_contract_path
            ),
            "structural_real_anchored_donor_commitments": (
                protocol.file_record(structural_donor_commitment_path)
            ),
            "structural_real_anchored_availability": protocol.file_record(
                structural_availability_path
            ),
            "real_anchored_reference_backgrounds": protocol.file_record(
                reference_background_path
            ),
            "structural_real_anchored_reference_backgrounds": (
                protocol.file_record(structural_reference_background_path)
            ),
            "real_anchored_reference_contracts": protocol.file_record(
                reference_contract_path
            ),
            "real_anchored_bank_split_audit": protocol.file_record(
                bank_split_path
            ),
            "real_anchored_qualification_policy": protocol.file_record(
                qualification_policy_path
            ),
            "structural_hierarchy_qualification": {
                **protocol.file_record(hierarchy_qualification_path),
                "row_count": hierarchy_qualification_count,
            },
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
                "real_anchored_background_count": len(
                    real_anchored_backgrounds
                ),
                "real_anchored_available_capabilities": [
                    cell["capability_id"]
                    for cell in real_anchored_availability["cells"]
                    if cell["status"] == "available"
                ],
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
