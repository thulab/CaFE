from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pytest

from cafe import protocol
from cafe.data.real import RealDatasetBundle, RealSeriesRecord
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_ALPHAS,
    REAL_ANCHORED_MASTER_SCHEMA,
    array_sha256,
    build_availability,
)
from cafe.generation.real_anchored_policy import (
    NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY,
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
)
from cafe.generation.reference_bank import (
    build_combined_real_anchored_bank_split_audit,
    freeze_real_anchored_qualification_policy,
    split_real_anchored_background_banks,
    unavailable_real_anchored_qualification_policy,
)
from cafe.generation.structural_real_counterfactuals import (
    STRUCTURAL_DONOR_COMMITMENT_POLICY,
    STRUCTURAL_DONOR_COMMITMENT_SCHEMA,
    _structural_donor_commitment_entry,
    build_matched_input_ablation_task,
    build_structural_donor_commitment_manifest,
    build_structural_real_anchored_backgrounds,
    fit_structural_capability_contracts,
    iter_mandatory_structural_input_ablation_tasks,
    iter_structural_real_anchored_samples,
    public_structural_background,
)
from cafe.validation import runner


def _real_anchored_rows() -> list[dict]:
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    length = context + protocol.HORIZON
    time = np.arange(length, dtype=float)
    baseline = (
        0.01 * time
        + np.sin(2.0 * np.pi * time / 24.0 + 0.2)
        + 0.2 * np.cos(2.0 * np.pi * time / 7.0)
    )
    controlled = 0.45 * np.sin(2.0 * np.pi * time / 12.0 + 0.4)
    history = baseline[:context]
    mase_scale = float(np.mean(np.abs(history[24:] - history[:-24])))
    baseline_hashes = {
        "baseline_history_sha256": array_sha256(baseline[:context]),
        "baseline_future_sha256": array_sha256(baseline[context:]),
        "baseline_target_sha256": array_sha256(baseline),
    }
    group_id = "real-cf-group-0"
    background_id = "background-0"
    contract_hash = "1" * 64
    capability_contract_hash = "2" * 64
    reference_hash = "3" * 64
    source_hash = "4" * 64
    decomposition_hash = "5" * 64
    standardization = {
        "scope": "shared_unmodified_real_l336_history",
        "location": 17.0,
        "scale": 2.5,
    }
    rows: list[dict] = []
    for dose_index, treatment_alpha in enumerate(
        REAL_ANCHORED_ALPHAS,
        start=1,
    ):
        pair_id = f"real-cf-pair-a{dose_index}"
        for member, alpha in ((0, 1.0), (1, treatment_alpha)):
            delta = (
                np.zeros((length, 1), dtype=float)
                if member == 0
                else ((alpha - 1.0) * controlled)[:, None]
            )
            target = baseline[:, None] + delta
            intervention_rms = float(np.sqrt(np.mean(delta**2)))
            visible_delta = target - baseline[:, None]
            metadata = {
                "capability_id": "multi_seasonal",
                "alpha": alpha,
                "contract_sha256": contract_hash,
                "capability_contract_sha256": capability_contract_hash,
                "source_history_sha256": source_hash,
                "decomposition_history_sha256": decomposition_hash,
                "intervention_rms": intervention_rms,
                "normalization_mean_by_target": [
                    float(np.mean(history))
                ],
                "normalization_scale_by_target": [
                    float(np.std(history))
                ],
                "normalization_policy": (
                    "baseline_history_shared_by_pair_v1"
                ),
                "mase_scale_by_target": [mase_scale],
                "mase_scale": mase_scale,
                "mase_period": 24,
                "mase_effective_period_by_target": [24],
                "mase_scale_source_by_target": ["seasonal_lag"],
                "mase_reference_policy": (
                    "baseline_history_shared_by_pair_v1"
                ),
                "reference_start": (
                    protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                    - context
                ),
                "reference_length": context,
                "reference_history_sha256": reference_hash,
                "reference_history_policy": (
                    "unmodified_fit_history_suffix_shared_by_pair_v1"
                ),
                "controlled_component": "secondary_harmonic_sum",
                "carrier_fixed": True,
            }
            sample_id = f"{pair_id}__m{member}"
            rows.append(
                {
                    "schema_version": REAL_ANCHORED_MASTER_SCHEMA,
                    "benchmark_track": "real_anchored_counterfactual",
                    "evaluation_table": "real_anchored_counterfactual",
                    "sample_id": sample_id,
                    "master_sample_id": sample_id,
                    "baseline_sample_id": f"{pair_id}__m0",
                    "paired_group_id": group_id,
                    "counterfactual_pair_id": pair_id,
                    "counterfactual_member": member,
                    "dataset_id": "test_hourly",
                    "capability_id": "multi_seasonal",
                    "generator_family_role": "real_anchored",
                    "anchor_id": background_id,
                    "background_id": background_id,
                    "seed_index": 0,
                    "context_length": context,
                    "horizon": protocol.HORIZON,
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "covariates": None,
                    "intensity": dose_index,
                    "intensity_lambda": alpha,
                    "dose_index": dose_index,
                    "dose_parameter": "alpha",
                    "dose_value": alpha,
                    "baseline_dose_value": 1.0,
                    "target_feature": "real_anchored_intervention_rms",
                    "intensity_target_feature_value": intervention_rms,
                    "target_feature_value": intervention_rms,
                    "intensity_calibration": {
                        "policy": (
                            "physical_component_amplitude_alpha_grid_v1"
                        ),
                        "scope": (
                            "real_anchored_history_only_decomposition"
                        ),
                        "selected_alphas": list(REAL_ANCHORED_ALPHAS),
                    },
                    "sampled_generator_parameters": {
                        "alpha": alpha,
                        "controlled_component": (
                            "secondary_harmonic_sum"
                        ),
                    },
                    "parameter_sampling": {
                        "policy": (
                            "real_background_contract_deterministic_selection_v1"
                        ),
                        "background_id": background_id,
                        "contract_sha256": contract_hash,
                    },
                    "generation_metadata": metadata,
                    "mase_period": 24,
                    "mase_scale": mase_scale,
                    "mase_scale_by_target": [mase_scale],
                    "mase_scale_effective_period_by_target": [24],
                    "mase_scale_fallback_target_indices": [],
                    "mase_scale_policy": "seasonal_lag_v1",
                    "mase_scale_source": (
                        "shared_unmodified_real_l336_history"
                    ),
                    "shared_standardization": copy.deepcopy(
                        standardization
                    ),
                    **baseline_hashes,
                    "intervention_delta_sha256": array_sha256(
                        visible_delta
                    ),
                    "target_sha256": (
                        protocol.target_and_covariate_sha256(target, None)
                    ),
                    "future_sha256": array_sha256(target[context:]),
                    "anti_copy_gate": {
                        "status": "not_applicable",
                        "reason_code": (
                            "intentional_real_anchor_counterfactual"
                        ),
                    },
                    "target": target.tolist(),
                }
            )
    return rows


def _rehash_target(row: dict, baseline_row: dict) -> None:
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    target = np.asarray(row["target"], dtype=float)
    baseline = np.asarray(baseline_row["target"], dtype=float)
    delta = target - baseline
    rms = float(np.sqrt(np.mean(delta**2)))
    row["target_sha256"] = protocol.target_and_covariate_sha256(
        target,
        None,
    )
    row["future_sha256"] = array_sha256(target[context:])
    row["intervention_delta_sha256"] = array_sha256(delta)
    row["target_feature_value"] = rms
    row["intensity_target_feature_value"] = rms
    row["generation_metadata"]["intervention_rms"] = rms


def _nonlinear_real_anchored_rows() -> list[dict]:
    rows = _real_anchored_rows()
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    baseline = np.asarray(rows[0]["target"], dtype=float)
    first_treatment = np.asarray(rows[1]["target"], dtype=float)
    unit_delta = (first_treatment - baseline) / (
        float(rows[1]["dose_value"]) - 1.0
    )
    qualification: list[dict[str, float]] = []
    for dose_index, alpha in enumerate(REAL_ANCHORED_ALPHAS):
        baseline_row = rows[2 * dose_index]
        treatment = rows[2 * dose_index + 1]
        nonlinear_scale = (alpha - 1.0) * (1.0 + 0.35 * (alpha - 1.0))
        treatment["target"] = (baseline + nonlinear_scale * unit_delta).tolist()
        _rehash_target(treatment, baseline_row)
        delta = np.asarray(treatment["target"], dtype=float) - baseline
        qualification.append(
            {
                "alpha": alpha,
                "intervention_rms": float(np.sqrt(np.mean(delta**2))),
                "visible_history_effect_rms": float(
                    np.sqrt(np.mean(delta[:context] ** 2))
                ),
                "future_effect_rms": float(
                    np.sqrt(np.mean(delta[context:] ** 2))
                ),
            }
        )
    for row in rows:
        metadata = row["generation_metadata"]
        metadata.update(
            {
                "capability_id": "nonlinear_persistence",
                "controlled_component": (
                    "bounded_nonlinear_autoregressive_gain"
                ),
                "dose_response_law": "dynamic_recursive_nonproportional",
                "dynamic_contract_replay_verified": True,
                "history_innovation_policy": (
                    "shared_observed_one_step_innovations"
                ),
                "history_innovation_sha256": "6" * 64,
                "future_innovation_policy": (
                    "zero_future_innovation_paired_rollout_v1"
                ),
                "future_innovation_sha256": "7" * 64,
                "future_component_source": (
                    "paired_zero_innovation_dynamic_rollout"
                ),
                "history_residual_replay_policy": (
                    "history_residual_replay_qualification_only_v1"
                ),
                "history_residual_replay_sensitivity_alpha2_rms": 0.31,
                "dose_response_qualification": copy.deepcopy(qualification),
            }
        )
        row["capability_id"] = "nonlinear_persistence"
        row["sampled_generator_parameters"]["controlled_component"] = (
            "bounded_nonlinear_autoregressive_gain"
        )
        target = np.asarray(row["target"], dtype=float)
        baseline_delta = target - baseline
        metadata["future_effect_rms"] = float(
            np.sqrt(np.mean(baseline_delta[context:] ** 2))
        )
    return rows


def _nonlinear_replay_sensitivity_rows(
    source_rows: list[dict],
) -> list[dict]:
    replay_rows: list[dict] = []
    for source in source_rows:
        row = copy.deepcopy(source)
        source_sample_id = str(source["sample_id"])
        source_pair_id = str(source["counterfactual_pair_id"])
        source_group_id = str(source["paired_group_id"])
        row["evaluation_table"] = (
            "real_anchored_nonlinear_replay_sensitivity"
        )
        row["sample_id"] = f"{source_sample_id}__nonlinear_replay"
        row["master_sample_id"] = row["sample_id"]
        row["counterfactual_pair_id"] = (
            f"{source_pair_id}__nonlinear_replay"
        )
        row["paired_group_id"] = f"{source_group_id}__nonlinear_replay"
        row["baseline_sample_id"] = (
            f"{source_pair_id}__m0__nonlinear_replay"
        )
        row["sensitivity_source_sample_id"] = source_sample_id
        row["sensitivity_source_pair_id"] = source_pair_id
        row["sensitivity_source_paired_group_id"] = source_group_id
        row["excluded_from_primary_score"] = True
        row["generation_metadata"]["future_innovation_policy"] = (
            NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
        )
        row["generation_metadata"]["future_innovation_sha256"] = "8" * 64
        row["generation_metadata"]["sensitivity_role"] = (
            "history_residual_replay_auxiliary"
        )
        replay_rows.append(row)
    return replay_rows


def _structural_common_rows() -> tuple[
    list[dict],
    list[dict],
    dict[str, dict],
    str,
    dict,
]:
    source_length = protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
    time = np.arange(source_length, dtype=float)

    def panel(phase: float) -> np.ndarray:
        factor = np.sin(2.0 * np.pi * time / 24.0 + phase) + 0.3 * np.cos(
            2.0 * np.pi * time / 12.0 - phase
        )
        return np.vstack(
            [
                loading * factor
                + 0.08 * np.sin(2.0 * np.pi * time / period + index)
                for index, (loading, period) in enumerate(
                    zip(
                        (1.0, -0.9, 1.2, -1.1),
                        (37.0, 41.0, 43.0, 47.0),
                        strict=True,
                    )
                )
            ]
        )

    dataset = protocol.DatasetSpec(
        dataset_id="validation_structural_fixture",
        logical_name="Validation structural fixture",
        config_id="fixture",
        asset_name="fixture",
        domain="Test",
        real_data_adapter="fixture",
    )
    known_future = np.column_stack(
        (
            np.sin(2.0 * np.pi * time / 31.0),
            ((time.astype(int) % 47) < 5).astype(float),
        )
    )
    bundle = RealDatasetBundle(
        frequency="h",
        records=(
            RealSeriesRecord(
                item_id="panel_a",
                values=panel(0.0),
                covariates=known_future,
                covariate_names=("weather", "event"),
                covariate_kind="known_future",
            ),
            RealSeriesRecord(
                item_id="panel_b",
                values=panel(0.4),
                covariates=np.roll(known_future, 3, axis=0),
                covariate_names=("weather", "event"),
                covariate_kind="known_future",
            ),
            RealSeriesRecord(
                item_id="panel_c",
                values=panel(0.8),
                covariates=np.roll(known_future, 7, axis=0),
                covariate_names=("weather", "event"),
                covariate_kind="known_future",
            ),
        ),
        asset_files=(),
        adapter_id="fixture",
        metadata={"fixture": True},
    )
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        dataset,
        source_root=Path("/unused"),
        maximum_backgrounds=3,
        real_bundle=bundle,
    )
    contracts, _availability = fit_structural_capability_contracts(
        backgrounds,
        capability_ids=("common_factor",),
    )
    main = list(
        iter_structural_real_anchored_samples(
            backgrounds,
            contracts,
            seed_indexes=range(len(backgrounds)),
        )
    )
    commitments = build_structural_donor_commitment_manifest(
        backgrounds,
        contracts,
        dataset_id="validation_structural_fixture",
    )
    ablations = list(
        iter_mandatory_structural_input_ablation_tasks(
            main,
            donor_commitment_manifest=commitments,
        )
    )
    entries = {
        str(entry["sample_id"]): entry for entry in commitments["entries"]
    }
    return (
        main,
        ablations,
        entries,
        str(commitments["commitment_root_sha256"]),
        commitments,
    )


def _structural_d2_sensitivity_rows(
    *,
    dataset_id: str = "validation_structural_d2_sensitivity",
) -> tuple[
    list[dict],
    list[dict],
    dict[str, dict],
    str,
]:
    source_length = protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
    time = np.arange(source_length, dtype=float)

    def panel(phase: float) -> np.ndarray:
        factor = np.sin(2.0 * np.pi * time / 24.0 + phase) + 0.25 * np.cos(
            2.0 * np.pi * time / 12.0 - phase
        )
        return np.vstack(
            (
                factor + 0.06 * np.sin(2.0 * np.pi * time / 37.0),
                -0.9 * factor + 0.06 * np.cos(2.0 * np.pi * time / 41.0),
            )
        )

    dataset = protocol.DatasetSpec(
        dataset_id=dataset_id,
        logical_name="Validation structural D2 sensitivity",
        config_id="fixture",
        asset_name="fixture",
        domain="Test",
        real_data_adapter="fixture",
    )
    bundle = RealDatasetBundle(
        frequency="h",
        records=tuple(
            RealSeriesRecord(
                item_id=f"panel_{index}",
                values=panel(0.31 * index),
            )
            for index in range(3)
        ),
        asset_files=(),
        adapter_id="fixture",
        metadata={"fixture": True},
    )
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        dataset,
        source_root=Path("/unused"),
        maximum_backgrounds=3,
        real_bundle=bundle,
    )
    contracts, _availability = fit_structural_capability_contracts(
        backgrounds,
        capability_ids=("common_factor",),
    )
    assert all(
        row["sensitivity_available"] is True
        and row["generation_eligible"] is False
        for row in contracts
    )
    main = list(
        iter_structural_real_anchored_samples(
            backgrounds,
            contracts,
            sensitivity=True,
            seed_indexes=range(len(backgrounds)),
        )
    )
    for row in main:
        row["excluded_from_primary_score"] = True

    entries = sorted(
        (_structural_donor_commitment_entry(row) for row in main),
        key=lambda entry: str(entry["sample_id"]),
    )
    sample_ids = [str(entry["sample_id"]) for entry in entries]
    commitment_manifest = {
        "schema_version": STRUCTURAL_DONOR_COMMITMENT_SCHEMA,
        "commitment_policy": STRUCTURAL_DONOR_COMMITMENT_POLICY,
        "dataset_id": dataset_id,
        "context_length": protocol.REAL_ANCHORED_CONTEXT_LENGTH,
        "source_structural_background_bank_sha256": protocol.json_sha256(
            [
                public_structural_background(background)
                for background in backgrounds
            ]
        ),
        "source_structural_contract_bank_sha256": protocol.json_sha256(
            contracts
        ),
        "entry_count": len(entries),
        "eligible_donor_sample_ids_sha256": protocol.json_sha256(sample_ids),
        "entries_sha256": protocol.json_sha256(entries),
        "entries": entries,
    }
    commitment_manifest["commitment_root_sha256"] = protocol.json_sha256(
        commitment_manifest
    )
    ablations = list(
        iter_mandatory_structural_input_ablation_tasks(
            main,
            donor_samples=main,
            donor_commitment_manifest=commitment_manifest,
        )
    )
    return (
        main,
        ablations,
        {str(entry["sample_id"]): entry for entry in entries},
        str(commitment_manifest["commitment_root_sha256"]),
    )


def _structural_covariate_d1_rows() -> list[dict]:
    source_length = protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
    time = np.arange(source_length, dtype=float)
    known_future = np.column_stack(
        (
            np.sin(2.0 * np.pi * time / 31.0),
            ((time.astype(int) % 47) < 5).astype(float),
        )
    )
    target = np.zeros(source_length, dtype=float)
    for index in range(2, source_length):
        target[index] = (
            0.3 * target[index - 1]
            + 1.1 * known_future[index, 0]
            + 0.7 * known_future[index - 1, 1]
            + 0.05 * np.sin(index / 7.0)
        )
    dataset = protocol.DatasetSpec(
        dataset_id="validation_covariate_d1_fixture",
        logical_name="Validation D1 covariate fixture",
        config_id="fixture",
        asset_name="fixture",
        domain="Test",
        real_data_adapter="fixture",
    )
    bundle = RealDatasetBundle(
        frequency="h",
        records=(
            RealSeriesRecord(
                item_id="target",
                values=target,
                covariates=known_future,
                covariate_names=("weather", "event"),
                covariate_kind="known_future",
            ),
        ),
        asset_files=(),
        adapter_id="fixture",
        metadata={"fixture": True},
    )
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        dataset,
        source_root=Path("/unused"),
        maximum_backgrounds=1,
        real_bundle=bundle,
    )
    contracts, _availability = fit_structural_capability_contracts(
        backgrounds,
        capability_ids=("covariate_response",),
    )
    assert contracts[0]["generation_eligible"] is True
    return list(iter_structural_real_anchored_samples(backgrounds, contracts))


def _structural_covariate_replay_bank(
    dataset_id: str,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    source_length = protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
    time = np.arange(source_length, dtype=float)
    records: list[RealSeriesRecord] = []
    for record_index in range(4):
        phase = 0.19 * record_index
        known_future = np.column_stack(
            (
                np.sin(2.0 * np.pi * time / 31.0 + phase),
                (((time.astype(int) + 3 * record_index) % 47) < 5).astype(
                    float
                ),
            )
        )
        target = np.zeros(source_length, dtype=float)
        for index in range(2, source_length):
            target[index] = (
                0.3 * target[index - 1]
                + 1.1 * known_future[index, 0]
                + 0.7 * known_future[index - 1, 1]
                + 0.05 * np.sin(index / 7.0 + phase)
            )
        records.append(
            RealSeriesRecord(
                item_id=f"target_{record_index}",
                values=target,
                covariates=known_future,
                covariate_names=("weather", "event"),
                covariate_kind="known_future",
            )
        )
    dataset = protocol.DatasetSpec(
        dataset_id=dataset_id,
        logical_name="Validation replay fixture",
        config_id="fixture",
        asset_name="fixture",
        domain="Test",
        real_data_adapter="fixture",
    )
    bundle = RealDatasetBundle(
        frequency="h",
        records=tuple(records),
        asset_files=(),
        adapter_id="fixture",
        metadata={"fixture": True},
    )
    private_backgrounds, _metadata = (
        build_structural_real_anchored_backgrounds(
            dataset,
            source_root=Path("/unused"),
            maximum_backgrounds=4,
            real_bundle=bundle,
        )
    )
    contracts, availability = fit_structural_capability_contracts(
        private_backgrounds,
        capability_ids=("covariate_response",),
    )
    assert availability["cells"][0]["status"] == "available"
    backgrounds = [
        public_structural_background(background)
        for background in private_backgrounds
    ]
    rows = list(
        iter_structural_real_anchored_samples(
            backgrounds,
            contracts,
            seed_indexes=(0,),
        )
    )
    return private_backgrounds, contracts, availability, rows


def _v3_generation_manifest_fixture(
    root: Path,
    *,
    dataset_id: str = "gift_electricity_h",
    seed_start: int = 4,
    seed_count: int = 2,
) -> tuple[dict, Path, Path]:
    dataset_root = root / dataset_id
    calibration_dir = dataset_root / "01_calibration"
    generation_dir = dataset_root / "02_generation"
    calibration_dir.mkdir(parents=True)
    generation_dir.mkdir(parents=True)

    base_split_audit = {
        "schema_version": "cafe.real_anchored_bank_split.v1",
        "policy": (
            "native_item_temporal_overlap_components_balanced_without_"
            "replacement_v1"
        ),
        "source_window_length": protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
        "source_background_count": 0,
        "temporal_overlap_component_count": 0,
        "evaluation_background_count": 0,
        "reference_background_count": 0,
        "dropped_background_count": 0,
        "cross_bank_temporal_overlap_count": 0,
        "evaluation_background_ids_sha256": protocol.json_sha256([]),
        "evaluation_background_ids": [],
        "reference_background_ids_sha256": protocol.json_sha256([]),
        "reference_background_ids": [],
        "component_assignment_sha256": protocol.json_sha256([]),
        "component_assignments": [],
        "threshold_tuning_policy": (
            "qualification_only_reference_bank_never_evaluation_origins"
        ),
    }
    bank_split_audit = build_combined_real_anchored_bank_split_audit(
        (),
        (),
        base_split_audit=base_split_audit,
    )
    policy = unavailable_real_anchored_qualification_policy(
        reference_background_ids=(),
        bank_split_audit=bank_split_audit,
    )
    policy_path = calibration_dir / "real_anchored_qualification_policy.json"
    protocol.write_json(policy_path, policy)
    donor_commitments = build_structural_donor_commitment_manifest(
        (),
        (),
        dataset_id=dataset_id,
    )
    donor_commitment_path = (
        calibration_dir / "structural_real_anchored_donor_commitments.json"
    )
    protocol.write_json(donor_commitment_path, donor_commitments)
    replay_files: dict[str, dict] = {}
    for name in (
        "real_anchored_backgrounds",
        "real_anchored_contracts",
        "structural_real_anchored_backgrounds",
        "structural_real_anchored_contracts",
        "real_anchored_reference_backgrounds",
        "structural_real_anchored_reference_backgrounds",
        "real_anchored_reference_contracts",
    ):
        path = calibration_dir / f"{name}.jsonl"
        protocol.write_jsonl(path, ())
        replay_files[name] = protocol.file_record(path)
    bank_split_path = calibration_dir / "real_anchored_bank_split_audit.json"
    protocol.write_json(bank_split_path, bank_split_audit)
    replay_files["real_anchored_bank_split_audit"] = protocol.file_record(
        bank_split_path
    )
    real_calibration_availability = build_availability(
        (),
        requested_capability_ids=(),
        minimum_eligible_backgrounds=4,
    )
    _unused_rows, structural_calibration_availability = (
        fit_structural_capability_contracts((), capability_ids=())
    )
    del _unused_rows
    for name, payload in (
        ("real_anchored_availability", real_calibration_availability),
        (
            "structural_real_anchored_availability",
            structural_calibration_availability,
        ),
    ):
        path = calibration_dir / f"{name}.json"
        protocol.write_json(path, payload)
        replay_files[name] = protocol.file_record(path)
    bundle = {
        "schema_version": "cafe.calibration_bundle.v3",
        "pipeline_schema_version": protocol.SCHEMA_VERSION,
        "generator_version": protocol.GENERATOR_VERSION,
        "dataset": {"dataset_id": dataset_id},
        "source": {"fixture": True},
        "files": {
            "real_anchored_qualification_policy": protocol.file_record(
                policy_path
            ),
            "structural_real_anchored_donor_commitments": (
                protocol.file_record(donor_commitment_path)
            ),
            **replay_files,
        },
        "real_anchored_qualification_policy_sha256": policy[
            "qualification_policy_sha256"
        ],
    }
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(calibration_dir / "calibration_bundle.json", bundle)

    seed_indexes = list(range(seed_start, seed_start + seed_count))
    real_availability = {
        "source_calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "requested_seed_indexes": seed_indexes,
        "generated_capabilities": [],
        "generated_master_count": 0,
        "generated_nonlinear_replay_sensitivity_count": 0,
        "hierarchical_coherence_generation_count": 0,
    }
    structural_availability = {
        "source_calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "requested_seed_indexes": seed_indexes,
        "generated_capabilities": [],
        "generated_main_master_count": 0,
        "generated_input_ablation_master_count": 0,
        "generated_sensitivity_capabilities": [],
        "generated_sensitivity_main_master_count": 0,
        "generated_sensitivity_input_ablation_master_count": 0,
        "hierarchical_coherence_generation_count": 0,
        "frozen_qualification_policy_sha256": policy[
            "qualification_policy_sha256"
        ],
    }
    real_availability_path = generation_dir / "real_availability.json"
    structural_availability_path = (
        generation_dir / "structural_availability.json"
    )
    protocol.write_json(real_availability_path, real_availability)
    protocol.write_json(
        structural_availability_path,
        structural_availability,
    )
    files: dict[str, dict] = {
        "real_anchored_availability": protocol.file_record(
            real_availability_path
        ),
        "structural_real_anchored_availability": protocol.file_record(
            structural_availability_path
        ),
        "structural_donor_commitments": {
            **bundle["files"][
                "structural_real_anchored_donor_commitments"
            ],
            "commitment_root_sha256": donor_commitments[
                "commitment_root_sha256"
            ],
            "source_calibration_bundle_sha256": bundle[
                "bundle_content_sha256"
            ],
        },
    }
    for key in (
        "clean",
        "robustness",
        "input_ablations",
        "real_anchored_counterfactuals",
    ):
        path = generation_dir / f"{key}.jsonl"
        protocol.write_jsonl(path, ())
        files[key] = {**protocol.file_record(path), "row_count": 0}
    config = {
        "schema_version": "cafe.generation_config.v3",
        "dataset_id": dataset_id,
        "calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "seed_start": seed_start,
        "seed_count": seed_count,
        "seed_indexes": seed_indexes,
        "requested_capabilities": [],
        "real_anchored_counterfactual": {
            "calibrated_available_capabilities": [],
            "generated_capabilities": [],
            "qualification_policy_sha256": policy[
                "qualification_policy_sha256"
            ],
            "qualification_threshold_source": (
                QUALIFICATION_THRESHOLD_SOURCE_POLICY
            ),
            "upstream_real_anchored_protocol": protocol.SCHEMA_VERSION,
            "legacy_upstream_component_policy": None,
            "included_in_synthetic_ranking": False,
            "formal_panel_minimum_dimension": 3,
            "hierarchy_policy": "qualification_only_zero_generation_rows",
            "structural_main_count": 0,
            "structural_input_ablation_count": 0,
            "structural_sensitivity_capabilities": [],
            "structural_sensitivity_main_count": 0,
            "structural_sensitivity_input_ablation_count": 0,
            "nonlinear_replay_sensitivity_count": 0,
            "structural_donor_commitment": {
                "schema_version": donor_commitments["schema_version"],
                "commitment_policy": donor_commitments[
                    "commitment_policy"
                ],
                "commitment_root_sha256": donor_commitments[
                    "commitment_root_sha256"
                ],
                "source_calibration_bundle_sha256": bundle[
                    "bundle_content_sha256"
                ],
                "source_file_sha256": bundle["files"][
                    "structural_real_anchored_donor_commitments"
                ]["sha256"],
            },
        },
    }
    manifest = {
        "schema_version": "cafe.generation_manifest.v3",
        "config": config,
        "config_sha256": protocol.json_sha256(config),
        "files": files,
    }
    shard_name = f"seed_{seed_start:06d}_{seed_start + seed_count:06d}"
    manifest_path = generation_dir / f"manifest__{shard_name}.json"
    protocol.write_json(manifest_path, manifest)
    return manifest, manifest_path, calibration_dir


def _install_structural_replay_fixture(
    manifest: dict,
    manifest_path: Path,
    calibration_dir: Path,
    *,
    dataset_id: str,
) -> list[dict]:
    candidate_backgrounds, _contracts, _availability, _rows = (
        _structural_covariate_replay_bank(dataset_id)
    )
    cloned_backgrounds = []
    for background in candidate_backgrounds:
        clone = copy.deepcopy(background)
        clone["background_id"] = f"{background['background_id']}__reference"
        clone["item_id"] = f"{background['item_id']}__reference"
        cloned_backgrounds.append(clone)
    backgrounds, reference_backgrounds, base_split_audit = (
        split_real_anchored_background_banks(
            [*candidate_backgrounds, *cloned_backgrounds],
            maximum_evaluation_backgrounds=4,
            maximum_reference_backgrounds=4,
            source_window_length=protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
        )
    )
    bank_split_audit = build_combined_real_anchored_bank_split_audit(
        backgrounds,
        reference_backgrounds,
        base_split_audit=base_split_audit,
    )
    reference_contracts, _reference_availability = (
        fit_structural_capability_contracts(
            reference_backgrounds,
            capability_ids=("covariate_response",),
        )
    )
    qualification_policy = freeze_real_anchored_qualification_policy(
        reference_contracts,
        reference_background_ids=[
            str(row["background_id"]) for row in reference_backgrounds
        ],
        bank_split_audit=bank_split_audit,
    )
    contracts, availability = fit_structural_capability_contracts(
        backgrounds,
        capability_ids=("covariate_response",),
        frozen_qualification_policy=qualification_policy,
    )
    backgrounds = [
        public_structural_background(background) for background in backgrounds
    ]
    reference_backgrounds = [
        public_structural_background(background)
        for background in reference_backgrounds
    ]
    rows = list(
        iter_structural_real_anchored_samples(
            backgrounds,
            contracts,
            seed_indexes=(0,),
        )
    )
    bundle_path = calibration_dir / "calibration_bundle.json"
    bundle = protocol.read_json(bundle_path)
    for key, payload in (
        ("structural_real_anchored_backgrounds", backgrounds),
        ("structural_real_anchored_contracts", contracts),
    ):
        path = Path(bundle["files"][key]["path"])
        protocol.write_jsonl(path, payload)
        bundle["files"][key] = protocol.file_record(path)
    for key, payload in (
        ("structural_real_anchored_reference_backgrounds", reference_backgrounds),
        ("real_anchored_reference_contracts", reference_contracts),
    ):
        path = Path(bundle["files"][key]["path"])
        protocol.write_jsonl(path, payload)
        bundle["files"][key] = protocol.file_record(path)
    bank_split_path = Path(
        bundle["files"]["real_anchored_bank_split_audit"]["path"]
    )
    protocol.write_json(bank_split_path, bank_split_audit)
    bundle["files"]["real_anchored_bank_split_audit"] = (
        protocol.file_record(bank_split_path)
    )
    policy_path = Path(
        bundle["files"]["real_anchored_qualification_policy"]["path"]
    )
    protocol.write_json(policy_path, qualification_policy)
    bundle["files"]["real_anchored_qualification_policy"] = (
        protocol.file_record(policy_path)
    )
    bundle["real_anchored_qualification_policy_sha256"] = (
        qualification_policy["qualification_policy_sha256"]
    )
    availability_path = Path(
        bundle["files"]["structural_real_anchored_availability"]["path"]
    )
    protocol.write_json(availability_path, availability)
    bundle["files"]["structural_real_anchored_availability"] = (
        protocol.file_record(availability_path)
    )
    donor_commitments = build_structural_donor_commitment_manifest(
        backgrounds,
        contracts,
        dataset_id=dataset_id,
    )
    donor_path = Path(
        bundle["files"][
            "structural_real_anchored_donor_commitments"
        ]["path"]
    )
    protocol.write_json(donor_path, donor_commitments)
    bundle["files"]["structural_real_anchored_donor_commitments"] = (
        protocol.file_record(donor_path)
    )
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(bundle_path, bundle)

    generation_files = manifest["files"]
    row_path = Path(generation_files["real_anchored_counterfactuals"]["path"])
    protocol.write_jsonl(row_path, rows)
    generation_files["real_anchored_counterfactuals"] = {
        **protocol.file_record(row_path),
        "row_count": len(rows),
    }
    real_availability_path = Path(
        generation_files["real_anchored_availability"]["path"]
    )
    real_availability = protocol.read_json(real_availability_path)
    real_availability.update(
        {
            "source_calibration_bundle_sha256": bundle[
                "bundle_content_sha256"
            ],
            "generated_capabilities": ["covariate_response"],
            "generated_master_count": len(rows),
        }
    )
    protocol.write_json(real_availability_path, real_availability)
    generation_files["real_anchored_availability"] = protocol.file_record(
        real_availability_path
    )
    structural_availability_path = Path(
        generation_files["structural_real_anchored_availability"]["path"]
    )
    structural_generation_availability = protocol.read_json(
        structural_availability_path
    )
    structural_generation_availability.update(
        {
            "source_calibration_bundle_sha256": bundle[
                "bundle_content_sha256"
            ],
            "generated_capabilities": ["covariate_response"],
            "generated_main_master_count": len(rows),
            "generated_input_ablation_master_count": 0,
            "frozen_qualification_policy_sha256": qualification_policy[
                "qualification_policy_sha256"
            ],
        }
    )
    protocol.write_json(
        structural_availability_path,
        structural_generation_availability,
    )
    generation_files["structural_real_anchored_availability"] = (
        protocol.file_record(structural_availability_path)
    )
    generation_files["structural_donor_commitments"] = {
        **bundle["files"]["structural_real_anchored_donor_commitments"],
        "commitment_root_sha256": donor_commitments[
            "commitment_root_sha256"
        ],
        "source_calibration_bundle_sha256": bundle["bundle_content_sha256"],
    }

    config = manifest["config"]
    config["calibration_bundle_sha256"] = bundle["bundle_content_sha256"]
    config["requested_capabilities"] = ["covariate_response"]
    real_config = config["real_anchored_counterfactual"]
    real_config["calibrated_available_capabilities"] = []
    real_config["generated_capabilities"] = ["covariate_response"]
    real_config["structural_main_count"] = len(rows)
    real_config["structural_input_ablation_count"] = 0
    real_config["qualification_policy_sha256"] = qualification_policy[
        "qualification_policy_sha256"
    ]
    real_config["structural_donor_commitment"] = {
        "schema_version": donor_commitments["schema_version"],
        "commitment_policy": donor_commitments["commitment_policy"],
        "commitment_root_sha256": donor_commitments[
            "commitment_root_sha256"
        ],
        "source_calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "source_file_sha256": bundle["files"][
            "structural_real_anchored_donor_commitments"
        ]["sha256"],
    }
    manifest["config_sha256"] = protocol.json_sha256(config)
    protocol.write_json(manifest_path, manifest)
    return rows


def test_v3_generation_manifest_contract_binds_config_bundle_and_policy(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, calibration_dir = (
        _v3_generation_manifest_fixture(tmp_path)
    )

    config = runner.validate_generation_manifest_contract(
        manifest,
        manifest_path=manifest_path,
        calibration_dir=calibration_dir,
        dataset_id="gift_electricity_h",
        seed_start=4,
        seed_count=2,
    )

    assert config == manifest["config"]


def test_main_row_replay_rejects_self_consistent_real_path_rewrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_id = "gift_electricity_h"
    manifest, manifest_path, calibration_dir = (
        _v3_generation_manifest_fixture(
            tmp_path,
            dataset_id=dataset_id,
            seed_start=0,
            seed_count=1,
        )
    )
    rows = _install_structural_replay_fixture(
        manifest,
        manifest_path,
        calibration_dir,
        dataset_id=dataset_id,
    )
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(
            dataset_id=dataset_id,
            output_root=tmp_path,
            seed_start=0,
            seed_count=1,
        ),
    )

    assert runner.main() == 0

    rewritten = copy.deepcopy(rows)
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    for row in rewritten:
        target = np.asarray(row["target"], dtype=float)
        target[:context, 0] += 0.25
        covariates = np.asarray(row["covariates"], dtype=float)
        row["target"] = target.tolist()
        row["target_sha256"] = protocol.target_and_covariate_sha256(
            target,
            covariates,
        )
        row["future_sha256"] = array_sha256(target[context:])
    row_path = Path(
        manifest["files"]["real_anchored_counterfactuals"]["path"]
    )
    protocol.write_jsonl(row_path, rewritten)
    manifest["files"]["real_anchored_counterfactuals"] = {
        **protocol.file_record(row_path),
        "row_count": len(rewritten),
    }
    protocol.write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="generation validation failed"):
        runner.main()
    report = protocol.read_json(
        manifest_path.parent / "validation__seed_000000_000001.json"
    )
    replay_failures = [
        failure
        for failure in report["real_anchored_validation"]["row_failures"]
        if not failure["checks"].get(
            "upstream_full_row_replay_exact",
            True,
        )
    ]
    assert len(replay_failures) == len(rewritten)
    assert any(
        not failure["checks"]["upstream_baseline_target_exact"]
        for failure in replay_failures
    )
    assert any(
        not failure["checks"]["upstream_treatment_target_replay_exact"]
        for failure in replay_failures
    )


def test_generation_main_rejects_stale_config_hash_before_row_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest, manifest_path, _calibration_dir = (
        _v3_generation_manifest_fixture(tmp_path)
    )
    manifest["config"]["dataset_id"] = "tampered"
    protocol.write_json(manifest_path, manifest)
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(
            dataset_id="gift_electricity_h",
            output_root=tmp_path,
            seed_start=4,
            seed_count=2,
        ),
    )
    monkeypatch.setattr(
        protocol,
        "iter_jsonl",
        lambda *_args, **_kwargs: pytest.fail(
            "generation rows were read before manifest validation"
        ),
    )

    with pytest.raises(ValueError, match="config hash mismatch"):
        runner.main()


def test_v3_generation_manifest_rejects_self_consistent_identity_tamper(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, calibration_dir = (
        _v3_generation_manifest_fixture(tmp_path)
    )
    manifest["config"]["dataset_id"] = "tampered"
    manifest["config_sha256"] = protocol.json_sha256(manifest["config"])

    with pytest.raises(ValueError, match="dataset_id disagrees"):
        runner.validate_generation_manifest_contract(
            manifest,
            manifest_path=manifest_path,
            calibration_dir=calibration_dir,
            dataset_id="gift_electricity_h",
            seed_start=4,
            seed_count=2,
        )


def test_v3_generation_manifest_rejects_qualification_binding_tamper(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, calibration_dir = (
        _v3_generation_manifest_fixture(tmp_path)
    )
    manifest["config"]["real_anchored_counterfactual"][
        "qualification_policy_sha256"
    ] = "f" * 64
    manifest["config_sha256"] = protocol.json_sha256(manifest["config"])

    with pytest.raises(ValueError, match="qualification policy mismatch"):
        runner.validate_generation_manifest_contract(
            manifest,
            manifest_path=manifest_path,
            calibration_dir=calibration_dir,
            dataset_id="gift_electricity_h",
            seed_start=4,
            seed_count=2,
        )


def test_v3_manifest_rejects_bundle_rehashed_wrong_split_audit(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, calibration_dir = (
        _v3_generation_manifest_fixture(tmp_path)
    )
    bundle_path = calibration_dir / "calibration_bundle.json"
    bundle = protocol.read_json(bundle_path)
    audit_path = Path(
        bundle["files"]["real_anchored_bank_split_audit"]["path"]
    )
    audit = protocol.read_json(audit_path)
    audit["combined_split"]["component_assignment_sha256"] = "f" * 64
    protocol.write_json(audit_path, audit)
    bundle["files"]["real_anchored_bank_split_audit"] = (
        protocol.file_record(audit_path)
    )
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(bundle_path, bundle)

    new_bundle_hash = bundle["bundle_content_sha256"]
    for key in (
        "real_anchored_availability",
        "structural_real_anchored_availability",
    ):
        path = Path(manifest["files"][key]["path"])
        availability = protocol.read_json(path)
        availability["source_calibration_bundle_sha256"] = new_bundle_hash
        protocol.write_json(path, availability)
        manifest["files"][key] = protocol.file_record(path)
    manifest["files"]["structural_donor_commitments"][
        "source_calibration_bundle_sha256"
    ] = new_bundle_hash
    config = manifest["config"]
    config["calibration_bundle_sha256"] = new_bundle_hash
    config["real_anchored_counterfactual"][
        "structural_donor_commitment"
    ]["source_calibration_bundle_sha256"] = new_bundle_hash
    manifest["config_sha256"] = protocol.json_sha256(config)

    with pytest.raises(ValueError, match="component assignment hash"):
        runner.validate_generation_manifest_contract(
            manifest,
            manifest_path=manifest_path,
            calibration_dir=calibration_dir,
            dataset_id="gift_electricity_h",
            seed_start=4,
            seed_count=2,
        )


def test_v3_generation_manifest_accepts_legacy_v2_upstream_without_rows(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, calibration_dir = (
        _v3_generation_manifest_fixture(tmp_path)
    )
    bundle_path = calibration_dir / "calibration_bundle.json"
    bundle = protocol.read_json(bundle_path)
    bundle["schema_version"] = "cafe.calibration_bundle.v2"
    bundle["pipeline_schema_version"] = "cafe.pipeline.v2"
    bundle["files"] = {}
    bundle.pop("real_anchored_qualification_policy_sha256")
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(bundle_path, bundle)

    generation_files = manifest["files"]
    generation_files.pop("structural_donor_commitments")
    for key in (
        "real_anchored_availability",
        "structural_real_anchored_availability",
    ):
        path = Path(generation_files[key]["path"])
        availability = protocol.read_json(path)
        availability["source_calibration_bundle_sha256"] = bundle[
            "bundle_content_sha256"
        ]
        if key == "structural_real_anchored_availability":
            availability.pop("frozen_qualification_policy_sha256")
        protocol.write_json(path, availability)
        generation_files[key] = protocol.file_record(path)

    config = manifest["config"]
    config["calibration_bundle_sha256"] = bundle["bundle_content_sha256"]
    real_config = config["real_anchored_counterfactual"]
    real_config["qualification_policy_sha256"] = None
    real_config["qualification_threshold_source"] = None
    real_config["upstream_real_anchored_protocol"] = "cafe.pipeline.v2"
    real_config["legacy_upstream_component_policy"] = (
        "validated_but_not_regenerated_or_ranked_as_v3"
    )
    real_config["structural_donor_commitment"] = None
    manifest["config_sha256"] = protocol.json_sha256(config)

    validated = runner.validate_generation_manifest_contract(
        manifest,
        manifest_path=manifest_path,
        calibration_dir=calibration_dir,
        dataset_id="gift_electricity_h",
        seed_start=4,
        seed_count=2,
    )

    assert validated["real_anchored_counterfactual"][
        "upstream_real_anchored_protocol"
    ] == "cafe.pipeline.v2"


def test_valid_real_anchored_component_allows_exact_repeated_baselines() -> None:
    rows = _real_anchored_rows()

    result = runner.real_anchored_counterfactual_checks(
        rows,
        expected_row_count=len(rows),
    )

    assert result["accepted"] is True
    assert result["sample_count"] == 2 * len(REAL_ANCHORED_ALPHAS)
    assert result["pair_count"] == len(REAL_ANCHORED_ALPHAS)
    assert result["paired_group_count"] == 1
    assert result["row_failures"] == []
    assert result["pair_failures"] == []
    assert result["paired_group_failures"] == []
    baselines = [
        np.asarray(row["target"])
        for row in rows
        if row["counterfactual_member"] == 0
    ]
    assert all(
        np.array_equal(values, baselines[0]) for values in baselines[1:]
    )


def test_univariate_replay_rejects_self_consistent_path_rewrite() -> None:
    expected_rows = _real_anchored_rows()
    replay_evidence = {
        "univariate_expected_rows": {
            row["sample_id"]: copy.deepcopy(row) for row in expected_rows
        },
        "structural_expected_rows": {},
    }
    assert runner.real_anchored_counterfactual_checks(
        copy.deepcopy(expected_rows),
        upstream_replay_evidence=replay_evidence,
    )["accepted"] is True

    rewritten = copy.deepcopy(expected_rows)
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    for dose_index in range(len(REAL_ANCHORED_ALPHAS)):
        baseline = rewritten[2 * dose_index]
        treatment = rewritten[2 * dose_index + 1]
        for row in (baseline, treatment):
            target = np.asarray(row["target"], dtype=float)
            target[:context, 0] += 0.25
            row["target"] = target.tolist()
        baseline_target = np.asarray(baseline["target"], dtype=float)[:, 0]
        baseline_hashes = {
            "baseline_history_sha256": array_sha256(
                baseline_target[:context]
            ),
            "baseline_future_sha256": array_sha256(
                baseline_target[context:]
            ),
            "baseline_target_sha256": array_sha256(baseline_target),
        }
        baseline.update(baseline_hashes)
        treatment.update(baseline_hashes)
        _rehash_target(baseline, baseline)
        _rehash_target(treatment, baseline)

    # The legacy self-consistency checks alone cannot establish that the path
    # still comes from the immutable calibration bank.
    assert runner.real_anchored_counterfactual_checks(rewritten)[
        "accepted"
    ] is True
    rejected = runner.real_anchored_counterfactual_checks(
        rewritten,
        upstream_replay_evidence=replay_evidence,
    )
    assert rejected["accepted"] is False
    replay_failures = [
        failure
        for failure in rejected["row_failures"]
        if not failure["checks"].get("upstream_full_row_replay_exact", True)
    ]
    assert len(replay_failures) == len(rewritten)
    assert any(
        not failure["checks"]["upstream_baseline_target_exact"]
        for failure in replay_failures
    )
    assert any(
        not failure["checks"]["upstream_treatment_target_replay_exact"]
        for failure in replay_failures
    )


def test_upstream_replay_requires_exact_main_sample_id_coverage() -> None:
    rows = _real_anchored_rows()
    expected = {
        row["sample_id"]: copy.deepcopy(row) for row in rows
    }

    result = runner.real_anchored_counterfactual_checks(
        rows[:-2],
        upstream_replay_evidence={
            "univariate_expected_rows": expected,
            "structural_expected_rows": {},
        },
    )

    assert result["accepted"] is False
    assert result["upstream_replay_coverage_failures"] == [
        {
            "reason": "missing_calibration_replay_rows",
            "sample_ids": sorted(
                (rows[-2]["sample_id"], rows[-1]["sample_id"])
            ),
        }
    ]


def test_hash_tampering_and_incomplete_pair_fail_closed() -> None:
    rows = _real_anchored_rows()
    rows[3]["target"][0][0] += 1.0
    rows.pop(-1)

    result = runner.real_anchored_counterfactual_checks(rows)

    assert result["accepted"] is False
    assert any(
        not failure["checks"]["target_hash_matches"]
        for failure in result["row_failures"]
        if failure["sample_id"] == rows[3]["sample_id"]
    )
    assert any(
        not failure["checks"]["exactly_two_members"]
        for failure in result["pair_failures"]
    )


def test_shared_references_anti_copy_and_manifest_count_are_hard_gates() -> None:
    rows = _real_anchored_rows()
    rows[1]["shared_standardization"]["scale"] = 9.0
    rows[2]["anti_copy_gate"]["status"] = "passed"

    result = runner.real_anchored_counterfactual_checks(
        rows,
        expected_row_count=len(rows) + 1,
    )

    assert result["accepted"] is False
    assert result["manifest_row_count_matches"] is False
    assert result["pair_failures"]
    assert result["paired_group_failures"]
    assert any(
        not failure["checks"][
            "anti_copy_explicitly_not_applicable"
        ]
        for failure in result["row_failures"]
    )


def test_nonmonotone_but_self_consistently_hashed_delta_fails() -> None:
    rows = _real_anchored_rows()
    first_treatment = rows[1]
    second_baseline = rows[2]
    second_treatment = rows[3]
    baseline = np.asarray(second_baseline["target"], dtype=float)
    first_delta = np.asarray(first_treatment["target"], dtype=float) - baseline
    second_treatment["target"] = (baseline + 0.5 * first_delta).tolist()
    _rehash_target(second_treatment, second_baseline)

    result = runner.real_anchored_counterfactual_checks(rows)

    assert result["accepted"] is False
    group_failure = result["paired_group_failures"][0]
    assert group_failure["checks"]["delta_rms_strictly_increases"] is False
    assert (
        group_failure["checks"]["delta_is_linear_in_alpha_minus_one"]
        is False
    )


def test_reusing_one_real_background_under_a_fresh_seed_fails() -> None:
    rows = _real_anchored_rows()
    recycled = copy.deepcopy(rows)
    for row in recycled:
        old_pair_id = str(row["counterfactual_pair_id"])
        new_pair_id = f"{old_pair_id}-recycled-seed-1"
        member = int(row["counterfactual_member"])
        row["seed_index"] = 1
        row["paired_group_id"] = "real-cf-group-1"
        row["counterfactual_pair_id"] = new_pair_id
        row["sample_id"] = f"{new_pair_id}__m{member}"
        row["master_sample_id"] = row["sample_id"]
        row["baseline_sample_id"] = f"{new_pair_id}__m0"

    result = runner.real_anchored_counterfactual_checks(rows + recycled)

    assert result["accepted"] is False
    assert result["repeated_background_failures"] == [
        {
            "dataset_id": "test_hourly",
            "capability_id": "multi_seasonal",
            "background_id": "background-0",
            "paired_group_ids": ["real-cf-group-0", "real-cf-group-1"],
        }
    ]


def test_legacy_v1_rows_without_evaluation_table_remain_accepted() -> None:
    rows = _real_anchored_rows()
    for row in rows:
        row["schema_version"] = (
            "cafe.real_anchored_counterfactual_master.v1"
        )
        row.pop("evaluation_table")

    result = runner.real_anchored_counterfactual_checks(rows)

    assert result["accepted"] is True
    assert result["schema_version"] == "cafe.real_anchored_validation.v3"


def test_current_univariate_grid_cannot_delete_a_frozen_dose() -> None:
    rows = _real_anchored_rows()[:-2]
    shortened_grid = list(REAL_ANCHORED_ALPHAS[:-1])
    for row in rows:
        row["intensity_calibration"]["selected_alphas"] = shortened_grid

    result = runner.real_anchored_counterfactual_checks(rows)

    assert result["accepted"] is False
    assert any(
        not failure["checks"]["alpha_grid_valid"]
        for failure in result["row_failures"]
    )
    assert result["paired_group_failures"][0]["checks"][
        "frozen_treatment_grid_exact"
    ] is False


def test_legacy_v1_univariate_grid_remains_self_declared() -> None:
    rows = _real_anchored_rows()[:-2]
    shortened_grid = list(REAL_ANCHORED_ALPHAS[:-1])
    for row in rows:
        row["schema_version"] = "cafe.real_anchored_counterfactual_master.v1"
        row.pop("evaluation_table")
        row["intensity_calibration"]["selected_alphas"] = shortened_grid

    result = runner.real_anchored_counterfactual_checks(rows)

    assert result["accepted"] is True


def test_current_univariate_grid_cannot_replace_a_frozen_dose() -> None:
    rows = _real_anchored_rows()
    replacement_grid = [*REAL_ANCHORED_ALPHAS[:-1], 2.2]
    for row in rows:
        row["intensity_calibration"]["selected_alphas"] = replacement_grid
    baseline = np.asarray(rows[-2]["target"], dtype=float)
    first_baseline = np.asarray(rows[0]["target"], dtype=float)
    first_treatment = np.asarray(rows[1]["target"], dtype=float)
    unit_delta = (first_treatment - first_baseline) / (
        float(rows[1]["dose_value"]) - 1.0
    )
    treatment = rows[-1]
    treatment["dose_value"] = 2.2
    treatment["intensity_lambda"] = 2.2
    treatment["sampled_generator_parameters"]["alpha"] = 2.2
    treatment["generation_metadata"]["alpha"] = 2.2
    treatment["target"] = (baseline + 1.2 * unit_delta).tolist()
    _rehash_target(treatment, rows[-2])

    result = runner.real_anchored_counterfactual_checks(rows)

    assert result["accepted"] is False
    assert any(
        not failure["checks"]["alpha_grid_valid"]
        for failure in result["row_failures"]
    )
    assert result["paired_group_failures"][0]["checks"][
        "frozen_treatment_grid_exact"
    ] is False


def test_nonlinear_dynamic_contract_is_not_forced_to_be_alpha_linear() -> None:
    rows = _nonlinear_real_anchored_rows()
    replay_evidence = {
        "univariate_expected_rows": {
            row["sample_id"]: copy.deepcopy(row) for row in rows
        },
        "structural_expected_rows": {},
    }
    baseline = np.asarray(rows[0]["target"], dtype=float)
    first_delta = np.asarray(rows[1]["target"], dtype=float) - baseline
    second_delta = np.asarray(rows[3]["target"], dtype=float) - baseline
    first_alpha = float(rows[1]["dose_value"])
    second_alpha = float(rows[3]["dose_value"])
    assert not np.allclose(
        first_delta / (first_alpha - 1.0),
        second_delta / (second_alpha - 1.0),
    )

    result = runner.real_anchored_counterfactual_checks(
        rows,
        upstream_replay_evidence=replay_evidence,
    )

    assert result["accepted"] is True
    assert result["paired_group_failures"] == []

    rows[1]["generation_metadata"]["future_innovation_policy"] = (
        "history_residual_replay"
    )
    rejected = runner.real_anchored_counterfactual_checks(rows)
    assert rejected["accepted"] is False
    assert any(
        not failure["checks"]["nonlinear_dynamic_contract_valid"]
        for failure in rejected["paired_group_failures"]
    )


def test_nonlinear_replay_sensitivity_is_source_bound_and_excluded() -> None:
    main = _nonlinear_real_anchored_rows()
    replay = _nonlinear_replay_sensitivity_rows(main)
    evidence = {
        "univariate_expected_rows": {
            row["sample_id"]: copy.deepcopy(row) for row in main
        },
        "nonlinear_replay_expected_rows": {
            row["sample_id"]: copy.deepcopy(row) for row in replay
        },
        "structural_expected_rows": {},
        "structural_sensitivity_expected_rows": {},
    }

    result = runner.real_anchored_counterfactual_checks(
        main + replay,
        upstream_replay_evidence=evidence,
    )

    assert result["accepted"] is True
    assert result["nonlinear_replay_sensitivity_sample_count"] == len(replay)
    assert result["effective_background_count_by_capability"] == {
        "nonlinear_persistence": 1
    }
    assert result["nonlinear_replay_sensitivity_validation"][
        "effective_background_count_by_capability"
    ] == {}


@pytest.mark.parametrize(
    "tamper",
    ("source", "policy", "not_excluded", "target", "missing_dose"),
)
def test_nonlinear_replay_sensitivity_fails_closed(tamper: str) -> None:
    main = _nonlinear_real_anchored_rows()
    replay = _nonlinear_replay_sensitivity_rows(main)
    evidence = {
        "univariate_expected_rows": {
            row["sample_id"]: copy.deepcopy(row) for row in main
        },
        "nonlinear_replay_expected_rows": {
            row["sample_id"]: copy.deepcopy(row) for row in replay
        },
        "structural_expected_rows": {},
        "structural_sensitivity_expected_rows": {},
    }
    if tamper == "source":
        replay[0]["sensitivity_source_sample_id"] = "unknown-main-row"
    elif tamper == "policy":
        replay[0]["generation_metadata"]["future_innovation_policy"] = (
            "zero_future_innovation_paired_rollout_v1"
        )
    elif tamper == "not_excluded":
        replay[0]["excluded_from_primary_score"] = False
    elif tamper == "target":
        replay[1]["target"][-1][0] += 0.25
        _rehash_target(replay[1], replay[0])
    else:
        replay = [
            row
            for row in replay
            if row["dose_index"] != len(REAL_ANCHORED_ALPHAS)
        ]

    result = runner.real_anchored_counterfactual_checks(
        main + replay,
        upstream_replay_evidence=evidence,
    )

    assert result["accepted"] is False
    replay_validation = result["nonlinear_replay_sensitivity_validation"]
    if tamper == "missing_dose":
        assert replay_validation["paired_group_failures"]
        assert replay_validation["upstream_replay_coverage_failures"]
    else:
        assert replay_validation["row_failures"]


def test_structural_main_and_mandatory_input_ablation_pass_v3() -> None:
    main, ablations, entries, root, _manifest = _structural_common_rows()

    result = runner.real_anchored_counterfactual_checks(
        main + ablations,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is True
    assert result["structural_main_sample_count"] == len(main)
    assert result["structural_input_ablation_sample_count"] == len(ablations)
    assert result["input_ablation_failures"] == []
    assert result["input_ablation_coverage_failures"] == []
    assert result["input_ablation_pair_failures"] == []
    for row in main:
        assert row["target_dim"] >= 3
        assert row["covariate_dim"] == 2
        assert np.asarray(row["covariates"]).shape == (384, 2)
        assert row["mase_scale"] == np.mean(row["mase_scale_by_target"])


def test_d2_structural_sensitivity_is_excluded_replayed_and_ablated() -> None:
    main, ablations, entries, root = _structural_d2_sensitivity_rows()
    expected = {
        row["sample_id"]: copy.deepcopy(row) for row in main
    }

    result = runner.real_anchored_counterfactual_checks(
        main + ablations,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
        upstream_replay_evidence={
            "univariate_expected_rows": {},
            "structural_expected_rows": {},
            "structural_sensitivity_expected_rows": expected,
        },
    )

    assert result["accepted"] is True
    assert result["structural_main_sample_count"] == 0
    assert result["structural_sensitivity_sample_count"] == len(main)
    assert result["effective_background_count_by_capability"] == {}
    assert result["structural_validation"][
        "sensitivity_background_count_by_capability"
    ] == {"common_factor": 3}
    assert result["input_ablation_coverage_failures"] == []
    assert all(
        row["target_dim"] == 2
        and row["evaluation_table"]
        == "real_anchored_structural_sensitivity"
        and row["excluded_from_primary_score"] is True
        for row in main
    )


def test_formal_and_d2_structural_tracks_scope_seed_assignment_by_table() -> None:
    formal, _formal_ablations, _formal_entries, _formal_root, _manifest = (
        _structural_common_rows()
    )
    sensitivity, _sensitivity_ablations, _entries, _root = (
        _structural_d2_sensitivity_rows(
            dataset_id="validation_structural_fixture"
        )
    )
    combined_main = [*formal, *sensitivity]
    entries = sorted(
        (_structural_donor_commitment_entry(row) for row in combined_main),
        key=lambda entry: str(entry["sample_id"]),
    )
    sample_ids = [str(entry["sample_id"]) for entry in entries]
    commitment_manifest = {
        "schema_version": STRUCTURAL_DONOR_COMMITMENT_SCHEMA,
        "commitment_policy": STRUCTURAL_DONOR_COMMITMENT_POLICY,
        "dataset_id": "validation_structural_fixture",
        "context_length": protocol.REAL_ANCHORED_CONTEXT_LENGTH,
        "source_structural_background_bank_sha256": protocol.json_sha256([]),
        "source_structural_contract_bank_sha256": protocol.json_sha256([]),
        "entry_count": len(entries),
        "eligible_donor_sample_ids_sha256": protocol.json_sha256(sample_ids),
        "entries_sha256": protocol.json_sha256(entries),
        "entries": entries,
    }
    commitment_manifest["commitment_root_sha256"] = protocol.json_sha256(
        commitment_manifest
    )
    ablations = list(
        iter_mandatory_structural_input_ablation_tasks(
            combined_main,
            donor_samples=combined_main,
            donor_commitment_manifest=commitment_manifest,
        )
    )

    result = runner.real_anchored_counterfactual_checks(
        combined_main + ablations,
        donor_commitment_entries={
            str(entry["sample_id"]): entry for entry in entries
        },
        donor_commitment_root_sha256=str(
            commitment_manifest["commitment_root_sha256"]
        ),
        upstream_replay_evidence={
            "univariate_expected_rows": {},
            "nonlinear_replay_expected_rows": {},
            "structural_expected_rows": {
                row["sample_id"]: copy.deepcopy(row) for row in formal
            },
            "structural_sensitivity_expected_rows": {
                row["sample_id"]: copy.deepcopy(row) for row in sensitivity
            },
        },
    )

    assert result["accepted"] is True
    assert result["seed_assignment_failures"] == []
    assert result["repeated_background_failures"] == []
    assert result["effective_background_count_by_capability"] == {
        "common_factor": 3
    }
    assert result["structural_validation"][
        "sensitivity_background_count_by_capability"
    ] == {"common_factor": 3}


@pytest.mark.parametrize(
    "tamper",
    ("missing_replay", "not_excluded", "wrong_dimension", "missing_ablation"),
)
def test_d2_structural_sensitivity_policy_fails_closed(tamper: str) -> None:
    main, ablations, entries, root = _structural_d2_sensitivity_rows()
    expected = {
        row["sample_id"]: copy.deepcopy(row) for row in main
    }
    evidence: dict | None = {
        "univariate_expected_rows": {},
        "structural_expected_rows": {},
        "structural_sensitivity_expected_rows": expected,
    }
    if tamper == "missing_replay":
        evidence = None
    elif tamper == "not_excluded":
        main[0]["excluded_from_primary_score"] = False
    elif tamper == "wrong_dimension":
        main[0]["target_dim"] = 3
    else:
        source_id = str(ablations[0]["input_ablation_source_sample_id"])
        ablations = [
            row
            for row in ablations
            if row["input_ablation_source_sample_id"] != source_id
        ]

    result = runner.real_anchored_counterfactual_checks(
        main + ablations,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
        upstream_replay_evidence=evidence,
    )

    assert result["accepted"] is False
    if tamper == "missing_replay":
        assert any(
            not failure["checks"]["sensitivity_contract_eligible"]
            for failure in result["row_failures"]
        )
    elif tamper == "not_excluded":
        assert any(
            not failure["checks"]["primary_score_policy_valid"]
            for failure in result["row_failures"]
        )
    elif tamper == "wrong_dimension":
        assert any(
            not failure["checks"]["formal_panel_dimension_valid"]
            for failure in result["row_failures"]
        )
    else:
        assert result["input_ablation_coverage_failures"]


def test_current_structural_grid_cannot_delete_a_frozen_dose() -> None:
    main, ablations, entries, root, _manifest = _structural_common_rows()
    rows = [
        row
        for row in (*main, *ablations)
        if int(row["dose_index"]) < len(REAL_ANCHORED_ALPHAS)
    ]
    shortened_grid = list(REAL_ANCHORED_ALPHAS[:-1])
    for row in rows:
        row["intensity_calibration"]["selected_alphas"] = shortened_grid

    result = runner.real_anchored_counterfactual_checks(
        rows,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is False
    assert any(
        not failure["checks"]["alpha_grid_valid"]
        for failure in result["row_failures"]
    )
    assert any(
        not failure["checks"]["frozen_treatment_grid_exact"]
        for failure in result["input_ablation_failures"]
    )


def test_covariate_response_d1_with_known_future_inputs_passes_v3() -> None:
    rows = _structural_covariate_d1_rows()

    result = runner.real_anchored_counterfactual_checks(rows)

    assert result["accepted"] is True
    assert result["policy_failures"] == []
    assert result["structural_main_sample_count"] == len(rows)
    assert result["structural_input_ablation_sample_count"] == 0
    for row in rows:
        assert row["capability_id"] == "covariate_response"
        assert row["target_dim"] == 1
        assert row["covariate_dim"] == 2
        assert row["covariate_column_names"] == ["weather", "event"]
        assert row["generation_metadata"][
            "known_future_covariate_path_used_for_delta"
        ] is True

    missing_declaration = copy.deepcopy(rows)
    missing_declaration[0]["generation_metadata"][
        "known_future_covariate_path_used_for_delta"
    ] = False
    rejected = runner.real_anchored_counterfactual_checks(
        missing_declaration
    )
    assert rejected["accepted"] is False
    assert any(
        not failure["checks"][
            "known_future_covariate_contract_valid"
        ]
        for failure in rejected["row_failures"]
    )


def test_structural_ablation_history_future_and_source_are_hard_gates() -> None:
    main, ablations, entries, root, _manifest = _structural_common_rows()
    tampered = copy.deepcopy(ablations)
    assessed = tampered[0]["input_ablation_metadata"][
        "assessed_target_indices"
    ][0]
    tampered_target = np.asarray(tampered[0]["target"], dtype=float)
    tampered_target[0, assessed] += 0.5
    tampered[0]["target"] = tampered_target.tolist()
    tampered[0]["target_sha256"] = protocol.target_and_covariate_sha256(
        tampered_target,
        np.asarray(tampered[0]["covariates"], dtype=float),
    )
    tampered[1]["input_ablation_source_pair_id"] = "wrong-main-pair"
    tampered[2]["excluded_from_primary_score"] = False

    result = runner.real_anchored_counterfactual_checks(
        main + tampered,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is False
    assert any(
        not failure["checks"].get("assessed_history_unchanged", True)
        for failure in result["input_ablation_failures"]
    )
    assert any(
        not failure["checks"].get("source_binding_exact", True)
        for failure in result["input_ablation_failures"]
    )
    assert any(
        not failure["checks"]["excluded_from_primary_score"]
        for failure in result["input_ablation_failures"]
    )


def test_structural_ablation_is_required_for_every_common_main_member() -> None:
    main, ablations, entries, root, _manifest = _structural_common_rows()
    missing_source = str(ablations[0]["input_ablation_source_sample_id"])
    incomplete = [
        row
        for row in ablations
        if row["input_ablation_source_sample_id"] != missing_source
    ]

    result = runner.real_anchored_counterfactual_checks(
        main + incomplete,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is False
    assert {
        failure["source_sample_id"]
        for failure in result["input_ablation_coverage_failures"]
    } == {missing_source}


def test_single_seed_shard_accepts_global_successor_donor_provenance() -> None:
    all_main, _all_ablations, entries, root, commitments = (
        _structural_common_rows()
    )
    source_seed = min(int(row["seed_index"]) for row in all_main)
    shard_main = [
        row for row in all_main if int(row["seed_index"]) == source_seed
    ]
    shard_ablations = list(
        iter_mandatory_structural_input_ablation_tasks(
            shard_main,
            donor_samples=all_main,
            donor_commitment_manifest=commitments,
        )
    )
    shard_sample_ids = {str(row["sample_id"]) for row in shard_main}
    assert shard_ablations
    assert all(
        row["donor_sample_id"] not in shard_sample_ids
        and row["donor_background_id"] != row["background_id"]
        and row["donor_seed_index"] != row["seed_index"]
        for row in shard_ablations
    )

    result = runner.real_anchored_counterfactual_checks(
        shard_main + shard_ablations,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is True
    assert result["input_ablation_failures"] == []


def test_authentic_committed_non_successor_donor_fails_closed() -> None:
    all_main, _ablations, entries, root, commitments = (
        _structural_common_rows()
    )
    source = next(
        row
        for row in all_main
        if row["dose_index"] == 1 and row["counterfactual_member"] == 0
    )
    cell = sorted(
        (
            row
            for row in all_main
            if row["dose_index"] == source["dose_index"]
            and row["counterfactual_member"]
            == source["counterfactual_member"]
        ),
        key=lambda row: (
            int(row["seed_index"]),
            str(row["background_id"]),
            str(row["sample_id"]),
        ),
    )
    source_index = next(
        index
        for index, row in enumerate(cell)
        if row["background_id"] == source["background_id"]
    )
    successor = cell[(source_index + 1) % len(cell)]
    wrong_authentic = next(
        row
        for row in cell
        if row["background_id"]
        not in {source["background_id"], successor["background_id"]}
    )
    forged = build_matched_input_ablation_task(
        source,
        wrong_authentic,
        donor_commitment_manifest=commitments,
    )

    result = runner.real_anchored_counterfactual_checks(
        [source, forged],
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is False
    assert result["input_ablation_failures"]
    assert result["input_ablation_failures"][0]["checks"][
        "donor_upstream_commitment_valid"
    ] is False


def test_self_consistent_forged_off_shard_history_fails_upstream_commitment() -> None:
    main, ablations, entries, root, _commitments = (
        _structural_common_rows()
    )
    forged_rows = copy.deepcopy(ablations)
    row = forged_rows[0]
    source = next(
        candidate
        for candidate in main
        if candidate["sample_id"] == row["input_ablation_source_sample_id"]
    )
    metadata = row["input_ablation_metadata"]
    ablated = [int(value) for value in metadata["ablated_input_indices"]]
    source_target = np.asarray(source["target"], dtype=float)
    forged_target = np.asarray(row["target"], dtype=float)
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    forged_channels: list[np.ndarray] = []
    for offset, channel in enumerate(ablated):
        donor_channel = np.asarray(
            metadata["donor_visible_history_by_channel"][str(channel)],
            dtype=float,
        ).copy()
        donor_channel += (offset + 1) * 0.03 * np.sin(
            np.arange(context, dtype=float) / 11.0
        )
        destination = source_target[:context, channel]
        donor_center = float(np.mean(donor_channel))
        donor_scale = max(float(np.std(donor_channel)), 1e-9)
        recipient_center = float(np.mean(destination))
        recipient_scale = max(float(np.std(destination)), 1e-9)
        forged_target[:context, channel] = (
            (donor_channel - donor_center)
            / donor_scale
            * recipient_scale
            + recipient_center
        )
        metadata["donor_visible_history_by_channel"][str(channel)] = (
            donor_channel.tolist()
        )
        metadata["affine_match_by_channel"][str(channel)] = {
            "donor_center": donor_center,
            "donor_scale": donor_scale,
            "recipient_center": recipient_center,
            "recipient_scale": recipient_scale,
        }
        forged_channels.append(donor_channel)
    metadata["donor_visible_history_sha256"] = array_sha256(
        np.column_stack(forged_channels)
    )
    metadata["ablated_visible_history_sha256"] = array_sha256(
        forged_target[:context, ablated]
    )
    row["target"] = forged_target.tolist()
    covariates = np.asarray(row["covariates"], dtype=float)
    row["target_sha256"] = protocol.target_and_covariate_sha256(
        forged_target,
        covariates,
    )

    result = runner.real_anchored_counterfactual_checks(
        main + forged_rows,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    failure = next(
        failure
        for failure in result["input_ablation_failures"]
        if failure["sample_id"] == row["sample_id"]
    )
    assert failure["checks"]["affine_replacement_valid"] is True
    assert failure["checks"]["donor_upstream_commitment_valid"] is False
    assert failure["checks"]["donor_provenance_valid"] is False


@pytest.mark.parametrize(
    "tamper",
    (
        "selection_policy",
        "donor_history_hash",
        "ablated_history_hash",
        "donor_background",
        "donor_seed",
        "donor_sample_id",
    ),
)
def test_off_shard_donor_provenance_tampering_fails_closed(
    tamper: str,
) -> None:
    all_main, _all_ablations, entries, root, commitments = (
        _structural_common_rows()
    )
    source_seed = min(int(row["seed_index"]) for row in all_main)
    shard_main = [
        row for row in all_main if int(row["seed_index"]) == source_seed
    ]
    shard_ablations = list(
        iter_mandatory_structural_input_ablation_tasks(
            shard_main,
            donor_samples=all_main,
            donor_commitment_manifest=commitments,
        )
    )
    row = shard_ablations[0]
    if tamper == "selection_policy":
        row["input_ablation_metadata"]["donor_selection_policy"] = "local"
    elif tamper == "donor_history_hash":
        row["input_ablation_metadata"][
            "donor_visible_history_sha256"
        ] = "malformed"
    elif tamper == "ablated_history_hash":
        row["input_ablation_metadata"][
            "ablated_visible_history_sha256"
        ] = "0" * 64
    elif tamper == "donor_background":
        row["donor_background_id"] = row["background_id"]
    elif tamper == "donor_seed":
        row["donor_seed_index"] = row["seed_index"]
    else:
        row["donor_sample_id"] = "wrong-donor-sample"

    result = runner.real_anchored_counterfactual_checks(
        shard_main + shard_ablations,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is False
    assert any(
        not failure["checks"]["donor_provenance_valid"]
        for failure in result["input_ablation_failures"]
    )


def test_arbitrary_ramp_cannot_masquerade_as_affine_donor_replacement() -> None:
    all_main, _all_ablations, entries, root, commitments = (
        _structural_common_rows()
    )
    source_seed = min(int(row["seed_index"]) for row in all_main)
    shard_main = [
        row for row in all_main if int(row["seed_index"]) == source_seed
    ]
    shard_ablations = list(
        iter_mandatory_structural_input_ablation_tasks(
            shard_main,
            donor_samples=all_main,
            donor_commitment_manifest=commitments,
        )
    )
    row = shard_ablations[0]
    metadata = row["input_ablation_metadata"]
    ablated = metadata["ablated_input_indices"]
    target = np.asarray(row["target"], dtype=float)
    ramp = np.linspace(-1.0, 1.0, protocol.REAL_ANCHORED_CONTEXT_LENGTH)
    for offset, channel in enumerate(ablated):
        target[: protocol.REAL_ANCHORED_CONTEXT_LENGTH, channel] = (
            ramp + 0.1 * offset
        )
    row["target"] = target.tolist()
    covariates = (
        None
        if row["covariates"] is None
        else np.asarray(row["covariates"], dtype=float)
    )
    row["target_sha256"] = protocol.target_and_covariate_sha256(
        target,
        covariates,
    )
    metadata["ablated_visible_history_sha256"] = array_sha256(
        target[: protocol.REAL_ANCHORED_CONTEXT_LENGTH, ablated]
    )

    result = runner.real_anchored_counterfactual_checks(
        shard_main + shard_ablations,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )

    assert result["accepted"] is False
    failure = next(
        failure
        for failure in result["input_ablation_failures"]
        if failure["sample_id"] == row["sample_id"]
    )
    assert failure["checks"]["affine_replacement_valid"] is False
    assert failure["checks"]["donor_provenance_valid"] is False


def test_hierarchy_and_d2_formal_rows_are_rejected() -> None:
    main, ablations, entries, root, _manifest = _structural_common_rows()
    d2_rows = copy.deepcopy(main + ablations)
    d2_rows[0]["target_dim"] = 2
    d2 = runner.real_anchored_counterfactual_checks(
        d2_rows,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )
    assert d2["accepted"] is False
    assert any(
        failure["policy"] == "formal_panel_dimension_at_least_three"
        for failure in d2["policy_failures"]
    )

    hierarchy_rows = copy.deepcopy(main + ablations)
    hierarchy_rows[0]["capability_id"] = "hierarchical_coherence"
    hierarchy = runner.real_anchored_counterfactual_checks(
        hierarchy_rows,
        donor_commitment_entries=entries,
        donor_commitment_root_sha256=root,
    )
    assert hierarchy["accepted"] is False
    assert any(
        failure["policy"]
        == "hierarchy_qualification_only_no_formal_rows"
        for failure in hierarchy["policy_failures"]
    )


def test_absent_real_anchored_manifest_component_is_backward_compatible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_id = "gift_electricity_h"
    generation_dir = tmp_path / dataset_id / "02_generation"
    generation_dir.mkdir(parents=True)
    shard_name = "seed_000000_000001"
    files: dict[str, dict] = {}
    for key in ("clean", "robustness", "input_ablations"):
        path = generation_dir / f"{key}.jsonl"
        protocol.write_jsonl(path, ())
        files[key] = {**protocol.file_record(path), "row_count": 0}
    manifest_path = generation_dir / f"manifest__{shard_name}.json"
    config = {
        "schema_version": "cafe.generation_config.v1",
        "dataset_id": dataset_id,
        "calibration_bundle_sha256": "a" * 64,
        "seed_start": 0,
        "seed_count": 1,
        "seed_indexes": [0],
    }
    protocol.write_json(
        manifest_path,
        {
            "schema_version": "cafe.generation_manifest.v1",
            "config": config,
            "config_sha256": protocol.json_sha256(config),
            "files": files,
        },
    )
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(
            dataset_id=dataset_id,
            output_root=tmp_path,
            seed_start=0,
            seed_count=1,
        ),
    )

    assert runner.main() == 0
    report = protocol.read_json(
        generation_dir / f"validation__{shard_name}.json"
    )
    assert report["schema_version"] == "cafe.generation_validation.v2"
    assert report["accepted"] is True
    assert report["real_anchored_sample_count"] == 0
    assert report["real_anchored_validation"]["status"] == "not_present"
    assert report["mase_scale_audit"]["sample_count"] == 0
