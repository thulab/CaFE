from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cafe import protocol
from cafe.data.real import RealDatasetBundle, RealSeriesRecord
from cafe.generation.reference_bank import (
    freeze_real_anchored_qualification_policy,
    split_real_anchored_background_banks,
)
from cafe.generation.structural_real_counterfactuals import (
    STRUCTURAL_ALPHAS,
    _design_holdout_gain,
    _incremental_gain,
    _safe_r2,
    apply_structural_contract,
    build_matched_input_ablation_task,
    build_structural_donor_commitment_manifest,
    build_structural_real_anchored_backgrounds,
    fit_common_factor_contract,
    fit_covariate_response_contract,
    fit_cross_series_contract,
    fit_hierarchy_qualification_contract,
    fit_structural_capability_contracts,
    iter_mandatory_structural_input_ablation_tasks,
    iter_structural_real_anchored_samples,
    public_structural_background,
    structural_threshold_contract,
    available_structural_capabilities,
    available_structural_sensitivity_capabilities,
    validate_structural_availability,
    validate_structural_contract,
    validate_structural_donor_commitment_manifest,
)


SOURCE_LENGTH = protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH


def test_degenerate_structural_gains_are_finite_json_zero() -> None:
    short_panel = np.zeros((20, 2), dtype=float)
    zero_error_panel = np.zeros((96, 2), dtype=float)
    short_base = np.ones((30, 1), dtype=float)
    short_response = np.zeros((30, 1), dtype=float)
    short_target = np.zeros(30, dtype=float)
    exact_base = np.ones((96, 1), dtype=float)
    exact_response = np.zeros((96, 1), dtype=float)
    exact_target = np.full(96, 3.5, dtype=float)

    diagnostics = {
        "cross_degenerate_holdout_gain": _incremental_gain(
            short_panel,
            source=0,
            destination=1,
            lag=1,
        ),
        "cross_zero_base_error_gain": _incremental_gain(
            zero_error_panel,
            source=0,
            destination=1,
            lag=1,
        ),
        "covariate_degenerate_holdout_gain": _design_holdout_gain(
            short_base,
            short_response,
            short_target,
        ),
        "covariate_zero_base_error_gain": _design_holdout_gain(
            exact_base,
            exact_response,
            exact_target,
        ),
        "constant_target_r2": _safe_r2(np.ones(16), np.ones(16)),
    }

    assert all(value == 0.0 and np.isfinite(value) for value in diagnostics.values())
    assert json.loads(json.dumps(diagnostics, allow_nan=False)) == diagnostics


def _spec(dataset_id: str) -> protocol.DatasetSpec:
    return protocol.DatasetSpec(
        dataset_id=dataset_id,
        logical_name="Structural fixture",
        config_id="fixture",
        asset_name="fixture",
        domain="Test",
        real_data_adapter="fixture",
    )


def _bundle(*records: RealSeriesRecord) -> RealDatasetBundle:
    return RealDatasetBundle(
        frequency="h",
        records=tuple(records),
        asset_files=(),
        adapter_id="fixture",
        metadata={"fixture": True},
    )


def _common_panel(*, phase: float = 0.0, dimension: int = 4) -> np.ndarray:
    time = np.arange(SOURCE_LENGTH, dtype=float)
    factor = (
        np.sin(2.0 * np.pi * time / 24.0 + phase)
        + 0.3 * np.cos(2.0 * np.pi * time / 12.0 - phase)
    )
    loadings = np.asarray([1.0, -0.9, 1.2, -1.1])[:dimension]
    periods = np.asarray([37.0, 41.0, 43.0, 47.0])[:dimension]
    return np.vstack(
        [
            loading * factor
            + 0.08
            * np.sin(2.0 * np.pi * time / period + index)
            for index, (loading, period) in enumerate(
                zip(loadings, periods, strict=True)
            )
        ]
    )


def _persistent_common_panel(*, dimension: int = 4) -> np.ndarray:
    time = np.arange(SOURCE_LENGTH, dtype=float)
    phase = np.pi / 2.0 - 2.0 * np.pi * 503.0 / 200.0
    factor = np.sin(2.0 * np.pi * time / 200.0 + phase)
    loadings = np.asarray([1.0, -0.9, 1.2, -1.1])[:dimension]
    return np.vstack(
        [
            loading * factor
            + 0.03 * np.sin(2.0 * np.pi * time / (41.0 + index))
            for index, loading in enumerate(loadings)
        ]
    )


def _cross_panel() -> np.ndarray:
    rng = np.random.default_rng(4)
    driver = np.empty(SOURCE_LENGTH, dtype=float)
    driver[0] = 0.0
    for index in range(1, SOURCE_LENGTH):
        driver[index] = 0.65 * driver[index - 1] + rng.normal()
    responders: list[np.ndarray] = []
    for gain in (1.2, -0.9, 0.7):
        response = np.zeros(SOURCE_LENGTH, dtype=float)
        for index in range(1, SOURCE_LENGTH):
            response[index] = (
                0.45 * response[index - 1]
                + gain * driver[max(0, index - 5)]
                + 0.12 * rng.normal()
            )
        responders.append(response)
    return np.vstack([driver, *responders])


def _covariate_record(*, future_target_offset: float = 0.0) -> RealSeriesRecord:
    time = np.arange(SOURCE_LENGTH, dtype=float)
    covariates = np.column_stack(
        [
            np.sin(2.0 * np.pi * time / 31.0),
            ((time.astype(int) % 47) < 5).astype(float),
        ]
    )
    target = np.zeros(SOURCE_LENGTH, dtype=float)
    for index in range(2, SOURCE_LENGTH):
        target[index] = (
            0.3 * target[index - 1]
            + 1.1 * covariates[index, 0]
            + 0.7 * covariates[index - 1, 1]
            + 0.05 * np.sin(index / 7.0)
        )
    target[
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH :
    ] += future_target_offset
    return RealSeriesRecord(
        item_id="covariate_item",
        values=target,
        covariates=covariates,
        covariate_names=("weather", "event"),
        covariate_kind="known_future",
    )


def _one_background(
    dataset_id: str,
    record: RealSeriesRecord,
) -> dict[str, object]:
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        _spec(dataset_id),
        source_root=Path("/unused"),
        maximum_backgrounds=1,
        real_bundle=_bundle(record),
    )
    assert len(backgrounds) == 1
    return backgrounds[0]


def test_structural_background_preserves_synchronized_paths_and_semantics() -> None:
    time = np.arange(SOURCE_LENGTH, dtype=float)
    parent = 20.0 + np.sin(2.0 * np.pi * time / 24.0)
    contrast = 3.0 * np.sin(2.0 * np.pi * time / 31.0)
    children = np.vstack([0.6 * parent + contrast, 0.4 * parent - contrast])
    promotions = np.column_stack(
        [
            ((time.astype(int) % 14) < 2).astype(float),
            ((time.astype(int) % 17) < 3).astype(float),
        ]
    )
    record = RealSeriesRecord(
        item_id="siblings",
        values=children,
        channel_ids=("child_a", "child_b"),
        covariates=promotions,
        covariate_names=("promo_a", "promo_b"),
        covariate_kind="known_future",
        hierarchy_values=children,
        hierarchy_kind="children_only_additive",
        structural_group_id="brand_a",
    )
    background = _one_background("structural_semantics", record)

    assert background["target_dim"] == 2
    assert background["panel_contract"]["formal_main_eligible"] is False
    assert background["panel_contract"]["sensitivity_only"] is True
    assert background["panel_contract"]["role"] == "d2_sensitivity_only"
    assert np.asarray(background["target"]).shape == (384, 2)
    assert np.asarray(background["_decomposition_target"]).shape == (504, 2)
    assert np.asarray(
        background["known_future_covariates"]["target"]
    ).shape == (384, 2)
    hierarchy = background["hierarchy"]
    assert hierarchy["parent_source"] == (
        "constructed_exact_sum_of_declared_children"
    )
    hierarchy_target = np.asarray(hierarchy["target"])
    np.testing.assert_allclose(
        hierarchy_target[:, 0],
        np.sum(hierarchy_target[:, 1:], axis=1),
        atol=1e-10,
    )
    public = public_structural_background(background)
    assert "_decomposition_target" not in public
    assert "_source_window" not in public["known_future_covariates"]
    assert "_raw_source_window" not in public["hierarchy"]


def test_d2_panel_contracts_are_sensitivity_only_never_formal() -> None:
    background = _one_background(
        "d2_panel",
        RealSeriesRecord(item_id="panel", values=_common_panel(dimension=2)),
    )
    common = fit_common_factor_contract(background)
    assert common["formal_main_eligible"] is False
    assert common["sensitivity_eligible"] is True
    with pytest.raises(ValueError, match="not eligible"):
        apply_structural_contract(background, common, alpha=1.2)
    target, _metadata = apply_structural_contract(
        background,
        common,
        alpha=1.2,
        allow_sensitivity=True,
    )
    assert target.shape == (384, 2)


@pytest.mark.parametrize(
    ("background_count", "expected"),
    [(1, ()), (2, ("common_factor",))],
)
def test_d2_sensitivity_requires_two_distinct_donor_backgrounds(
    background_count: int,
    expected: tuple[str, ...],
) -> None:
    records = tuple(
        RealSeriesRecord(
            item_id=f"panel_{index}",
            values=_common_panel(phase=0.3 * index, dimension=2),
        )
        for index in range(background_count)
    )
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        _spec("d2_sensitivity_n_fixture"),
        source_root=Path("/unused"),
        maximum_backgrounds=background_count,
        real_bundle=_bundle(*records),
    )
    rows, availability = fit_structural_capability_contracts(
        backgrounds,
        capability_ids=("common_factor",),
    )

    assert (
        available_structural_sensitivity_capabilities(availability)
        == expected
    )
    assert availability["cells"][0]["sensitivity_generation_eligible"] is bool(
        expected
    )
    validate_structural_availability(availability, rows)


def test_common_factor_is_history_only_and_requires_matched_ablation() -> None:
    records = (
        RealSeriesRecord(item_id="panel_a", values=_common_panel(phase=0.0)),
        RealSeriesRecord(item_id="panel_b", values=_common_panel(phase=0.4)),
    )
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        _spec("common_factor_fixture"),
        source_root=Path("/unused"),
        maximum_backgrounds=2,
        real_bundle=_bundle(*records),
    )
    rows, availability = fit_structural_capability_contracts(
        backgrounds,
        capability_ids=("common_factor",),
    )
    assert availability["formal_background_count_by_capability"] == {
        "common_factor": 2
    }
    assert availability["cells"][0]["status"] == "unavailable"
    assert availability["cells"][0]["reason_codes"] == [
        "insufficient_eligible_backgrounds"
    ]
    assert available_structural_capabilities(availability) == ()
    validate_structural_availability(availability, rows)
    for row, background in zip(rows, backgrounds, strict=True):
        contract = row["contract"]
        assert contract["mandatory_input_ablation"]["required"] is True
        assert contract["mandatory_input_ablation"]["excluded_from_primary_score"] is True
        assert contract["dose_design_reference"][
            "affected_channel_indices"
        ] == contract["fit_diagnostics"]["nondegenerate_loading_indices"]
        assert contract["dose_design_reference"]["evidence_role"] == "formal"
        validate_structural_contract(contract, background)
        identity, identity_meta = apply_structural_contract(
            background,
            contract,
            alpha=1.0,
        )
        np.testing.assert_array_equal(identity, background["target"])
        assert identity_meta["target_future_used_for_delta"] is False

    samples = list(
        iter_structural_real_anchored_samples(
            backgrounds,
            rows,
            seed_indexes=range(len(backgrounds)),
        )
    )
    assert len(samples) == 2 * len(STRUCTURAL_ALPHAS) * 2
    required_generic_fields = {
        "master_sample_id",
        "config_id",
        "task_id",
        "profile_id",
        "generator_family_role",
        "intensity",
        "seed_index",
        "mase_period",
        "mase_scale_by_target",
        "future_sha256",
    }
    assert required_generic_fields.issubset(samples[0])
    assert samples[0]["evaluation_table"] == "real_anchored_counterfactual"
    assert "protected_target_index" in samples[0]["generation_metadata"]
    donor_commitments = build_structural_donor_commitment_manifest(
        backgrounds,
        rows,
        dataset_id="common_factor_fixture",
    )
    validate_structural_donor_commitment_manifest(
        donor_commitments,
        backgrounds,
        rows,
        dataset_id="common_factor_fixture",
    )
    ablation_rows = list(
        iter_mandatory_structural_input_ablation_tasks(
            samples,
            donor_commitment_manifest=donor_commitments,
        )
    )
    assert len(ablation_rows) == len(samples)
    assert all(
        row["evaluation_table"] == "real_anchored_input_ablation"
        for row in ablation_rows
    )
    first = next(
        row
        for row in samples
        if row["background_id"] == backgrounds[0]["background_id"]
        and row["dose_index"] == 1
        and row["counterfactual_member"] == 1
    )
    donor = next(
        row
        for row in samples
        if row["background_id"] == backgrounds[1]["background_id"]
        and row["dose_index"] == 1
        and row["counterfactual_member"] == 1
    )
    ablation = build_matched_input_ablation_task(
        first,
        donor,
        donor_commitment_manifest=donor_commitments,
    )
    assert ablation["evaluation_table"] == "real_anchored_input_ablation"
    assert ablation["input_ablation_source_pair_id"] == (
        first["counterfactual_pair_id"]
    )
    assert ablation["master_sample_id"] == ablation["sample_id"]
    assert ablation["baseline_sample_id"].endswith("__input_ablation")
    assert ablation["excluded_from_primary_score"] is True
    assert ablation["structural_donor_commitment_root_sha256"] == (
        donor_commitments["commitment_root_sha256"]
    )
    assert ablation["input_ablation_metadata"][
        "donor_upstream_commitment"
    ]["entry_sha256"] == ablation[
        "donor_structural_commitment_entry_sha256"
    ]
    assessed = ablation["input_ablation_metadata"]["assessed_target_indices"]
    np.testing.assert_array_equal(
        np.asarray(ablation["target"])[:336, assessed],
        np.asarray(first["target"])[:336, assessed],
    )
    np.testing.assert_array_equal(
        np.asarray(ablation["target"])[336:],
        np.asarray(first["target"])[336:],
    )

    first_seed = list(
        iter_structural_real_anchored_samples(
            backgrounds,
            rows,
            seed_indexes=(0,),
        )
    )
    second_seed = list(
        iter_structural_real_anchored_samples(
            backgrounds,
            rows,
            seed_indexes=(1,),
        )
    )
    assert len(first_seed) == len(STRUCTURAL_ALPHAS) * 2
    assert len(second_seed) == len(STRUCTURAL_ALPHAS) * 2
    assert {row["seed_index"] for row in first_seed} == {0}
    assert {row["seed_index"] for row in second_seed} == {1}
    assert {row["background_id"] for row in first_seed}.isdisjoint(
        {row["background_id"] for row in second_seed}
    )
    donor_population = [*first_seed, *second_seed]
    first_ablation = list(
        iter_mandatory_structural_input_ablation_tasks(
            first_seed,
            donor_samples=donor_population,
            donor_commitment_manifest=donor_commitments,
        )
    )
    second_ablation = list(
        iter_mandatory_structural_input_ablation_tasks(
            second_seed,
            donor_samples=donor_population,
            donor_commitment_manifest=donor_commitments,
        )
    )
    full_ablation = list(
        iter_mandatory_structural_input_ablation_tasks(
            donor_population,
            donor_commitment_manifest=donor_commitments,
        )
    )
    assert len(first_ablation) == len(first_seed)
    assert len(second_ablation) == len(second_seed)
    assert {
        row["sample_id"]: row["donor_background_id"]
        for row in [*first_ablation, *second_ablation]
    } == {
        row["sample_id"]: row["donor_background_id"]
        for row in full_ablation
    }
    assert all(
        row["background_id"] != row["donor_background_id"]
        and row["input_ablation_metadata"]["donor_selection_policy"]
        == "global_eligible_background_successor_shard_invariant_v1"
        for row in full_ablation
    )

    forged_donor = dict(donor)
    forged_target = np.asarray(donor["target"], dtype=float).copy()
    forged_target[0, 0] += 0.25
    forged_donor["target"] = forged_target.tolist()
    forged_donor["target_sha256"] = protocol.target_and_covariate_sha256(
        forged_target,
        None,
    )
    with pytest.raises(ValueError, match="history differs from commitment"):
        build_matched_input_ablation_task(
            first,
            forged_donor,
            donor_commitment_manifest=donor_commitments,
        )
    self_consistent_forgery = copy.deepcopy(donor_commitments)
    forged_entry = next(
        entry
        for entry in self_consistent_forgery["entries"]
        if entry["sample_id"] == donor["sample_id"]
    )
    forged_entry["visible_history_by_channel_sha256"]["0"] = "f" * 64
    forged_entry.pop("entry_sha256")
    forged_entry["entry_sha256"] = protocol.json_sha256(forged_entry)
    self_consistent_forgery["entries_sha256"] = protocol.json_sha256(
        self_consistent_forgery["entries"]
    )
    self_consistent_forgery.pop("commitment_root_sha256")
    self_consistent_forgery["commitment_root_sha256"] = protocol.json_sha256(
        self_consistent_forgery
    )
    with pytest.raises(ValueError, match="disagree with calibration banks"):
        validate_structural_donor_commitment_manifest(
            self_consistent_forgery,
            backgrounds,
            rows,
            dataset_id="common_factor_fixture",
        )


def test_cross_series_contract_controls_only_predictive_transfer() -> None:
    background = _one_background(
        "cross_series_fixture",
        RealSeriesRecord(item_id="panel", values=_cross_panel()),
    )
    contract = fit_cross_series_contract(background)
    assert contract["formal_main_eligible"] is True
    diagnostics = contract["fit_diagnostics"]
    assert diagnostics["source"] == 0
    assert diagnostics["lag"] == 5
    assert diagnostics["interpretation"] == (
        "directed_predictive_transfer_not_causal_scm"
    )
    assert diagnostics["causal_identification_claimed"] is False
    component = np.asarray(contract["component"])
    np.testing.assert_array_equal(component[:, diagnostics["source"]], 0.0)
    assert contract["dose_design_reference"][
        "affected_channel_indices"
    ] == diagnostics["responders"]
    treatment, metadata = apply_structural_contract(
        background,
        contract,
        alpha=2.0,
    )
    baseline = np.asarray(background["target"])
    np.testing.assert_array_equal(
        treatment[:, diagnostics["source"]],
        baseline[:, diagnostics["source"]],
    )
    assert metadata["mandatory_input_ablation"]["required"] is True


def test_covariate_contract_uses_known_future_path_not_target_future() -> None:
    first = _one_background(
        "covariate_fixture",
        _covariate_record(future_target_offset=0.0),
    )
    second = _one_background(
        "covariate_fixture",
        _covariate_record(future_target_offset=100.0),
    )
    first_contract = fit_covariate_response_contract(first)
    second_contract = fit_covariate_response_contract(second)
    assert first_contract["formal_main_eligible"] is True
    assert first_contract["contract_sha256"] == second_contract["contract_sha256"]
    assert first_contract["component_sha256"] == second_contract["component_sha256"]
    assert first_contract["fit_diagnostics"]["causal_identification_claimed"] is False
    assert first_contract["dose_design_reference"][
        "affected_channel_indices"
    ] == first_contract["fit_diagnostics"]["eligible_target_indices"]
    first_treatment, first_metadata = apply_structural_contract(
        first,
        first_contract,
        alpha=1.6,
    )
    second_treatment, second_metadata = apply_structural_contract(
        second,
        second_contract,
        alpha=1.6,
    )
    first_delta = first_treatment - np.asarray(first["target"])
    second_delta = second_treatment - np.asarray(second["target"])
    np.testing.assert_allclose(first_delta, second_delta, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(
        first_metadata["truth_delta"], second_metadata["truth_delta"]
    )
    assert first_metadata["known_future_covariate_path_used_for_delta"] is True
    assert second_metadata["target_future_used_for_delta"] is False


def test_hierarchy_is_zero_sum_qualification_only_with_negativity_audit() -> None:
    time = np.arange(SOURCE_LENGTH, dtype=float)
    parent = 12.0 + np.sin(2.0 * np.pi * time / 24.0)
    contrast = 4.5 * np.sin(2.0 * np.pi * time / 31.0)
    children = np.vstack([0.55 * parent + contrast, 0.45 * parent - contrast])
    assert float(np.min(children)) > 0.0
    background = _one_background(
        "hierarchy_fixture",
        RealSeriesRecord(
            item_id="siblings",
            values=children,
            hierarchy_values=children,
            hierarchy_kind="children_only_additive",
            structural_group_id="brand",
        ),
    )
    contract = fit_hierarchy_qualification_contract(background)
    assert contract["fit_diagnostics"]["qualification_passed"] is True
    assert contract["dose_design_reference"]["evidence_role"] == (
        "qualification_only"
    )
    assert contract["dose_design_reference"][
        "affected_channel_indices"
    ] == background["hierarchy"]["child_indices"]
    assert contract["generation_eligible"] is False
    assert contract["ranking_eligible"] is False
    component = np.asarray(contract["component"])
    np.testing.assert_allclose(np.sum(component[:, 1:], axis=1), 0.0, atol=1e-12)
    negativity = contract["fit_diagnostics"]["raw_negativity_audit_by_alpha"]
    assert negativity["2.0"]["total_negative_value_count"] > 0
    assert contract["fit_diagnostics"][
        "raw_negativity_affects_fit_or_thresholds"
    ] is False
    with pytest.raises(ValueError, match="qualification-only"):
        apply_structural_contract(background, contract, alpha=1.2)
    rows, _availability = fit_structural_capability_contracts(
        [background],
        capability_ids=("hierarchical_coherence",),
    )
    assert rows[0]["qualification_available"] is True
    assert rows[0]["generation_eligible"] is False
    assert list(iter_structural_real_anchored_samples([background], rows)) == []


def test_reference_policy_is_json_safe_and_reused_by_evaluation_rows() -> None:
    records = [
        RealSeriesRecord(
            item_id=f"panel_{index}",
            values=_common_panel(phase=0.2 * index),
        )
        for index in range(4)
    ]
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        _spec("reference_policy_fixture"),
        source_root=Path("/unused"),
        maximum_backgrounds=4,
        real_bundle=_bundle(*records),
    )
    evaluation, reference, split_audit = split_real_anchored_background_banks(
        backgrounds,
        maximum_evaluation_backgrounds=2,
        maximum_reference_backgrounds=2,
        source_window_length=SOURCE_LENGTH,
    )
    reference_rows, _availability = fit_structural_capability_contracts(
        reference,
        capability_ids=("common_factor",),
    )
    policy = freeze_real_anchored_qualification_policy(
        reference_rows,
        reference_background_ids=[row["background_id"] for row in reference],
        bank_split_audit=split_audit,
    )
    dose_calibration = policy["capabilities"]["common_factor"][
        "dose_calibration"
    ]
    assert dose_calibration["status"] == "unavailable"
    evaluation_rows, evaluation_availability = (
        fit_structural_capability_contracts(
            evaluation,
            capability_ids=("common_factor",),
            frozen_qualification_policy=policy,
        )
    )
    assert all(
        row["frozen_qualification_policy_sha256"]
        == policy["qualification_policy_sha256"]
        for row in evaluation_rows
    )
    assert all(
        row["formal_main_available"] is False
        and row["generation_eligible"] is False
        and row["contract"]["dose_pairing_eligible"] is False
        and row["contract"]["applied_alpha_grid"] == []
        for row in evaluation_rows
    )
    assert evaluation_availability["frozen_qualification_policy_sha256"] == (
        policy["qualification_policy_sha256"]
    )
    assert structural_threshold_contract()[
        "evaluation_origin_adaptation_allowed"
    ] is False


def test_frozen_structural_dose_grid_drives_rows_gates_and_commitments() -> None:
    records = [
        RealSeriesRecord(
            item_id=f"panel_{index}",
            values=_persistent_common_panel(),
        )
        for index in range(8)
    ]
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        _spec("structural_dose_fixture"),
        source_root=Path("/unused"),
        maximum_backgrounds=8,
        real_bundle=_bundle(*records),
    )
    evaluation, reference, split_audit = split_real_anchored_background_banks(
        backgrounds,
        maximum_evaluation_backgrounds=4,
        maximum_reference_backgrounds=4,
        source_window_length=SOURCE_LENGTH,
    )
    reference_rows, _reference_availability = (
        fit_structural_capability_contracts(
            reference,
            capability_ids=("common_factor",),
        )
    )
    policy = freeze_real_anchored_qualification_policy(
        reference_rows,
        reference_background_ids=[row["background_id"] for row in reference],
        bank_split_audit=split_audit,
    )
    calibration = policy["capabilities"]["common_factor"][
        "dose_calibration"
    ]
    assert calibration["status"] == "available"
    assert calibration["applied_alpha_grid"] != list(STRUCTURAL_ALPHAS)

    rows, availability = fit_structural_capability_contracts(
        evaluation,
        capability_ids=("common_factor",),
        frozen_qualification_policy=policy,
    )
    assert availability["formal_background_count_by_capability"] == {
        "common_factor": 4
    }
    for row, background in zip(rows, evaluation, strict=True):
        contract = row["contract"]
        assert contract["dose_calibration"]["dose_policy_sha256"] == (
            calibration["policy_sha256"]
        )
        assert contract["canonical_strength_grid"] == calibration[
            "strength_grid"
        ]
        assert len(contract["applied_alpha_grid"]) == 5
        assert all(
            gate["accepted"] is True
            for gate in contract["paired_minimum_separation_gate"]
        )
        validate_structural_contract(contract, background)

    # A caller-supplied legacy alpha is ignored once a reference-frozen grid
    # is attached to an evaluation row.
    first_seed_samples = list(
        iter_structural_real_anchored_samples(
            evaluation,
            rows,
            alphas=(1.2,),
            seed_indexes=(0,),
        )
    )
    assert len(first_seed_samples) == 10
    for dose_index in range(1, 6):
        pair = [
            sample
            for sample in first_seed_samples
            if sample["dose_index"] == dose_index
        ]
        baseline = next(
            sample for sample in pair if sample["counterfactual_member"] == 0
        )
        treatment = next(
            sample for sample in pair if sample["counterfactual_member"] == 1
        )
        canonical_strength = calibration["strength_grid"][dose_index - 1]
        applied_alpha = treatment["applied_alpha"]
        assert baseline["dose_value"] == baseline["intensity_lambda"] == 0.0
        assert baseline["applied_alpha"] == 1.0
        assert treatment["dose_value"] == canonical_strength
        assert treatment["intensity_lambda"] == canonical_strength
        assert treatment["applied_alpha"] == applied_alpha
        assert baseline["paired_treatment_strength"] == canonical_strength
        assert treatment["paired_treatment_strength"] == canonical_strength
        assert baseline["paired_treatment_applied_alpha"] == applied_alpha
        assert treatment["paired_treatment_applied_alpha"] == applied_alpha
        assert baseline["paired_minimum_separation_gate"]["status"] == (
            "not_applicable"
        )
        assert baseline["paired_minimum_separation_gate"][
            "paired_treatment_gate_status"
        ] == "passed"
        assert treatment["paired_minimum_separation_gate"]["accepted"] is True
        assert treatment["paired_minimum_separation_gate"][
            "adjacent_distance_role"
        ] == "diagnostic_only"
        assert treatment["anti_copy_gate"]["status"] == "not_applicable"

    commitments = build_structural_donor_commitment_manifest(
        evaluation,
        rows,
        dataset_id="structural_dose_fixture",
    )
    assert commitments[
        "dose_calibration_policy_sha256_by_capability"
    ] == {"common_factor": calibration["policy_sha256"]}
    assert all(
        entry["dose_calibration_policy_sha256"] == calibration["policy_sha256"]
        and entry["paired_minimum_separation_gate_sha256"] is not None
        for entry in commitments["entries"]
    )
    validate_structural_donor_commitment_manifest(
        commitments,
        evaluation,
        rows,
        dataset_id="structural_dose_fixture",
    )


def test_structural_formal_cell_requires_four_independent_backgrounds() -> None:
    records = tuple(
        RealSeriesRecord(
            item_id=f"panel_{index}",
            values=_common_panel(phase=0.3 * index),
        )
        for index in range(4)
    )
    backgrounds, _metadata = build_structural_real_anchored_backgrounds(
        _spec("formal_n_fixture"),
        source_root=Path("/unused"),
        maximum_backgrounds=4,
        real_bundle=_bundle(*records),
    )
    rows, availability = fit_structural_capability_contracts(
        backgrounds,
        capability_ids=("common_factor",),
    )

    assert availability["cells"][0]["status"] == "available"
    assert availability["cells"][0]["formal_background_count"] == 4
    assert available_structural_capabilities(availability) == (
        "common_factor",
    )
    validate_structural_availability(availability, rows)

    tampered = dict(availability)
    tampered["formal_background_count_by_capability"] = {
        "common_factor": 3
    }
    with pytest.raises(ValueError, match="formal counts"):
        validate_structural_availability(tampered, rows)
