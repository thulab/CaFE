from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cafe import protocol
from cafe.data.real import RealDatasetBundle, RealSeriesRecord
from cafe.generation.families import generate_deterministic_sample
from cafe.generation.real_anchored_policy import (
    NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY,
    NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY,
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
)
from cafe.generation.real_counterfactuals import (
    fit_background_capability_contracts,
    iter_nonlinear_replay_sensitivity_samples,
    iter_real_anchored_samples,
    public_background,
)
from cafe.generation.real_path_dynamics import (
    apply_real_path_dynamic_contract,
    default_dynamic_qualification_policy,
    fit_real_path_dynamic_contract,
    validate_real_path_dynamic_contract,
)
from cafe.generation.reference_bank import (
    freeze_real_anchored_qualification_policy,
    split_real_anchored_background_banks,
    validate_evaluation_qualification_policy,
)
from cafe.validation.runner import real_anchored_counterfactual_checks


def _mase_scale(history: np.ndarray) -> float:
    return max(float(np.mean(np.abs(history[24:] - history[:-24]))), 1e-3)


def _intermittent_path(seed: int) -> np.ndarray:
    target, _metadata, _covariates = generate_deterministic_sample(
        "predictable_intermittency",
        552,
        504,
        1,
        24,
        5,
        np.random.default_rng(seed),
    )
    return np.asarray(target[:, 0], dtype=float)


def _nonlinear_path(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    length = 552
    state = np.zeros(length, dtype=float)
    state[:24] = rng.normal(0.0, 1.0, 24)
    innovations = rng.normal(0.0, 0.3, length)
    for index in range(24, length):
        delayed = float(np.clip(state[index - 6], -3.0, 3.0))
        nonlinear = delayed**2 / (1.0 + delayed**2) - 0.5
        state[index] = (
            0.15 * state[index - 1]
            + 0.05 * state[index - 24]
            + 0.8 * nonlinear
            + innovations[index]
        )
    time = np.arange(length, dtype=float)
    return state + 0.05 * np.sin(2.0 * np.pi * time / 24.0)


def _fit(path: np.ndarray, capability_id: str) -> dict[str, object]:
    history = path[:504]
    reference = history[-336:]
    return fit_real_path_dynamic_contract(
        history,
        capability_id=capability_id,
        carrier_period=24.0,
        secondary_periods=(),
        reference_history=reference,
        mase_period=24,
        mase_scale=_mase_scale(reference),
        mase_effective_period=24,
        mase_scale_source="seasonal_history",
    )


def _bundle(paths: list[np.ndarray]) -> RealDatasetBundle:
    return RealDatasetBundle(
        frequency="h",
        records=tuple(
            RealSeriesRecord(item_id=f"item_{index}", values=path)
            for index, path in enumerate(paths)
        ),
        asset_files=(),
        adapter_id="fixture",
    )


def _backgrounds(paths: list[np.ndarray]) -> list[dict[str, object]]:
    dataset = protocol.DatasetSpec(
        "real_path_dynamic_fixture",
        "Real path dynamic fixture",
        "fixture",
        "unused",
        "Test",
        real_data_adapter="fixture",
    )
    backgrounds, _metadata = protocol.build_real_anchored_backgrounds(
        dataset,
        source_root=Path("/unused"),
        maximum_backgrounds=len(paths),
        sample_seed=1729,
        real_bundle=_bundle(paths),
    )
    return backgrounds


def test_predictable_intermittency_is_history_only_and_exactly_additive() -> None:
    baseline = _intermittent_path(2)
    contract = _fit(baseline, "predictable_intermittency")

    assert contract["available"] is True
    validate_real_path_dynamic_contract(contract)
    identity, identity_metadata = apply_real_path_dynamic_contract(
        baseline,
        contract,
        alpha=1.0,
        context_length=504,
    )
    np.testing.assert_array_equal(identity, baseline)
    assert identity_metadata["intervention_rms"] == 0.0

    member_14, metadata_14 = apply_real_path_dynamic_contract(
        baseline,
        contract,
        alpha=1.4,
        context_length=504,
    )
    member_18, metadata_18 = apply_real_path_dynamic_contract(
        baseline,
        contract,
        alpha=1.8,
        context_length=504,
    )
    np.testing.assert_allclose(
        (member_14 - baseline) / 0.4,
        (member_18 - baseline) / 0.8,
        rtol=1e-12,
        atol=1e-12,
    )
    assert metadata_14["dose_response_law"] == (
        "additive_linear_in_alpha_minus_one"
    )
    assert metadata_18["event_clock_holdout_r2"] >= 0.10

    changed_future = baseline.copy()
    changed_future[504:] += np.linspace(-20.0, 20.0, 48)
    changed_member, _metadata = apply_real_path_dynamic_contract(
        changed_future,
        contract,
        alpha=1.8,
        context_length=504,
    )
    np.testing.assert_allclose(
        changed_member - changed_future,
        member_18 - baseline,
        rtol=1e-13,
        atol=1e-13,
    )


def test_nonlinear_persistence_uses_dynamic_zero_innovation_rollout() -> None:
    baseline = _nonlinear_path(4)
    contract = _fit(baseline, "nonlinear_persistence")

    assert contract["available"] is True
    assert contract["future_component_source"] == (
        "paired_zero_innovation_dynamic_rollout"
    )
    assert contract["history_residual_replay_policy"] == (
        NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
    )
    validate_real_path_dynamic_contract(contract)

    identity, _metadata = apply_real_path_dynamic_contract(
        baseline,
        contract,
        alpha=1.0,
        context_length=504,
    )
    np.testing.assert_array_equal(identity, baseline)
    members: dict[float, np.ndarray] = {}
    rms_values: list[float] = []
    for alpha in (1.2, 1.4, 1.6, 1.8, 2.0):
        member, metadata = apply_real_path_dynamic_contract(
            baseline,
            contract,
            alpha=alpha,
            context_length=504,
        )
        members[alpha] = np.asarray(member)
        rms_values.append(float(metadata["intervention_rms"]))
        assert metadata["dose_response_law"] == (
            "dynamic_recursive_nonproportional"
        )
        assert metadata["future_innovation_policy"] == (
            NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY
        )
    assert all(
        right > left
        for left, right in zip(rms_values, rms_values[1:], strict=False)
    )
    assert not np.allclose(
        (members[1.4] - baseline) / 0.4,
        (members[2.0] - baseline),
        rtol=1e-6,
        atol=1e-8,
    )

    changed_future = baseline.copy()
    changed_future[504:] = -100.0
    changed_member, _metadata = apply_real_path_dynamic_contract(
        changed_future,
        contract,
        alpha=2.0,
        context_length=504,
    )
    np.testing.assert_allclose(
        changed_member - changed_future,
        members[2.0] - baseline,
        rtol=1e-12,
        atol=1e-12,
    )

    replay_member, replay_metadata = apply_real_path_dynamic_contract(
        baseline,
        contract,
        alpha=2.0,
        context_length=504,
        future_innovation_policy=(
            NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
        ),
    )
    assert replay_metadata["future_innovation_policy"] == (
        NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
    )
    np.testing.assert_array_equal(
        replay_member[:504] - baseline[:504],
        members[2.0][:504] - baseline[:504],
    )
    assert not np.array_equal(
        replay_member[504:] - baseline[504:],
        members[2.0][504:] - baseline[504:],
    )


def test_dynamic_contract_requires_reference_bank_threshold_provenance() -> None:
    baseline = _nonlinear_path(4)
    default_policy = default_dynamic_qualification_policy()
    thresholds = default_policy["qualification_thresholds"]
    frozen = {
        "schema_version": REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
        "threshold_source_policy": QUALIFICATION_THRESHOLD_SOURCE_POLICY,
        "qualification_policy_sha256": "a" * 64,
        "capabilities": {
            capability_id: {
                "qualification_policy_id": f"frozen-{capability_id}",
                "qualification_thresholds": values,
            }
            for capability_id, values in thresholds.items()
        },
    }
    history = baseline[:504]
    reference = history[-336:]
    contract = fit_real_path_dynamic_contract(
        history,
        capability_id="nonlinear_persistence",
        carrier_period=24.0,
        secondary_periods=(),
        reference_history=reference,
        mase_period=24,
        mase_scale=_mase_scale(reference),
        mase_effective_period=24,
        mase_scale_source="seasonal_history",
        qualification_policy=frozen,
    )
    assert contract["qualification_policy_id"] == (
        "frozen-nonlinear_persistence"
    )
    assert contract["qualification_thresholds"] == thresholds[
        "nonlinear_persistence"
    ]
    assert contract["qualification_threshold_source"] == (
        QUALIFICATION_THRESHOLD_SOURCE_POLICY
    )

    invalid = dict(frozen)
    invalid["threshold_source_policy"] = "evaluation_origins"
    with pytest.raises(ValueError, match="independent"):
        fit_real_path_dynamic_contract(
            history,
            capability_id="nonlinear_persistence",
            carrier_period=24.0,
            secondary_periods=(),
            reference_history=reference,
            mase_period=24,
            mase_scale=_mase_scale(reference),
            mase_effective_period=24,
            mase_scale_source="seasonal_history",
            qualification_policy=invalid,
        )


def test_dynamic_thresholds_freeze_on_disjoint_reference_bank() -> None:
    candidates = _backgrounds(
        [_intermittent_path(seed) for seed in range(1, 9)]
    )
    evaluation, reference, split_audit = split_real_anchored_background_banks(
        candidates,
        maximum_evaluation_backgrounds=4,
        maximum_reference_backgrounds=4,
        source_window_length=protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
    )
    reference_rows, _availability = fit_background_capability_contracts(
        reference,
        capability_ids=("predictable_intermittency",),
    )
    frozen = freeze_real_anchored_qualification_policy(
        reference_rows,
        reference_background_ids=[
            str(background["background_id"]) for background in reference
        ],
        bank_split_audit=split_audit,
    )
    evaluation_rows, _availability = fit_background_capability_contracts(
        evaluation,
        capability_ids=("predictable_intermittency",),
        qualification_policy=frozen,
    )
    validate_evaluation_qualification_policy(evaluation_rows, frozen)
    frozen_cell = frozen["capabilities"]["predictable_intermittency"]
    assert all(
        row["qualification_policy_id"]
        == frozen_cell["qualification_policy_id"]
        and row["qualification_thresholds"]
        == frozen_cell["qualification_thresholds"]
        for row in evaluation_rows
    )


@pytest.mark.parametrize(
    ("capability_id", "paths"),
    [
        (
            "predictable_intermittency",
            [_intermittent_path(seed) for seed in (1, 2, 3, 4)],
        ),
        (
            "nonlinear_persistence",
            [_nonlinear_path(seed) for seed in (8, 9, 10, 12)],
        ),
    ],
)
def test_dynamic_capabilities_integrate_with_real_anchored_generation_and_validation(
    capability_id: str,
    paths: list[np.ndarray],
) -> None:
    private_backgrounds = _backgrounds(paths)
    contracts, availability = fit_background_capability_contracts(
        private_backgrounds,
        capability_ids=(capability_id,),
    )

    assert all(row["available"] is True for row in contracts)
    assert availability["cells"][0]["status"] == "available"
    assert availability["cells"][0]["eligible_background_count"] == 4
    public_backgrounds = [
        public_background(background) for background in private_backgrounds
    ]
    samples = list(
        iter_real_anchored_samples(
            public_backgrounds,
            contracts,
            capability_ids=(capability_id,),
            seed_indexes=range(4),
        )
    )
    assert len(samples) == 4 * 5 * 2
    validation = real_anchored_counterfactual_checks(
        samples,
        expected_row_count=len(samples),
    )
    assert validation["accepted"] is True

    if capability_id == "nonlinear_persistence":
        replay = list(
            iter_nonlinear_replay_sensitivity_samples(
                public_backgrounds,
                contracts,
                seed_indexes=range(4),
            )
        )
        assert len(replay) == len(samples)
        assert all(
            row["evaluation_table"]
            == "real_anchored_nonlinear_replay_sensitivity"
            and row["excluded_from_primary_score"] is True
            and row["generation_metadata"]["future_innovation_policy"]
            == NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
            for row in replay
        )
