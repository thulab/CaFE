from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from cafe import protocol
from cafe.data.real import RealDatasetBundle, RealSeriesRecord
from cafe.features.primitives import gift_eval_short_term_test_holdout_steps
from cafe.generation.real_counterfactuals import (
    build_availability,
    default_four_capability_qualification_policy,
    fit_background_capability_contracts,
    iter_real_anchored_samples,
    public_background,
    real_anchored_assignments,
    reconstruct_source_baseline,
    resolve_history_periods,
    resolve_modulation_period,
    resolve_regime_joinpoint,
    validate_availability_contract,
    validate_contract_integrity,
)
from cafe.generation.real_anchored_policy import (
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
    TIME_VARYING_SEASONALITY_BASIS_POLICY,
)


def _hourly_path(length: int, *, phase: float = 0.0) -> np.ndarray:
    time = np.arange(length, dtype=float)
    local = np.maximum(
        (time - (protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH - 96))
        / 96.0,
        0.0,
    )
    return (
        10.0
        + 0.18 * local
        + 0.95 * local**2
        + 2.0 * np.sin(2.0 * np.pi * time / 24.0 + phase)
        + 1.2 * np.sin(2.0 * np.pi * time / 168.0 - 0.3 + phase)
        + 0.08 * np.cos(2.0 * np.pi * time / 61.0 + phase)
    )


def _modulated_regime_path(
    length: int,
    *,
    modulation_phase: float = 0.0,
    future_offset: float = 0.0,
) -> np.ndarray:
    time = np.arange(length, dtype=float)
    envelope = 2.0 + 0.3 * np.sin(
        2.0 * np.pi * time / 168.0 + modulation_phase
    )
    values = (
        10.0
        + envelope * np.sin(2.0 * np.pi * time / 24.0 + 0.2)
        + 0.2 * np.sin(2.0 * np.pi * time / 84.0 - 0.1)
        + 4.0 * (time >= 432.0)
        + 0.02 * np.cos(time / 5.0)
    )
    values[protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH :] += (
        future_offset
    )
    return values


def _strong_modulated_path(length: int) -> np.ndarray:
    time = np.arange(length, dtype=float)
    envelope = 2.0 + 1.4 * np.sin(2.0 * np.pi * time / 168.0 + 0.2)
    return (
        10.0
        + envelope * np.sin(2.0 * np.pi * time / 24.0 + 0.3)
        + 0.8 * np.sin(2.0 * np.pi * time / 84.0 - 0.4)
        + 0.02 * np.cos(time / 7.0)
    )


def _bundle(
    paths: list[np.ndarray],
    *,
    adapter_id: str,
) -> RealDatasetBundle:
    return RealDatasetBundle(
        frequency="h",
        records=tuple(
            RealSeriesRecord(item_id=f"item_{index}", values=values)
            for index, values in enumerate(paths)
        ),
        asset_files=(),
        adapter_id=adapter_id,
        metadata={"fixture": True},
    )


def _spec(
    dataset_id: str = "fixture_hourly",
    *,
    adapter_id: str = "fixture",
) -> protocol.DatasetSpec:
    return protocol.DatasetSpec(
        dataset_id=dataset_id,
        logical_name="Anchored fixture",
        config_id="fixture/H",
        asset_name="fixture/H",
        domain="Test",
        real_data_adapter=adapter_id,
    )


def _four_background_bank() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    paths = [
        _hourly_path(protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH, phase=0.17 * index)
        for index in range(4)
    ]
    backgrounds, _metadata = protocol.build_real_anchored_backgrounds(
        _spec(),
        source_root=Path("/unused"),
        maximum_backgrounds=4,
        sample_seed=1729,
        real_bundle=_bundle(paths, adapter_id="fixture"),
    )
    contract_rows, availability = fit_background_capability_contracts(
        backgrounds,
        capability_ids=("multi_seasonal", "trend"),
    )
    return (
        [public_background(background) for background in backgrounds],
        contract_rows,
        availability,
    )


def _four_extended_background_bank() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    paths = [
        _modulated_regime_path(
            protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
            modulation_phase=0.1 * index,
        )
        for index in range(4)
    ]
    backgrounds, _metadata = protocol.build_real_anchored_backgrounds(
        _spec("fixture_extended_hourly"),
        source_root=Path("/unused"),
        maximum_backgrounds=4,
        sample_seed=2718,
        real_bundle=_bundle(paths, adapter_id="fixture"),
    )
    contract_rows, availability = fit_background_capability_contracts(
        backgrounds,
        capability_ids=(
            "time_varying_seasonality",
            "regime_switching",
        ),
    )
    return (
        [public_background(background) for background in backgrounds],
        contract_rows,
        availability,
    )


@pytest.mark.parametrize(
    ("dataset_id", "expected_policy"),
    (
        (
            "gift_fixture_hourly",
            "gift_eval_short_term_official_test_tail_excluded",
        ),
        ("gift_m4_hourly", "m4_official_single_h48_test_tail_excluded"),
    ),
)
def test_real_backgrounds_exclude_official_gift_tail_before_sampling(
    dataset_id: str,
    expected_policy: str,
) -> None:
    total_length = 4_000
    clean_values = _hourly_path(total_length)
    if dataset_id == "gift_m4_hourly":
        expected_holdout = protocol.HORIZON
    else:
        expected_holdout = gift_eval_short_term_test_holdout_steps(
            "h",
            [("item_0", clean_values)],
        )
    values = clean_values.copy()
    values[-expected_holdout:] = 1e9 + np.arange(expected_holdout)
    backgrounds, metadata = protocol.build_real_anchored_backgrounds(
        _spec(dataset_id, adapter_id="gift_arrow"),
        source_root=Path("/unused"),
        maximum_backgrounds=4,
        sample_seed=20260812,
        real_bundle=_bundle([values], adapter_id="gift_arrow"),
    )

    assert len(backgrounds) == 4
    assert metadata["official_holdout"] == {
        "policy": expected_policy,
        "excluded_tail_steps": expected_holdout,
    }
    training_end = total_length - expected_holdout
    for background in backgrounds:
        start = int(background["decomposition_start"])
        assert start + protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH <= training_end

        raw_fit = values[
            start : start + protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        ]
        raw_visible = raw_fit[-protocol.REAL_ANCHORED_CONTEXT_LENGTH :]
        location = float(np.mean(raw_visible))
        scale = float(np.std(raw_visible))
        raw_future = values[
            start + protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH :
            start + protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
        ]
        expected_fit = (raw_fit - location) / scale
        expected_target = np.concatenate(
            ((raw_visible - location) / scale, (raw_future - location) / scale)
        )

        np.testing.assert_allclose(
            background["_decomposition_history"],
            expected_fit,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(background["target"])[:, 0],
            expected_target,
            rtol=0.0,
            atol=1e-12,
        )
        assert float(background["standardization"]["location"]) == pytest.approx(
            location
        )
        assert float(background["standardization"]["scale"]) == pytest.approx(
            scale
        )


def test_l504_contract_is_history_only_and_normalization_uses_trailing_l336(
) -> None:
    length = protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
    first = _hourly_path(length)
    first[:168] += 500.0
    changed_future = first.copy()
    changed_future[-protocol.HORIZON :] += 1e6

    first_backgrounds, _ = protocol.build_real_anchored_backgrounds(
        _spec(),
        source_root=Path("/unused"),
        maximum_backgrounds=1,
        real_bundle=_bundle([first], adapter_id="fixture"),
    )
    changed_backgrounds, _ = protocol.build_real_anchored_backgrounds(
        _spec(),
        source_root=Path("/unused"),
        maximum_backgrounds=1,
        real_bundle=_bundle([changed_future], adapter_id="fixture"),
    )
    first_background = first_backgrounds[0]
    changed_background = changed_backgrounds[0]
    raw_fit = first[: protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH]
    raw_visible = raw_fit[-protocol.REAL_ANCHORED_CONTEXT_LENGTH :]

    assert first_background["standardization"]["scope"] == (
        "shared_unmodified_real_l336_history"
    )
    assert first_background["standardization"]["location"] == pytest.approx(
        np.mean(raw_visible)
    )
    assert first_background["standardization"]["location"] != pytest.approx(
        np.mean(raw_fit)
    )
    visible_standardized = np.asarray(first_background["target"])[:336, 0]
    assert np.mean(visible_standardized) == pytest.approx(0.0, abs=1e-12)
    assert np.std(visible_standardized) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(
        first_background["_decomposition_history"],
        changed_background["_decomposition_history"],
    )
    assert first_background["decomposition_history_sha256"] == (
        changed_background["decomposition_history_sha256"]
    )

    first_rows, _ = fit_background_capability_contracts(
        first_backgrounds,
        capability_ids=("multi_seasonal", "trend"),
    )
    changed_rows, _ = fit_background_capability_contracts(
        changed_backgrounds,
        capability_ids=("multi_seasonal", "trend"),
    )
    assert first_rows == changed_rows


def test_period_resolution_keeps_declared_carrier_and_finds_weekly_component(
) -> None:
    history = _hourly_path(
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
    )

    resolved = resolve_history_periods(
        history,
        declared_carrier_period=24.0,
    )

    assert resolved["history_only"] is True
    assert resolved["history_length"] == 504
    assert resolved["carrier_period"] == 24.0
    assert resolved["carrier_source"] == (
        "visible_calibration_feature_period"
    )
    assert resolved["carrier_visibility_passed"] is True
    assert resolved["carrier_rms_ratio"] >= resolved[
        "minimum_carrier_rms_ratio"
    ]
    assert resolved["secondary_periods"] == [168.0]
    assert resolved["secondary_peaks"][0]["frequency_bin"] == 3.0
    assert resolved["secondary_peaks"][0]["power_share"] > 0.01


def test_period_resolution_rejects_invisible_declared_carrier() -> None:
    time = np.arange(
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH,
        dtype=float,
    )
    history = 2.0 * np.sin(2.0 * np.pi * time / 24.0 + 0.3)

    resolved = resolve_history_periods(
        history,
        declared_carrier_period=48.0,
    )

    assert resolved["carrier_source"] == "history_spectral_peak_fallback"
    assert resolved["carrier_period"] == pytest.approx(24.0)
    assert resolved["declared_carrier_visibility"][
        "carrier_rms_ratio"
    ] < resolved["minimum_carrier_rms_ratio"]
    with pytest.raises(ValueError, match="visibility gates"):
        resolve_history_periods(
            np.ones_like(history),
            declared_carrier_period=24.0,
        )


def test_modulation_and_regime_resolvers_are_history_only_and_recover_signal(
) -> None:
    baseline = _modulated_regime_path(
        protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
    )
    changed_future = _modulated_regime_path(
        protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
        future_offset=1e6,
    )
    history = baseline[: protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH]
    changed_history = changed_future[
        : protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
    ]
    periods = resolve_history_periods(
        history,
        declared_carrier_period=24.0,
    )
    changed_periods = resolve_history_periods(
        changed_history,
        declared_carrier_period=24.0,
    )
    modulation = resolve_modulation_period(
        history,
        carrier_period=float(periods["carrier_period"]),
    )
    changed_modulation = resolve_modulation_period(
        changed_history,
        carrier_period=float(changed_periods["carrier_period"]),
    )
    regime = resolve_regime_joinpoint(
        history,
        carrier_period=float(periods["carrier_period"]),
        secondary_periods=periods["secondary_periods"],
    )
    changed_regime = resolve_regime_joinpoint(
        changed_history,
        carrier_period=float(changed_periods["carrier_period"]),
        secondary_periods=changed_periods["secondary_periods"],
    )

    np.testing.assert_array_equal(history, changed_history)
    assert periods == changed_periods
    assert modulation == changed_modulation
    assert regime == changed_regime
    assert modulation["history_only"] is True
    assert modulation["available"] is True
    assert float(modulation["modulation_period"]) == pytest.approx(
        168.0,
        abs=1.0,
    )
    assert modulation["envelope_frequency_bin"] == 3
    assert regime["history_only"] is True
    assert regime["available"] is True
    assert int(regime["regime_join_index"]) == pytest.approx(432, abs=1)
    assert int(regime["regime_join_index"]) in range(360, 481)
    assert regime["step_over_ramp_sse_advantage"] >= regime[
        "minimum_step_over_ramp_advantage"
    ]
    assert regime["join_stability_width"] <= regime[
        "maximum_join_stability_width"
    ]


def test_regime_resolver_rejects_a_smooth_ramp() -> None:
    length = protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
    time = np.arange(length, dtype=float)
    ramp = 4.0 * np.clip((time - 408.0) / 48.0, 0.0, 1.0)
    history = (
        10.0
        + 2.0 * np.sin(2.0 * np.pi * time / 24.0 + 0.2)
        + ramp
    )

    resolved = resolve_regime_joinpoint(
        history,
        carrier_period=24.0,
        secondary_periods=(),
    )

    assert resolved["available"] is False
    assert resolved["unavailable_reason"] in {
        "continuous_ramp_preferred_over_level_step",
        "regime_joinpoint_not_locally_stable",
    }


def test_shared_ownership_excludes_am_sidebands_from_secondary_periods() -> None:
    paths = [
        _strong_modulated_path(protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH)
    ]
    backgrounds, _ = protocol.build_real_anchored_backgrounds(
        _spec("fixture_strong_am_hourly"),
        source_root=Path("/unused"),
        maximum_backgrounds=1,
        sample_seed=811,
        real_bundle=_bundle(paths, adapter_id="fixture"),
    )

    rows, _ = fit_background_capability_contracts(
        backgrounds,
        capability_ids=("multi_seasonal", "time_varying_seasonality"),
    )

    assert all(row["available"] is True for row in rows)
    for row in rows:
        ownership = row["component_ownership"]
        assert ownership["am_sideband_owned_peak_count"] >= 1
        assert ownership["modulation_basis"] == (
            TIME_VARYING_SEASONALITY_BASIS_POLICY
        )
        secondary = row["period_resolution"]["secondary_periods"]
        assert any(float(period) == pytest.approx(84.0) for period in secondary)
        assert not any(
            float(period) == pytest.approx(21.0, abs=1.0)
            or float(period) == pytest.approx(28.0, abs=1.0)
            for period in secondary
        )


def test_frozen_qualification_policy_is_reused_by_available_and_unavailable_rows(
) -> None:
    paths = [_hourly_path(protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH)]
    backgrounds, _ = protocol.build_real_anchored_backgrounds(
        _spec("fixture_policy_hourly"),
        source_root=Path("/unused"),
        maximum_backgrounds=1,
        sample_seed=812,
        real_bundle=_bundle(paths, adapter_id="fixture"),
    )
    defaults = default_four_capability_qualification_policy()
    default_thresholds = defaults["qualification_thresholds"]
    frozen_policy: dict[str, object] = {
        "schema_version": REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
        "threshold_source_policy": QUALIFICATION_THRESHOLD_SOURCE_POLICY,
        "capabilities": {
            capability_id: {
                "qualification_policy_id": (
                    f"fixture.{capability_id}.reference.v1"
                ),
                "qualification_thresholds": dict(
                    default_thresholds[capability_id]
                ),
            }
            for capability_id in ("trend", "regime_switching")
        },
    }
    frozen_policy["qualification_policy_sha256"] = protocol.json_sha256(
        frozen_policy
    )

    rows, _ = fit_background_capability_contracts(
        backgrounds,
        capability_ids=("trend", "regime_switching"),
        qualification_policy=frozen_policy,
    )

    assert {bool(row["available"]) for row in rows} == {False, True}
    for row in rows:
        capability_id = str(row["capability_id"])
        expected = frozen_policy["capabilities"][capability_id]
        assert row["qualification_policy_id"] == expected[
            "qualification_policy_id"
        ]
        assert row["qualification_thresholds"] == expected[
            "qualification_thresholds"
        ]


def test_public_background_reconstructs_full_source_and_contracts_are_available(
) -> None:
    private_paths = [
        _hourly_path(protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH, phase=0.17 * index)
        for index in range(4)
    ]
    private_backgrounds, _ = protocol.build_real_anchored_backgrounds(
        _spec(),
        source_root=Path("/unused"),
        maximum_backgrounds=4,
        sample_seed=1729,
        real_bundle=_bundle(private_paths, adapter_id="fixture"),
    )
    contract_rows, availability = fit_background_capability_contracts(
        private_backgrounds,
        capability_ids=("multi_seasonal", "trend"),
    )

    assert len(private_backgrounds) == 4
    assert {cell["status"] for cell in availability["cells"]} == {"available"}
    assert all(row["available"] is True for row in contract_rows)
    for row in contract_rows:
        validate_contract_integrity(row)
        assert row["qualification_policy_id"]
        assert row["qualification_threshold_source"] == (
            QUALIFICATION_THRESHOLD_SOURCE_POLICY
        )
        assert row["qualification_thresholds"][
            "visible_context_length"
        ] == protocol.FIXED_CONTEXT_LENGTH
        assert row["controlled_component_visible_context_length"] == (
            protocol.FIXED_CONTEXT_LENGTH
        )
    for background_id in {
        str(row["background_id"]) for row in contract_rows
    }:
        selected = [
            row
            for row in contract_rows
            if str(row["background_id"]) == background_id
        ]
        assert len(
            {
                row["contract"]["decomposition_contract"][
                    "contract_sha256"
                ]
                for row in selected
            }
        ) == 1
        assert len(
            {
                row["contract"]["decomposition_contract"][
                    "spectral_component_ownership"
                ]
                for row in selected
            }
        ) == 1

    for private in private_backgrounds:
        public = public_background(private)
        assert "_decomposition_history" not in public
        assert len(public["decomposition_prefix"]) == 168
        reconstructed = reconstruct_source_baseline(public)
        expected = np.concatenate(
            (
                np.asarray(private["_decomposition_history"]),
                np.asarray(private["target"], dtype=float)[-protocol.HORIZON :, 0],
            )
        )
        np.testing.assert_array_equal(reconstructed, expected)
        np.testing.assert_array_equal(
            reconstructed[168:],
            np.asarray(public["target"], dtype=float)[:, 0],
        )


def test_real_anchored_generation_is_deterministic_exactly_paired_and_monotone(
) -> None:
    backgrounds, contract_rows, availability = _four_background_bank()
    assert {cell["status"] for cell in availability["cells"]} == {"available"}
    for cell in availability["cells"]:
        rms_gate = cell["controlled_component_rms_gate"]
        assert rms_gate["evaluated_background_count"] == len(backgrounds)
        assert rms_gate["future_horizon"] == protocol.HORIZON
        assert rms_gate["history_minimum_rms_ratios"] == [0.01]
        assert rms_gate["visible_history_minimum_rms_ratios"] == [0.01]
        assert rms_gate["visible_context_lengths"] == [
            protocol.FIXED_CONTEXT_LENGTH
        ]
        assert rms_gate["future_minimum_rms_ratios"] == [0.01]
        assert rms_gate["history_rms_range"] is not None
        assert rms_gate["future_rms_range"] is not None
        assert rms_gate["visible_history_rms_range"] is not None
        assert rms_gate["future_threshold_range"] is not None
    alphas = (1.2, 1.6, 2.0)
    arguments = {
        "capability_ids": ("multi_seasonal", "trend"),
        "seed_indexes": (0, 1),
        "alphas": alphas,
    }

    first = list(iter_real_anchored_samples(backgrounds, contract_rows, **arguments))
    second = list(iter_real_anchored_samples(backgrounds, contract_rows, **arguments))

    assert first == second
    assert len(first) == 2 * 2 * len(alphas) * 2
    by_background = {
        str(background["background_id"]): background for background in backgrounds
    }
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in first:
        by_pair[str(sample["counterfactual_pair_id"])].append(sample)
    assert all(len(pair) == 2 for pair in by_pair.values())

    treatment_deltas: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = (
        defaultdict(list)
    )
    for pair in by_pair.values():
        baseline = next(row for row in pair if row["counterfactual_member"] == 0)
        treatment = next(row for row in pair if row["counterfactual_member"] == 1)
        background = by_background[str(baseline["background_id"])]
        expected_baseline = np.asarray(background["target"], dtype=float)
        baseline_target = np.asarray(baseline["target"], dtype=float)
        treatment_target = np.asarray(treatment["target"], dtype=float)

        np.testing.assert_array_equal(baseline_target, expected_baseline)
        assert baseline["dose_value"] == 1.0
        assert baseline["generation_metadata"]["intervention_rms"] == 0.0
        assert baseline["mase_scale"] == treatment["mase_scale"]
        assert baseline["mase_scale"] == background["mase_scale"]
        assert baseline["mase_scale_by_target"] == treatment["mase_scale_by_target"]
        assert baseline["shared_standardization"] == treatment["shared_standardization"]
        assert baseline["baseline_history_sha256"] == (
            treatment["baseline_history_sha256"]
        )
        assert baseline["baseline_future_sha256"] == (
            treatment["baseline_future_sha256"]
        )
        assert treatment["anti_copy_gate"] == {
            "status": "not_applicable",
            "reason_code": "intentional_real_anchor_counterfactual",
        }
        delta = treatment_target - baseline_target
        assert np.linalg.norm(delta) > 0.0
        treatment_deltas[
            (str(treatment["capability_id"]), int(treatment["seed_index"]))
        ].append((float(treatment["dose_value"]), delta))

    assert {capability for capability, _seed in treatment_deltas} == {
        "multi_seasonal",
        "trend",
    }
    for rows in treatment_deltas.values():
        rows.sort(key=lambda row: row[0])
        norms = [float(np.linalg.norm(delta)) for _alpha, delta in rows]
        assert all(left < right for left, right in zip(norms, norms[1:]))
        reference_alpha, reference_delta = rows[0]
        reference_unit_delta = reference_delta / (reference_alpha - 1.0)
        for alpha, delta in rows[1:]:
            np.testing.assert_allclose(
                delta / (alpha - 1.0),
                reference_unit_delta,
                rtol=0.0,
                atol=2e-14,
            )


def test_real_anchored_backgrounds_are_never_recycled_as_new_seeds() -> None:
    backgrounds, contract_rows, _availability = _four_background_bank()
    capability_ids = ("trend",)
    all_assignments = real_anchored_assignments(
        contract_rows,
        capability_ids=capability_ids,
        seed_indexes=range(10),
    )["trend"]
    first_shard = real_anchored_assignments(
        contract_rows,
        capability_ids=capability_ids,
        seed_indexes=range(2),
    )["trend"]
    second_shard = real_anchored_assignments(
        contract_rows,
        capability_ids=capability_ids,
        seed_indexes=range(2, 10),
    )["trend"]

    assert [seed for seed, _row in all_assignments] == [0, 1, 2, 3]
    assert [
        (seed, str(row["background_id"]))
        for seed, row in first_shard + second_shard
    ] == [
        (seed, str(row["background_id"]))
        for seed, row in all_assignments
    ]
    assert len(
        {str(row["background_id"]) for _seed, row in all_assignments}
    ) == len(all_assignments)

    samples = list(
        iter_real_anchored_samples(
            backgrounds,
            contract_rows,
            capability_ids=capability_ids,
            seed_indexes=range(10),
            alphas=(2.0,),
        )
    )
    assert len(samples) == len(backgrounds) * 2
    treatment_backgrounds = {
        str(row["background_id"])
        for row in samples
        if row["counterfactual_member"] == 1
    }
    assert len(treatment_backgrounds) == len(backgrounds)


def test_extended_real_anchored_pairs_are_exact_shared_and_alpha_monotone(
) -> None:
    backgrounds, contract_rows, availability = _four_extended_background_bank()
    expected_capabilities = {
        "time_varying_seasonality",
        "regime_switching",
    }
    assert {
        str(cell["capability_id"])
        for cell in availability["cells"]
        if cell["status"] == "available"
    } == expected_capabilities
    assert len(contract_rows) == 8
    assert all(row["available"] is True for row in contract_rows)
    for row in contract_rows:
        assert row["modulation_resolution"]["history_only"] is True
        assert row["modulation_resolution"]["available"] is True
        assert float(
            row["modulation_resolution"]["modulation_period"]
        ) == pytest.approx(168.0, abs=1.0)
        assert row["regime_resolution"]["history_only"] is True
        assert row["regime_resolution"]["available"] is True
        assert int(row["regime_resolution"]["regime_join_index"]) == pytest.approx(
            432,
            abs=1,
        )
        validate_contract_integrity(row)
    for background_id in {
        str(row["background_id"]) for row in contract_rows
    }:
        hashes = {
            row["contract"]["decomposition_contract"]["contract_sha256"]
            for row in contract_rows
            if str(row["background_id"]) == background_id
        }
        assert len(hashes) == 1
        assert all(
            row["component_ownership"]["policy"]
            == "shared_background_joint_design_v1"
            for row in contract_rows
            if str(row["background_id"]) == background_id
        )

    alphas = (1.2, 1.6, 2.0)
    arguments = {
        "capability_ids": tuple(sorted(expected_capabilities)),
        "seed_indexes": (0, 1),
        "alphas": alphas,
    }
    first = list(
        iter_real_anchored_samples(backgrounds, contract_rows, **arguments)
    )
    second = list(
        iter_real_anchored_samples(backgrounds, contract_rows, **arguments)
    )
    assert first == second
    assert len(first) == len(expected_capabilities) * 2 * len(alphas) * 2

    by_background = {
        str(background["background_id"]): background
        for background in backgrounds
    }
    by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in first:
        by_pair[str(sample["counterfactual_pair_id"])].append(sample)
    dose_deltas: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = (
        defaultdict(list)
    )
    for pair in by_pair.values():
        assert len(pair) == 2
        baseline = next(row for row in pair if row["counterfactual_member"] == 0)
        treatment = next(row for row in pair if row["counterfactual_member"] == 1)
        expected_baseline = np.asarray(
            by_background[str(baseline["background_id"])]["target"],
            dtype=float,
        )
        baseline_target = np.asarray(baseline["target"], dtype=float)
        treatment_target = np.asarray(treatment["target"], dtype=float)
        np.testing.assert_array_equal(baseline_target, expected_baseline)
        assert baseline["generation_metadata"]["intervention_rms"] == 0.0
        assert baseline["mase_scale"] == treatment["mase_scale"]
        assert baseline["mase_scale_by_target"] == treatment["mase_scale_by_target"]
        assert baseline["shared_standardization"] == treatment[
            "shared_standardization"
        ]
        assert baseline["baseline_history_sha256"] == treatment[
            "baseline_history_sha256"
        ]
        assert baseline["baseline_future_sha256"] == treatment[
            "baseline_future_sha256"
        ]
        metadata = treatment["generation_metadata"]
        if treatment["capability_id"] == "time_varying_seasonality":
            assert metadata["controlled_component"] == (
                "carrier_phase_locked_symmetric_amplitude_modulation"
            )
            assert metadata["modulation_basis"] == (
                TIME_VARYING_SEASONALITY_BASIS_POLICY
            )
            assert metadata["modulation_period"] == pytest.approx(168.0, abs=1.0)
            assert metadata["amplitude_modulation_fixed"] is False
        else:
            assert treatment["capability_id"] == "regime_switching"
            assert metadata["controlled_component"] == (
                "history_joinpoint_level_shift"
            )
            assert metadata["regime_join_index"] == pytest.approx(432, abs=1)
            assert metadata["regime_level_shift_fixed"] is False
        delta = treatment_target - baseline_target
        assert np.linalg.norm(delta) > 0.0
        dose_deltas[
            (str(treatment["capability_id"]), int(treatment["seed_index"]))
        ].append((float(treatment["dose_value"]), delta))

    assert {capability for capability, _seed in dose_deltas} == expected_capabilities
    for rows in dose_deltas.values():
        rows.sort(key=lambda row: row[0])
        norms = [float(np.linalg.norm(delta)) for _alpha, delta in rows]
        assert all(left < right for left, right in zip(norms, norms[1:]))
        alpha, delta = rows[0]
        unit_delta = delta / (alpha - 1.0)
        for next_alpha, next_delta in rows[1:]:
            np.testing.assert_allclose(
                next_delta / (next_alpha - 1.0),
                unit_delta,
                rtol=0.0,
                atol=2e-14,
            )


def test_legacy_v1_availability_is_validated_without_redefining_it() -> None:
    _backgrounds, contract_rows, availability = _four_background_bank()
    legacy = build_availability(
        contract_rows,
        requested_capability_ids=("trend", "multi_seasonal"),
        minimum_eligible_backgrounds=4,
    )
    legacy["schema_version"] = "cafe.real_anchored_availability.v1"
    for cell in legacy["cells"]:
        gate = cell["controlled_component_rms_gate"]
        for field in (
            "visible_history_source",
            "visible_history_minimum_rms_ratios",
            "visible_context_lengths",
            "visible_history_rms_range",
            "visible_history_threshold_range",
        ):
            gate.pop(field, None)

    validate_availability_contract(legacy, contract_rows)
    tampered = dict(legacy)
    tampered["cells"] = [dict(cell) for cell in legacy["cells"]]
    tampered["cells"][0]["eligible_background_count"] += 1
    with pytest.raises(ValueError, match="disagree"):
        validate_availability_contract(tampered, contract_rows)
