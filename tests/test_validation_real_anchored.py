from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from cafe import protocol
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_ALPHAS,
    REAL_ANCHORED_MASTER_SCHEMA,
    array_sha256,
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
    protocol.write_json(
        manifest_path,
        {
            "schema_version": "cafe.generation_manifest.v1",
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
