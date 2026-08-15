from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from cafe import protocol
from cafe.analysis import runner
from cafe.generation.real_anchored_policy import (
    REAL_ANCHORED_CANONICAL_STRENGTH_GRID,
)


def anchored_sample(
    *,
    member: int,
    intensity: int = 4,
    mase_scale: float = 2.5,
    background_index: int = 0,
) -> dict:
    context = runner.FIXED_CONTEXT_LENGTH
    horizon = 4
    time = np.arange(context + horizon, dtype=float)
    baseline = np.sin(2.0 * np.pi * time / 12.0)
    treatment_delta = np.zeros_like(baseline)
    treatment_delta[context:] = np.asarray([1.0, 2.0, 3.0, 4.0])
    target = baseline + member * treatment_delta
    pair_id = f"real_pair_b{background_index}_i{intensity}__L{context}"
    master_pair_id = f"real_pair_b{background_index}_i{intensity}"
    return {
        "schema_version": "cafe.forecast_view.v1",
        "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "sample_id": (
            f"real_sample_b{background_index}_i{intensity}_m{member}"
            f"__L{context}"
        ),
        "master_sample_id": (
            f"real_sample_b{background_index}_i{intensity}_m{member}"
        ),
        "dataset_id": "gift_electricity_h",
        "config_id": f"real_background_{background_index}",
        "capability_id": "multi_seasonal",
        "generator_family_role": "real_anchored",
        "evaluation_table": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "intensity": intensity,
        "seed_index": 7 + background_index,
        "background_id": f"real_background_{background_index}",
        "dose_parameter": "canonical_strength_lambda",
        "dose_value": 0.0 if member == 0 else intensity / 5.0,
        "paired_treatment_strength": intensity / 5.0,
        "applied_alpha": 1.0 if member == 0 else 1.0 + 0.3 * intensity,
        "paired_treatment_applied_alpha": 1.0 + 0.3 * intensity,
        "dose_calibration_policy_sha256": "d" * 64,
        "context_length": context,
        "horizon": horizon,
        "target_dim": 1,
        "target": target[:, None].tolist(),
        "covariates": None,
        "season_length": 12,
        "mase_period": 12,
        "mase_scale": mase_scale,
        "counterfactual_pair_id": pair_id,
        "master_counterfactual_pair_id": master_pair_id,
        "counterfactual_member": member,
        "generation_metadata": {"periods": [12.0, 24.0]},
    }


def file_record(path: Path, *, row_count: int) -> dict:
    return {**protocol.file_record(path), "row_count": row_count}


def test_synthetic_covariate_effect_defaults_to_all_target_channels() -> None:
    sample = {
        "capability_id": "covariate_response",
        "generator_family_role": "primary",
        "target_dim": 1,
        "generation_metadata": {},
    }

    assert runner.effect_channels(sample) == [0]


def test_hierarchy_qualification_is_projected_but_never_ranked(
    tmp_path: Path,
) -> None:
    dataset_id = "gift_hierarchical_sales_t"
    calibration_dir = tmp_path / dataset_id / "01_calibration"
    qualification_path = (
        calibration_dir / "structural_hierarchy_qualification.jsonl"
    )
    row = {
        "dataset_id": dataset_id,
        "background_id": "hierarchy-background-0",
        "capability_id": "hierarchical_coherence",
        "unavailable_reason": (
            "qualification_only_generation_and_ranking_prohibited"
        ),
        "contract": {
            "qualification_only": True,
            "generation_eligible": False,
            "ranking_eligible": False,
            "contract_sha256": "b" * 64,
            "fit_diagnostics": {
                "qualification_passed": True,
                "contrast_one_step_holdout_r2": [0.4, 0.6],
                "zero_sum_component_max_abs": 1e-14,
                "raw_negativity_audit_by_alpha": {
                    "2.0": {
                        "negative_value_count_by_child": [2, 0],
                        "total_negative_value_count": 2,
                        "minimum_augmented_child_value": -0.5,
                    }
                },
            },
        },
    }
    unavailable = {
        "dataset_id": dataset_id,
        "background_id": "hierarchy-background-unavailable",
        "capability_id": "hierarchical_coherence",
        "available": False,
        "qualification_available": False,
        "generation_eligible": False,
        "ranking_eligible": False,
        "unavailable_reason": "hierarchy qualification requires a declared hierarchy",
        "contract": None,
    }
    protocol.write_jsonl(qualification_path, [unavailable, row])
    files = {
        "structural_hierarchy_qualification": file_record(
            qualification_path,
            row_count=2,
        )
    }
    bundle = {
        "dataset": {"dataset_id": dataset_id},
        "source": {"kind": "fixture"},
        "files": files,
        "generator_version": protocol.GENERATOR_VERSION,
    }
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": files,
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(calibration_dir / "calibration_bundle.json", bundle)

    summary = runner.hierarchy_qualification_summary(
        tmp_path,
        dataset_id=dataset_id,
    )

    assert summary["status"] == "qualified"
    assert summary["qualification_background_count"] == 1
    assert summary["passed_background_count"] == 1
    assert summary["unavailable_background_count"] == 1
    assert summary["reason_counts"][
        "hierarchy qualification requires a declared hierarchy"
    ] == 1
    assert summary["included_in_generation_or_ranking"] is False
    assert summary["rows"][0][
        "mean_contrast_one_step_holdout_r2"
    ] == pytest.approx(0.5)
    assert summary["raw_negativity_by_alpha"]["2.0"] == {
        "backgrounds_with_negative_values": 1,
        "total_negative_value_count": 2,
        "minimum_augmented_child_value": -0.5,
    }

    failed = dict(row)
    failed["contract"] = {
        **row["contract"],
        "fit_diagnostics": {
            **row["contract"]["fit_diagnostics"],
            "qualification_passed": False,
        },
    }
    protocol.write_jsonl(qualification_path, [failed])
    files["structural_hierarchy_qualification"] = file_record(
        qualification_path,
        row_count=1,
    )
    bundle["files"] = files
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": files,
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(calibration_dir / "calibration_bundle.json", bundle)
    failed_summary = runner.hierarchy_qualification_summary(
        tmp_path,
        dataset_id=dataset_id,
    )
    assert failed_summary["status"] == "qualification_failed"
    assert failed_summary["passed_background_count"] == 0


def test_optional_component_is_backward_compatible_when_absent(
    tmp_path: Path,
) -> None:
    assert (
        runner.validated_optional_real_anchored_task_path(
            tmp_path,
            {},
            {"task_components": {}},
        )
        is None
    )


def test_empty_legacy_optional_component_is_treated_as_absent(
    tmp_path: Path,
) -> None:
    inference_dir = tmp_path / "inference"
    generation_path = tmp_path / "empty_real_anchored_masters.jsonl"
    task_path = inference_dir / "empty_real_anchored_views.jsonl"
    protocol.write_jsonl(generation_path, ())
    protocol.write_jsonl(task_path, ())
    component = {
        **file_record(task_path, row_count=0),
        "generation_component": file_record(
            generation_path,
            row_count=0,
        ),
    }

    assert (
        runner.validated_optional_real_anchored_task_path(
            inference_dir,
            {},
            {
                "real_anchored_view_count": 0,
                "task_components": {
                    "real_anchored_counterfactuals": component,
                },
            },
        )
        is None
    )


def test_legacy_real_anchored_generation_cannot_enter_v4_ranking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest__seed_000000_000001.json"
    config = {
        "schema_version": "cafe.generation_config.v2",
        "real_anchored_counterfactual": {
            "qualification_policy_sha256": None,
        },
    }
    protocol.write_json(
        path,
        {
            "schema_version": "cafe.generation_manifest.v2",
            "config": config,
            "config_sha256": protocol.json_sha256(config),
        },
    )

    with pytest.raises(ValueError, match="legacy real-anchored generation"):
        runner.validated_current_real_anchored_generation_protocol(path)


def test_current_real_anchored_generation_binds_frozen_dose_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest__seed_000000_000001.json"
    alpha_grid = [1.2, 1.4, 1.6, 1.8, 2.0]
    config = {
        "schema_version": "cafe.generation_config.v5",
        "real_anchored_counterfactual": {
            "upstream_real_anchored_protocol": protocol.SCHEMA_VERSION,
            "qualification_policy_sha256": "a" * 64,
            "legacy_upstream_component_policy": None,
            "generated_capabilities": ["multi_seasonal"],
            "dose_parameter": "canonical_strength_lambda",
            "canonical_strength_grid": list(
                REAL_ANCHORED_CANONICAL_STRENGTH_GRID
            ),
            "applied_alpha_grid_by_capability": {
                "multi_seasonal": [],
            },
            "applied_alpha_scope": "contract_specific_history_only",
            "applied_alpha_range_by_capability": {
                "multi_seasonal": [
                    {"minimum": value, "maximum": value}
                    for value in alpha_grid
                ],
            },
            "dose_calibration_sha256_by_capability": {
                "multi_seasonal": "b" * 64,
            },
            "dose_policy_sha256": "c" * 64,
            "pairing": (
                "baseline_lambda0_alpha1_vs_treatment_"
                "contract_resolved_alpha"
            ),
            "paired_minimum_separation": (
                "mandatory_treatment_source_l168_distance_with_budget_v1"
            ),
            "anti_copy": (
                "not_applicable_intentional_real_anchor_counterfactual"
            ),
        },
    }
    protocol.write_json(
        path,
        {
            "schema_version": "cafe.generation_manifest.v5",
            "config": config,
            "config_sha256": protocol.json_sha256(config),
        },
    )

    dose_provenance = {
        key: config["real_anchored_counterfactual"].get(key)
        for key in (
            "dose_parameter",
            "canonical_strength_grid",
            "applied_alpha_grid_by_capability",
            "applied_alpha_scope",
            "applied_alpha_range_by_capability",
            "dose_calibration_sha256_by_capability",
            "dose_policy_sha256",
            "qualification_policy_sha256",
        )
    }
    task_manifest = {
        "generation_config_sha256": protocol.json_sha256(config),
        "real_anchored_dose_provenance": dose_provenance,
    }
    binding = runner.validated_current_real_anchored_generation_protocol(
        path,
        task_manifest=task_manifest,
    )

    assert binding["canonical_strength_grid"] == list(
        REAL_ANCHORED_CANONICAL_STRENGTH_GRID
    )
    assert binding["applied_alpha_grid_by_capability"] == {
        "multi_seasonal": []
    }
    assert binding["applied_alpha_range_by_capability"] == {
        "multi_seasonal": [
            {"minimum": value, "maximum": value} for value in alpha_grid
        ]
    }
    assert binding["dose_calibration_sha256_by_capability"] == {
        "multi_seasonal": "b" * 64
    }
    task_manifest["real_anchored_dose_provenance"] = {
        **dose_provenance,
        "dose_policy_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="dose provenance differs"):
        runner.validated_current_real_anchored_generation_protocol(
            path,
            task_manifest=task_manifest,
        )


def test_component_validation_binds_generation_and_shared_mase(
    tmp_path: Path,
) -> None:
    inference_dir = tmp_path / "03_inference" / "seed_000000_000001"
    generation_path = tmp_path / "real_anchored_masters.jsonl"
    protocol.write_jsonl(generation_path, [{"master_sample_id": "master"}])
    task_path = inference_dir / "real_anchored_forecast_views.jsonl"
    samples = [anchored_sample(member=1), anchored_sample(member=0)]
    protocol.write_jsonl(task_path, samples)
    component = {
        **file_record(task_path, row_count=2),
        "generation_component": file_record(
            generation_path,
            row_count=1,
        ),
    }
    manifest = {
        "real_anchored_view_count": 2,
        "task_components": {
            "real_anchored_counterfactuals": component,
        },
    }

    resolved = runner.validated_optional_real_anchored_task_path(
        inference_dir,
        {},
        manifest,
    )

    assert resolved == task_path


def test_component_validation_rejects_background_recycled_as_new_seed(
    tmp_path: Path,
) -> None:
    inference_dir = tmp_path / "inference"
    generation_path = tmp_path / "real_anchored_masters.jsonl"
    protocol.write_jsonl(generation_path, [{"master_sample_id": "master"}])
    samples = [anchored_sample(member=member) for member in (0, 1)]
    recycled = []
    for sample in samples:
        row = dict(sample)
        row["sample_id"] = str(row["sample_id"]).replace("real_sample", "recycled")
        row["counterfactual_pair_id"] = str(
            row["counterfactual_pair_id"]
        ).replace("real_pair", "recycled_pair")
        row["seed_index"] = 8
        recycled.append(row)
    task_path = inference_dir / "real_anchored_forecast_views.jsonl"
    protocol.write_jsonl(task_path, [*samples, *recycled])
    manifest = {
        "real_anchored_view_count": 4,
        "task_components": {
            "real_anchored_counterfactuals": {
                **file_record(task_path, row_count=4),
                "generation_component": file_record(
                    generation_path,
                    row_count=1,
                ),
            },
        },
    }

    with pytest.raises(ValueError, match="recycled across seeds"):
        runner.validated_optional_real_anchored_task_path(
            inference_dir,
            {},
            manifest,
        )


def test_component_validation_separates_formal_and_d2_seed_assignments(
    tmp_path: Path,
) -> None:
    inference_dir = tmp_path / "inference"
    generation_path = tmp_path / "real_anchored_masters.jsonl"
    protocol.write_jsonl(generation_path, [{"master_sample_id": "master"}])
    formal = [anchored_sample(member=member) for member in (0, 1)]
    sensitivity = []
    for sample in formal:
        row = dict(sample)
        row["sample_id"] = f"{sample['sample_id']}__d2"
        row["master_sample_id"] = f"{sample['master_sample_id']}__d2"
        row["counterfactual_pair_id"] = (
            f"{sample['counterfactual_pair_id']}__d2"
        )
        row["master_counterfactual_pair_id"] = (
            f"{sample['master_counterfactual_pair_id']}__d2"
        )
        row["background_id"] = "d2_background"
        row["evaluation_table"] = (
            runner.REAL_ANCHORED_STRUCTURAL_SENSITIVITY_TABLE
        )
        row["generator_family_role"] = "real_anchored_structural"
        target = np.asarray(row["target"], dtype=float)
        row["target"] = np.repeat(target, 2, axis=1).tolist()
        row["target_dim"] = 2
        row["mase_scale_by_target"] = [2.5, 2.5]
        row["excluded_from_primary_score"] = True
        sensitivity.append(row)
    task_path = inference_dir / "real_anchored_forecast_views.jsonl"
    protocol.write_jsonl(task_path, [*formal, *sensitivity])
    manifest = {
        "real_anchored_view_count": 4,
        "task_components": {
            "real_anchored_counterfactuals": {
                **file_record(task_path, row_count=4),
                "generation_component": file_record(
                    generation_path,
                    row_count=1,
                ),
            },
        },
    }

    assert runner.validated_optional_real_anchored_task_path(
        inference_dir,
        {},
        manifest,
    ) == task_path


def test_effect_is_treatment_minus_baseline_even_when_rows_are_reversed(
    tmp_path: Path,
) -> None:
    treated = anchored_sample(member=1)
    baseline = anchored_sample(member=0)
    task_path = tmp_path / "tasks.jsonl"
    protocol.write_jsonl(task_path, [treated, baseline])
    predictions = []
    for sample in (treated, baseline):
        context = int(sample["context_length"])
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "forecast": sample["target"][context:],
            }
        )
    prediction_path = tmp_path / "predictions.jsonl"
    protocol.write_jsonl(prediction_path, predictions)

    metric_rows, effects, missing = runner.analyze_one_model(
        task_path,
        model_id="test-model",
        prediction_path=prediction_path,
    )

    assert missing == 0
    assert len(metric_rows) == 2
    assert len(effects) == 1
    assert effects[0]["benchmark_track"] == (
        runner.REAL_ANCHORED_BENCHMARK_TRACK
    )
    assert effects[0]["effect_orientation"] == (
        "treatment_member_1_minus_baseline_member_0"
    )
    assert effects[0]["counterfactual_effect_nrmse"] == pytest.approx(0.0)
    assert effects[0]["shared_baseline_mase_scale"] == pytest.approx(2.5)
    assert effects[0]["effect_mae_shared_baseline_mase"] == pytest.approx(0.0)
    assert effects[0]["dose_parameter"] == "canonical_strength_lambda"
    assert effects[0]["paired_treatment_strength"] == pytest.approx(0.8)
    assert effects[0]["applied_alpha"] == pytest.approx(2.2)
    assert effects[0]["dose_calibration_policy_sha256"] == "d" * 64
    assert {
        row["dose_calibration_policy_sha256"] for row in metric_rows
    } == {"d" * 64}


def test_effect_rejects_pair_specific_mase_normalization() -> None:
    baseline = anchored_sample(member=0, mase_scale=2.5)
    treated = anchored_sample(member=1, mase_scale=3.0)
    first_forecast = np.asarray(baseline["target"], dtype=float)[-4:]
    second_forecast = np.asarray(treated["target"], dtype=float)[-4:]

    with pytest.raises(ValueError, match="share the baseline MASE"):
        runner.effect_row(
            baseline,
            first_forecast,
            treated,
            second_forecast,
            model_id="test-model",
        )


def test_covariate_effect_scores_only_eligible_target_channels() -> None:
    context = 3
    baseline_target = np.zeros((context + 2, 2), dtype=float)
    treatment_target = baseline_target.copy()
    treatment_target[context:, 0] = 1.0
    treatment_target[context:, 1] = 100.0
    common = {
        "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "dataset_id": "gift_fixture",
        "capability_id": "covariate_response",
        "generator_family_role": "real_anchored_structural",
        "evaluation_table": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "intensity": 4,
        "seed_index": 0,
        "context_length": context,
        "target_dim": 2,
        "master_counterfactual_pair_id": "covariate-pair",
        "background_id": "background-0",
        "dose_value": 1.0,
        "mase_scale": 1.0,
        "generation_metadata": {"eligible_target_indices": [0]},
    }
    baseline = {
        **common,
        "counterfactual_member": 0,
        "target": baseline_target.tolist(),
    }
    treatment = {
        **common,
        "counterfactual_member": 1,
        "dose_value": 1.8,
        "target": treatment_target.tolist(),
    }
    baseline_forecast = np.zeros((2, 2), dtype=float)
    treatment_forecast = np.zeros((2, 2), dtype=float)
    treatment_forecast[:, 0] = 1.0

    effect = runner.effect_row(
        baseline,
        baseline_forecast,
        treatment,
        treatment_forecast,
        model_id="test-model",
    )

    assert effect["counterfactual_effect_nrmse"] == pytest.approx(0.0)
    assert effect["truth_effect_rms"] == pytest.approx(1.0)


def test_real_anchored_scores_use_maximum_available_dose_and_write_separately(
    tmp_path: Path,
) -> None:
    metrics = []
    effects = []
    for background_index in range(4):
        for intensity, nrmse in ((2, 0.8), (4, 0.2)):
            for member in (0, 1):
                member_sample = anchored_sample(
                    member=member,
                    intensity=intensity,
                    background_index=background_index,
                )
                metric = runner.metric_row(
                    member_sample,
                    model_id="test-model",
                    forecast=np.asarray(
                        member_sample["target"],
                        dtype=float,
                    )[-4:],
                    input_adaptation=None,
                )
                metrics.append(
                    {**metric, "context_policy": runner.FIXED_CONTEXT_POLICY}
                )
            effects.append(
                {
                "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
                "dataset_id": "gift_electricity_h",
                "context_policy": runner.FIXED_CONTEXT_POLICY,
                "evaluation_table": runner.REAL_ANCHORED_BENCHMARK_TRACK,
                "generator_family_role": "real_anchored",
                "capability_id": "multi_seasonal",
                "model_id": "test-model",
                "context_length": runner.FIXED_CONTEXT_LENGTH,
                "intensity": intensity,
                "seed_index": 7 + background_index,
                "background_id": f"real_background_{background_index}",
                "dose_parameter": "canonical_strength_lambda",
                "dose_value": intensity / 5.0,
                "paired_treatment_strength": intensity / 5.0,
                "applied_alpha": (
                    1.0 + 0.3 * intensity + 0.1 * background_index
                ),
                "paired_treatment_applied_alpha": (
                    1.0 + 0.3 * intensity + 0.1 * background_index
                ),
                "dose_calibration_policy_sha256": "d" * 64,
                "counterfactual_effect_nrmse": nrmse,
                "shared_baseline_mase_scale": 2.5,
                }
            )

    scores = runner.real_anchored_score_table(metrics, effects)
    assert len(scores) == 1
    assert scores[0]["mechanism_intensity"] == 4
    assert scores[0]["mechanism_canonical_strength"] == pytest.approx(0.8)
    assert scores[0]["mechanism_applied_alpha"] is None
    assert scores[0]["mechanism_applied_alpha_minimum"] == pytest.approx(2.2)
    assert scores[0]["mechanism_applied_alpha_maximum"] == pytest.approx(2.5)
    assert scores[0]["mechanism_applied_alpha_scope"] == (
        "contract_specific_history_only"
    )
    assert scores[0]["mechanism_applied_alpha_by_background_sha256"] == (
        protocol.json_sha256(
            {
                f"real_background_{index}": 2.2 + 0.1 * index
                for index in range(4)
            }
        )
    )
    assert scores[0]["dose_calibration_policy_sha256"] == "d" * 64
    assert scores[0]["mechanism_score"] == pytest.approx(0.2)
    assert scores[0]["ranking_scope"].endswith("never_synthetic")

    files = runner.write_real_anchored_analysis(
        tmp_path,
        {
            "prediction_metrics": metrics,
            "counterfactual_effects": effects,
            "scores": scores,
        },
    )
    assert set(files) == {
        "prediction_metrics",
        "counterfactual_effects",
        "scores",
        "input_ablation_attribution",
        "input_ablation_summary",
        "sensitivity_effects",
        "sensitivity_summary",
    }
    assert (tmp_path / "real_anchored_prediction_metrics.jsonl").is_file()
    assert (
        tmp_path / "real_anchored_counterfactual_effects.jsonl"
    ).is_file()
    payload = protocol.read_json(tmp_path / "real_anchored_scores.json")
    assert payload["ranking_scope"] == (
        "independent_from_synthetic_capability_scores_and_ranks"
    )
    assert payload["scores"] == scores


def test_dataset_score_collapses_repeated_baseline_to_one_background_path(
) -> None:
    metric_rows = []
    effect_rows = []
    for background_index in range(4):
        for dose, treatment_mase, effect_nrmse in (
            (1, 3.0, 0.5),
            (2, 6.0, 0.25),
        ):
            for member, mase in ((0, 0.0), (1, treatment_mase)):
                metric_rows.append(
                {
                    "benchmark_track": (
                        runner.REAL_ANCHORED_BENCHMARK_TRACK
                    ),
                    "dataset_id": "d1",
                    "context_policy": runner.FIXED_CONTEXT_POLICY,
                    "evaluation_table": (
                        runner.REAL_ANCHORED_BENCHMARK_TRACK
                    ),
                    "generator_family_role": "real_anchored",
                    "capability_id": "trend",
                    "model_id": "m1",
                    "context_length": runner.FIXED_CONTEXT_LENGTH,
                    "intensity": dose,
                    "seed_index": background_index,
                    "background_id": f"background-{background_index}",
                    "counterfactual_member": member,
                    "metrics": {
                        "mase": mase,
                        "normalized_mae_history_std": mase,
                    },
                    }
                )
            effect_rows.append(
                {
                "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
                "dataset_id": "d1",
                "context_policy": runner.FIXED_CONTEXT_POLICY,
                "evaluation_table": runner.REAL_ANCHORED_BENCHMARK_TRACK,
                "generator_family_role": "real_anchored",
                "capability_id": "trend",
                "model_id": "m1",
                "context_length": runner.FIXED_CONTEXT_LENGTH,
                "intensity": dose,
                "seed_index": background_index,
                "background_id": f"background-{background_index}",
                "counterfactual_effect_nrmse": effect_nrmse,
                "shared_baseline_mase_scale": 1.0,
                }
            )

    scores = runner.real_anchored_score_table(metric_rows, effect_rows)

    assert len(scores) == 1
    # One baseline plus the two treatment paths: (0 + 3 + 6) / 3.
    assert scores[0]["accuracy_score"] == pytest.approx(3.0)
    assert scores[0]["effective_background_count"] == 4
    assert scores[0]["seed_count"] == 4
    assert scores[0]["serialized_metric_row_count"] == 16
    assert scores[0]["unique_forecast_path_count"] == 12
    assert scores[0]["mechanism_background_count"] == 4
    assert scores[0]["mechanism_score"] == pytest.approx(0.25)


def test_dataset_score_does_not_rank_fewer_than_four_backgrounds() -> None:
    metric_rows = []
    effect_rows = []
    for background_index in range(3):
        for member in (0, 1):
            sample = anchored_sample(
                member=member,
                background_index=background_index,
            )
            metric_rows.append(
                {
                    **runner.metric_row(
                        sample,
                        model_id="test-model",
                        forecast=np.asarray(sample["target"], dtype=float)[-4:],
                        input_adaptation=None,
                    ),
                    "context_policy": runner.FIXED_CONTEXT_POLICY,
                }
            )
        effect_rows.append(
            {
                "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
                "dataset_id": "gift_electricity_h",
                "context_policy": runner.FIXED_CONTEXT_POLICY,
                "evaluation_table": runner.REAL_ANCHORED_BENCHMARK_TRACK,
                "generator_family_role": "real_anchored",
                "capability_id": "multi_seasonal",
                "model_id": "test-model",
                "context_length": runner.FIXED_CONTEXT_LENGTH,
                "intensity": 4,
                "seed_index": 7 + background_index,
                "background_id": f"real_background_{background_index}",
                "counterfactual_effect_nrmse": 0.0,
                "shared_baseline_mase_scale": 2.5,
            }
        )

    assert runner.real_anchored_score_table(metric_rows, effect_rows) == []


def test_dataset_score_rejects_background_reuse_across_seeds() -> None:
    rows = [
        {
            "background_id": "same-background",
            "seed_index": seed_index,
        }
        for seed_index in (0, 1)
    ]

    with pytest.raises(ValueError, match="reused by seeds"):
        runner._real_anchored_background_groups(
            rows,
            label="test",
        )


def experiment_anchored_score(
    dataset_id: str,
    model_id: str,
    *,
    accuracy: float,
    mechanism: float,
) -> dict:
    return {
        "schema_version": "cafe.real_anchored_score.v1",
        "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "ranking_scope": "real_anchored_counterfactual_only_never_synthetic",
        "dataset_id": dataset_id,
        "context_policy": runner.FIXED_CONTEXT_POLICY,
        "evaluation_table": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "generator_family_role": "real_anchored",
        "capability_id": "multi_seasonal",
        "model_id": model_id,
        "accuracy_score": accuracy,
        "accuracy_metric": "mase",
        "history_std_normalized_mae": accuracy + 0.25,
        "mechanism_metric": "counterfactual_effect_nrmse",
        "mechanism_score": mechanism,
        "effective_background_count": 4,
        "effective_background_ids_sha256": protocol.json_sha256(
            [f"{dataset_id}-background-{index}" for index in range(4)]
        ),
        "mechanism_background_count": 4,
        "seed_count": 4,
        "intensities": [1, 2, 3, 4, 5],
        "is_reference_baseline": False,
    }


def test_experiment_real_anchored_macro_uses_common_model_intersection() -> None:
    scores = [
        experiment_anchored_score("d1", "m1", accuracy=1.0, mechanism=0.4),
        experiment_anchored_score("d1", "m2", accuracy=2.0, mechanism=0.1),
        experiment_anchored_score("d1", "m3", accuracy=0.1, mechanism=0.1),
        experiment_anchored_score("d2", "m1", accuracy=3.0, mechanism=0.2),
        experiment_anchored_score("d2", "m2", accuracy=1.0, mechanism=0.3),
    ]

    rows, summary = runner.experiment_real_anchored_capability_rows(
        scores,
        dataset_ids=["d1", "d2"],
        models=["m1", "m2", "m3"],
        capabilities=["multi_seasonal"],
    )

    assert summary["status"] == "aggregated"
    assert summary["capabilities"][0]["common_models"] == ["m1", "m2"]
    assert summary["capabilities"][0][
        "effective_background_count_by_dataset"
    ] == {"d1": 4, "d2": 4}
    assert summary["capabilities"][0][
        "total_authentic_background_units"
    ] == 8
    assert {row["model_id"] for row in rows} == {"m1", "m2"}
    by_model = {row["model_id"]: row for row in rows}
    assert by_model["m1"]["macro_mean_accuracy_score"] == pytest.approx(2.0)
    assert by_model["m2"]["macro_mean_accuracy_score"] == pytest.approx(1.5)
    assert by_model["m2"]["accuracy_rank"] == 1
    assert by_model["m2"]["macro_mean_mechanism_score"] == pytest.approx(0.2)
    assert by_model["m2"]["mechanism_rank"] == 1
    assert all(
        row["ranking_scope"].endswith("never_synthetic") for row in rows
    )


def synthetic_score(
    dataset_id: str,
    model_id: str,
    context_policy: str,
) -> dict:
    model_offset = 0.0 if model_id == "m1" else 1.0
    return {
        "dataset_id": dataset_id,
        "context_policy": context_policy,
        "evaluation_table": "main",
        "generator_family_role": "primary",
        "capability_id": "multi_seasonal",
        "model_id": model_id,
        "accuracy_score": 10.0 + model_offset,
        "history_std_normalized_mae": 11.0 + model_offset,
        "mechanism_score": 12.0 + model_offset,
        "accuracy_rank": 1 + int(model_id == "m2"),
        "mechanism_rank": 1 + int(model_id == "m2"),
    }


def prepare_aggregate_experiment(
    root: Path,
    *,
    include_real_anchored: bool,
) -> Namespace:
    dataset_ids = ["d1", "d2"]
    models = ["m1", "m2"]
    shard_name = "seed_000000_000002"
    protocol.write_json(
        root / "experiment.json",
        {"experiment_id": "test-real-anchored-aggregate"},
    )
    protocol.write_json(
        root / "stage_contracts" / "analysis.json",
        {
            "config": {
                "dataset_ids": dataset_ids,
                "models": models,
                "capabilities": ["multi_seasonal"],
                "analysis_profile": "scores_only",
                "seed_start": 0,
                "seed_count": 2,
            }
        },
    )
    for dataset_index, dataset_id in enumerate(dataset_ids):
        analysis_dir = root / dataset_id / "04_analysis" / shard_name
        score_path = analysis_dir / "scores.json"
        synthetic_scores = [
            synthetic_score(dataset_id, model_id, context_policy)
            for context_policy in (
                runner.FIXED_CONTEXT_POLICY,
                "oracle_context",
            )
            for model_id in models
        ]
        protocol.write_json(score_path, {"scores": synthetic_scores})
        effect_path = analysis_dir / "counterfactual_effects.jsonl"
        protocol.write_jsonl(effect_path, ())
        files = {
            "scores": protocol.file_record(score_path),
            "counterfactual_effects": protocol.file_record(effect_path),
        }
        if include_real_anchored:
            real_score_path = analysis_dir / "real_anchored_scores.json"
            real_scores = [
                experiment_anchored_score(
                    dataset_id,
                    model_id,
                    accuracy=(dataset_index + 1) * (model_index + 1),
                    mechanism=(dataset_index + model_index + 1) / 10.0,
                )
                for model_index, model_id in enumerate(models)
            ]
            protocol.write_json(
                real_score_path,
                {
                    "schema_version": "cafe.real_anchored_scores.v1",
                    "benchmark_track": (
                        runner.REAL_ANCHORED_BENCHMARK_TRACK
                    ),
                    "scores": real_scores,
                },
            )
            files["real_anchored_scores"] = {
                **protocol.file_record(real_score_path),
                "row_count": len(real_scores),
            }
        protocol.write_json(
            analysis_dir / "analysis_manifest.json",
            {
                "schema_version": "cafe.analysis_manifest.v2",
                "dataset_id": dataset_id,
                "models": models,
                "analysis_profile": "scores_only",
                "files": files,
            },
        )
        generation_config = {
            "schema_version": "cafe.generation_config.v5",
            "capabilities": ["multi_seasonal"],
            "real_anchored_counterfactual": {
                "upstream_real_anchored_protocol": protocol.SCHEMA_VERSION,
                "qualification_policy_sha256": "a" * 64,
                "legacy_upstream_component_policy": None,
                "generated_capabilities": ["multi_seasonal"],
                "dose_parameter": "canonical_strength_lambda",
                "canonical_strength_grid": list(
                    REAL_ANCHORED_CANONICAL_STRENGTH_GRID
                ),
                "applied_alpha_grid_by_capability": {
                    "multi_seasonal": [],
                },
                "applied_alpha_scope": "contract_specific_history_only",
                "applied_alpha_range_by_capability": {
                    "multi_seasonal": [
                        {"minimum": value, "maximum": value}
                        for value in [1.2, 1.4, 1.6, 1.8, 2.0]
                    ],
                },
                "dose_calibration_sha256_by_capability": {
                    "multi_seasonal": "b" * 64,
                },
                "dose_policy_sha256": "c" * 64,
                "pairing": (
                    "baseline_lambda0_alpha1_vs_treatment_"
                    "contract_resolved_alpha"
                ),
                "paired_minimum_separation": (
                    "mandatory_treatment_source_l168_distance_with_budget_v1"
                ),
                "anti_copy": (
                    "not_applicable_intentional_real_anchor_counterfactual"
                ),
            },
        }
        protocol.write_json(
            root
            / dataset_id
            / "02_generation"
            / f"manifest__{shard_name}.json",
            {
                "schema_version": "cafe.generation_manifest.v5",
                "config": generation_config,
                "config_sha256": protocol.json_sha256(generation_config),
            },
        )
    return Namespace(
        output_root=root,
        source_experiment_root=None,
        analysis_profile="scores_only",
        models=models,
        seed_start=0,
        seed_count=2,
        reuse_existing_aggregate=False,
    )


def test_experiment_aggregate_writes_separate_real_anchored_table(
    tmp_path: Path,
) -> None:
    args = prepare_aggregate_experiment(
        tmp_path,
        include_real_anchored=True,
    )

    assert runner.aggregate_experiment_analysis(args) == 0

    analysis_dir = tmp_path / "04_analysis" / "seed_000000_000002"
    real_path = (
        analysis_dir / "capability_scores_real_anchored_fixed_l168.json"
    )
    real_payload = protocol.read_json(real_path)
    fixed_payload = protocol.read_json(
        analysis_dir / "capability_scores_fixed_l168.json"
    )
    manifest = protocol.read_json(analysis_dir / "analysis_manifest.json")
    assert real_payload["ranking_scope"] == (
        "independent_from_synthetic_fixed_and_oracle_rankings"
    )
    assert len(real_payload["scores"]) == 2
    assert all(
        row["macro_mean_accuracy_score"] < 10.0
        for row in real_payload["scores"]
    )
    assert all(
        row["macro_mean_accuracy_score"] >= 10.0
        for row in fixed_payload["scores"]
    )
    assert manifest["schema_version"] == (
        "cafe.experiment_analysis_manifest.v2"
    )
    assert manifest["real_anchored_counterfactual"]["status"] == (
        "aggregated"
    )
    assert "real_anchored_fixed_scores" in manifest["files"]
    assert runner.reusable_experiment_analysis_manifest(
        analysis_dir,
        stage_contract_path=(
            tmp_path / "stage_contracts" / "analysis.json"
        ),
        dataset_ids=["d1", "d2"],
        models=["m1", "m2"],
        capabilities=["multi_seasonal"],
        analysis_profile="scores_only",
    )


def test_experiment_aggregate_without_component_remains_backward_compatible(
    tmp_path: Path,
) -> None:
    args = prepare_aggregate_experiment(
        tmp_path,
        include_real_anchored=False,
    )

    assert runner.aggregate_experiment_analysis(args) == 0

    analysis_dir = tmp_path / "04_analysis" / "seed_000000_000002"
    manifest = protocol.read_json(analysis_dir / "analysis_manifest.json")
    assert manifest["real_anchored_counterfactual"]["status"] == (
        "component_absent"
    )
    assert "real_anchored_fixed_scores" not in manifest["files"]
    assert not (
        analysis_dir / "capability_scores_real_anchored_fixed_l168.json"
    ).exists()
