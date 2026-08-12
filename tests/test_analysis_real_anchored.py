from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from cafe import protocol
from cafe.analysis import runner


def anchored_sample(
    *,
    member: int,
    intensity: int = 4,
    mase_scale: float = 2.5,
) -> dict:
    context = runner.FIXED_CONTEXT_LENGTH
    horizon = 4
    time = np.arange(context + horizon, dtype=float)
    baseline = np.sin(2.0 * np.pi * time / 12.0)
    treatment_delta = np.zeros_like(baseline)
    treatment_delta[context:] = np.asarray([1.0, 2.0, 3.0, 4.0])
    target = baseline + member * treatment_delta
    pair_id = f"real_pair_i{intensity}__L{context}"
    master_pair_id = f"real_pair_i{intensity}"
    return {
        "schema_version": "cafe.forecast_view.v1",
        "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "sample_id": f"real_sample_i{intensity}_m{member}__L{context}",
        "master_sample_id": f"real_sample_i{intensity}_m{member}",
        "dataset_id": "gift_electricity_h",
        "config_id": "real_background_0",
        "capability_id": "multi_seasonal",
        "generator_family_role": "real_anchored",
        "evaluation_table": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "intensity": intensity,
        "seed_index": 7,
        "background_id": "real_background_0",
        "dose_value": 1.0 if member == 0 else 1.0 + 0.2 * intensity,
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


def test_real_anchored_scores_use_maximum_available_dose_and_write_separately(
    tmp_path: Path,
) -> None:
    metrics = []
    effects = []
    for intensity, nrmse in ((2, 0.8), (4, 0.2)):
        for member in (0, 1):
            member_sample = anchored_sample(
                member=member,
                intensity=intensity,
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
                "seed_index": 7,
                "background_id": "real_background_0",
                "dose_value": 1.0 + 0.2 * intensity,
                "counterfactual_effect_nrmse": nrmse,
                "shared_baseline_mase_scale": 2.5,
            }
        )

    scores = runner.real_anchored_score_table(metrics, effects)
    assert len(scores) == 1
    assert scores[0]["mechanism_intensity"] == 4
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
                    "seed_index": 0,
                    "background_id": "background-0",
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
                "seed_index": 0,
                "background_id": "background-0",
                "counterfactual_effect_nrmse": effect_nrmse,
                "shared_baseline_mase_scale": 1.0,
            }
        )

    scores = runner.real_anchored_score_table(metric_rows, effect_rows)

    assert len(scores) == 1
    # One baseline plus the two treatment paths: (0 + 3 + 6) / 3.
    assert scores[0]["accuracy_score"] == pytest.approx(3.0)
    assert scores[0]["effective_background_count"] == 1
    assert scores[0]["seed_count"] == 1
    assert scores[0]["serialized_metric_row_count"] == 4
    assert scores[0]["unique_forecast_path_count"] == 3
    assert scores[0]["mechanism_background_count"] == 1
    assert scores[0]["mechanism_score"] == pytest.approx(0.25)


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
        "effective_background_count": 2,
        "effective_background_ids_sha256": protocol.json_sha256(
            [f"{dataset_id}-background-0", f"{dataset_id}-background-1"]
        ),
        "mechanism_background_count": 2,
        "seed_count": 2,
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
    ] == {"d1": 2, "d2": 2}
    assert summary["capabilities"][0][
        "total_authentic_background_units"
    ] == 4
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
        protocol.write_json(
            root
            / dataset_id
            / "02_generation"
            / f"manifest__{shard_name}.json",
            {"config": {"capabilities": ["multi_seasonal"]}},
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
