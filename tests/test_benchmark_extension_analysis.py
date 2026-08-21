from __future__ import annotations

from pathlib import Path

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension import analysis as analysis_module
from cafe.benchmark_extension.analysis import (
    _accuracy_rows,
    _aggregate_effects,
    _effect_measurement,
    _input_ablation_rows,
    analyse_model,
    run_analysis,
)
from cafe.benchmark_extension.inference import INFERENCE_SCHEMA
from cafe.benchmark_extension.generation import PIPELINE_SCHEMA, _baseline_row
from cafe.benchmark_extension.gift_eval import GiftEvalInstance
from cafe.benchmark_extension.storage import (
    PredictionParquetWriter,
    parquet_file_record,
)


def test_effect_metric_uses_shared_official_baseline_prediction() -> None:
    history = np.arange(20.0)[:, None]
    future = np.arange(20.0, 24.0)[:, None]
    baseline = {
        "sample_id": "baseline",
        "dataset_id": "gift_fixture",
        "official_instance_id": "official",
        "context_length": 20,
        "target": np.vstack((history, future)).tolist(),
        "future_observed_mask": np.ones((4, 1), dtype=bool).tolist(),
        "mase_scale_by_target": [1.0],
    }
    truth_delta = np.ones((4, 1)) * 2.0
    treatment = {
        **baseline,
        "sample_id": "treatment",
        "baseline_sample_id": "baseline",
        "dataset_id": "gift_fixture",
        "capability_id": "trend",
        "capability_level": 1,
        "controlled_coordinate": "distance",
        "sampled_coordinate": 0.1,
        "affected_target_indices": [0],
        "target": np.vstack((history, future + truth_delta)).tolist(),
    }
    predictions = {
        "baseline": future.copy(),
        "treatment": future + truth_delta,
    }
    summary, effects = analyse_model(
        "model",
        {"baseline": baseline},
        [treatment],
        predictions,
    )
    assert summary["official_mase_mean"] == 0.0
    assert effects[0]["effect_nrmse"] == 0.0
    assert effects[0]["effect_amplitude_ratio"] == 1.0
    assert effects[0]["treatment_mase"] == 0.0
    accuracy = _accuracy_rows(
        "model", {"baseline": baseline}, [treatment], predictions
    )
    assert [row["sample_kind"] for row in accuracy] == [
        "official_baseline",
        "capability_treatment",
    ]
    assert [row["mase"] for row in accuracy] == [0.0, 0.0]


def test_low_truth_effect_is_unavailable_without_epsilon_division() -> None:
    truth_delta = np.full((4, 1), 0.01)
    forecast_delta = np.full((4, 1), 0.05)
    measurement = _effect_measurement(
        truth_delta,
        forecast_delta,
        np.ones((4, 1), dtype=bool),
        [0],
        np.ones(1),
    )
    assert measurement["status"] == "unavailable_low_truth_effect"
    assert measurement["nrmse"] is None
    assert measurement["truth_mase_rms"] == 0.01

    history = np.arange(20.0)[:, None]
    future = np.arange(20.0, 24.0)[:, None]
    baseline = {
        "sample_id": "baseline",
        "dataset_id": "gift_fixture",
        "official_instance_id": "official",
        "context_length": 20,
        "target": np.vstack((history, future)).tolist(),
        "future_observed_mask": np.ones((4, 1), dtype=bool).tolist(),
        "mase_scale_by_target": [1.0],
    }
    treatment = {
        **baseline,
        "sample_id": "treatment",
        "baseline_sample_id": "baseline",
        "capability_id": "predictable_intermittency",
        "capability_level": 5,
        "controlled_coordinate": "gap",
        "sampled_coordinate": 0.9,
        "affected_target_indices": [0],
        "target": np.vstack((history, future + 0.01)).tolist(),
    }
    _, effects = analyse_model(
        "model",
        {"baseline": baseline},
        [treatment],
        {"baseline": future, "treatment": future + 0.05},
    )
    assert effects[0]["effect_score_status"] == "unavailable_low_truth_effect"
    assert effects[0]["effect_nrmse"] is None
    assert effects[0]["treatment_mase"] > 0.0


def test_effect_measurement_preserves_zero_and_one_interpretation() -> None:
    truth_delta = np.full((4, 2), 2.0)
    mask = np.ones((4, 2), dtype=bool)
    scales = np.asarray([2.0, 4.0])
    perfect = _effect_measurement(
        truth_delta, truth_delta, mask, [0, 1], scales
    )
    no_response = _effect_measurement(
        truth_delta, np.zeros_like(truth_delta), mask, [0, 1], scales
    )
    assert perfect["status"] == "scored"
    assert perfect["nrmse"] == 0.0
    assert no_response["nrmse"] == 1.0


def test_effect_summary_uses_pooled_standardized_energy() -> None:
    base = {
        "model_id": "model",
        "capability_id": "trend",
        "capability_level": 1,
        "effect_score_status": "scored",
        "effect_correlation": None,
        "effect_amplitude_ratio": 1.0,
    }
    summary = _aggregate_effects(
        [
            {
                **base,
                "official_instance_id": "large-effect",
                "effect_nrmse": 0.0,
                "standardized_squared_error_sum": 0.0,
                "standardized_truth_squared_sum": 100.0,
            },
            {
                **base,
                "official_instance_id": "small-effect",
                "effect_nrmse": 2.0,
                "standardized_squared_error_sum": 4.0,
                "standardized_truth_squared_sum": 1.0,
            },
        ]
    )[0]
    assert np.isclose(summary["effect_nrmse_mean"], 1.0)
    assert np.isclose(summary["effect_nrmse_pooled"], np.sqrt(4.0 / 101.0))


def test_input_ablation_reports_measured_degradation_without_model_type_shortcut() -> None:
    history = np.arange(20.0)[:, None]
    future = np.arange(20.0, 24.0)[:, None]
    baseline = {
        "sample_id": "baseline",
        "official_instance_id": "official",
        "dataset_id": "gift_fixture",
        "context_length": 20,
        "target": np.vstack((history, future)).tolist(),
        "future_observed_mask": np.ones((4, 1), dtype=bool).tolist(),
        "mase_scale_by_target": [1.0],
    }
    treatment = {
        **baseline,
        "sample_id": "treatment",
        "baseline_sample_id": "baseline",
        "capability_id": "cross_series_dependence",
        "capability_level": 1,
        "target": np.vstack((history, future + 2.0)).tolist(),
    }
    ablation = {
        **treatment,
        "sample_id": "ablation",
        "input_ablation_source_sample_id": "treatment",
        "assessed_target_indices": [0],
        "ablated_input_indices": [1],
    }
    perfect = future + 2.0
    rows = _input_ablation_rows(
        "independent-univariate",
        {"baseline": baseline},
        {"treatment": treatment},
        [ablation],
        {"treatment": perfect, "ablation": perfect.copy()},
    )
    assert rows[0]["input_ablation_mase_degradation"] == 0.0
    assert rows[0]["input_ablation_response_ratio"] == 0.0

    rows = _input_ablation_rows(
        "native-panel",
        {"baseline": baseline},
        {"treatment": treatment},
        [ablation],
        {"treatment": perfect, "ablation": perfect - 1.0},
    )
    assert rows[0]["full_input_mase"] == 0.0
    assert rows[0]["ablated_input_mase"] == 1.0
    assert rows[0]["input_ablation_mase_degradation"] == 1.0


def test_analysis_writes_separate_accuracy_effect_and_ablation_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "gift_fixture"
    generation_dir = dataset_root / "01_generation"
    inference_dir = dataset_root / "03_inference"
    history = np.arange(20.0)[:, None]
    future = np.arange(20.0, 24.0)[:, None]
    baseline = {
        "evaluation_table": "gift_eval_official_baseline",
        "source_shard_index": 0,
        "sample_id": "baseline",
        "official_instance_id": "official",
        "dataset_id": "gift_fixture",
        "context_length": 20,
        "target": np.vstack((history, future)).tolist(),
        "future_observed_mask": np.ones((4, 1), dtype=bool).tolist(),
        "mase_scale_by_target": [1.0],
        "capability_id": None,
        "capability_level": 0,
    }
    treatment = {
        **baseline,
        "sample_id": "treatment",
        "baseline_sample_id": "baseline",
        "capability_id": "cross_series_dependence",
        "capability_level": 1,
        "controlled_coordinate": "strength",
        "sampled_coordinate": 0.2,
        "affected_target_indices": [0],
        "target": np.vstack((history, future + 2.0)).tolist(),
        "evaluation_table": "gift_eval_capability_treatment",
    }
    ablation = {
        **treatment,
        "sample_id": "ablation",
        "input_ablation_source_sample_id": "treatment",
        "assessed_target_indices": [0],
        "ablated_input_indices": [1],
        "evaluation_table": "gift_eval_capability_input_ablation",
    }
    protocol.write_json(
        generation_dir / "manifest.json",
        {"config": {"dataset_id": "gift_fixture"}},
    )
    replay_calls = 0

    def replay_once(*_args, **_kwargs):
        nonlocal replay_calls
        replay_calls += 1
        yield None, {"source_shard_index": 0}, [], []

    monkeypatch.setattr(
        analysis_module, "iter_replay_contract_work_items", replay_once
    )
    monkeypatch.setattr(
        analysis_module,
        "_replay_contract_instance",
        lambda *_args: [baseline, treatment, ablation],
    )
    prediction_path = inference_dir / "models" / "model" / "predictions" / "part_000000.parquet"
    writer = PredictionParquetWriter(prediction_path)
    writer.write(model_id="model", sample_id="baseline", forecast=future)
    writer.write(model_id="model", sample_id="treatment", forecast=future + 2.0)
    writer.write(model_id="model", sample_id="ablation", forecast=future + 1.0)
    prediction_count = writer.close()
    prediction_record = {
        **parquet_file_record(prediction_path, row_count=prediction_count),
        "source_shard_index": 0,
    }
    second_prediction_path = (
        inference_dir / "models" / "model2" / "predictions" / "part_000000.parquet"
    )
    second_writer = PredictionParquetWriter(second_prediction_path)
    second_writer.write(model_id="model2", sample_id="baseline", forecast=future)
    second_writer.write(
        model_id="model2", sample_id="treatment", forecast=future + 2.0
    )
    second_writer.write(
        model_id="model2", sample_id="ablation", forecast=future + 1.0
    )
    second_prediction_count = second_writer.close()
    second_prediction_record = {
        **parquet_file_record(
            second_prediction_path, row_count=second_prediction_count
        ),
        "source_shard_index": 0,
    }
    protocol.write_json(
        inference_dir / "manifest.json",
        {
            "schema_version": INFERENCE_SCHEMA,
            "dataset_id": "gift_fixture",
            "config": {
                "pipeline_schema_version": PIPELINE_SCHEMA,
                "models": ["model", "model2"],
            },
            "model_predictions": {
                "model": {
                    "format": "partitioned_parquet",
                    "row_count": 3,
                    "parts": [prediction_record],
                },
                "model2": {
                    "format": "partitioned_parquet",
                    "row_count": 3,
                    "parts": [second_prediction_record],
                },
            },
            "complete": True,
        },
    )
    manifest = run_analysis(dataset_root)
    assert replay_calls == 1
    assert manifest["files"]["accuracy_rows"]["row_count"] == 4
    assert manifest["files"]["capability_effect_rows"]["row_count"] == 2
    assert manifest["files"]["input_ablation_rows"]["row_count"] == 2
    accuracy = protocol.read_json(dataset_root / "04_analysis" / "accuracy_summary.json")
    assert {row["sample_kind"] for row in accuracy["rows"]} == {
        "official_baseline",
        "capability_treatment",
    }
    effects = protocol.read_json(
        dataset_root / "04_analysis" / "capability_effect_summary.json"
    )["rows"]
    assert len(effects) == 2
    assert all(row["effect_nrmse_pooled"] == 0.0 for row in effects)
    assert all(row["effect_scoring_coverage"] == 1.0 for row in effects)
    assert all(row["effect_score_metric"] == "mase_standardized_pooled_nrmse_v1" for row in effects)


def test_analysis_skips_an_official_instance_without_observed_future(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "gift_fixture"
    generation_dir = dataset_root / "01_generation"
    inference_dir = dataset_root / "03_inference"
    future = np.arange(4.0)[:, None]
    baseline = {
        "evaluation_table": "gift_eval_official_baseline",
        "source_shard_index": 0,
        "sample_id": "unobserved",
        "official_instance_id": "official",
        "dataset_id": "gift_fixture",
        "context_length": 2,
        "target": np.vstack((np.zeros((2, 1)), future)).tolist(),
        "future_observed_mask": np.zeros((4, 1), dtype=bool).tolist(),
        "mase_scale_by_target": [1.0],
        "capability_id": None,
        "capability_level": 0,
    }
    protocol.write_json(
        generation_dir / "manifest.json",
        {"config": {"dataset_id": "gift_fixture"}},
    )
    monkeypatch.setattr(
        analysis_module,
        "iter_replay_contract_work_items",
        lambda *_args, **_kwargs: iter(
            ((None, {"source_shard_index": 0}, [], []),)
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "_replay_contract_instance",
        lambda *_args: [baseline],
    )
    prediction_path = (
        inference_dir / "models" / "model" / "predictions" / "part_000000.parquet"
    )
    writer = PredictionParquetWriter(prediction_path)
    writer.write(model_id="model", sample_id="unobserved", forecast=future)
    prediction_count = writer.close()
    prediction_record = {
        **parquet_file_record(prediction_path, row_count=prediction_count),
        "source_shard_index": 0,
    }
    protocol.write_json(
        inference_dir / "manifest.json",
        {
            "schema_version": INFERENCE_SCHEMA,
            "dataset_id": "gift_fixture",
            "config": {
                "pipeline_schema_version": PIPELINE_SCHEMA,
                "models": ["model"],
            },
            "model_predictions": {
                "model": {
                    "format": "partitioned_parquet",
                    "row_count": 1,
                    "parts": [prediction_record],
                }
            },
            "complete": True,
        },
    )
    manifest = run_analysis(dataset_root)
    assert manifest["files"]["accuracy_rows"]["row_count"] == 0
    summary = protocol.read_json(dataset_root / "04_analysis" / "official_accuracy.json")
    assert summary["models"][0]["official_instance_count"] == 0
    assert summary["models"][0]["official_mase_mean"] is None


def test_analysis_parallelizes_source_shards_without_repeating_source_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "gift_fixture"
    generation_dir = dataset_root / "01_generation"
    inference_dir = dataset_root / "03_inference"
    protocol.write_json(
        generation_dir / "manifest.json",
        {"config": {"dataset_id": "gift_fixture"}},
    )
    work_items = []
    prediction_records = []
    for shard in range(2):
        history = (np.arange(8.0) + shard)[:, None]
        future = (np.arange(8.0, 12.0) + shard)[:, None]
        instance = GiftEvalInstance(
            dataset_id="gift_fixture",
            config_id="fixture/H",
            item_id=f"item_{shard}",
            official_instance_id=f"official_{shard}",
            frequency="H",
            term="short",
            window_index=0,
            window_count=1,
            forecast_origin=8,
            prediction_length=4,
            history=history,
            future=future,
            future_observed_mask=np.ones((4, 1), dtype=bool),
            history_covariates=np.empty((8, 0)),
            future_covariates=np.empty((4, 0)),
            covariate_column_names=(),
            target_column_names=("target",),
            source_target_length=12,
            history_imputation={"policy": "none"},
        )
        baseline = {**_baseline_row(instance), "source_shard_index": shard}
        work_items.append((instance, baseline, [], []))
        prediction_path = (
            inference_dir
            / "models"
            / "model"
            / "predictions"
            / f"part_{shard:06d}.parquet"
        )
        writer = PredictionParquetWriter(prediction_path)
        writer.write(
            model_id="model",
            sample_id=str(baseline["sample_id"]),
            forecast=future,
        )
        prediction_records.append(
            {
                **parquet_file_record(prediction_path, row_count=writer.close()),
                "source_shard_index": shard,
            }
        )
    source_scans = 0

    def iter_work_items(*_args, **_kwargs):
        nonlocal source_scans
        source_scans += 1
        yield from work_items

    monkeypatch.setattr(
        analysis_module,
        "iter_replay_contract_work_items",
        iter_work_items,
    )
    protocol.write_json(
        inference_dir / "manifest.json",
        {
            "schema_version": INFERENCE_SCHEMA,
            "dataset_id": "gift_fixture",
            "config": {
                "pipeline_schema_version": PIPELINE_SCHEMA,
                "models": ["model"],
            },
            "model_predictions": {
                "model": {
                    "format": "partitioned_parquet",
                    "row_count": 2,
                    "parts": prediction_records,
                }
            },
            "complete": True,
        },
    )
    manifest = run_analysis(dataset_root, replay_workers=2)
    assert source_scans == 1
    assert manifest["config"]["source_shard_count"] == 2
    assert manifest["config"]["shard_workers"] == 2
    assert manifest["files"]["accuracy_rows"]["row_count"] == 2
    assert not (dataset_root / "04_analysis" / ".source_shard_parts").exists()
