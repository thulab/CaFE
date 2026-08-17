from __future__ import annotations

from pathlib import Path

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension import analysis as analysis_module
from cafe.benchmark_extension.analysis import (
    _accuracy_rows,
    _input_ablation_rows,
    analyse_model,
    run_analysis,
)
from cafe.benchmark_extension.inference import INFERENCE_SCHEMA
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
    monkeypatch.setattr(
        analysis_module,
        "iter_replayed_samples",
        lambda *_args, **_kwargs: iter((baseline, treatment, ablation)),
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
    protocol.write_json(
        inference_dir / "manifest.json",
        {
            "schema_version": INFERENCE_SCHEMA,
            "dataset_id": "gift_fixture",
            "config": {"models": ["model"]},
            "model_predictions": {
                "model": {
                    "format": "partitioned_parquet",
                    "row_count": 3,
                    "parts": [prediction_record],
                }
            },
            "complete": True,
        },
    )
    manifest = run_analysis(dataset_root)
    assert manifest["files"]["accuracy_rows"]["row_count"] == 2
    assert manifest["files"]["capability_effect_rows"]["row_count"] == 1
    assert manifest["files"]["input_ablation_rows"]["row_count"] == 1
    accuracy = protocol.read_json(dataset_root / "04_analysis" / "accuracy_summary.json")
    assert {row["sample_kind"] for row in accuracy["rows"]} == {
        "official_baseline",
        "capability_treatment",
    }
