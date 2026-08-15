from __future__ import annotations

import numpy as np

from cafe.benchmark_extension.analysis import analyse_model


def test_effect_metric_uses_shared_official_baseline_prediction() -> None:
    history = np.arange(20.0)[:, None]
    future = np.arange(20.0, 24.0)[:, None]
    baseline = {
        "sample_id": "baseline",
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
