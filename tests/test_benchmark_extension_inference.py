from __future__ import annotations

import numpy as np

from cafe.benchmark_extension.inference import model_task_row


def _sample() -> dict:
    target = np.arange(250 * 3.0).reshape(250, 3)
    return {
        "schema_version": "cafe.benchmark_extension_sample.v1",
        "sample_id": "sample",
        "context_length": 200,
        "horizon": 50,
        "target_dim": 3,
        "covariate_dim": 0,
        "covariates": None,
        "frequency": "H",
        "target": target.tolist(),
    }


def test_model_context_truncation_happens_after_full_history_treatment() -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {"max_input_length": 96, "min_input_length": 1},
    }
    row = model_task_row(_sample(), model)
    assert row["source_context_length"] == 200
    assert row["context_length"] == 96
    assert np.asarray(row["target"]).shape == (146, 3)
    np.testing.assert_array_equal(
        np.asarray(row["target"]),
        np.asarray(_sample()["target"])[104:],
    )


def test_native_panel_is_preserved_in_generation_task() -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {"max_input_length": -1, "min_input_length": 1},
    }
    row = model_task_row(_sample(), model)
    assert row["target_dim"] == 3
    assert np.asarray(row["target"]).shape == (250, 3)
