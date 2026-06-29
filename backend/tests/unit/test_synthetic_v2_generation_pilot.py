from __future__ import annotations

import numpy as np

from app.services.synthetic_generation_service import (
    PILOT_ACCEPTANCE_CAPS,
    _generate_accepted_sample_values,
    _seed_for,
)


def test_trend_pilot_features_are_monotonic_and_capped():
    summaries = []
    for difficulty in range(1, 6):
        rows = [
            _generate_accepted_sample_values(
                "trend",
                192,
                168,
                1,
                24,
                difficulty,
                _seed_for(321, "trend", difficulty * 1000 + sample_index),
            )[3]
            for sample_index in range(96)
        ]
        summaries.append(
            {
                "trend_strength": float(np.mean([row["trend_strength"] for row in rows])),
                "slope_abs": float(np.mean([row["slope_abs"] for row in rows])),
                "curvature_abs": float(np.mean([row["curvature_abs"] for row in rows])),
                "max_slope_abs": float(np.max([row["slope_abs"] for row in rows])),
                "max_noise_ratio": float(np.max([row["noise_ratio"] for row in rows])),
            }
        )

    for feature in ("trend_strength", "slope_abs", "curvature_abs"):
        values = [summary[feature] for summary in summaries]
        assert values == sorted(values)
    assert summaries[-1]["max_slope_abs"] <= PILOT_ACCEPTANCE_CAPS["trend"]["slope_abs"] + 1e-6
    assert summaries[-1]["max_noise_ratio"] <= PILOT_ACCEPTANCE_CAPS["trend"]["noise_ratio"]


def test_multi_seasonal_pilot_degrades_single_period_seasonal_naive():
    seasonal_naive_mae = []
    for difficulty in range(1, 6):
        errors = []
        for sample_index in range(96):
            values, latent_params, _, features = _generate_accepted_sample_values(
                "multi_seasonal",
                192,
                168,
                1,
                24,
                difficulty,
                _seed_for(321, "multi_seasonal", difficulty * 1000 + sample_index),
            )
            history = values[:168, 0]
            actual = values[168:, 0]
            errors.append(float(np.mean(np.abs(actual - history[-24:]))))
            assert latent_params["acceptance"]["accepted"] is True
            assert features["noise_ratio"] <= PILOT_ACCEPTANCE_CAPS["multi_seasonal"]["noise_ratio"]
        seasonal_naive_mae.append(float(np.mean(errors)))

    assert seasonal_naive_mae == sorted(seasonal_naive_mae)
    assert seasonal_naive_mae[-1] > seasonal_naive_mae[0] * 4
