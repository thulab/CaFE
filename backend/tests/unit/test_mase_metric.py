"""
Unit tests for MASE (Mean Absolute Scaled Error) in compute_sample_metrics.

Hand-computed reference:
  target_history = [[10], [12], [14]]
    → flattened by dim: dim-0 series = [10, 12, 14]
    → consecutive abs diffs = [|12-10|, |14-12|] = [2, 2]
    → scale = mean([2, 2]) = 2.0
  target_future = [[16], [18]], forecast = [[16], [17]]
    → errors = [|16-16|, |17-18|] = [0, 1]
    → MAE (numerator) = 0.5
    → MASE = 0.5 / 2.0 = 0.25
"""

import pytest
from app.models.dataset import Shard
from app.models.sample import SampleIndex
from app.services.metric_service import (
    compute_sample_metrics,
    resolve_mase_period,
    seasonal_period_for_frequency,
)
from app.services.run_executor import _mase_period_for_sample


def test_mase_hand_computed():
    target_history = [[10.0], [12.0], [14.0]]
    target_future  = [[16.0], [18.0]]
    forecast       = [[16.0], [17.0]]

    result = compute_sample_metrics(target_future, forecast, target_history=target_history)

    assert "mse" in result
    assert "mae" in result
    assert "mase" in result
    assert result["mae"] == pytest.approx(0.5)
    # mse = (0^2 + 1^2) / 2 = 0.5
    assert result["mse"] == pytest.approx(0.5)
    assert result["mase"] == pytest.approx(0.25)


def test_mase_uses_seasonal_lag_p_not_consecutive_differences():
    # Lag-1 differences are large, while the period-4 seasonal differences are
    # exactly 1. A unit MAE must therefore produce MASE(4) == 1.
    target_history = [[0.0], [10.0], [20.0], [30.0], [1.0], [11.0], [21.0], [31.0]]
    target_future = [[2.0], [12.0]]
    forecast = [[3.0], [13.0]]

    result = compute_sample_metrics(
        target_future,
        forecast,
        target_history=target_history,
        seasonal_period=4,
    )

    assert result["mae"] == pytest.approx(1.0)
    assert result["mase"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [("1h", 24), ("30m", 48), ("1d", 7), ("7d", 52), ("weekly", 52), ("2d", 1)],
)
def test_default_seasonal_period_is_derived_from_frequency(frequency, expected):
    assert seasonal_period_for_frequency(frequency) == expected


def test_short_history_uses_explicit_lag1_compatibility_fallback():
    assert resolve_mase_period(explicit_period=None, frequency="1h", history_length=12) == 1
    assert resolve_mase_period(explicit_period=None, frequency="1h", history_length=168) == 24


def test_run_uses_sample_level_synthetic_period_before_shard_frequency():
    shard = Shard(
        dataset_manifest_id="manifest",
        source_uri="synthetic://unit",
        frequency="1h",
        generation_config={"season_length": None},
    )
    sample = SampleIndex(
        shard_id=shard.shard_id,
        sample_index=0,
        context_start=0,
        context_end=167,
        horizon_start=168,
        horizon_end=191,
        context_length=168,
        horizon=24,
        sample_metadata={"season_length": 7},
    )

    assert _mase_period_for_sample(shard, sample) == 7


def test_mase_absent_when_history_is_flat():
    """Flat history → scale == 0 → MASE undefined → key must be absent."""
    target_history = [[5.0], [5.0], [5.0]]
    target_future  = [[5.0], [6.0]]
    forecast       = [[5.0], [5.0]]

    result = compute_sample_metrics(target_future, forecast, target_history=target_history)

    assert "mse" in result
    assert "mae" in result
    assert "mase" not in result


def test_mase_absent_when_history_has_single_row():
    """Single-row history → no consecutive diffs → MASE undefined → key must be absent."""
    target_history = [[10.0]]
    target_future  = [[12.0]]
    forecast       = [[11.0]]

    result = compute_sample_metrics(target_future, forecast, target_history=target_history)

    assert "mase" not in result


def test_backward_compat_no_history():
    """Calling without target_history must return only mse and mae (no mase key)."""
    target_future = [[1.0], [3.0], [5.0]]
    forecast      = [[2.0], [1.0], [5.0]]

    result = compute_sample_metrics(target_future, forecast)

    assert "mse" in result
    assert "mae" in result
    assert "mase" not in result
