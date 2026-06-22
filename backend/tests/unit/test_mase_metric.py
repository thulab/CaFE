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
from app.services.metric_service import compute_sample_metrics


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
