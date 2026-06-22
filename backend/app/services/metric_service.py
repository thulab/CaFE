from collections.abc import Iterable

# Reason codes for an undefined MASE on a sample.
MASE_REASON_FLAT_HISTORY = "flat_history"  # in-sample naive MAE scale == 0
MASE_REASON_NO_HISTORY_DIFFS = "no_history_diffs"  # < 2 history rows → no diffs


class SampleMetrics(dict):
    """A sample's metrics dict that also carries an out-of-band MASE reason.

    The numeric metrics (``mse``/``mae``/``mase`` when defined) live as normal
    dict items, so the existing per-key persistence loop that iterates
    ``metrics.items()`` and writes one float ``MetricResult`` per key keeps
    seeing only floats. When MASE is undefined we record WHY on the
    ``mase_unavailable_reason`` ATTRIBUTE (not as a dict item), making the
    absence visible to reporting without ever offering a non-float metric to
    persist.
    """

    mase_unavailable_reason: str | None = None


def _flatten(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]


def _mase_scale(target_history: list[list[float]]) -> tuple[float | None, str | None]:
    """Return the naive (last-value, m=1) in-sample MAE scale for MASE.

    Iterates each dimension column independently and collects all
    consecutive absolute differences |h[t][d] - h[t-1][d]|, then
    returns their mean alongside a reason when the scale is undefined.

    Returns ``(scale, None)`` for a usable positive scale. Returns
    ``(None, reason)`` when MASE is undefined: ``no_history_diffs`` when the
    history has fewer than 2 rows (no diffs possible) and ``flat_history``
    when the computed mean is 0 (flat / stationary history). The reason lets
    the caller make the absence VISIBLE instead of dividing by zero and
    silently dropping the metric.
    """
    if len(target_history) < 2:
        return None, MASE_REASON_NO_HISTORY_DIFFS

    n_dims = len(target_history[0])
    abs_diffs: list[float] = []
    for d in range(n_dims):
        for t in range(1, len(target_history)):
            abs_diffs.append(abs(target_history[t][d] - target_history[t - 1][d]))

    if not abs_diffs:
        return None, MASE_REASON_NO_HISTORY_DIFFS

    scale = sum(abs_diffs) / len(abs_diffs)
    if scale > 0:
        return scale, None
    return None, MASE_REASON_FLAT_HISTORY


def compute_sample_metrics(
    target_future: list[list[float]],
    forecast: list[list[float]],
    target_history: list[list[float]] | None = None,
) -> dict[str, float]:
    expected = _flatten(target_future)
    predicted = _flatten(forecast)
    if len(expected) != len(predicted):
        raise ValueError("target_future and forecast must have the same flattened length")
    errors = [prediction - target for prediction, target in zip(predicted, expected, strict=True)]
    mae = sum(abs(error) for error in errors) / len(errors)
    result = SampleMetrics(
        {
            "mse": sum(error * error for error in errors) / len(errors),
            "mae": mae,
        }
    )

    if target_history is not None:
        scale, reason = _mase_scale(target_history)
        if scale is not None:
            result["mase"] = mae / scale
        else:
            # MASE is undefined. We intentionally do NOT add a "mase" key (so
            # result.get("mase") is None, and the per-key float persistence loop
            # never tries to store None). Instead we record WHY on an attribute
            # so reporting can surface the absence rather than hiding it.
            result.mase_unavailable_reason = reason

    return result


def mase_unavailable_reason(result: dict) -> str | None:
    """Return why a sample has no MASE, or None when MASE is available.

    ``compute_sample_metrics`` records the reason on the ``SampleMetrics``
    attribute when scale is undefined; plain dicts (no attribute) read as None.
    """
    return getattr(result, "mase_unavailable_reason", None)


def aggregate_metric(results: Iterable[dict[str, float] | None], metric_name: str) -> dict[str, float | int] | None:
    values: list[float] = []
    failure_count = 0
    for result in results:
        if result is None or metric_name not in result:
            failure_count += 1
            continue
        values.append(float(result[metric_name]))
    if not values:
        return None
    return {
        "value": sum(values) / len(values),
        "success_count": len(values),
        "failure_count": failure_count,
    }


def ranking_metric_for_unit(unit_status: str, unit_metrics: dict[str, float], metric_name: str) -> float | None:
    if unit_status != "succeeded":
        return None
    return unit_metrics.get(metric_name)
