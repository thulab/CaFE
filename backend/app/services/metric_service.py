from collections.abc import Iterable


def _flatten(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]


def _mase_scale(target_history: list[list[float]]) -> float | None:
    """Return the naive (last-value, m=1) in-sample MAE scale for MASE.

    Iterates each dimension column independently and collects all
    consecutive absolute differences |h[t][d] - h[t-1][d]|, then
    returns their mean.

    Returns None when the history has fewer than 2 rows (no diffs
    possible) or when the computed mean is 0 (flat history), so the
    caller can skip the "mase" key rather than dividing by zero.
    """
    if len(target_history) < 2:
        return None

    n_dims = len(target_history[0])
    abs_diffs: list[float] = []
    for d in range(n_dims):
        for t in range(1, len(target_history)):
            abs_diffs.append(abs(target_history[t][d] - target_history[t - 1][d]))

    if not abs_diffs:
        return None

    scale = sum(abs_diffs) / len(abs_diffs)
    return scale if scale > 0 else None


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
    result: dict[str, float] = {
        "mse": sum(error * error for error in errors) / len(errors),
        "mae": mae,
    }

    if target_history is not None:
        scale = _mase_scale(target_history)
        if scale is not None:
            result["mase"] = mae / scale

    return result


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
