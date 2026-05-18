from collections.abc import Iterable


def _flatten(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]


def compute_sample_metrics(target_future: list[list[float]], forecast: list[list[float]]) -> dict[str, float]:
    expected = _flatten(target_future)
    predicted = _flatten(forecast)
    if len(expected) != len(predicted):
        raise ValueError("target_future and forecast must have the same flattened length")
    errors = [prediction - target for prediction, target in zip(predicted, expected, strict=True)]
    return {
        "mse": sum(error * error for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
    }


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
