import re
import math
from collections.abc import Iterable

import numpy as np

# Reason codes for an undefined MASE on a sample.
MASE_REASON_FLAT_HISTORY = "flat_history"  # in-sample naive MAE scale == 0
MASE_REASON_NO_HISTORY_DIFFS = "no_history_diffs"  # history length <= seasonal period


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


def seasonal_period_for_frequency(frequency: str | None) -> int:
    """Return the default seasonal-naive period for an equidistant frequency.

    Sub-daily data use a daily period, daily data use a weekly period, and
    weekly data use an annual period. Frequencies without an unambiguous
    calendar season fall back to the non-seasonal lag-1 baseline.
    """
    text = (frequency or "").strip().lower()
    aliases = {
        "hour": "1h",
        "hourly": "1h",
        "day": "1d",
        "daily": "1d",
        "week": "1w",
        "weekly": "1w",
        "minute": "1m",
        "minutely": "1m",
    }
    text = aliases.get(text, text)
    match = re.fullmatch(r"(\d+)?\s*(s|m|min|h|d|w)", text)
    if match is None:
        return 1
    magnitude = int(match.group(1) or 1)
    if magnitude <= 0:
        return 1
    unit = match.group(2)
    unit_seconds = {"s": 1, "m": 60, "min": 60, "h": 3600, "d": 86400, "w": 604800}
    interval_seconds = magnitude * unit_seconds[unit]
    day_seconds = 86400
    week_seconds = 7 * day_seconds
    if interval_seconds < day_seconds and day_seconds % interval_seconds == 0:
        return day_seconds // interval_seconds
    if interval_seconds == day_seconds:
        return 7
    if interval_seconds == week_seconds:
        return 52
    return 1


def safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            delta_left = np.sign(left[first] - left[second])
            delta_right = np.sign(right[first] - right[second])
            if delta_left == 0 and delta_right == 0:
                continue
            if delta_left == 0:
                ties_left += 1
            elif delta_right == 0:
                ties_right += 1
            elif delta_left == delta_right:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    return float((concordant - discordant) / denominator) if denominator else 0.0


def resolve_mase_period(
    *,
    explicit_period: int | None,
    frequency: str | None,
    history_length: int,
) -> int:
    """Resolve the period actually usable by a sample's MASE denominator.

    Synthetic samples provide their calibrated ``season_length`` explicitly;
    real shards normally derive it from frequency. A history must contain at
    least one seasonal difference (``len(history) > P``). For legacy or tiny
    platform samples that cannot support their calendar period, use lag 1
    instead of silently dropping the primary metric. Paper experiments use
    contexts longer than ``P`` and therefore never take this fallback.
    """
    if explicit_period is not None:
        period = int(explicit_period)
        if period <= 0:
            raise ValueError("seasonal_period must be positive")
    else:
        period = seasonal_period_for_frequency(frequency)
    if history_length <= period:
        return 1
    return period


def _mase_scale(
    target_history: list[list[float]],
    seasonal_period: int = 1,
) -> tuple[float | None, str | None]:
    """Return the seasonal-naive in-sample MAE scale for MASE(P).

    Iterates each dimension column independently and collects all
    lag-P absolute differences |h[t][d] - h[t-P][d]|, then
    returns their mean alongside a reason when the scale is undefined.

    Returns ``(scale, None)`` for a usable positive scale. Returns
    ``(None, reason)`` when MASE is undefined: ``no_history_diffs`` when the
    history is not longer than ``seasonal_period`` and ``flat_history`` when
    the computed mean is 0 (flat / stationary history). The reason lets
    the caller make the absence VISIBLE instead of dividing by zero and
    silently dropping the metric.
    """
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")
    if len(target_history) <= seasonal_period:
        return None, MASE_REASON_NO_HISTORY_DIFFS

    n_dims = len(target_history[0])
    abs_diffs: list[float] = []
    for d in range(n_dims):
        for t in range(seasonal_period, len(target_history)):
            abs_diffs.append(abs(target_history[t][d] - target_history[t - seasonal_period][d]))

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
    *,
    seasonal_period: int = 1,
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
        scale, reason = _mase_scale(target_history, seasonal_period)
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
