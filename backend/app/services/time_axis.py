"""Reader-agnostic validation of a dataset's time axis.

Centralises the time-axis checks that every dataset reader shares so the rules
stay in one place and the error codes stay stable across readers:

- timezone consistency (all naive or all aware; no mix)
- strictly increasing timestamps (no going backwards)
- no duplicate timestamps
- equally spaced timestamps (equidistant)
- frequency inference from the common interval
- declared-frequency cross-check by *duration* (so "60m" matches inferred "1h")
"""

from datetime import datetime

from app.core.errors import ApiError


def validate_time_axis(timestamps: list[datetime], declared_freq: str | None) -> str:
    """Validate a time axis and return its inferred canonical frequency.

    Args:
        timestamps: parsed timestamps in row order.
        declared_freq: frequency declared by the caller/manifest, or ``None``.
            Compared against the inferred frequency by duration, not by string.

    Returns:
        The inferred canonical frequency string (e.g. ``"1h"``, ``"7d"``).

    Raises:
        ApiError: with a stable ``error_code`` for any axis violation.
    """
    _check_timezone_consistency(timestamps)
    _check_monotonic_unique(timestamps)
    inferred_frequency = _infer_frequency(timestamps)

    if declared_freq is not None:
        declared_seconds = _frequency_to_seconds(declared_freq)
        inferred_seconds = _frequency_to_seconds(inferred_frequency)
        if declared_seconds != inferred_seconds:
            raise ApiError(
                "csv_frequency_mismatch",
                "provided frequency does not match inferred frequency",
                {"provided": declared_freq, "inferred": inferred_frequency},
            )

    return inferred_frequency


def _check_timezone_consistency(timestamps: list[datetime]) -> None:
    if not timestamps:
        return
    aware = [timestamp.tzinfo is not None for timestamp in timestamps]
    if any(aware) and not all(aware):
        raise ApiError(
            "csv_mixed_timezone",
            "time_column must not mix timezone-aware and naive timestamps",
        )


def _check_monotonic_unique(timestamps: list[datetime]) -> None:
    seen: set[datetime] = set()
    previous: datetime | None = None
    for index, timestamp in enumerate(timestamps):
        # Row index in the original file is 1-based with a header row, so the
        # first data row is row 2.
        row_index = index + 2
        if timestamp in seen:
            raise ApiError(
                "csv_duplicate_timestamp",
                "time_column must not contain duplicate timestamps",
                {"row_index": row_index},
            )
        if previous is not None and timestamp < previous:
            raise ApiError(
                "csv_time_not_monotonic",
                "time_column must be strictly increasing",
                {"row_index": row_index},
            )
        seen.add(timestamp)
        previous = timestamp


def _infer_frequency(timestamps: list[datetime]) -> str:
    if len(timestamps) < 2:
        raise ApiError("csv_frequency_not_inferable", "at least two timestamps are required")
    intervals = [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))]
    first = intervals[0]
    if any(interval != first for interval in intervals):
        raise ApiError("csv_time_not_equidistant", "time_column must be equally spaced")
    total_seconds = int(first.total_seconds())
    if total_seconds <= 0:
        raise ApiError("csv_time_not_monotonic", "time_column must be strictly increasing")
    return _seconds_to_frequency(total_seconds)


def _seconds_to_frequency(total_seconds: int) -> str:
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"


_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def _frequency_to_seconds(frequency: str) -> int:
    """Parse a frequency string (e.g. ``"60m"``, ``"1H"``) into seconds.

    Raises:
        ApiError: ``csv_frequency_unparseable`` when the string cannot be parsed.
    """
    text = frequency.strip().lower()
    if not text:
        raise ApiError("csv_frequency_unparseable", "frequency string is empty", {"provided": frequency})
    unit = text[-1]
    if unit not in _UNIT_SECONDS:
        raise ApiError(
            "csv_frequency_unparseable",
            "frequency unit is not supported",
            {"provided": frequency},
        )
    magnitude_text = text[:-1] or "1"
    try:
        magnitude = int(magnitude_text)
    except ValueError as exc:
        raise ApiError(
            "csv_frequency_unparseable",
            "frequency magnitude must be an integer",
            {"provided": frequency},
        ) from exc
    if magnitude <= 0:
        raise ApiError(
            "csv_frequency_unparseable",
            "frequency magnitude must be positive",
            {"provided": frequency},
        )
    return magnitude * _UNIT_SECONDS[unit]
