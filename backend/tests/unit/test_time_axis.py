from datetime import datetime, timedelta, timezone

import pytest

from app.core.errors import ApiError
from app.services.time_axis import validate_time_axis


def _hourly(count: int, *, tz: timezone | None = None) -> list[datetime]:
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
    return [base + timedelta(hours=index) for index in range(count)]


def test_infers_frequency_for_equidistant_axis():
    assert validate_time_axis(_hourly(3), None) == "1h"


def test_returns_declared_frequency_when_matching_by_duration():
    # "60m" and inferred "1h" are the same duration (3600s) -> must accept.
    assert validate_time_axis(_hourly(3), "60m") == "1h"


def test_daily_and_weekly_inference():
    base = datetime(2026, 1, 1, 0, 0, 0)
    daily = [base + timedelta(days=index) for index in range(3)]
    weekly = [base + timedelta(weeks=index) for index in range(3)]
    assert validate_time_axis(daily, None) == "1d"
    assert validate_time_axis(weekly, None) == "7d"


def test_rejects_non_monotonic_axis():
    base = datetime(2026, 1, 1, 0, 0, 0)
    axis = [base, base + timedelta(hours=2), base + timedelta(hours=1)]
    with pytest.raises(ApiError) as exc:
        validate_time_axis(axis, None)
    assert exc.value.error_code == "csv_time_not_monotonic"


def test_rejects_duplicate_timestamps():
    base = datetime(2026, 1, 1, 0, 0, 0)
    axis = [base, base + timedelta(hours=1), base + timedelta(hours=1)]
    with pytest.raises(ApiError) as exc:
        validate_time_axis(axis, None)
    assert exc.value.error_code == "csv_duplicate_timestamp"


def test_rejects_non_equidistant_axis():
    base = datetime(2026, 1, 1, 0, 0, 0)
    axis = [base, base + timedelta(hours=1), base + timedelta(hours=3)]
    with pytest.raises(ApiError) as exc:
        validate_time_axis(axis, None)
    assert exc.value.error_code == "csv_time_not_equidistant"


def test_rejects_frequency_mismatch_by_duration():
    with pytest.raises(ApiError) as exc:
        validate_time_axis(_hourly(3), "1d")
    assert exc.value.error_code == "csv_frequency_mismatch"


def test_rejects_mixed_timezone_axis():
    base_naive = datetime(2026, 1, 1, 0, 0, 0)
    base_aware = datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    with pytest.raises(ApiError) as exc:
        validate_time_axis([base_naive, base_aware], None)
    assert exc.value.error_code == "csv_mixed_timezone"


def test_aware_axis_is_supported():
    aware = _hourly(3, tz=timezone(timedelta(hours=8)))
    assert validate_time_axis(aware, "60m") == "1h"
