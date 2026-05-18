import pytest

from app.core.errors import ApiError
from app.services.dataset_load_service import build_windows


def test_windowing_uses_stride_until_future_would_overrun():
    windows = build_windows(row_count=20, context_length=6, horizon=3, stride=3)

    assert [window.source_row_start for window in windows] == [0, 3, 6, 9]
    assert [(window.context_start, window.context_end, window.horizon_start, window.horizon_end) for window in windows] == [
        (0, 5, 6, 8),
        (3, 8, 9, 11),
        (6, 11, 12, 14),
        (9, 14, 15, 17),
    ]


def test_windowing_fails_when_split_exceeds_row_count():
    with pytest.raises(ApiError) as exc:
        build_windows(row_count=8, context_length=6, horizon=3, stride=3)

    assert exc.value.error_code == "split_length_exceeds_rows"
