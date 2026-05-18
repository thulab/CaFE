import pytest

from app.core.errors import ApiError
from app.services.csv_dataset_reader import CsvDatasetReader
from tests.fixtures.csv_factory import write_csv


@pytest.mark.parametrize(
    "time_value",
    ["2026-01-01", "2026-01-01 00:00:00", "2026-01-01T00:00:00"],
)
def test_supported_time_formats(tmp_path, time_value):
    path = write_csv(tmp_path, f"time,target\n{time_value},1\n2026-01-02 00:00:00,2\n")

    result = CsvDatasetReader().read(path, "time", ["target"])

    assert result.row_count == 2


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("time,target\nnot-a-date,1\n2026-01-01 01:00:00,2\n", "csv_time_parse_failed"),
        ("time,target\n2026-01-01 00:00:00,1\n2026-01-01 00:00:00,2\n", "csv_duplicate_timestamp"),
        ("time,target\n2026-01-01 01:00:00,1\n2026-01-01 00:00:00,2\n", "csv_time_not_monotonic"),
        ("time,target\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n2026-01-01 03:00:00,3\n", "csv_time_not_equidistant"),
    ],
)
def test_time_validation_errors(tmp_path, content, code):
    path = write_csv(tmp_path, content)

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, "time", ["target"])

    assert exc.value.error_code == code
