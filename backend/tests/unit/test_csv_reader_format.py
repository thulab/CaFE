import pytest

from app.core.errors import ApiError
from app.services.csv_dataset_reader import CsvDatasetReader
from tests.fixtures.csv_factory import write_csv


@pytest.mark.parametrize("delimiter", [",", "\t", ";"])
def test_supported_delimiters_are_detected(tmp_path, delimiter):
    path = write_csv(tmp_path, f"time{delimiter}target\n2026-01-01 00:00:00{delimiter}1\n2026-01-01 01:00:00{delimiter}2\n")

    result = CsvDatasetReader().read(path, "time", ["target"])

    assert result.delimiter == delimiter


def test_utf8_bom_is_supported(tmp_path):
    path = write_csv(tmp_path, "\ufefftime,target\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n")

    result = CsvDatasetReader().read(path, "time", ["target"])

    assert result.encoding == "utf-8-sig"
    assert result.columns == ["time", "target"]


@pytest.mark.parametrize(
    ("content", "time_column", "target_columns", "code"),
    [
        ("2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n", "time", ["target"], "csv_missing_header"),
        ("time,target,target\n2026-01-01 00:00:00,1,2\n2026-01-01 01:00:00,3,4\n", "time", ["target"], "csv_duplicate_columns"),
        ("timestamp,target\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n", "time", ["target"], "csv_time_column_missing"),
        ("time,value\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n", "time", ["target"], "csv_target_column_missing"),
    ],
)
def test_format_validation_errors(tmp_path, content, time_column, target_columns, code):
    path = write_csv(tmp_path, content)

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, time_column, target_columns)

    assert exc.value.error_code == code


def test_mixed_timezone_raises_api_error_not_typeerror(tmp_path):
    # First row naive, second row carries a +08:00 offset. Comparing aware vs
    # naive datetimes would raise TypeError and bypass the error envelope; the
    # reader must detect the mix and raise a controlled ApiError instead.
    path = write_csv(
        tmp_path,
        "time,target\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00+08:00,2\n",
    )

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, "time", ["target"])

    assert exc.value.error_code == "csv_mixed_timezone"
