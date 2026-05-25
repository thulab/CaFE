"""全列数值摄入（plan Task B2.1）：reader 不再强制单目标，所有数值列都被摄入并校验。"""
import pytest

from app.core.errors import ApiError
from app.services.csv_dataset_reader import CsvDatasetReader
from tests.fixtures.csv_factory import write_csv


def test_reads_all_value_columns_by_default(tmp_path):
    path = write_csv(
        tmp_path,
        "time,target,extra\n"
        "2026-01-01 00:00:00,100.0,20.0\n"
        "2026-01-01 01:00:00,101.5,20.4\n"
        "2026-01-01 02:00:00,103.0,20.8\n",
    )

    result = CsvDatasetReader().read(path, time_column="time")

    assert result.value_columns == ["target", "extra"]
    assert result.values == [[100.0, 20.0], [101.5, 20.4], [103.0, 20.8]]
    assert result.column_matrix(["extra"]) == [[20.0], [20.4], [20.8]]


def test_explicit_value_columns_subset(tmp_path):
    path = write_csv(
        tmp_path,
        "time,target,extra\n2026-01-01 00:00:00,1,2\n2026-01-01 01:00:00,3,4\n",
    )

    result = CsvDatasetReader().read(path, time_column="time", value_columns=["extra"])

    assert result.value_columns == ["extra"]
    assert result.values == [[2.0], [4.0]]


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("time,target,extra\n2026-01-01 00:00:00,1,\n2026-01-01 01:00:00,3,4\n", "csv_value_missing"),
        ("time,target,extra\n2026-01-01 00:00:00,1,nope\n2026-01-01 01:00:00,3,4\n", "csv_value_not_float"),
        ("time,target,extra\n2026-01-01 00:00:00,1,NaN\n2026-01-01 01:00:00,3,4\n", "csv_value_not_finite"),
    ],
)
def test_any_value_column_failing_is_reported(tmp_path, content, code):
    path = write_csv(tmp_path, content)

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, time_column="time")

    assert exc.value.error_code == code
    assert exc.value.details.get("column") == "extra"
