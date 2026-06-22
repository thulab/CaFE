"""CSV reader only reads the selected target column for the single-variable MVP."""
import pytest

from app.core.errors import ApiError
from app.services.csv_dataset_reader import CsvDatasetReader
from tests.fixtures.csv_factory import write_csv


def test_reads_selected_target_column_only(tmp_path):
    path = write_csv(
        tmp_path,
        "time,target,extra\n"
        "2026-01-01 00:00:00,100.0,20.0\n"
        "2026-01-01 01:00:00,101.5,20.4\n"
        "2026-01-01 02:00:00,103.0,20.8\n",
    )

    result = CsvDatasetReader().read(path, time_column="time", target_columns=["target"])

    assert result.target_columns == ["target"]
    assert result.values == [[100.0], [101.5], [103.0]]
    assert result.column_matrix(["target"]) == [[100.0], [101.5], [103.0]]


def test_non_target_columns_are_not_parsed_or_validated(tmp_path):
    path = write_csv(
        tmp_path,
        "time,target,extra\n2026-01-01 00:00:00,1,nope\n2026-01-01 01:00:00,3,NaN\n",
    )

    result = CsvDatasetReader().read(path, time_column="time", target_columns=["target"])

    assert result.target_columns == ["target"]
    assert result.values == [[1.0], [3.0]]


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("time,target\n2026-01-01 00:00:00,\n2026-01-01 01:00:00,2\n", "csv_value_missing"),
        ("time,target\n2026-01-01 00:00:00,nope\n2026-01-01 01:00:00,2\n", "csv_value_not_float"),
        ("time,target\n2026-01-01 00:00:00,NaN\n2026-01-01 01:00:00,2\n", "csv_value_not_finite"),
    ],
)
def test_target_value_validation_errors(tmp_path, content, code):
    path = write_csv(tmp_path, content)

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, time_column="time", target_columns=["target"])

    assert exc.value.error_code == code
    assert exc.value.details.get("column") == "target"
