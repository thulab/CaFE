import pytest

from app.core.errors import ApiError
from app.services.csv_dataset_reader import CsvDatasetReader
from tests.fixtures.csv_factory import write_csv


def test_string_values_are_converted_to_float(tmp_path):
    path = write_csv(tmp_path, 'time,target\n2026-01-01 00:00:00,"1.25"\n2026-01-01 01:00:00,"2.5"\n')

    result = CsvDatasetReader().read(path, "time", ["target"])

    assert result.column_matrix(["target"]) == [[1.25], [2.5]]


def test_multiple_target_columns_are_rejected(tmp_path):
    path = write_csv(tmp_path, "time,a,b\n2026-01-01 00:00:00,1,2\n2026-01-01 01:00:00,3,4\n")

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, "time", ["a", "b"])

    assert exc.value.error_code == "csv_target_columns_invalid"


@pytest.mark.parametrize(
    ("content", "target_columns", "code"),
    [
        ("time,target\n2026-01-01 00:00:00,\n2026-01-01 01:00:00,2\n", ["target"], "csv_value_missing"),
        ("time,target\n2026-01-01 00:00:00,nope\n2026-01-01 01:00:00,2\n", ["target"], "csv_value_not_float"),
        ("time,target\n2026-01-01 00:00:00,NaN\n2026-01-01 01:00:00,2\n", ["target"], "csv_value_not_finite"),
        ("time,target\n2026-01-01 00:00:00,Inf\n2026-01-01 01:00:00,2\n", ["target"], "csv_value_not_finite"),
    ],
)
def test_value_validation_errors(tmp_path, content, target_columns, code):
    path = write_csv(tmp_path, content)

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, "time", target_columns)

    assert exc.value.error_code == code
