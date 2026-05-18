import pytest

from app.core.errors import ApiError
from app.services.csv_dataset_reader import CsvDatasetReader
from tests.fixtures.csv_factory import write_csv


def test_explicit_frequency_must_match_inferred_interval(tmp_path):
    path = write_csv(tmp_path, "time,target\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n")

    with pytest.raises(ApiError) as exc:
        CsvDatasetReader().read(path, "time", ["target"], frequency="1d")

    assert exc.value.error_code == "csv_frequency_mismatch"
