from pathlib import Path

from tests.fixtures.csv_factory import write_csv

from app.services.csv_dataset_reader import CsvDatasetReader


def test_csv_reader_reads_valid_hourly_csv():
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"

    result = CsvDatasetReader().read(path, time_column="time", target_columns=["target"])

    assert result.row_count == 20
    assert result.frequency == "1h"
    assert result.encoding == "utf-8"
    assert result.delimiter == ","
    assert result.columns == ["time", "target", "extra"]
    assert result.target_columns == ["target"]
    assert result.column_matrix(["target"])[0] == [10.0]
    assert result.column_matrix(["target"])[-1] == [29.0]


def test_declared_frequency_matching_by_duration_is_accepted(tmp_path):
    # Manifest declares "60m" while the data is hourly. They share the same
    # duration (3600s), so the load must succeed instead of reporting
    # csv_frequency_mismatch on a string inequality.
    path = write_csv(
        tmp_path,
        "time,target\n2026-01-01 00:00:00,1\n2026-01-01 01:00:00,2\n2026-01-01 02:00:00,3\n",
    )

    result = CsvDatasetReader().read(path, "time", ["target"], frequency="60m")

    assert result.frequency == "1h"
