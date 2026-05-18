from pathlib import Path

from app.services.csv_dataset_reader import CsvDatasetReader


def test_csv_reader_reads_valid_hourly_csv():
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"

    result = CsvDatasetReader().read(path, time_column="time", target_columns=["target"])

    assert result.row_count == 20
    assert result.frequency == "1h"
    assert result.encoding == "utf-8"
    assert result.delimiter == ","
    assert result.columns == ["time", "target", "extra"]
    assert result.target_values[0] == [10.0]
    assert result.target_values[-1] == [29.0]
