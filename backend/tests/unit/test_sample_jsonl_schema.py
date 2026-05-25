from pathlib import Path

from app.services.csv_dataset_reader import CsvDatasetReader
from app.services.dataset_load_service import build_windows
from app.services.sample_store import SampleStore
from app.services.tsfile_store import TsFileStore


def test_sample_store_reconstructs_expected_sample_v1_schema(tmp_path):
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    read_result = CsvDatasetReader().read(path, "time", ["target"])
    windows = build_windows(read_result.row_count, context_length=6, horizon=3, stride=3)
    dataset_id = "ds1"
    tsfile_path = TsFileStore(tmp_path).write(
        "shard-1", dataset_id, read_result.timestamps, read_result.value_columns, read_result.values
    )

    samples = SampleStore().write_samples("shard-1", dataset_id, tsfile_path, windows[:1], ["target"])
    record = SampleStore().read_by_ref(samples[0].materialized_sample_uri, samples[0].storage_ref)

    assert record["schema_version"] == "sample.v1"
    assert record["sample_id"] == samples[0].sample_id
    assert record["shard_id"] == "shard-1"
    assert record["target_column_names"] == ["target"]
    assert record["target_history"] == [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]
    assert record["target_future"] == [[16.0], [17.0], [18.0]]
    assert len(record["history_timestamps"]) == 6
    assert len(record["future_timestamps"]) == 3
    assert record["history_cov"] == []
    assert record["future_cov"] == []
    assert record["source_row_start"] == 0
    assert record["source_row_end"] == 8
