from pathlib import Path

from app.services.csv_dataset_reader import CsvDatasetReader
from app.services.dataset_load_service import build_windows
from app.services.sample_store import SampleStore


def test_sample_store_materializes_expected_jsonl_schema(tmp_path):
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    read_result = CsvDatasetReader().read(path, "time", ["target"])
    windows = build_windows(read_result.row_count, context_length=6, horizon=3, stride=3)

    samples = SampleStore(tmp_path).write_samples("shard-1", read_result, windows[:1], ["target"])
    record = SampleStore(tmp_path).read_by_ref(samples[0].materialized_sample_uri, samples[0].storage_ref)

    assert record["schema_version"] == "sample.v1"
    assert record["sample_id"] == samples[0].sample_id
    assert record["shard_id"] == "shard-1"
    assert record["target_column_names"] == ["target"]
    assert len(record["target_history"]) == 6
    assert all(len(row) == 1 for row in record["target_history"])
    assert len(record["target_future"]) == 3
    assert all(len(row) == 1 for row in record["target_future"])
    assert len(record["history_timestamps"]) == 6
    assert len(record["future_timestamps"]) == 3
    assert record["history_cov"] == []
    assert record["future_cov"] == []
    assert record["source_row_start"] == 0
    assert record["source_row_end"] == 8
