from pathlib import Path

from app.services.csv_dataset_reader import CsvDatasetReader
from app.services.dataset_load_service import build_windows
from app.services.sample_store import SampleStore, canonical_json_checksum
from app.services.tsfile_store import TsFileStore


def test_sample_index_checksum_matches_canonical_json_and_pointer_ref(tmp_path):
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    read_result = CsvDatasetReader().read(path, "time", ["target"])
    windows = build_windows(read_result.row_count, context_length=6, horizon=3, stride=3)
    dataset_id = "ds1"
    tsfile_path = TsFileStore(tmp_path).write(
        "shard-1", dataset_id, read_result.timestamps, read_result.value_columns, read_result.values
    )

    samples = SampleStore().write_samples("shard-1", dataset_id, tsfile_path, windows[:2], ["target"])
    first_record = SampleStore().read_by_ref(samples[0].materialized_sample_uri, samples[0].storage_ref)

    assert samples[0].checksum == canonical_json_checksum(first_record)
    # 指针化切片器：storage_ref 记录行号区间，而非 JSONL 行号。
    assert samples[0].storage_ref["context"] == [0, 5]
    assert samples[0].storage_ref["horizon"] == [6, 8]
    assert samples[1].storage_ref["context"] == [3, 8]
    assert samples[1].storage_ref["horizon"] == [9, 11]
