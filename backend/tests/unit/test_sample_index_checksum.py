from pathlib import Path

from app.services.csv_dataset_reader import CsvDatasetReader
from app.services.dataset_load_service import build_windows
from app.services.sample_store import SampleStore


def test_checksum_is_content_based_and_cross_load_stable():
    """checksum 只对样本内容算、不含随机 sample_id/shard_id → 同内容跨加载相等（#7）。"""
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    read_result = CsvDatasetReader().read(path, "time", ["target"])
    windows = build_windows(read_result.row_count, context_length=6, horizon=3, stride=3)

    a = SampleStore().write_samples("shard-A", windows[:2], ["target"], read_result)
    b = SampleStore().write_samples("shard-B", windows[:2], ["target"], read_result)

    # 随机 sample_id 不同，但内容相同 → checksum 相同
    assert a[0].sample_id != b[0].sample_id
    assert a[0].checksum == b[0].checksum
    assert a[1].checksum == b[1].checksum
    # 不同窗口内容不同 → checksum 不同
    assert a[0].checksum != a[1].checksum


def test_storage_ref_records_row_ranges_not_jsonl_line():
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    read_result = CsvDatasetReader().read(path, "time", ["target"])
    windows = build_windows(read_result.row_count, context_length=6, horizon=3, stride=3)

    samples = SampleStore().write_samples("shard-1", windows[:2], ["target"], read_result)

    assert samples[0].materialized is False
    assert samples[0].storage_ref["context"] == [0, 5]
    assert samples[0].storage_ref["horizon"] == [6, 8]
    assert samples[1].storage_ref["context"] == [3, 8]
    assert samples[1].storage_ref["horizon"] == [9, 11]
