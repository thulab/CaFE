from pathlib import Path

from app.services.csv_dataset_reader import CsvDatasetReader
from app.services.dataset_load_service import build_windows
from app.services.sample_store import SampleStore, canonical_json_checksum


def test_sample_index_checksum_matches_canonical_json_and_line_ref(tmp_path):
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    read_result = CsvDatasetReader().read(path, "time", ["target"])
    windows = build_windows(read_result.row_count, context_length=6, horizon=3, stride=3)

    samples = SampleStore(tmp_path).write_samples("shard-1", read_result, windows[:2], ["target"])
    first_record = SampleStore(tmp_path).read_by_ref(samples[0].materialized_sample_uri, samples[0].storage_ref)

    assert samples[0].checksum == canonical_json_checksum(first_record)
    assert samples[0].storage_ref == {"line": 0}
    assert samples[1].storage_ref == {"line": 1}
