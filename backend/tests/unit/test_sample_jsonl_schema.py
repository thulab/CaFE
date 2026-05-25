from pathlib import Path

from sqlmodel import Session, SQLModel

from app.core.config import get_settings
from app.db.session import create_db_engine
from app.models.series_point import SeriesPoint  # noqa: F401  (注册建表)
from app.services.csv_dataset_reader import CsvDatasetReader
from app.services.dataset_load_service import build_windows
from app.services.sample_store import SampleStore
from app.services.series_store import SeriesStore


def test_sample_store_reconstructs_expected_sample_v1_schema():
    get_settings().runtime_dir.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    read_result = CsvDatasetReader().read(path, "time", ["target"])
    windows = build_windows(read_result.row_count, context_length=6, horizon=3, stride=3)

    engine = create_db_engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        SeriesStore().write(
            session, "shard-1", read_result.timestamps, read_result.value_columns, read_result.values
        )
        samples = SampleStore().write_samples("shard-1", windows[:1], ["target"], read_result)
        storage_ref = dict(samples[0].storage_ref)
        sample_id = samples[0].sample_id
        for sample in samples:
            session.add(sample)
        session.commit()

    # 用新 session 读（已提交的 SeriesPoint 可见）→ 重拼 sample.v1
    with Session(engine) as session:
        record = SampleStore().read_by_ref(session, storage_ref)

    assert record["schema_version"] == "sample.v1"
    assert record["sample_id"] == sample_id
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
