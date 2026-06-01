import os
import time

import pandas as pd
import pytest
import tsfile
from sqlmodel import Session, SQLModel, select

from app.core.config import get_settings
from app.core.errors import ApiError
from app.db.session import create_db_engine
from app.models.dataset import DatasetManifest, Shard
from app.models.sample import SampleIndex  # noqa: F401  (注册建表)
from app.models.series_point import SeriesPoint
from app.services.dataset_load_service import DatasetLoadService
from app.services.tsfile_dataset_reader import TsFileDatasetReader

_BASE_MS = 1_700_000_000_000  # 任意 ms epoch 基准


def _write_tsfile(path, devices=("dev1",), n=8):
    rows = []
    for device in devices:
        for i in range(n):
            rows.append(
                {"time": _BASE_MS + i * 3600_000, "target": 100.0 + i, "extra": 20.0 + i, "dataset_id": device}
            )
    tsfile.dataframe_to_tsfile(
        pd.DataFrame(rows), str(path), table_name="tsbench", time_column="time", tag_column=["dataset_id"]
    )


def test_reads_single_device_tsfile_to_dataset_read_result(tmp_path):
    path = tmp_path / "in.tsfile"
    _write_tsfile(path, n=8)
    result = TsFileDatasetReader().read(path, "time", target_columns=["target"])
    assert result.target_columns == ["target"]
    assert result.row_count == 8
    assert result.frequency == "1h"
    assert result.column_matrix(["target"])[0] == [100.0]


def test_target_column_selects_and_orders(tmp_path):
    path = tmp_path / "in.tsfile"
    _write_tsfile(path, n=6)
    result = TsFileDatasetReader().read(path, "time", target_columns=["target"])
    assert result.target_columns == ["target"]
    assert len(result.values[0]) == 1


def test_multiple_devices_rejected(tmp_path):
    path = tmp_path / "multi.tsfile"
    _write_tsfile(path, devices=("dev1", "dev2"), n=4)
    with pytest.raises(ApiError) as exc:
        TsFileDatasetReader().read(path, "time", target_columns=["target"])
    assert exc.value.error_code == "tsfile_multiple_devices"


def test_full_series_path_selects_one_device_from_multi_device_tsfile(tmp_path):
    path = tmp_path / "multi.tsfile"
    _write_tsfile(path, devices=("dev1", "dev2"), n=4)

    result = TsFileDatasetReader().read(path, "time", target_columns=["tsbench.dev2.target"])

    assert result.target_columns == ["tsbench.dev2.target"]
    assert result.row_count == 4
    assert result.values[0] == [100.0]


def test_full_series_paths_must_share_one_device(tmp_path):
    path = tmp_path / "multi.tsfile"
    _write_tsfile(path, devices=("dev1", "dev2"), n=4)

    with pytest.raises(ApiError) as exc:
        TsFileDatasetReader().read(path, "time", target_columns=["tsbench.dev1.target", "tsbench.dev2.target"])

    assert exc.value.error_code == "tsfile_target_columns_invalid"


def test_tsfile_epoch_timestamps_are_not_converted_through_local_timezone(tmp_path, monkeypatch):
    path = tmp_path / "dst.tsfile"
    rows = [
        {"time": 527_011_200_000, "target": 1.0, "dataset_id": "dev1"},
        {"time": 527_014_800_000, "target": 2.0, "dataset_id": "dev1"},
    ]
    tsfile.dataframe_to_tsfile(
        pd.DataFrame(rows), str(path), table_name="tsbench", time_column="time", tag_column=["dataset_id"]
    )
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()

    try:
        result = TsFileDatasetReader().read(path, "time", target_columns=["target"])
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()

    assert result.frequency == "1h"
    assert [timestamp.isoformat() for timestamp in result.timestamps] == [
        "1986-09-13T16:00:00",
        "1986-09-13T17:00:00",
    ]


def test_tsfile_input_load_flow_stores_into_sqlite(tmp_path):
    path = tmp_path / "flow.tsfile"
    _write_tsfile(path, n=20)

    get_settings().runtime_dir.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        manifest = DatasetManifest(
            name="x", domain="d", source_uri=str(path), file_format="tsfile", time_column="time"
        )
        session.add(manifest)
        session.commit()
        session.refresh(manifest)

        job = DatasetLoadService(tmp_path).create_load_job(
            session,
            manifest.dataset_manifest_id,
            {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]},
        )
        assert job.status == "succeeded", job.error_message
        shard = session.get(Shard, job.output_shard_id)
        assert shard.target_columns == ["target"]

        rows = session.exec(select(SeriesPoint).where(SeriesPoint.shard_id == shard.shard_id)).all()
        assert len(rows) == 20
        assert rows[0].values_json["target"] == 100.0
        assert "extra" not in rows[0].values_json


def test_tsfile_load_flow_selects_full_series_target_without_extra_manifest_columns(tmp_path):
    path = tmp_path / "multi-flow.tsfile"
    _write_tsfile(path, devices=("dev1", "dev2"), n=20)

    get_settings().runtime_dir.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        manifest = DatasetManifest(
            name="x", domain="d", source_uri=str(path), file_format="tsfile", time_column="time"
        )
        session.add(manifest)
        session.commit()
        session.refresh(manifest)

        job = DatasetLoadService(tmp_path).create_load_job(
            session,
            manifest.dataset_manifest_id,
            {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["tsbench.dev2.target"]},
        )

        assert job.status == "succeeded", job.error_message
        shard = session.get(Shard, job.output_shard_id)
        assert shard.target_columns == ["tsbench.dev2.target"]
        assert "value_columns" not in shard.model_dump()
