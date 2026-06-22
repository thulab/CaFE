from datetime import datetime

from sqlmodel import Session, SQLModel

from app.core.config import get_settings
from app.db.session import create_db_engine
from app.models.series_point import SeriesPoint  # noqa: F401  (注册建表)
from app.services.series_store import SeriesStore


def _session() -> Session:
    get_settings().runtime_dir.mkdir(parents=True, exist_ok=True)  # 测试隔离 runtime 目录需先建
    engine = create_db_engine()
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_write_then_slice_inclusive_range():
    timestamps = [datetime(2026, 1, 1, hour) for hour in range(5)]
    values = [[100.0 + i, 20.0 + i] for i in range(5)]  # [N=5, C=2]
    with _session() as session:
        SeriesStore().write(session, "shard-1", timestamps, ["target", "extra"], values)

        # 闭区间 [1,3] → 行 1,2,3
        target = SeriesStore().slice(session, "shard-1", ["target"], 1, 3)
        assert target == [[101.0], [102.0], [103.0]]

        both = SeriesStore().slice(session, "shard-1", ["target", "extra"], 0, 1)
        assert both == [[100.0, 20.0], [101.0, 21.0]]

        ts = SeriesStore().slice_timestamps(session, "shard-1", 1, 3)
        assert ts == ["2026-01-01T01:00:00", "2026-01-01T02:00:00", "2026-01-01T03:00:00"]


def test_slice_isolated_by_shard():
    ts2 = [datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 1)]
    with _session() as session:
        SeriesStore().write(session, "shard-A", ts2, ["v"], [[1.0], [2.0]])
        SeriesStore().write(session, "shard-B", ts2, ["v"], [[9.0], [8.0]])
        assert SeriesStore().slice(session, "shard-A", ["v"], 0, 1) == [[1.0], [2.0]]
        assert SeriesStore().slice(session, "shard-B", ["v"], 0, 1) == [[9.0], [8.0]]
