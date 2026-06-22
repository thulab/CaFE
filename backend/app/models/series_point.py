from datetime import datetime
from typing import Any

from sqlalchemy import Column, Index, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class SeriesPoint(SQLModel, table=True):
    """per-shard 原始序列的逐点行存储（SQLite 单一真值源）。

    一行 = 一个时间点；`values_json` 为该点目标列的 {列名: 值}，自描述。
    样本读取按 `(shard_id, row_index)` 范围查询。
    """

    series_point_id: str = Field(default_factory=new_id, primary_key=True)
    shard_id: str = Field(index=True)
    row_index: int
    ts: str  # ISO 8601 原文（保留原始 offset，不经 ms-epoch 往返）
    values_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)

    __table_args__ = (Index("ix_seriespoint_shard_row", "shard_id", "row_index"),)
