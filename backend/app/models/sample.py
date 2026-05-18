from datetime import datetime
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class SampleIndex(SQLModel, table=True):
    sample_id: str = Field(default_factory=new_id, primary_key=True)
    shard_id: str = Field(index=True)
    sample_index: int
    context_start: int
    context_end: int
    horizon_start: int
    horizon_end: int
    target_columns: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    context_length: int = 0
    horizon: int = 0
    storage_ref: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    materialized: bool = True
    materialized_sample_uri: str | None = None
    checksum: str | None = None
    sample_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
