from datetime import datetime
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class DatasetManifest(SQLModel, table=True):
    dataset_manifest_id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    domain: str
    source_type: str = "managed_file"
    source_uri: str
    file_format: str = "csv"
    time_column: str
    frequency: str | None = None
    timezone: str | None = None
    schema_version: str = "dataset_manifest.v1"
    status: str = "ready_to_load"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DatasetLoadJob(SQLModel, table=True):
    load_job_id: str = Field(default_factory=new_id, primary_key=True)
    dataset_manifest_id: str = Field(index=True)
    status: str = "created"
    current_step: str | None = None
    reader_type: str = "csv_dataset_reader"
    reader_version: str = "1"
    validation_summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    split_config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    seed: int = 0
    error_code: str | None = None
    error_message: str | None = None
    output_shard_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Shard(SQLModel, table=True):
    shard_id: str = Field(default_factory=new_id, primary_key=True)
    name: str | None = None
    shard_type: str = "real"
    dataset_manifest_id: str = Field(index=True)
    load_job_id: str | None = Field(default=None, index=True)
    capability_block_id: str | None = Field(default=None, index=True)
    source_uri: str
    storage_uri: str | None = None
    checksum: str | None = None
    time_range_start: str | None = None
    time_range_end: str | None = None
    row_count: int = 0
    target_columns: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    target_dim: int = 1
    frequency: str | None = None
    context_length: int = 0
    horizon: int = 0
    stride: int = 0
    sample_count: int = 0
    status: str = "created"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
