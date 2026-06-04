from datetime import datetime
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class CapabilityBlock(SQLModel, table=True):
    capability_block_id: str = Field(default_factory=new_id, primary_key=True)
    track_id: str | None = Field(default=None, index=True)
    block_type: str = "real"
    capability_type: str = "real_data"
    name: str
    task_type: str = "univariate_forecast"
    target_dim: int = 1
    covariate_dim: int = 0
    shard_count: int = 0
    sample_count: int = 0
    aggregation_policy: str = "mean_over_shards"
    status: str = "ready"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CapabilityBlockShard(SQLModel, table=True):
    capability_block_id: str = Field(primary_key=True)
    shard_id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)


class Track(SQLModel, table=True):
    track_id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    track_type: str = "real_dataset"
    description: str | None = None
    primary_metric_id: str = "mase"
    default_ranking_policy: str = "latest_valid_result"
    benchmark_version: str = "mvp"
    data_version: str = "v1"
    status: str = "ready"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class BenchmarkingRun(SQLModel, table=True):
    benchmarking_run_id: str = Field(default_factory=new_id, primary_key=True)
    track_id: str = Field(index=True)
    model_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    benchmark_version: str = "mvp"
    data_version: str = "v1"
    status: str = "created"
    execution_mode: str = "background_thread"
    cancel_requested: bool = False
    cancel_requested_at: datetime | None = None
    model_count: int = 0
    task_count: int = 0
    sample_count: int = 0
    metric_set: list[str] = Field(default_factory=lambda: ["mase", "mse", "mae"], sa_column=Column(JSON))
    report_id: str | None = None
    ranking_list_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Unit(SQLModel, table=True):
    unit_id: str = Field(default_factory=new_id, primary_key=True)
    benchmarking_run_id: str = Field(index=True)
    model_id: str = Field(index=True)
    status: str = "created"
    task_count: int = 0
    sample_count: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Task(SQLModel, table=True):
    task_id: str = Field(default_factory=new_id, primary_key=True)
    benchmarking_run_id: str = Field(index=True)
    unit_id: str = Field(index=True)
    model_id: str = Field(index=True)
    capability_block_id: str = Field(index=True)
    status: str = "created"
    shard_count: int = 0
    sample_count: int = 0
    processed_sample_count: int = 0
    failed_sample_count: int = 0
    aggregation_policy: str = "mean_over_shards"
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ForecastArtifact(SQLModel, table=True):
    forecast_artifact_id: str = Field(default_factory=new_id, primary_key=True)
    benchmarking_run_id: str = Field(index=True)
    unit_id: str = Field(index=True)
    task_id: str = Field(index=True)
    model_id: str = Field(index=True)
    shard_id: str = Field(index=True)
    storage_uri: str
    schema_version: str = "forecast.v1"
    sample_count: int = 0
    checksum: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RunEvent(SQLModel, table=True):
    run_event_id: str = Field(default_factory=new_id, primary_key=True)
    benchmarking_run_id: str = Field(index=True)
    unit_id: str | None = Field(default=None, index=True)
    task_id: str | None = Field(default=None, index=True)
    level: str = "info"
    event_type: str = "status_changed"
    message: str
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
