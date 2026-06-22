from datetime import datetime

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class MetricDefinition(SQLModel, table=True):
    metric_id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    display_name: str
    direction: str = "lower_is_better"
    supported_levels: list[str] = Field(
        default_factory=lambda: ["sample", "shard", "task", "unit", "run", "ranking"],
        sa_column=Column(JSON),
    )
    status: str = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MetricResult(SQLModel, table=True):
    metric_result_id: str = Field(default_factory=new_id, primary_key=True)
    metric_id: str = Field(index=True)
    result_level: str
    benchmarking_run_id: str = Field(index=True)
    unit_id: str | None = Field(default=None, index=True)
    task_id: str | None = Field(default=None, index=True)
    sample_id: str | None = Field(default=None, index=True)
    shard_id: str | None = Field(default=None, index=True)
    model_id: str = Field(index=True)
    capability_block_id: str | None = Field(default=None, index=True)
    value: float
    aggregation: str = "raw"
    metadata_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
