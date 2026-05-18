from datetime import datetime

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class Model(SQLModel, table=True):
    model_id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    model_family: str
    model_version: str
    adapter_type: str = "timer_service"
    endpoint_uri: str | None = None
    supported_task_types: list[str] = Field(default_factory=lambda: ["univariate_forecast"], sa_column=Column(JSON))
    input_schema_version: str = "sample.v1"
    stub_seed: int = 0
    status: str = "available"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
