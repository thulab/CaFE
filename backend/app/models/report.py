from datetime import datetime
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class Report(SQLModel, table=True):
    report_id: str = Field(default_factory=new_id, primary_key=True)
    report_type: str = "run_summary"
    benchmarking_run_id: str = Field(index=True)
    track_id: str = Field(index=True)
    status: str = "created"
    storage_uri: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
