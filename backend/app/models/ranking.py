from datetime import datetime

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class RankingList(SQLModel, table=True):
    ranking_list_id: str = Field(default_factory=new_id, primary_key=True)
    track_id: str = Field(index=True)
    default_metric_id: str
    default_policy: str = "latest_valid_result"
    supported_policies: list[str] = Field(default_factory=lambda: ["latest_valid_result", "best_result"], sa_column=Column(JSON))
    public_visible: bool = True
    status: str = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RankingEntry(SQLModel, table=True):
    ranking_entry_id: str = Field(default_factory=new_id, primary_key=True)
    ranking_list_id: str = Field(index=True)
    track_id: str = Field(index=True)
    metric_id: str = Field(index=True)
    policy: str = "latest_valid_result"
    model_id: str = Field(index=True)
    benchmarking_run_id: str = Field(index=True)
    unit_id: str = Field(index=True)
    metric_value: float
    rank: int
    status: str = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
