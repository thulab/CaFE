from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..datasets.domain import TrackKind, TrackSpec
from ..domain.common import utc_now
from ..models.domain import ModelRecord
from ..tasks.domain import TaskStatus


class LeaderboardEntry(BaseModel):
    rank: int
    task_id: str
    model_id: str
    model_name: str
    batch_id: str
    track: TrackKind
    track_variant_id: str
    track_label: str
    metric_id: str
    metric_value: float
    metric_snapshot: dict[str, float] = Field(default_factory=dict)
    sample_count: int = 0

    @property
    def composite_score(self) -> float | None:
        return self.metric_snapshot.get("composite_score")

    @property
    def mse(self) -> float | None:
        return self.metric_snapshot.get("mse")

    @property
    def smape(self) -> float | None:
        return self.metric_snapshot.get("smape")

    @property
    def mean_latency_ms(self) -> float | None:
        return self.metric_snapshot.get("latency_ms")


class BatchSummary(BaseModel):
    batch_id: str
    track: TrackKind
    track_variant_id: str | None = None
    policy: str
    created_at: datetime
    sample_count: int
    context_length: int
    horizon: int
    validation_passed: bool


class TaskSummary(BaseModel):
    task_id: str
    model_id: str
    model_name: str
    batch_id: str
    track: TrackKind
    track_variant_id: str | None = None
    status: TaskStatus
    created_at: datetime
    primary_metric_id: str | None = None
    primary_metric_value: float | None = None
    composite_score: float | None = None
    execution_repeat_count: int | None = None
    task_run_count: int = 0
    report_id: str | None = None
    error_message: str | None = None
    model_runtime_parameters: dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: list[str] = Field(default_factory=list)


class TrackLeaderboard(BaseModel):
    track: str
    track_label: str
    metric_id: str
    ranking_strategy: str
    entries: list[LeaderboardEntry]


class OverallLeaderboardEntry(BaseModel):
    rank: int
    model_id: str
    model_name: str
    metric_id: str
    rank_sum: int
    covered_tracks: int
    mean_metric_value: float
    track_ranks: dict[str, int] = Field(default_factory=dict)
    track_values: dict[str, float] = Field(default_factory=dict)

    @property
    def mean_composite_score(self) -> float:
        return self.mean_metric_value

    @property
    def track_scores(self) -> dict[str, float]:
        return self.track_values


class UserDashboardOverview(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    tracks: list[TrackSpec]
    models: list[ModelRecord]
    overall_leaderboard_strategy: str
    overall_metric_id: str = "mse"
    overall_leaderboard: list[OverallLeaderboardEntry]
    track_leaderboards: list[TrackLeaderboard]


class AdminDashboardOverview(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    tracks: list[TrackSpec]
    models: list[ModelRecord]
    batches: list[BatchSummary]
    recent_tasks: list[TaskSummary]
    overall_leaderboard_strategy: str
    overall_metric_id: str = "mse"
    leaderboard: list[OverallLeaderboardEntry]
