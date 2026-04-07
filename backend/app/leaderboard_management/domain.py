from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..data_management.domain import TrackKind, TrackSpec
from ..domain.common import utc_now
from ..model_management.domain import ModelRecord
from ..task_management.domain import TaskStatus


class LeaderboardEntry(BaseModel):
    rank: int
    task_id: str
    model_id: str
    model_name: str
    batch_id: str
    track: TrackKind
    composite_score: float
    mse: float
    smape: float
    mean_latency_ms: float


class BatchSummary(BaseModel):
    batch_id: str
    track: TrackKind
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
    status: TaskStatus
    created_at: datetime
    composite_score: float | None = None
    report_id: str | None = None
    error_message: str | None = None
    model_runtime_parameters: dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: list[str] = Field(default_factory=list)


class TrackLeaderboard(BaseModel):
    track: TrackKind
    scoring_strategy: str
    entries: list[LeaderboardEntry]


class OverallLeaderboardEntry(BaseModel):
    rank: int
    model_id: str
    model_name: str
    rank_sum: int
    covered_tracks: int
    mean_composite_score: float
    track_ranks: dict[str, int] = Field(default_factory=dict)
    track_scores: dict[str, float] = Field(default_factory=dict)


class UserDashboardOverview(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    tracks: list[TrackSpec]
    models: list[ModelRecord]
    overall_leaderboard_strategy: str
    overall_leaderboard: list[OverallLeaderboardEntry]
    track_leaderboards: list[TrackLeaderboard]


class AdminDashboardOverview(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    tracks: list[TrackSpec]
    models: list[ModelRecord]
    batches: list[BatchSummary]
    recent_tasks: list[TaskSummary]
    overall_leaderboard_strategy: str
    leaderboard: list[OverallLeaderboardEntry]
