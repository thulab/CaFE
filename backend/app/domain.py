from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrackKind(str, Enum):
    FORECAST_ACCURACY = "forecast_accuracy"
    COVARIATE_ROBUSTNESS = "covariate_robustness"
    NOISE_ROBUSTNESS = "noise_robustness"
    COST_INTENSIVE = "cost_intensive"


class ModelAdapter(str, Enum):
    SEASONAL_NAIVE = "seasonal_naive"
    RECENT_MEAN = "recent_mean"
    COVARIATE_TRAP = "covariate_trap"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TrackSpec(BaseModel):
    track: TrackKind
    name: str
    description: str
    fairness_policy: str
    default_context_length: int
    default_horizon: int
    suggested_sample_count: int
    knobs: list[str] = Field(default_factory=list)


class SeriesTruth(BaseModel):
    trend_type: str
    periods: list[int]
    dominant_period: int
    amplitude_mode: str
    phase_shift: bool
    noise_level: float
    difficulty: str


class SeriesSample(BaseModel):
    sample_id: str
    history: list[float]
    target: list[float]
    covariates: dict[str, list[float]] = Field(default_factory=dict)
    track_tags: list[str] = Field(default_factory=list)
    truth: SeriesTruth
    notes: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)


class DatasetBatch(BaseModel):
    batch_id: str
    track: TrackKind
    policy: str
    seed: int
    created_at: datetime = Field(default_factory=utc_now)
    sample_count: int
    context_length: int
    horizon: int
    samples: list[SeriesSample]
    validation: ValidationReport


class ModelRecord(BaseModel):
    model_id: str
    name: str
    adapter: ModelAdapter
    source_type: str
    manual: str
    created_at: datetime = Field(default_factory=utc_now)
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleOutcome(BaseModel):
    sample_id: str
    mse: float
    mae: float
    smape: float
    latency_ms: float
    token_count: int
    prediction: list[float]
    notes: dict[str, Any] = Field(default_factory=dict)


class AggregatedMetrics(BaseModel):
    mse: float
    mae: float
    smape: float
    mean_latency_ms: float
    mean_token_count: float
    composite_score: float


class BenchmarkReport(BaseModel):
    report_id: str
    task_id: str
    created_at: datetime = Field(default_factory=utc_now)
    summary: str
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    bad_cases: list[str] = Field(default_factory=list)
    distribution: dict[str, float] = Field(default_factory=dict)


class EvaluationTask(BaseModel):
    task_id: str
    model_id: str
    batch_id: str
    track: TrackKind
    status: TaskStatus
    created_at: datetime = Field(default_factory=utc_now)
    metrics: AggregatedMetrics | None = None
    report_id: str | None = None
    sample_outcomes: list[SampleOutcome] = Field(default_factory=list)


class ModelRegistrationRequest(BaseModel):
    model_id: str
    name: str
    adapter: ModelAdapter
    source_type: str = "uploaded_stub"
    manual: str
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchGenerationRequest(BaseModel):
    track: TrackKind
    sample_count: int = 12
    context_length: int | None = None
    horizon: int | None = None
    seed: int = 7


class TaskRunRequest(BaseModel):
    model_id: str
    batch_id: str


class LeaderboardEntry(BaseModel):
    task_id: str
    model_id: str
    model_name: str
    batch_id: str
    track: TrackKind
    composite_score: float
    mse: float
    smape: float
    mean_latency_ms: float


class DashboardOverview(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    tracks: list[TrackSpec]
    models: list[ModelRecord]
    batches: list[DatasetBatch]
    recent_tasks: list[EvaluationTask]
    leaderboard: list[LeaderboardEntry]
