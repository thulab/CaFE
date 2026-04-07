from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..data_management.domain import TrackKind
from ..domain.common import utc_now

DEFAULT_EVALUATION_METRICS = ["mse", "mae", "smape", "latency_ms", "token_count", "composite_score"]


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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


class TaskDatasetSpec(BaseModel):
    batch_id: str
    track: TrackKind
    sample_count: int
    context_length: int
    horizon: int


class TaskSpec(BaseModel):
    model_id: str
    model_runtime_parameters: dict[str, Any] = Field(default_factory=dict)
    dataset: TaskDatasetSpec
    evaluation_metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_EVALUATION_METRICS))


class EvaluationTask(BaseModel):
    task_id: str
    model_id: str
    batch_id: str
    track: TrackKind
    status: TaskStatus
    spec: TaskSpec | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metrics: AggregatedMetrics | None = None
    report_id: str | None = None
    sample_outcomes: list[SampleOutcome] = Field(default_factory=list)
    error_message: str | None = None


class TaskRunRequest(BaseModel):
    model_id: str
    batch_id: str
    model_runtime_parameters: dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_EVALUATION_METRICS))
