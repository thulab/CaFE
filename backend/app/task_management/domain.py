from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, root_validator

from ..data_management.domain import TrackKind
from ..domain.common import utc_now

DEFAULT_EXECUTION_REPEAT_COUNT = 3
DEFAULT_EVALUATION_METRICS = ["mse", "mae", "smape", "latency_ms", "token_count", "composite_score"]


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskRunStatus(str, Enum):
    QUEUED = "queued"
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
    run_count: int = 1
    mse_std: float | None = None
    mae_std: float | None = None
    smape_std: float | None = None
    latency_ms_std: float | None = None
    token_count_std: float | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class AggregatedMetrics(BaseModel):
    mse: float
    mae: float
    smape: float
    mean_latency_ms: float
    mean_token_count: float
    composite_score: float
    stability_stats: dict[str, float] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    report_id: str
    task_id: str
    created_at: datetime = Field(default_factory=utc_now)
    summary: str
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    bad_cases: list[str] = Field(default_factory=list)
    distribution: dict[str, float] = Field(default_factory=dict)
    run_ids: list[str] = Field(default_factory=list)


class TaskDatasetSpec(BaseModel):
    batch_id: str
    track: TrackKind
    track_variant_id: str | None = None
    sample_count: int
    input_length: int | None = None
    prediction_length: int | None = None
    context_length: int | None = None
    horizon: int | None = None

    @root_validator
    def _sync_lengths(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["input_length"] = values.get("input_length") or values.get("context_length")
        values["prediction_length"] = values.get("prediction_length") or values.get("horizon")
        values["context_length"] = values.get("context_length") or values.get("input_length")
        values["horizon"] = values.get("horizon") or values.get("prediction_length")
        return values


class TaskSpec(BaseModel):
    model_id: str
    model_runtime_parameters: dict[str, Any] = Field(default_factory=dict)
    dataset: TaskDatasetSpec
    evaluation_metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_EVALUATION_METRICS))
    execution_repeat_count: int = DEFAULT_EXECUTION_REPEAT_COUNT


class TaskRunRecord(BaseModel):
    run_id: str
    task_id: str
    run_no: int
    status: TaskRunStatus
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: AggregatedMetrics | None = None
    sample_outcomes: list[SampleOutcome] = Field(default_factory=list)
    error_message: str | None = None


class EvaluationTask(BaseModel):
    task_id: str
    model_id: str
    batch_id: str
    track: TrackKind
    track_variant_id: str | None = None
    status: TaskStatus
    spec: TaskSpec | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metrics: AggregatedMetrics | None = None
    report_id: str | None = None
    sample_outcomes: list[SampleOutcome] = Field(default_factory=list)
    task_runs: list[TaskRunRecord] = Field(default_factory=list)
    error_message: str | None = None


class TaskRunRequest(BaseModel):
    model_id: str
    batch_id: str
    model_runtime_parameters: dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_EVALUATION_METRICS))
    execution_repeat_count: int | None = None
