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


class DatasetSourceType(str, Enum):
    CSV = "csv"


class DataProcessorType(str, Enum):
    IDENTITY = "identity"
    SCALE = "scale"
    CLIP = "clip"
    COVARIATE_FILTER = "covariate_filter"


class ModelAdapter(str, Enum):
    SEASONAL_NAIVE = "seasonal_naive"
    RECENT_MEAN = "recent_mean"
    COVARIATE_TRAP = "covariate_trap"
    HUGGINGFACE_TEXT_GENERATION = "huggingface_text_generation"
    HUGGINGFACE_CHRONOS2 = "huggingface_chronos2"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelRuntimeStatus(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    LOAD_FAILED = "load_failed"


class HuggingFaceTask(str, Enum):
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    CHRONOS2 = "chronos-2"


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


class HuggingFaceConfig(BaseModel):
    repo_id: str
    task: HuggingFaceTask = HuggingFaceTask.TEXT_GENERATION
    revision: str | None = None
    trust_remote_code: bool = False
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    device: int = -1
    device_map: str | None = None
    torch_dtype: str | None = None
    attn_implementation: str | None = None
    batch_size: int = 1
    context_length: int | None = None
    use_covariates: bool = True
    cross_learning: bool = False
    max_output_patches: int | None = None
    load_retries: int = 3
    load_retry_backoff_seconds: float = 1.0


class ModelRecord(BaseModel):
    model_id: str
    name: str
    adapter: ModelAdapter
    source_type: str
    manual: str
    created_at: datetime = Field(default_factory=utc_now)
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    huggingface: HuggingFaceConfig | None = None
    runtime_status: ModelRuntimeStatus = ModelRuntimeStatus.REGISTERED
    last_loaded_at: datetime | None = None
    last_error: str | None = None


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
    error_message: str | None = None


class ModelRegistrationRequest(BaseModel):
    model_id: str
    name: str
    adapter: ModelAdapter
    source_type: str = "uploaded_stub"
    manual: str
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HuggingFaceModelRegistrationRequest(BaseModel):
    repo_id: str
    name: str | None = None
    model_id: str | None = None
    manual: str
    task: HuggingFaceTask = HuggingFaceTask.TEXT_GENERATION
    revision: str | None = None
    trust_remote_code: bool = False
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    capabilities: list[str] = Field(default_factory=lambda: ["forecast", "huggingface"])
    metadata: dict[str, Any] = Field(default_factory=dict)
    device_map: str | None = None
    torch_dtype: str | None = None
    attn_implementation: str | None = None
    batch_size: int = 1
    context_length: int | None = None
    use_covariates: bool = True
    cross_learning: bool = False
    max_output_patches: int | None = None
    load_retries: int = 3
    load_retry_backoff_seconds: float = 1.0


class BatchGenerationRequest(BaseModel):
    track: TrackKind
    sample_count: int = 12
    context_length: int | None = None
    horizon: int | None = None
    seed: int = 7


class DatasetLoadRequest(BaseModel):
    source_type: DatasetSourceType
    track: TrackKind = TrackKind.FORECAST_ACCURACY
    context_length: int
    horizon: int
    max_samples: int | None = None
    batch_id_prefix: str = "load"
    processors: list["DataProcessorConfig"] = Field(default_factory=list)


class DataProcessorConfig(BaseModel):
    processor_type: DataProcessorType
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class CsvBatchLoadRequest(DatasetLoadRequest):
    source_type: DatasetSourceType = DatasetSourceType.CSV
    csv_path: str
    sample_id_column: str = "sample_id"
    step_column: str = "step"
    target_column: str = "target"
    covariate_columns: list[str] = Field(default_factory=list)
    delimiter: str = ","
    batch_id_prefix: str = "csv"


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


class TrackLeaderboard(BaseModel):
    track: TrackKind
    entries: list[LeaderboardEntry]


class UserDashboardOverview(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    tracks: list[TrackSpec]
    models: list[ModelRecord]
    overall_leaderboard: list[LeaderboardEntry]
    track_leaderboards: list[TrackLeaderboard]


class AdminDashboardOverview(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    tracks: list[TrackSpec]
    models: list[ModelRecord]
    batches: list[BatchSummary]
    recent_tasks: list[TaskSummary]
    leaderboard: list[LeaderboardEntry]
