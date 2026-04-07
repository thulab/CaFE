from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..domain.common import admin_batch_defaults, utc_now


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


class BatchGenerationRequest(BaseModel):
    track: TrackKind
    sample_count: int = Field(default_factory=lambda: admin_batch_defaults().sample_count)
    context_length: int | None = None
    horizon: int | None = None
    seed: int = Field(default_factory=lambda: admin_batch_defaults().seed)


class DataProcessorConfig(BaseModel):
    processor_type: DataProcessorType
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetLoadRequest(BaseModel):
    source_type: DatasetSourceType
    track: TrackKind = TrackKind.FORECAST_ACCURACY
    context_length: int
    horizon: int
    max_samples: int | None = None
    batch_id_prefix: str = "load"
    processors: list[DataProcessorConfig] = Field(default_factory=list)


class CsvBatchLoadRequest(DatasetLoadRequest):
    source_type: DatasetSourceType = DatasetSourceType.CSV
    csv_path: str
    sample_id_column: str = "sample_id"
    step_column: str = "step"
    target_column: str = "target"
    covariate_columns: list[str] = Field(default_factory=list)
    delimiter: str = ","
    batch_id_prefix: str = "csv"
