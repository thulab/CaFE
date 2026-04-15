from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, root_validator

from ..domain.common import admin_batch_defaults, utc_now


class TrackKind(str, Enum):
    FORECAST_ACCURACY = "forecast_accuracy"
    COVARIATE_ROBUSTNESS = "covariate_robustness"
    NOISE_ROBUSTNESS = "noise_robustness"
    COST_INTENSIVE = "cost_intensive"


class TrackTemplateKind(str, Enum):
    UNIVARIATE_FORECAST = "univariate_forecast"
    MULTIVARIATE_FORECAST_ALL_TO_ALL = "multivariate_forecast_all_to_all"
    MULTIVARIATE_FORECAST_ALL_TO_SUBSET = "multivariate_forecast_all_to_subset"
    MULTIVARIATE_FORECAST_WITH_FUTURE_COVARIATES = "multivariate_forecast_with_future_covariates"
    MULTIVARIATE_FORECAST_VIA_UNIVARIATE = "multivariate_forecast_via_univariate"


class NoiseMode(str, Enum):
    CLEAN = "clean"
    NOISY = "noisy"


class ExecutionConstraint(str, Enum):
    JOINT_MULTIVARIATE = "joint_multivariate"
    PER_CHANNEL_UNIVARIATE = "per_channel_univariate"


class DatasetSourceType(str, Enum):
    SYNTHETIC = "synthetic"
    CSV = "csv"
    TSFILE = "tsfile"


class DataProcessorType(str, Enum):
    IDENTITY = "identity"
    SCALE = "scale"
    CLIP = "clip"
    COVARIATE_FILTER = "covariate_filter"


class TrackSpec(BaseModel):
    track: TrackKind
    track_variant_id: str
    track_template_kind: TrackTemplateKind
    noise_mode: NoiseMode
    execution_constraint: ExecutionConstraint
    name: str
    description: str
    fairness_policy: str
    default_context_length: int
    default_horizon: int
    suggested_sample_count: int
    input_channels: list[str] = Field(default_factory=list)
    target_channels: list[str] = Field(default_factory=list)
    future_known_channels: list[str] = Field(default_factory=list)
    knobs: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class ChannelLayout(BaseModel):
    primary_target_channel: str = "target"
    input_channels: list[str] = Field(default_factory=list)
    target_channels: list[str] = Field(default_factory=list)
    future_known_channels: list[str] = Field(default_factory=list)


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
    input_channel_values: dict[str, list[float]] = Field(default_factory=dict)
    target_channel_values: dict[str, list[float]] = Field(default_factory=dict)
    future_known_channel_values: dict[str, list[float]] = Field(default_factory=dict)
    channel_layout: ChannelLayout = Field(default_factory=ChannelLayout)
    track_tags: list[str] = Field(default_factory=list)
    truth: SeriesTruth
    notes: dict[str, Any] = Field(default_factory=dict)

    @root_validator
    def _sync_channel_views(cls, values: dict[str, Any]) -> dict[str, Any]:
        channel_layout = values.get("channel_layout") or ChannelLayout()
        primary = channel_layout.primary_target_channel or "target"
        if not channel_layout.target_channels:
            channel_layout.target_channels = [primary]
        if not channel_layout.input_channels:
            channel_layout.input_channels = [primary]

        target_channel_values = dict(values.get("target_channel_values") or {})
        input_channel_values = dict(values.get("input_channel_values") or {})
        future_known_channel_values = dict(values.get("future_known_channel_values") or {})
        covariates = dict(values.get("covariates") or {})

        target_channel_values.setdefault(primary, list(values.get("target") or []))
        input_channel_values.setdefault(primary, list(values.get("history") or []))
        if not covariates:
            for name, channel_values in input_channel_values.items():
                if name != primary:
                    covariates[name] = list(channel_values)
            for name, channel_values in future_known_channel_values.items():
                covariates[name] = list(channel_values)

        values["channel_layout"] = channel_layout
        values["target_channel_values"] = target_channel_values
        values["input_channel_values"] = input_channel_values
        values["future_known_channel_values"] = future_known_channel_values
        values["covariates"] = covariates
        return values


class ValidationReport(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)


class DatasetFeatureProfile(BaseModel):
    trend_tags: list[str] = Field(default_factory=list)
    seasonality_tags: list[str] = Field(default_factory=list)
    dominant_periods: list[int] = Field(default_factory=list)
    noise_level: float = 0.0
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    feature_summary: dict[str, Any] = Field(default_factory=dict)


class DatasetSourceRecord(BaseModel):
    source_id: str
    source_type: DatasetSourceType
    source_path: str | None = None
    source_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DatasetRecord(BaseModel):
    dataset_id: str
    source_id: str
    batch_id: str
    track_variant_id: str
    sample_count: int
    input_length: int
    prediction_length: int
    feature_profile: DatasetFeatureProfile
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DatasetBatch(BaseModel):
    batch_id: str
    track: TrackKind
    track_variant_id: str
    track_template_kind: TrackTemplateKind
    noise_mode: NoiseMode
    execution_constraint: ExecutionConstraint
    input_channels: list[str] = Field(default_factory=list)
    target_channels: list[str] = Field(default_factory=list)
    future_known_channels: list[str] = Field(default_factory=list)
    policy: str
    seed: int
    source_type: DatasetSourceType
    source_id: str
    dataset_id: str
    created_at: datetime = Field(default_factory=utc_now)
    sample_count: int
    input_length: int
    prediction_length: int
    context_length: int
    horizon: int
    samples: list[SeriesSample]
    validation: ValidationReport
    feature_profile: DatasetFeatureProfile = Field(default_factory=DatasetFeatureProfile)

    @root_validator
    def _sync_lengths(cls, values: dict[str, Any]) -> dict[str, Any]:
        input_length = values.get("input_length") or values.get("context_length")
        prediction_length = values.get("prediction_length") or values.get("horizon")
        values["input_length"] = input_length
        values["prediction_length"] = prediction_length
        values["context_length"] = values.get("context_length") or input_length
        values["horizon"] = values.get("horizon") or prediction_length
        return values


class BatchGenerationRequest(BaseModel):
    track: TrackKind = Field(default_factory=lambda: TrackKind(admin_batch_defaults().track))
    track_variant_id: str | None = None
    sample_count: int = Field(default_factory=lambda: admin_batch_defaults().sample_count)
    input_length: int | None = None
    context_length: int | None = None
    prediction_length: int | None = None
    horizon: int | None = None
    seed: int = Field(default_factory=lambda: admin_batch_defaults().seed)

    @root_validator
    def _sync_lengths(cls, values: dict[str, Any]) -> dict[str, Any]:
        input_length = values.get("input_length") or values.get("context_length")
        prediction_length = values.get("prediction_length") or values.get("horizon")
        values["input_length"] = input_length
        values["context_length"] = values.get("context_length") or input_length
        values["prediction_length"] = prediction_length
        values["horizon"] = values.get("horizon") or prediction_length
        return values


class DataProcessorConfig(BaseModel):
    processor_type: DataProcessorType
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetLoadRequest(BaseModel):
    source_type: DatasetSourceType
    track: TrackKind = Field(default_factory=lambda: TrackKind(admin_batch_defaults().track))
    track_variant_id: str | None = None
    input_length: int | None = None
    context_length: int
    prediction_length: int | None = None
    horizon: int
    max_samples: int | None = None
    batch_id_prefix: str = "load"
    processors: list[DataProcessorConfig] = Field(default_factory=list)

    @root_validator
    def _sync_lengths(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["input_length"] = values.get("input_length") or values.get("context_length")
        values["prediction_length"] = values.get("prediction_length") or values.get("horizon")
        return values


class CsvBatchLoadRequest(DatasetLoadRequest):
    source_type: DatasetSourceType = DatasetSourceType.CSV
    csv_path: str
    sample_id_column: str = "sample_id"
    step_column: str = "step"
    target_column: str = "target"
    covariate_columns: list[str] = Field(default_factory=list)
    input_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    future_known_columns: list[str] = Field(default_factory=list)
    primary_target_column: str | None = None
    delimiter: str = ","
    batch_id_prefix: str = "csv"

    @root_validator
    def _sync_columns(cls, values: dict[str, Any]) -> dict[str, Any]:
        target_columns = list(values.get("target_columns") or [])
        if not target_columns:
            target_columns = [values.get("target_column") or "target"]
        values["target_columns"] = target_columns
        if values.get("primary_target_column") is None:
            values["primary_target_column"] = target_columns[0]
        if not values.get("input_columns"):
            values["input_columns"] = [values["primary_target_column"]]
        return values
