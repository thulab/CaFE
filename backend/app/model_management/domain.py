from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..domain.common import user_submission_defaults, utc_now


class ModelAdapter(str, Enum):
    SEASONAL_NAIVE = "seasonal_naive"
    RECENT_MEAN = "recent_mean"
    COVARIATE_TRAP = "covariate_trap"
    HUGGINGFACE_TEXT_GENERATION = "huggingface_text_generation"
    HUGGINGFACE_CHRONOS2 = "huggingface_chronos2"


class ModelRuntimeStatus(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    LOAD_FAILED = "load_failed"


class HuggingFaceTask(str, Enum):
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    CHRONOS2 = "chronos-2"


class HuggingFaceConfig(BaseModel):
    repo_id: str
    task: HuggingFaceTask = HuggingFaceTask.TEXT_GENERATION
    revision: str | None = None
    trust_remote_code: bool = False
    max_new_tokens: int = Field(default_factory=lambda: user_submission_defaults().max_new_tokens)
    do_sample: bool = Field(default_factory=lambda: user_submission_defaults().do_sample)
    temperature: float = Field(default_factory=lambda: user_submission_defaults().temperature)
    top_p: float = Field(default_factory=lambda: user_submission_defaults().top_p)
    device: int = -1
    device_map: str | None = None
    torch_dtype: str | None = None
    attn_implementation: str | None = None
    batch_size: int = Field(default_factory=lambda: user_submission_defaults().batch_size)
    context_length: int | None = None
    use_covariates: bool = Field(default_factory=lambda: user_submission_defaults().use_covariates)
    cross_learning: bool = Field(default_factory=lambda: user_submission_defaults().cross_learning)
    max_output_patches: int | None = None
    load_retries: int = Field(default_factory=lambda: user_submission_defaults().load_retries)
    load_retry_backoff_seconds: float = Field(default_factory=lambda: user_submission_defaults().load_retry_backoff_seconds)


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
    trust_remote_code: bool = Field(default_factory=lambda: user_submission_defaults().trust_remote_code)
    max_new_tokens: int = Field(default_factory=lambda: user_submission_defaults().max_new_tokens)
    do_sample: bool = Field(default_factory=lambda: user_submission_defaults().do_sample)
    temperature: float = Field(default_factory=lambda: user_submission_defaults().temperature)
    top_p: float = Field(default_factory=lambda: user_submission_defaults().top_p)
    capabilities: list[str] = Field(default_factory=lambda: ["forecast", "huggingface"])
    metadata: dict[str, Any] = Field(default_factory=dict)
    device_map: str | None = None
    torch_dtype: str | None = None
    attn_implementation: str | None = None
    batch_size: int = Field(default_factory=lambda: user_submission_defaults().batch_size)
    context_length: int | None = None
    use_covariates: bool = Field(default_factory=lambda: user_submission_defaults().use_covariates)
    cross_learning: bool = Field(default_factory=lambda: user_submission_defaults().cross_learning)
    max_output_patches: int | None = None
    load_retries: int = Field(default_factory=lambda: user_submission_defaults().load_retries)
    load_retry_backoff_seconds: float = Field(default_factory=lambda: user_submission_defaults().load_retry_backoff_seconds)
