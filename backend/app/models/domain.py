from __future__ import annotations

from datetime import datetime
from enum import Enum
from urllib.parse import urlparse
from typing import Any

from pydantic import BaseModel, Field

from ..domain.common import user_submission_defaults, utc_now


class ModelAdapter(str, Enum):
    SEASONAL_NAIVE = "seasonal_naive"
    RECENT_MEAN = "recent_mean"
    COVARIATE_TRAP = "covariate_trap"
    HUGGINGFACE_TEXT_GENERATION = "huggingface_text_generation"
    HUGGINGFACE_CHRONOS2 = "huggingface_chronos2"
    HUGGINGFACE_SUNDIAL = "huggingface_sundial"


class ModelRuntimeStatus(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    LOAD_FAILED = "load_failed"


class HuggingFaceTask(str, Enum):
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    CHRONOS2 = "chronos-2"
    SUNDIAL = "sundial"


class ParameterValueType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class ModelRuntimeParameterDefinition(BaseModel):
    name: str
    label: str
    value_type: ParameterValueType
    required: bool = False
    description: str = ""


class ModelSourceSpec(BaseModel):
    huggingface_repo_id: str | None = None
    huggingface_url: str | None = None
    local_weight_path: str | None = None
    revision: str | None = None


class ModelSpec(BaseModel):
    source: ModelSourceSpec = Field(default_factory=ModelSourceSpec)
    runtime_parameter_definitions: list[ModelRuntimeParameterDefinition] = Field(default_factory=list)


class HuggingFaceConfig(BaseModel):
    repo_id: str
    task: HuggingFaceTask = HuggingFaceTask.TEXT_GENERATION
    revision: str | None = None
    weights_path: str | None = None
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
    spec: ModelSpec = Field(default_factory=ModelSpec)
    runtime_status: ModelRuntimeStatus = ModelRuntimeStatus.REGISTERED
    last_loaded_at: datetime | None = None
    last_error: str | None = None


class ModelRegistrationRequest(BaseModel):
    model_id: str
    name: str
    adapter: ModelAdapter
    source_type: str = "uploaded_manual"
    manual: str
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HuggingFaceModelRegistrationRequest(BaseModel):
    huggingface_url: str | None = None
    repo_id: str | None = None
    name: str | None = None
    model_id: str | None = None
    manual: str
    task: HuggingFaceTask | None = None
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


def build_huggingface_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def normalize_huggingface_repo_id(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        raise ValueError("empty Hugging Face repo value")
    if "://" not in value and not value.startswith("huggingface.co/"):
        return value.strip("/")

    candidate = value if "://" in value else f"https://{value}"
    parsed = urlparse(candidate)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"unsupported Hugging Face host: {parsed.netloc or candidate}")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"invalid Hugging Face model url: {value}")
    return "/".join(parts[:2])


def infer_huggingface_task(repo_id: str) -> HuggingFaceTask:
    normalized = repo_id.strip().lower()
    if normalized == "amazon/chronos-2":
        return HuggingFaceTask.CHRONOS2
    if normalized == "thuml/sundial-base-128m":
        return HuggingFaceTask.SUNDIAL
    return HuggingFaceTask.TEXT_GENERATION


def build_runtime_parameter_definitions(task: HuggingFaceTask) -> list[ModelRuntimeParameterDefinition]:
    shared = [
        ModelRuntimeParameterDefinition(
            name="batch_size",
            label="Batch Size",
            value_type=ParameterValueType.INTEGER,
            description="任务执行时单轮批量推理的样本数。",
        )
    ]
    if task == HuggingFaceTask.CHRONOS2:
        return shared + [
            ModelRuntimeParameterDefinition(
                name="context_length",
                label="Context Length",
                value_type=ParameterValueType.INTEGER,
                description="推理时传给 Chronos-2 的上下文截断长度。",
            ),
            ModelRuntimeParameterDefinition(
                name="use_covariates",
                label="Use Covariates",
                value_type=ParameterValueType.BOOLEAN,
                description="是否在任务运行时启用协变量输入。",
            ),
            ModelRuntimeParameterDefinition(
                name="cross_learning",
                label="Cross Learning",
                value_type=ParameterValueType.BOOLEAN,
                description="是否在批任务中启用跨序列联合预测。",
            ),
            ModelRuntimeParameterDefinition(
                name="max_output_patches",
                label="Max Output Patches",
                value_type=ParameterValueType.INTEGER,
                description="限制 Chronos-2 单次预测的输出 patch 数量。",
            ),
        ]
    if task == HuggingFaceTask.SUNDIAL:
        return shared + [
            ModelRuntimeParameterDefinition(
                name="do_sample",
                label="Do Sample",
                value_type=ParameterValueType.BOOLEAN,
                description="是否在任务运行时开启采样生成。",
            ),
            ModelRuntimeParameterDefinition(
                name="temperature",
                label="Temperature",
                value_type=ParameterValueType.FLOAT,
                description="Sundial 采样生成温度。",
            ),
            ModelRuntimeParameterDefinition(
                name="top_p",
                label="Top P",
                value_type=ParameterValueType.FLOAT,
                description="Sundial nucleus sampling 参数。",
            ),
        ]
    return shared + [
        ModelRuntimeParameterDefinition(
            name="max_new_tokens",
            label="Max New Tokens",
            value_type=ParameterValueType.INTEGER,
            description="文本生成模型单次预测的最大输出长度。",
        ),
        ModelRuntimeParameterDefinition(
            name="do_sample",
            label="Do Sample",
            value_type=ParameterValueType.BOOLEAN,
            description="是否在任务运行时开启采样生成。",
        ),
        ModelRuntimeParameterDefinition(
            name="temperature",
            label="Temperature",
            value_type=ParameterValueType.FLOAT,
            description="文本生成任务的采样温度。",
        ),
        ModelRuntimeParameterDefinition(
            name="top_p",
            label="Top P",
            value_type=ParameterValueType.FLOAT,
            description="文本生成任务的 nucleus sampling 参数。",
        ),
    ]
