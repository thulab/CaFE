from typing import Protocol

from app.core.config import Settings
from app.services.stub_timer_adapter import StubTimerAdapter
from app.services.timer_rest_adapter import TimerRestAdapter, TimerServiceError


class ModelAdapter(Protocol):
    def ensure_model_loaded(self, model: dict, timeout_seconds: int) -> None:
        ...

    def forecast(self, sample: dict, model: dict, timeout_seconds: int) -> list[list[float]]:
        ...


def get_model_adapter(settings: Settings) -> ModelAdapter:
    """根据配置返回模型适配器：HTTP REST（默认）或进程内桩。"""
    if settings.model_adapter == "stub":
        return StubTimerAdapter()
    return TimerRestAdapter(
        base_url=settings.timer_service_base_url,
        api_prefix=settings.timer_service_api_prefix,
    )


def remote_model_id(model) -> str:
    """把本地 Model 映射为 REST 服务使用的 model_id（如 "Timer-3.5"）。"""
    endpoint = (getattr(model, "endpoint_uri", None) or "").strip()
    if endpoint.startswith("timer://"):
        return endpoint.removeprefix("timer://")
    family = (getattr(model, "model_family", None) or "").strip()
    version = (getattr(model, "model_version", None) or "").strip()
    if family and version:
        return f"{family}-{version}"
    return getattr(model, "name", None) or str(model.model_id)


def service_loaded_model_ids(settings: Settings, adapter: TimerRestAdapter | None = None) -> set[str] | None:
    """查询 timer-rest-service 已加载的 model_id 集合。

    返回 None 表示状态未知（adapter 为 stub 模式，或服务不可达）——调用方应优雅降级。
    """
    if settings.model_adapter != "rest" and adapter is None:
        return None
    if adapter is None:
        adapter = TimerRestAdapter(
            base_url=settings.timer_service_base_url,
            api_prefix=settings.timer_service_api_prefix,
        )
    try:
        models = adapter.list_models()
    except TimerServiceError:
        return None
    return {m["model_id"] for m in models if m.get("loaded")}
