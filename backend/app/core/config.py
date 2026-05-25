from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TSBENCHMARK_")

    runtime_dir: Path = Path("runtime")
    database_url: str = "sqlite:///runtime/tsbenchmark.db"
    sample_forecast_timeout_seconds: int = 300

    # 推理/模型服务（timer-rest-service）的接入配置。
    # base_url 是 REST API 参考文档里的 `http://<host>:<port>` 前缀，
    # api_prefix 是文档约定的统一路径前缀；两者拼出业务端点根地址。
    timer_service_base_url: str = "http://127.0.0.1:10810"
    timer_service_api_prefix: str = "/ai/api/v1"
    # 模型适配器选择：
    #   "rest" —— 通过 HTTP 调用 timer-rest-service（生产默认，本地指向桩服务）。
    #   "stub" —— 进程内确定性桩，无需网络（单测 / 离线）。
    model_adapter: str = "rest"

    @computed_field
    @property
    def timer_service_url(self) -> str:
        """业务端点根地址，如 http://127.0.0.1:10810/ai/api/v1。"""
        return self.timer_service_base_url.rstrip("/") + "/" + self.timer_service_api_prefix.strip("/")

    @computed_field
    @property
    def uploads_dir(self) -> Path:
        return self.runtime_dir / "uploads"

    @computed_field
    @property
    def samples_dir(self) -> Path:
        return self.runtime_dir / "samples"

    @computed_field
    @property
    def tsfiles_dir(self) -> Path:
        return self.runtime_dir / "tsfiles"

    @computed_field
    @property
    def forecasts_dir(self) -> Path:
        return self.runtime_dir / "forecasts"

    @computed_field
    @property
    def reports_dir(self) -> Path:
        return self.runtime_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
