from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TSBENCHMARK_")

    runtime_dir: Path = Path("runtime")
    database_url: str = "sqlite:///runtime/tsbenchmark.db"
    sample_forecast_timeout_seconds: int = 300

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
    def forecasts_dir(self) -> Path:
        return self.runtime_dir / "forecasts"

    @computed_field
    @property
    def reports_dir(self) -> Path:
        return self.runtime_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
