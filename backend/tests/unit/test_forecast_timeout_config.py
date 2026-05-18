from app.core.config import get_settings


def test_default_sample_forecast_timeout_is_300_seconds(monkeypatch):
    monkeypatch.delenv("TSBENCHMARK_SAMPLE_FORECAST_TIMEOUT_SECONDS", raising=False)
    get_settings.cache_clear()

    assert get_settings().sample_forecast_timeout_seconds == 300


def test_sample_forecast_timeout_can_be_configured(monkeypatch):
    monkeypatch.setenv("TSBENCHMARK_SAMPLE_FORECAST_TIMEOUT_SECONDS", "12")
    get_settings.cache_clear()

    assert get_settings().sample_forecast_timeout_seconds == 12
