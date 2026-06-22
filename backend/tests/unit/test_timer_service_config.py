from app.core.config import get_settings


def test_timer_service_defaults(monkeypatch):
    for key in (
        "TSBENCHMARK_TIMER_SERVICE_BASE_URL",
        "TSBENCHMARK_TIMER_SERVICE_API_PREFIX",
        "TSBENCHMARK_MODEL_ADAPTER",
        "TSBENCHMARK_MODEL_LIFECYCLE_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.timer_service_base_url == "http://127.0.0.1:10810"
    assert settings.timer_service_api_prefix == "/ai/api/v1"
    assert settings.timer_service_url == "http://127.0.0.1:10810/ai/api/v1"
    assert settings.model_adapter == "rest"
    assert settings.model_lifecycle_mode == "sequential_unload"


def test_timer_service_base_url_is_configurable(monkeypatch):
    monkeypatch.setenv("TSBENCHMARK_TIMER_SERVICE_BASE_URL", "http://gpu-host:9000/")
    monkeypatch.setenv("TSBENCHMARK_MODEL_ADAPTER", "stub")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.timer_service_base_url == "http://gpu-host:9000/"
    assert settings.timer_service_url == "http://gpu-host:9000/ai/api/v1"
    assert settings.model_adapter == "stub"


def test_model_lifecycle_mode_is_configurable(monkeypatch):
    monkeypatch.setenv("TSBENCHMARK_MODEL_LIFECYCLE_MODE", "keep_loaded")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.model_lifecycle_mode == "keep_loaded"
