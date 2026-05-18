from app.core.config import get_settings


def test_settings_use_isolated_runtime_and_derived_paths(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "isolated-runtime"
    db_url = f"sqlite:///{runtime_dir / 'metadata.db'}"
    monkeypatch.setenv("TSBENCHMARK_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("TSBENCHMARK_DATABASE_URL", db_url)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.runtime_dir == runtime_dir
    assert settings.database_url == db_url
    assert settings.uploads_dir == runtime_dir / "uploads"
    assert settings.samples_dir == runtime_dir / "samples"
    assert settings.forecasts_dir == runtime_dir / "forecasts"
    assert settings.reports_dir == runtime_dir / "reports"
