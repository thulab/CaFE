import os
from collections.abc import Iterator

import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    get_settings.cache_clear()
    runtime_dir = tmp_path / "runtime"
    db_path = runtime_dir / "tsbenchmark.db"
    monkeypatch.setenv("TSBENCHMARK_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("TSBENCHMARK_DATABASE_URL", f"sqlite:///{db_path}")
    yield
    get_settings.cache_clear()
    os.environ.pop("TSBENCHMARK_RUNTIME_DIR", None)
    os.environ.pop("TSBENCHMARK_DATABASE_URL", None)
