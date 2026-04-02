from __future__ import annotations

from pathlib import Path

import uvicorn

from .config import AppSettings, get_settings
from .api import create_api
from .services import BenchmarkEngine


def create_backend_app(runtime_root: Path | None = None, settings: AppSettings | None = None):
    repo_root = Path(__file__).resolve().parents[2]
    app_settings = settings or get_settings()
    engine = BenchmarkEngine(runtime_root=runtime_root or app_settings.runtime_root(repo_root), settings=app_settings)
    return create_api(engine)


app = create_backend_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.service.backend.host,
        port=settings.service.backend.port,
        reload=settings.service.backend.reload,
    )
