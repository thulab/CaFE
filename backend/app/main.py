from __future__ import annotations

from pathlib import Path

import uvicorn

from .api import create_api
from .services import BenchmarkEngine


def create_backend_app(runtime_root: Path | None = None):
    repo_root = Path(__file__).resolve().parents[2]
    engine = BenchmarkEngine(runtime_root=runtime_root or repo_root / "runtime")
    return create_api(engine)


app = create_backend_app()


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
