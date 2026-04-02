from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from .domain import (
    BatchGenerationRequest,
    CsvBatchLoadRequest,
    HuggingFaceModelRegistrationRequest,
    ModelRegistrationRequest,
    TaskRunRequest,
    TrackKind,
)
from .errors import BenchmarkError, InternalBenchmarkError, NotFoundError
from .services import BenchmarkEngine


def create_api(engine: BenchmarkEngine) -> FastAPI:
    app = FastAPI(title="TS Dynamic Benchmark Backend", version="0.1.0")
    app.state.engine = engine

    def _raise_http_error(exc: Exception) -> None:
        if isinstance(exc, NotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, InternalBenchmarkError):
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if isinstance(exc, BenchmarkError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "backend"}

    @app.get("/api/v1/tracks")
    async def list_tracks():
        return engine.list_tracks()

    @app.get("/api/v1/models")
    async def list_models():
        return engine.list_models()

    @app.post("/api/v1/models/register")
    async def register_model(request: ModelRegistrationRequest):
        try:
            return engine.register_model(request)
        except Exception as exc:
            _raise_http_error(exc)

    @app.post("/api/v1/models/register/huggingface")
    async def register_huggingface_model(request: HuggingFaceModelRegistrationRequest):
        try:
            return engine.register_huggingface_model(request)
        except Exception as exc:
            _raise_http_error(exc)

    @app.post("/api/v1/models/{model_id}/load")
    async def load_model(model_id: str):
        try:
            return engine.load_model(model_id)
        except Exception as exc:
            _raise_http_error(exc)

    @app.get("/api/v1/datasets/batches")
    async def list_batches():
        return engine.list_batches()

    @app.post("/api/v1/datasets/generate")
    async def generate_batch(request: BatchGenerationRequest):
        try:
            return engine.generate_batch(request)
        except Exception as exc:
            _raise_http_error(exc)

    @app.post("/api/v1/datasets/load/csv")
    async def load_csv_batch(request: CsvBatchLoadRequest):
        try:
            return engine.load_batch(request)
        except Exception as exc:
            _raise_http_error(exc)

    @app.get("/api/v1/tasks")
    async def list_tasks():
        return engine.list_tasks()

    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str):
        try:
            return engine.get_task(task_id)
        except Exception as exc:
            _raise_http_error(exc)

    @app.post("/api/v1/tasks/run")
    async def run_task(request: TaskRunRequest):
        try:
            return engine.run_task(request)
        except Exception as exc:
            _raise_http_error(exc)

    @app.get("/api/v1/reports/{report_id}")
    async def get_report(report_id: str):
        try:
            return engine.get_report(report_id)
        except Exception as exc:
            _raise_http_error(exc)

    @app.get("/api/v1/leaderboard")
    async def leaderboard(track: TrackKind | None = Query(default=None)):
        return engine.leaderboard(track=track)

    @app.get("/api/v1/overview")
    async def overview():
        return engine.overview()

    @app.get("/api/v1/overview/user")
    async def user_overview():
        return engine.user_overview()

    @app.get("/api/v1/overview/admin")
    async def admin_overview():
        return engine.admin_overview()

    return app
