from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from .domain import BatchGenerationRequest, ModelRegistrationRequest, TaskRunRequest, TrackKind
from .services import BenchmarkEngine, NotFoundError


def create_api(engine: BenchmarkEngine) -> FastAPI:
    app = FastAPI(title="TS Dynamic Benchmark Backend", version="0.1.0")
    app.state.engine = engine

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "backend"}

    @app.get("/api/v1/tracks")
    def list_tracks():
        return engine.list_tracks()

    @app.get("/api/v1/models")
    def list_models():
        return engine.list_models()

    @app.post("/api/v1/models/register")
    def register_model(request: ModelRegistrationRequest):
        return engine.register_model(request)

    @app.get("/api/v1/datasets/batches")
    def list_batches():
        return engine.list_batches()

    @app.post("/api/v1/datasets/generate")
    def generate_batch(request: BatchGenerationRequest):
        return engine.generate_batch(request)

    @app.get("/api/v1/tasks")
    def list_tasks():
        return engine.list_tasks()

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(task_id: str):
        try:
            return engine.get_task(task_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/tasks/run")
    def run_task(request: TaskRunRequest):
        try:
            return engine.run_task(request)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/reports/{report_id}")
    def get_report(report_id: str):
        try:
            return engine.get_report(report_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/leaderboard")
    def leaderboard(track: TrackKind | None = Query(default=None)):
        return engine.leaderboard(track=track)

    @app.get("/api/v1/overview")
    def overview():
        return engine.overview()

    return app
