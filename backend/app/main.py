from fastapi import FastAPI

from sqlmodel import Session

from app.api.routes import auth, benchmarking_runs, capability_blocks, dataset_load_jobs, dataset_manifests, models, ranking_lists, reports, roles, samples, shards, synthetic, tracks, users, wizard
from app.core.config import get_settings
from app.core.errors import ApiError, api_error_handler
from app.db.init_db import init_db
from app.db.session import create_db_engine
from app.services.auth_service import seed_admin_if_empty, seed_permissions, seed_system_roles
from app.services.track_service import seed_mvp_models
from app.workers.lifecycle import recover_runs_on_startup
from app.workers.run_queue import RunQueue


def create_app() -> FastAPI:
    app = FastAPI(title="TSBenchmark")
    app.add_exception_handler(ApiError, api_error_handler)
    settings = get_settings()
    if not settings.auth_secret:
        raise RuntimeError(
            "TSBENCHMARK_AUTH_SECRET is required; set it in env before starting the app"
        )
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.samples_dir.mkdir(parents=True, exist_ok=True)
    settings.forecasts_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.runtime_dir / "synthetic").mkdir(parents=True, exist_ok=True)
    app.state.engine = create_db_engine(settings.database_url)
    app.state.run_queue = RunQueue()
    init_db(app.state.engine)
    with Session(app.state.engine) as session:
        seed_permissions(session)
        seed_system_roles(session)
        seed_admin_if_empty(session, settings.admin_password)
        seed_mvp_models(session)
        recover_runs_on_startup(session)
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(roles.router)
    app.include_router(dataset_manifests.router)
    app.include_router(dataset_load_jobs.router)
    app.include_router(shards.router)
    app.include_router(synthetic.router)
    app.include_router(samples.router)
    app.include_router(capability_blocks.router)
    app.include_router(tracks.router)
    app.include_router(models.router)
    app.include_router(wizard.router)
    app.include_router(benchmarking_runs.router)
    app.include_router(reports.router)
    app.include_router(ranking_lists.router)

    @app.get("/__test__/error-contract")
    def error_contract_probe() -> None:
        raise ApiError("invalid_test_input", "invalid input", {"field": "name"})

    return app
