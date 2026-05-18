from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import ForecastArtifact, Task, Unit
from app.models.metric import MetricResult
from app.services.run_executor import create_benchmarking_run, execute_run
from tests.run_helpers import create_loaded_track_with_models


def test_run_execution_succeeds_and_persists_forecasts_and_metrics(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status == "succeeded"
        assert session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).one().status == "succeeded"
        assert session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).one().status == "succeeded"
        assert session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)).one()
        metric_levels = {metric.result_level for metric in session.exec(select(MetricResult)).all()}
        assert {"sample", "shard", "task", "unit"} <= metric_levels
