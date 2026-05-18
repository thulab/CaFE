from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import BenchmarkingRun, Unit
from app.models.metric import MetricResult
from app.models.model_registry import Model
from app.models.ranking import RankingEntry
from app.services.ranking_service import refresh_ranking
from tests.run_helpers import create_loaded_track_with_models


def test_latest_valid_result_is_not_overwritten_by_failed_or_partial_units(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        model = models[0]
        old_run = BenchmarkingRun(track_id=track.track_id, model_ids=[model.model_id], status="succeeded")
        old_unit = Unit(benchmarking_run_id=old_run.benchmarking_run_id, model_id=model.model_id, status="succeeded")
        new_run = BenchmarkingRun(track_id=track.track_id, model_ids=[model.model_id], status="partial_succeeded")
        new_unit = Unit(benchmarking_run_id=new_run.benchmarking_run_id, model_id=model.model_id, status="partial_succeeded")
        session.add(old_run)
        session.add(old_unit)
        session.add(new_run)
        session.add(new_unit)
        session.commit()
        session.add(MetricResult(metric_id="mse", result_level="unit", benchmarking_run_id=old_run.benchmarking_run_id, unit_id=old_unit.unit_id, model_id=model.model_id, value=0.4))
        session.add(MetricResult(metric_id="mse", result_level="unit", benchmarking_run_id=new_run.benchmarking_run_id, unit_id=new_unit.unit_id, model_id=model.model_id, value=0.1))
        session.commit()

        refresh_ranking(session, track.track_id, "mse")

        entry = session.exec(select(RankingEntry).where(RankingEntry.ranking_list_id == ranking.ranking_list_id, RankingEntry.policy == "latest_valid_result")).one()
        assert entry.benchmarking_run_id == old_run.benchmarking_run_id
        assert entry.metric_value == 0.4
