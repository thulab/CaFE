from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import BenchmarkingRun, Unit
from app.models.metric import MetricResult
from app.models.ranking import RankingEntry
from app.services.ranking_service import refresh_ranking
from tests.run_helpers import create_loaded_track_with_models


def test_best_result_picks_lowest_metric_value(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        model = models[0]
        runs = [BenchmarkingRun(track_id=track.track_id, model_ids=[model.model_id], status="succeeded") for _ in range(2)]
        units = [Unit(benchmarking_run_id=run.benchmarking_run_id, model_id=model.model_id, status="succeeded") for run in runs]
        for item in [*runs, *units]:
            session.add(item)
        session.commit()
        session.add(MetricResult(metric_id="mse", result_level="unit", benchmarking_run_id=runs[0].benchmarking_run_id, unit_id=units[0].unit_id, model_id=model.model_id, value=0.5))
        session.add(MetricResult(metric_id="mse", result_level="unit", benchmarking_run_id=runs[1].benchmarking_run_id, unit_id=units[1].unit_id, model_id=model.model_id, value=0.2))
        session.commit()

        refresh_ranking(session, track.track_id, "mse")

        entry = session.exec(select(RankingEntry).where(RankingEntry.ranking_list_id == ranking.ranking_list_id, RankingEntry.policy == "best_result")).one()
        assert entry.benchmarking_run_id == runs[1].benchmarking_run_id
        assert entry.metric_value == 0.2
