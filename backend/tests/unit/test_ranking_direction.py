from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, Track, Unit
from app.models.metric import MetricDefinition, MetricResult
from app.models.model_registry import Model
from app.models.ranking import RankingEntry, RankingList
from app.services.ranking_service import refresh_ranking


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return Session(engine)


def test_higher_is_better_metric_ranks_largest_value_first(tmp_path):
    with make_session(tmp_path) as session:
        # 注册一个 higher_is_better 的指标定义（按 name 作业务键）。
        session.add(MetricDefinition(name="r2", display_name="R2", direction="higher_is_better"))

        block = CapabilityBlock(name="block", block_type="real", shard_count=1)
        session.add(block)
        session.commit()
        track = Track(name="track", primary_metric_id="r2")
        session.add(track)
        session.commit()
        session.refresh(track)
        ranking = RankingList(track_id=track.track_id, default_metric_id="r2")
        session.add(ranking)

        models = [Model(name=f"M{i}", model_family="F", model_version=str(i)) for i in range(2)]
        for model in models:
            session.add(model)
        session.commit()
        for model in models:
            session.refresh(model)

        # 两个模型各一条 unit 级指标：低分 0.4 与高分 0.9。
        values = {models[0].model_id: 0.4, models[1].model_id: 0.9}
        for model in models:
            run = BenchmarkingRun(track_id=track.track_id, model_ids=[model.model_id], status="succeeded")
            session.add(run)
            session.commit()
            session.refresh(run)
            unit = Unit(benchmarking_run_id=run.benchmarking_run_id, model_id=model.model_id, status="succeeded")
            session.add(unit)
            session.commit()
            session.refresh(unit)
            session.add(
                MetricResult(
                    metric_id="r2",
                    result_level="unit",
                    benchmarking_run_id=run.benchmarking_run_id,
                    unit_id=unit.unit_id,
                    model_id=model.model_id,
                    value=values[model.model_id],
                )
            )
        session.commit()

        refresh_ranking(session, track.track_id, "r2")

        entries = session.exec(
            select(RankingEntry)
            .where(
                RankingEntry.ranking_list_id == ranking.ranking_list_id,
                RankingEntry.policy == "latest_valid_result",
            )
            .order_by(RankingEntry.rank)
        ).all()

        assert [entry.metric_value for entry in entries] == [0.9, 0.4]
        assert entries[0].rank == 1
        assert entries[0].metric_value == 0.9
        assert entries[0].model_id == models[1].model_id
