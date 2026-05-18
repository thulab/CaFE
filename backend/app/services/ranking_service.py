from sqlmodel import Session, select

from app.models.benchmark import BenchmarkingRun, Unit
from app.models.metric import MetricResult
from app.models.ranking import RankingEntry, RankingList


def refresh_ranking(session: Session, track_id: str, metric_id: str = "mse") -> RankingList:
    ranking = session.exec(select(RankingList).where(RankingList.track_id == track_id)).one()
    valid_rows = _valid_unit_metric_rows(session, track_id, metric_id)
    for policy in ["latest_valid_result", "best_result"]:
        for entry in session.exec(
            select(RankingEntry).where(
                RankingEntry.ranking_list_id == ranking.ranking_list_id,
                RankingEntry.metric_id == metric_id,
                RankingEntry.policy == policy,
            )
        ).all():
            session.delete(entry)
        selected = _select_latest(valid_rows) if policy == "latest_valid_result" else _select_best(valid_rows)
        for rank, row in enumerate(sorted(selected, key=lambda item: item["value"]), start=1):
            session.add(
                RankingEntry(
                    ranking_list_id=ranking.ranking_list_id,
                    track_id=track_id,
                    metric_id=metric_id,
                    policy=policy,
                    model_id=row["model_id"],
                    benchmarking_run_id=row["run_id"],
                    unit_id=row["unit_id"],
                    metric_value=row["value"],
                    rank=rank,
                )
            )
    session.commit()
    session.refresh(ranking)
    return ranking


def query_ranking(session: Session, track_id: str, metric_id: str, policy: str) -> list[RankingEntry]:
    ranking = session.exec(select(RankingList).where(RankingList.track_id == track_id)).one()
    return session.exec(
        select(RankingEntry)
        .where(
            RankingEntry.ranking_list_id == ranking.ranking_list_id,
            RankingEntry.metric_id == metric_id,
            RankingEntry.policy == policy,
        )
        .order_by(RankingEntry.rank)
    ).all()


def _valid_unit_metric_rows(session: Session, track_id: str, metric_id: str) -> list[dict]:
    rows = []
    metrics = session.exec(select(MetricResult).where(MetricResult.result_level == "unit", MetricResult.metric_id == metric_id)).all()
    for metric in metrics:
        run = session.get(BenchmarkingRun, metric.benchmarking_run_id)
        unit = session.get(Unit, metric.unit_id)
        if run and unit and run.track_id == track_id and run.status in {"succeeded", "partial_succeeded"} and unit.status == "succeeded":
            rows.append(
                {
                    "model_id": metric.model_id,
                    "run_id": run.benchmarking_run_id,
                    "unit_id": unit.unit_id,
                    "value": metric.value,
                    "created_at": run.created_at,
                }
            )
    return rows


def _select_latest(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        existing = latest.get(row["model_id"])
        if existing is None or row["created_at"] > existing["created_at"]:
            latest[row["model_id"]] = row
    return list(latest.values())


def _select_best(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        existing = best.get(row["model_id"])
        if existing is None or row["value"] < existing["value"]:
            best[row["model_id"]] = row
    return list(best.values())
