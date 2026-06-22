from sqlmodel import Session, select

from app.models.benchmark import BenchmarkingRun, Unit
from app.models.metric import MetricDefinition, MetricResult
from app.models.ranking import RankingEntry, RankingList


def refresh_ranking(session: Session, track_id: str, metric_id: str = "mse", commit: bool = True) -> RankingList:
    ranking = session.exec(select(RankingList).where(RankingList.track_id == track_id)).one()
    valid_rows = _valid_unit_metric_rows(session, track_id, metric_id)
    descending = _metric_higher_is_better(session, metric_id)
    for policy in ["latest_valid_result", "best_result"]:
        for entry in session.exec(
            select(RankingEntry).where(
                RankingEntry.ranking_list_id == ranking.ranking_list_id,
                RankingEntry.metric_id == metric_id,
                RankingEntry.policy == policy,
            )
        ).all():
            session.delete(entry)
        selected = _select_latest(valid_rows) if policy == "latest_valid_result" else _select_best(valid_rows, descending)
        for rank, row in enumerate(sorted(selected, key=lambda item: item["value"], reverse=descending), start=1):
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
    # commit=False：调用方（run_executor 终态收尾）需要把 status / report_id / 多张榜单
    # 攒进同一次提交，避免轮询看到 succeeded 却查不到榜单的竞态窗口。
    if commit:
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


def _select_best(rows: list[dict], descending: bool = False) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        existing = best.get(row["model_id"])
        if existing is None:
            best[row["model_id"]] = row
            continue
        better = row["value"] > existing["value"] if descending else row["value"] < existing["value"]
        if better:
            best[row["model_id"]] = row
    return list(best.values())


def _metric_higher_is_better(session: Session, metric_id: str) -> bool:
    # 执行链路里 metric 以 name（如 "mse"/"mase"）作业务键，MetricDefinition 按 name 查方向；
    # 未注册定义时（MVP 默认三指标）退回 lower_is_better。
    definition = session.exec(select(MetricDefinition).where(MetricDefinition.name == metric_id)).first()
    return definition is not None and definition.direction == "higher_is_better"
