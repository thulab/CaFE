from fastapi import Depends
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.models.benchmark import Track
from app.models.model_registry import Model
from app.models.ranking import RankingEntry, RankingList
from app.services.ranking_service import query_ranking

router = make_router(tags=["ranking-lists"])


@router.get("/tracks/{track_id}/ranking", tier="public")
def get_track_ranking(track_id: str, metric: str | None = None, policy: str | None = None, session: Session = Depends(get_db_session)) -> dict:
    track = session.get(Track, track_id)
    resolved_metric = metric or (track.primary_metric_id if track else "mase")
    resolved_policy = policy or (track.default_ranking_policy if track else "latest_valid_result")
    entries = query_ranking(session, track_id, resolved_metric, resolved_policy)
    return {"track_id": track_id, "metric": resolved_metric, "policy": resolved_policy, "items": entries}


@router.get("/ranking-lists", tier="public")
def list_ranking_lists(session: Session = Depends(get_db_session)) -> dict:
    """榜单总览：每个 active 的 RankingList 一张卡，附 Top 3 + 计数。

    供匿名访客 LeaderboardsPage 直接渲染。
    """
    ranking_lists = session.exec(
        select(RankingList).where(RankingList.status == "active").order_by(RankingList.updated_at.desc())
    ).all()

    # 匿名用户也走这个端点（tier=public），无法再调 tier=authed 的 /models 拿名字，
    # 所以在这里就把 top 项里出现的 model_id 一次性解析成 model_name 内嵌返回。
    model_name_by_id: dict[str, str] = {
        m.model_id: m.name for m in session.exec(select(Model)).all()
    }

    items: list[dict] = []
    for rl in ranking_lists:
        track = session.get(Track, rl.track_id)
        if track is None:
            continue
        # 复用 query_ranking 避免另起一份查询路径；取前 3。
        try:
            entries = query_ranking(session, rl.track_id, track.primary_metric_id, rl.default_policy)
        except Exception:
            entries = []
        all_entries = session.exec(
            select(RankingEntry).where(
                RankingEntry.ranking_list_id == rl.ranking_list_id,
                RankingEntry.metric_id == track.primary_metric_id,
                RankingEntry.policy == rl.default_policy,
            )
        ).all()
        model_count = len({e.model_id for e in all_entries})
        run_count = len({e.benchmarking_run_id for e in all_entries})
        items.append(
            {
                "ranking_list_id": rl.ranking_list_id,
                "track_id": rl.track_id,
                "track_name": track.name,
                "track_type": track.track_type,
                "primary_metric_id": track.primary_metric_id,
                "default_policy": rl.default_policy,
                "updated_at": rl.updated_at,
                "model_count": model_count,
                "run_count": run_count,
                "top": [
                    {
                        "rank": e.rank,
                        "model_id": e.model_id,
                        "model_name": model_name_by_id.get(e.model_id, e.model_id),
                        "metric_value": e.metric_value,
                    }
                    for e in entries[:3]
                ],
            }
        )
    return {"items": items}
