from sqlmodel import Session, select

from app.core.errors import ApiError
from app.models.benchmark import BenchmarkingRun, ForecastArtifact, Task, Unit
from app.models.dataset import Shard
from app.models.model_registry import Model
from app.models.ranking import RankingList
from app.models.sample import SampleIndex
from app.services.forecast_store import ForecastStore
from app.services.report_service import _latest_success_units_by_model, _track_terminal_runs, _track_terminal_units
from app.services.sample_store import SampleStore


def build_sample_forecast(session: Session, sample_id: str, run_id: str) -> dict:
    sample_index = session.get(SampleIndex, sample_id)
    sample, shard = _sample_window(session, sample_index)
    run = session.get(BenchmarkingRun, run_id)
    ranking_list = session.exec(select(RankingList).where(RankingList.track_id == run.track_id)).first() if run else None
    artifacts = session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id, ForecastArtifact.shard_id == shard.shard_id)).all()
    models, capability_block_id = _forecast_models(session, sample_id, artifacts, shard.capability_block_id)
    return _sample_forecast_payload(
        sample_id,
        sample_index,
        sample,
        shard,
        capability_block_id,
        models,
        {
            "run": run_id,
            "report": run.report_id if run else None,
            "ranking": ranking_list.ranking_list_id if ranking_list else None,
        },
        benchmarking_run_id=run_id,
    )


def build_track_sample_forecast(session: Session, sample_id: str, track_id: str) -> dict:
    sample_index = session.get(SampleIndex, sample_id)
    sample, shard = _sample_window(session, sample_index)
    run_ids, run_by_id = _track_terminal_runs(session, track_id)
    terminal_units = _track_terminal_units(session, run_ids) if run_ids else []
    latest_success_units = _latest_success_units_by_model(terminal_units, run_by_id)
    unit_ids = [unit.unit_id for unit in latest_success_units.values()]
    artifacts = session.exec(
        select(ForecastArtifact).where(
            ForecastArtifact.unit_id.in_(unit_ids),
            ForecastArtifact.shard_id == shard.shard_id,
        )
    ).all() if unit_ids else []
    models, capability_block_id = _forecast_models(session, sample_id, artifacts, shard.capability_block_id)
    ranking_list = session.exec(select(RankingList).where(RankingList.track_id == track_id)).first()
    return _sample_forecast_payload(
        sample_id,
        sample_index,
        sample,
        shard,
        capability_block_id,
        models,
        {
            "run": None,
            "report": None,
            "ranking": ranking_list.ranking_list_id if ranking_list else None,
            "track": track_id,
        },
    )


def _sample_window(session: Session, sample_index: SampleIndex | None) -> tuple[dict, Shard]:
    if sample_index is None:
        raise ApiError("sample_not_found", "sample not found", {}, 404)
    sample = SampleStore().read_by_ref(session, sample_index.storage_ref)
    shard = session.get(Shard, sample_index.shard_id)
    if shard is None:
        raise ApiError("shard_not_found", "sample shard not found", {"shard_id": sample_index.shard_id}, 404)
    return sample, shard


def _forecast_models(session: Session, sample_id: str, artifacts: list[ForecastArtifact], default_capability_block_id: str | None) -> tuple[list[dict], str | None]:
    models = []
    capability_block_id = default_capability_block_id
    for artifact in artifacts:
        matching = [row for row in ForecastStore(".").read_forecasts(artifact.storage_uri) if row["sample_id"] == sample_id]
        if not matching:
            continue
        row = matching[0]
        model = session.get(Model, artifact.model_id)
        unit = session.get(Unit, artifact.unit_id)
        task = session.get(Task, artifact.task_id)
        capability_block_id = task.capability_block_id if task else capability_block_id
        models.append(
            {
                "model_id": artifact.model_id,
                "model_name": model.name if model else artifact.model_id,
                "unit_id": artifact.unit_id,
                "task_id": artifact.task_id,
                "status": row["status"],
                "forecast": row["forecast"],
                "metrics": row["metrics"],
                "forecast_artifact_id": artifact.forecast_artifact_id,
                "error_code": row["error_code"],
                "error_message": row["error_message"],
                "unit_status": unit.status if unit else None,
                "task_status": task.status if task else None,
            }
        )
    return models, capability_block_id


def _sample_forecast_payload(
    sample_id: str,
    sample_index: SampleIndex,
    sample: dict,
    shard: Shard,
    capability_block_id: str | None,
    models: list[dict],
    links: dict,
    benchmarking_run_id: str | None = None,
) -> dict:
    payload = {
        "sample_id": sample_id,
        "shard_id": shard.shard_id,
        "capability_block_id": capability_block_id,
        "sample_index": sample_index.sample_index,
        "context_start": sample_index.context_start,
        "context_end": sample_index.context_end,
        "horizon_start": sample_index.horizon_start,
        "horizon_end": sample_index.horizon_end,
        "history_start_at": sample["history_timestamps"][0] if sample["history_timestamps"] else None,
        "history_end_at": sample["history_timestamps"][-1] if sample["history_timestamps"] else None,
        "forecast_start_at": sample["future_timestamps"][0] if sample["future_timestamps"] else None,
        "forecast_end_at": sample["future_timestamps"][-1] if sample["future_timestamps"] else None,
        "history_timestamps": sample["history_timestamps"],
        "future_timestamps": sample["future_timestamps"],
        "target_column_names": sample["target_column_names"],
        "covariate_column_names": sample.get("covariate_column_names", []),
        "target_history": sample["target_history"],
        "target_future": sample["target_future"],
        "history_cov": sample.get("history_cov", []),
        "future_cov": sample.get("future_cov", []),
        "models": models,
        "links": links,
    }
    if benchmarking_run_id is not None:
        payload["benchmarking_run_id"] = benchmarking_run_id
    return {
        **payload,
    }
