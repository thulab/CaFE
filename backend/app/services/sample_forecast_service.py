from sqlmodel import Session, select

from app.models.benchmark import CapabilityBlock, ForecastArtifact, Task, Unit
from app.models.dataset import Shard
from app.models.model_registry import Model
from app.models.sample import SampleIndex
from app.services.forecast_store import ForecastStore
from app.services.sample_store import SampleStore


def build_sample_forecast(session: Session, sample_id: str, run_id: str) -> dict:
    sample_index = session.get(SampleIndex, sample_id)
    sample = SampleStore().read_by_ref(sample_index.materialized_sample_uri, sample_index.storage_ref)
    shard = session.get(Shard, sample_index.shard_id)
    artifacts = session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id, ForecastArtifact.shard_id == shard.shard_id)).all()
    models = []
    for artifact in artifacts:
        matching = [row for row in ForecastStore(".").read_forecasts(artifact.storage_uri) if row["sample_id"] == sample_id]
        if not matching:
            continue
        row = matching[0]
        model = session.get(Model, artifact.model_id)
        unit = session.get(Unit, artifact.unit_id)
        task = session.get(Task, artifact.task_id)
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
    return {
        "sample_id": sample_id,
        "benchmarking_run_id": run_id,
        "shard_id": shard.shard_id,
        "capability_block_id": shard.capability_block_id,
        "history_timestamps": sample["history_timestamps"],
        "future_timestamps": sample["future_timestamps"],
        "target_column_names": sample["target_column_names"],
        "target_history": sample["target_history"],
        "target_future": sample["target_future"],
        "models": models,
        "links": {"run": run_id, "report": None, "ranking": None},
    }
