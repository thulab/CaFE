from sqlmodel import Session, SQLModel, select

from app.core.errors import ApiError
from app.models.auth import Permission, Role, RolePermission, User, UserRole
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, CapabilityBlockShard, ForecastArtifact, RunEvent, Task, Track, Unit
from app.models.dataset import DatasetLoadJob, DatasetManifest, Shard
from app.models.metric import MetricDefinition, MetricResult
from app.models.model_registry import Model
from app.models.ranking import RankingEntry, RankingList
from app.models.report import Report
from app.models.sample import SampleIndex
from app.models.series_point import SeriesPoint


def init_db(engine) -> None:
    # Imports above register all SQLModel table metadata.
    SQLModel.metadata.create_all(engine)


def assert_manifest_can_succeed_load(session: Session, dataset_manifest_id: str) -> None:
    existing = session.exec(
        select(DatasetLoadJob).where(
            DatasetLoadJob.dataset_manifest_id == dataset_manifest_id,
            DatasetLoadJob.status == "succeeded",
        )
    ).first()
    if existing is not None:
        raise ApiError(
            "dataset_manifest_already_loaded",
            "dataset manifest already has a successful load job",
            {"dataset_manifest_id": dataset_manifest_id},
        )


def assert_manifest_can_create_successful_real_shard(session: Session, dataset_manifest_id: str) -> None:
    existing = session.exec(
        select(Shard).where(
            Shard.dataset_manifest_id == dataset_manifest_id,
            Shard.shard_type == "real",
            Shard.status == "ready",
        )
    ).first()
    if existing is not None:
        raise ApiError(
            "dataset_manifest_already_has_real_shard",
            "dataset manifest already has a successful real shard",
            {"dataset_manifest_id": dataset_manifest_id},
        )
