from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from app.core.errors import ApiError
from app.core.time import utc_now
from app.db.init_db import assert_manifest_can_create_successful_real_shard, assert_manifest_can_succeed_load
from app.models.dataset import DatasetLoadJob, DatasetManifest, Shard
from app.models.sample import SampleIndex
from app.services.csv_dataset_reader import CsvDatasetReader
from app.services.tsfile_store import TsFileStore


@dataclass(frozen=True)
class SampleWindow:
    source_row_start: int
    source_row_end: int
    context_start: int
    context_end: int
    horizon_start: int
    horizon_end: int
    context_length: int
    horizon: int


def build_windows(row_count: int, context_length: int, horizon: int, stride: int | None = None) -> list[SampleWindow]:
    stride = stride or horizon
    if context_length <= 0 or horizon <= 0 or stride <= 0:
        raise ApiError("split_config_invalid", "context_length, horizon, and stride must be positive")
    required = context_length + horizon
    if required > row_count:
        raise ApiError(
            "split_length_exceeds_rows",
            "context_length + horizon exceeds available rows",
            {"required": required, "actual": row_count},
        )

    windows: list[SampleWindow] = []
    for start in range(0, row_count - required + 1, stride):
        windows.append(
            SampleWindow(
                source_row_start=start,
                source_row_end=start + required - 1,
                context_start=start,
                context_end=start + context_length - 1,
                horizon_start=start + context_length,
                horizon_end=start + required - 1,
                context_length=context_length,
                horizon=horizon,
            )
        )
    if not windows:
        raise ApiError("sample_count_empty", "split configuration produced no samples")
    return windows


def subsample_windows(windows: list[SampleWindow], max_samples: int | None) -> list[SampleWindow]:
    """超量时沿序列均匀采样（含首尾，索引近似等距），可复现。max_samples 缺省或不小于窗数时全取。"""
    if not max_samples or max_samples >= len(windows):
        return windows
    positions = np.linspace(0, len(windows) - 1, max_samples)
    picked = sorted({int(round(position)) for position in positions})
    return [windows[index] for index in picked]


class DatasetLoadService:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = Path(runtime_dir)
        self.tsfiles_dir = self.runtime_dir / "tsfiles"
        self.tsfiles_dir.mkdir(parents=True, exist_ok=True)

    def create_load_job(
        self,
        session: Session,
        dataset_manifest_id: str,
        split_config: dict,
        seed: int = 0,
    ) -> DatasetLoadJob:
        manifest = session.get(DatasetManifest, dataset_manifest_id)
        if manifest is None:
            raise ApiError("dataset_manifest_not_found", "dataset manifest not found", {"dataset_manifest_id": dataset_manifest_id}, 404)

        assert_manifest_can_succeed_load(session, dataset_manifest_id)
        assert_manifest_can_create_successful_real_shard(session, dataset_manifest_id)

        job = DatasetLoadJob(
            dataset_manifest_id=dataset_manifest_id,
            split_config=split_config,
            seed=seed,
            started_at=utc_now(),
            current_step="validating",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        try:
            return self._execute_job(session, manifest, job)
        except ApiError as exc:
            self._cleanup_job_artifacts(job)
            job.status = "failed"
            job.error_code = exc.error_code
            job.error_message = exc.message
            job.finished_at = utc_now()
            job.updated_at = utc_now()
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def _execute_job(self, session: Session, manifest: DatasetManifest, job: DatasetLoadJob) -> DatasetLoadJob:
        config = job.split_config
        read_result = CsvDatasetReader().read(
            Path(manifest.source_uri),
            time_column=manifest.time_column,
            value_columns=manifest.value_columns or None,
            frequency=manifest.frequency,
        )

        target_columns = list(config.get("target_columns") or [])
        if len(target_columns) != 1:
            raise ApiError(
                "load_target_columns_invalid",
                "exactly one target column must be selected",
                {"target_columns": target_columns},
            )
        for target_column in target_columns:
            if target_column not in read_result.value_columns:
                raise ApiError(
                    "load_target_columns_invalid",
                    "target column must be among the dataset value columns",
                    {"target_column": target_column, "value_columns": read_result.value_columns},
                )

        windows = build_windows(
            read_result.row_count,
            context_length=int(config["context_length"]),
            horizon=int(config["horizon"]),
            stride=int(config.get("stride") or config["horizon"]),
        )
        max_samples = config.get("max_samples")
        windows = subsample_windows(windows, int(max_samples) if max_samples else None)

        job.status = "loading"
        job.current_step = "materializing_samples"
        session.add(job)
        session.commit()

        shard = Shard(
            dataset_manifest_id=manifest.dataset_manifest_id,
            load_job_id=job.load_job_id,
            source_uri=manifest.source_uri,
            time_range_start=read_result.timestamps[0].isoformat(),
            time_range_end=read_result.timestamps[-1].isoformat(),
            row_count=read_result.row_count,
            value_columns=read_result.value_columns,
            target_columns=target_columns,
            target_dim=len(target_columns),
            frequency=read_result.frequency,
            context_length=int(config["context_length"]),
            horizon=int(config["horizon"]),
            stride=int(config.get("stride") or config["horizon"]),
            sample_count=len(windows),
            status="ready",
        )
        dataset_id = shard.shard_id.replace("-", "")
        shard.dataset_id = dataset_id
        session.add(shard)
        session.commit()
        session.refresh(shard)

        tsfile_path = TsFileStore(self.tsfiles_dir).write(
            shard.shard_id,
            dataset_id,
            read_result.timestamps,
            read_result.value_columns,
            read_result.values,
        )

        from app.services.sample_store import SampleStore

        sample_indexes = SampleStore().write_samples(
            shard.shard_id,
            dataset_id,
            tsfile_path,
            windows,
            target_columns,
        )
        for sample_index in sample_indexes:
            session.add(sample_index)

        shard.tsfile_uri = str(tsfile_path)
        shard.storage_uri = str(tsfile_path)
        manifest.status = "loaded"
        manifest.frequency = read_result.frequency
        manifest.updated_at = utc_now()
        job.status = "succeeded"
        job.current_step = "succeeded"
        job.output_shard_id = shard.shard_id
        job.validation_summary = {
            "row_count": read_result.row_count,
            "sample_count": len(windows),
            "frequency": read_result.frequency,
            "columns": read_result.columns,
            "value_columns": read_result.value_columns,
            "target_columns": target_columns,
        }
        job.finished_at = utc_now()
        job.updated_at = utc_now()
        session.add(manifest)
        session.add(shard)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job

    def _cleanup_job_artifacts(self, job: DatasetLoadJob) -> None:
        if job.output_shard_id:
            path = self.tsfiles_dir / f"{job.output_shard_id}.tsfile"
            if path.exists():
                path.unlink()


def sample_count_for_shard(session: Session, shard_id: str) -> int:
    return len(session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard_id)).all())
