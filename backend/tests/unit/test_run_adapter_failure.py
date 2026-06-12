"""#16 — adapter errors must not crash the run nor wedge the queue.

When ``adapter.forecast`` raises for a sample, the executor must:
- record a *failed* forecast row for that sample (``status == "failed"``
  with an ``error_code``) instead of letting the exception escape;
- still drive the run to a terminal status (never stuck in ``running``);
- and the background queue must still be drained so a subsequently
  submitted run can run to a terminal status too.
"""
from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import FailedSampleRerunJob, ForecastArtifact, Task, Unit
from app.models.dataset import DatasetManifest
from app.models.model_registry import Model
from app.services.dataset_load_service import DatasetLoadService
from app.services.forecast_store import ForecastStore
from app.services.ranking_service import query_ranking
from app.services.run_executor import (
    create_benchmarking_run,
    execute_failed_sample_rerun_job,
    execute_run,
    get_active_failed_sample_rerun_job,
    list_failed_samples,
    rerun_failed_samples,
    start_failed_sample_rerun,
)
from app.services.stub_timer_adapter import StubTimerAdapter
from app.services.track_service import create_real_capability_block, create_track_with_blocks
from app.workers.run_queue import RunQueue
from tests.run_helpers import create_loaded_track_with_models

TERMINAL = {"succeeded", "partial_succeeded", "failed", "cancelled"}


class _AlwaysRaisingAdapter:
    """Adapter whose ``forecast`` raises for every sample."""

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        raise RuntimeError("adapter boom")


class _FirstBadShapeAdapter:
    def __init__(self):
        self._calls = 0
        self._stub = StubTimerAdapter()

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        self._calls += 1
        if self._calls == 1:
            return []
        return self._stub.forecast(sample, model, timeout_seconds)


class _LoadFailingAdapter:
    def ensure_model_loaded(self, model, timeout_seconds):  # noqa: ANN001, ANN201
        raise RuntimeError("load boom")

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        raise AssertionError("forecast should not be called when model load fails")


class _LifecycleCountingAdapter:
    def __init__(self):
        self._stub = StubTimerAdapter()
        self.load_calls = 0
        self.unload_calls = 0

    def ensure_model_loaded(self, model, timeout_seconds):  # noqa: ANN001, ANN201, ARG002
        self.load_calls += 1

    def unload_model(self, model, timeout_seconds):  # noqa: ANN001, ANN201, ARG002
        self.unload_calls += 1

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        return self._stub.forecast(sample, model, timeout_seconds)


def _all_forecast_rows(forecasts_dir, artifacts: list[ForecastArtifact]) -> list[dict]:
    store = ForecastStore(forecasts_dir)
    rows: list[dict] = []
    for artifact in artifacts:
        rows.extend(store.read_forecasts(artifact.storage_uri))
    return rows


def _create_loaded_track_with_two_shards(session: Session, runtime_dir):
    source = runtime_dir.parent / "valid_hourly_20.csv"
    source.write_text(
        "time,target\n"
        + "\n".join(f"2024-01-01 {hour:02d}:00:00,{hour}" for hour in range(20)),
        encoding="utf-8",
    )
    shard_ids = []
    for index in range(2):
        manifest = DatasetManifest(
            name=f"multi-shard-demo-{index}",
            domain="energy",
            source_uri=str(source),
            time_column="time",
        )
        session.add(manifest)
        session.commit()
        session.refresh(manifest)
        job = DatasetLoadService(runtime_dir).create_load_job(
            session,
            manifest.dataset_manifest_id,
            {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]},
        )
        shard_ids.append(job.output_shard_id)
    block = create_real_capability_block(session, "real block", shard_ids)
    track, ranking = create_track_with_blocks(session, "multi shard track", [block.capability_block_id], "mase")
    return track, ranking


def test_adapter_failure_does_not_crash_run_and_marks_sample_failed(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _AlwaysRaisingAdapter(),
        )

        # Must not raise even though every adapter.forecast call raises.
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status in TERMINAL
        assert run.status != "running"

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)
        ).all()
        rows = _all_forecast_rows(tmp_path / "runtime" / "forecasts", artifacts)
        assert rows, "expected forecast rows to be written even on adapter failure"
        assert all(row["status"] == "failed" for row in rows)
        assert all(row["error_code"] for row in rows)


def test_sample_metric_error_marks_only_that_sample_failed_and_continues(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _FirstBadShapeAdapter(),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        unit = session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).one()
        task = session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).one()
        assert run.status == "partial_succeeded"
        assert unit.status == "partial_succeeded"
        assert task.status == "partial_succeeded"

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)
        ).all()
        rows = _all_forecast_rows(tmp_path / "runtime" / "forecasts", artifacts)
        assert [row["status"] for row in rows].count("failed") == 1
        assert [row["status"] for row in rows].count("succeeded") > 0
        failed = next(row for row in rows if row["status"] == "failed")
        assert failed["error_code"] == "metric_error"
        assert "same flattened length" in failed["error_message"]
        assert query_ranking(session, track.track_id, "mse", "latest_valid_result") == []


def test_failed_samples_can_be_inspected_and_rerun_into_valid_ranking(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _FirstBadShapeAdapter(),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        failures = list_failed_samples(session, run.benchmarking_run_id)
        assert failures["total"] == 1
        assert failures["items"][0]["model_id"] == models[0].model_id
        assert failures["items"][0]["error_code"] == "metric_error"
        assert "same flattened length" in failures["items"][0]["error_message"]
        assert query_ranking(session, track.track_id, "mse", "latest_valid_result") == []

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: StubTimerAdapter(),
        )

        result = rerun_failed_samples(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        unit = session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).one()
        task = session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).one()
        assert result == {"rerun_samples": 1, "remaining_failed_samples": 0}
        assert run.status == "succeeded"
        assert unit.status == "succeeded"
        assert task.status == "succeeded"
        assert task.failed_sample_count == 0
        remaining = list_failed_samples(session, run.benchmarking_run_id)
        assert remaining["items"] == []
        assert remaining["total"] == 0
        assert remaining["summary"] == []
        assert query_ranking(session, track.track_id, "mse", "latest_valid_result")

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)
        ).all()
        rows = _all_forecast_rows(tmp_path / "runtime" / "forecasts", artifacts)
        assert all(row["status"] == "succeeded" for row in rows)


def test_failed_sample_summary_groups_and_paginates_details(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _AlwaysRaisingAdapter(),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        failures = list_failed_samples(session, run.benchmarking_run_id, limit=2, offset=1)
        assert failures["total"] == 4
        assert failures["limit"] == 2
        assert failures["offset"] == 1
        assert len(failures["items"]) == 2
        assert failures["summary"] == [
            {
                "error_code": "adapter_error",
                "error_message": "adapter boom",
                "count": 4,
                "model_count": 1,
                "capability_count": 1,
                "sample_count": 4,
            }
        ]

        filtered = list_failed_samples(
            session,
            run.benchmarking_run_id,
            limit=50,
            offset=0,
            error_code="adapter_error",
            error_message="adapter boom",
        )
        assert filtered["total"] == 4
        assert {item["error_code"] for item in filtered["items"]} == {"adapter_error"}


def test_failed_sample_rerun_job_tracks_progress_and_blocks_duplicate(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _FirstBadShapeAdapter(),
        )
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        job = start_failed_sample_rerun(session, run.benchmarking_run_id)
        assert job.status == "queued"
        assert job.total_samples == 1
        assert get_active_failed_sample_rerun_job(session, run.benchmarking_run_id).rerun_job_id == job.rerun_job_id

        try:
            start_failed_sample_rerun(session, run.benchmarking_run_id)
        except Exception as error:  # noqa: BLE001
            assert getattr(error, "error_code", "") == "failed_sample_rerun_active"
        else:
            raise AssertionError("expected duplicate rerun to be rejected")

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: StubTimerAdapter(),
        )
        execute_failed_sample_rerun_job(session, job.rerun_job_id, tmp_path / "runtime")

        session.refresh(job)
        assert job.status == "succeeded"
        assert job.activity_status == "succeeded"
        assert job.processed_samples == 1
        assert job.succeeded_samples == 1
        assert job.failed_samples == 0
        assert get_active_failed_sample_rerun_job(session, run.benchmarking_run_id) is None
        assert session.exec(select(FailedSampleRerunJob)).one().rerun_job_id == job.rerun_job_id


def test_failed_sample_rerun_loads_model_once_per_unit_across_artifacts(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking = _create_loaded_track_with_two_shards(session, tmp_path / "runtime")
        model = Model(name="Lifecycle Model", model_family="Timer", model_version="life", endpoint_uri="stub://lifecycle")
        session.add(model)
        session.commit()
        session.refresh(model)
        run = create_benchmarking_run(session, track.track_id, [model.model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _AlwaysRaisingAdapter(),
        )
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)
        ).all()
        assert len(artifacts) == 2
        assert list_failed_samples(session, run.benchmarking_run_id, limit=0)["total"] > 1

        adapter = _LifecycleCountingAdapter()
        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: adapter,
        )

        result = rerun_failed_samples(session, run.benchmarking_run_id, tmp_path / "runtime")

        assert result["remaining_failed_samples"] == 0
        assert adapter.load_calls == 1
        assert adapter.unload_calls == 1


def test_model_load_failure_marks_unit_failed_without_forecast(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _LoadFailingAdapter(),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status == "failed"
        unit = session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).one()
        task = session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).one()
        assert unit.status == "failed"
        assert task.status == "failed"
        assert task.error_code == "model_load_error"
        assert "load boom" in task.error_message
        assert not session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)).all()


def test_queue_drains_after_adapter_failure_so_next_run_completes(tmp_path, monkeypatch):
    """A failing run must not wedge the queue: a later run still reaches terminal."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        run_id = run.benchmarking_run_id

    monkeypatch.setattr(
        "app.services.run_executor.get_model_adapter",
        lambda settings: _AlwaysRaisingAdapter(),
    )

    queue = RunQueue()

    # Exercise the real background entrypoint under test.
    from app.api.routes.benchmarking_runs import _execute_in_background

    assert queue.submit(run_id) == "running"
    _execute_in_background(engine, run_id, tmp_path / "runtime", queue)

    # complete() must have been called via try/finally despite the failing run.
    assert queue.running_run_id is None

    # And the next submission can now run, proving the queue isn't wedged.
    assert queue.submit("next-run") == "running"
