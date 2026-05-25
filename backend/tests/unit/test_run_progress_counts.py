"""#20 — run progress must report real per-sample counts.

``build_run_progress`` previously hardcoded ``completed_samples: 0`` and
``failed_samples: 0``. They must instead reflect the per-sample forecast
rows actually written during the run: ``completed_samples`` = samples that
produced a forecast (status ``succeeded``); ``failed_samples`` = samples
whose status is ``failed``.
"""
from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import ForecastArtifact
from app.services.forecast_store import ForecastStore
from app.services.run_executor import build_run_progress, create_benchmarking_run, execute_run
from tests.run_helpers import create_loaded_track_with_models


class _AlwaysRaisingAdapter:
    """Adapter whose ``forecast`` raises for every sample (all failed)."""

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        raise RuntimeError("adapter boom")


def _row_status_counts(forecasts_dir, artifacts):
    store = ForecastStore(forecasts_dir)
    succeeded = 0
    failed = 0
    for artifact in artifacts:
        for row in store.read_forecasts(artifact.storage_uri):
            if row["status"] == "succeeded":
                succeeded += 1
            elif row["status"] == "failed":
                failed += 1
    return succeeded, failed


def test_progress_counts_completed_samples_on_success(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        run_id = run.benchmarking_run_id

        execute_run(session, run_id, tmp_path / "runtime")

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)
        ).all()
        expected_completed, expected_failed = _row_status_counts(tmp_path / "runtime" / "forecasts", artifacts)
        assert expected_completed > 0  # sanity: the flow produced real samples

        progress = build_run_progress(session, run_id)
        assert progress["progress"]["completed_samples"] == expected_completed
        assert progress["progress"]["failed_samples"] == expected_failed
        assert progress["progress"]["failed_samples"] == 0


def test_progress_counts_failed_samples_on_adapter_failure(tmp_path, monkeypatch):
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

        execute_run(session, run_id, tmp_path / "runtime")

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)
        ).all()
        expected_completed, expected_failed = _row_status_counts(tmp_path / "runtime" / "forecasts", artifacts)
        assert expected_failed > 0  # sanity: every sample failed

        progress = build_run_progress(session, run_id)
        assert progress["progress"]["failed_samples"] == expected_failed
        assert progress["progress"]["completed_samples"] == expected_completed
