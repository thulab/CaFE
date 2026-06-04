import pytest
from sqlmodel import Session, create_engine, select

from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.report import Report
from app.services.report_service import generate_run_report
from app.services.run_executor import cancel_run, create_benchmarking_run, execute_run
from tests.run_helpers import create_loaded_track_with_models


def test_cancelled_run_execution_does_not_create_report(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        cancel_run(session, run.benchmarking_run_id)
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status == "cancelled"
        assert run.report_id is None
        assert not session.exec(select(Report).where(Report.benchmarking_run_id == run.benchmarking_run_id)).all()
        assert not (tmp_path / "runtime" / "reports" / f"{run.benchmarking_run_id}.json").exists()

        with pytest.raises(ApiError) as exc_info:
            generate_run_report(session, run.benchmarking_run_id, tmp_path / "runtime")
        assert exc_info.value.error_code == "cancelled_run_has_no_report"
