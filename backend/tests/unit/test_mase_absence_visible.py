"""Item #14: a flat (stationary) target history makes MASE undefined (scale == 0).

Previously ``compute_sample_metrics`` silently omitted the ``mase`` key, so a
"succeeded" model vanished from the MASE (primary) ranking with no explanation.
These tests pin the NEW behaviour: the absence must be VISIBLE.

  - ``compute_sample_metrics`` exposes mase as ``None`` WITH a structured reason
    (``mase_unavailable_reason(result) == "flat_history"``) instead of omitting it.
  - The run report surfaces the unavailability and its reason for the affected
    unit, rather than the metric silently missing.
"""

import json

from sqlmodel import Session, create_engine

from app.db.init_db import init_db
from app.services.metric_service import (
    compute_sample_metrics,
    mase_unavailable_reason,
)
from app.services.report_service import generate_run_report
from app.services.run_executor import create_benchmarking_run, execute_run


def test_flat_history_exposes_mase_none_with_reason():
    """scale == 0 → mase value is None AND the reason is discoverable."""
    target_history = [[5.0], [5.0], [5.0]]  # flat → naive MAE scale == 0
    target_future = [[5.0], [6.0]]
    forecast = [[5.0], [5.0]]

    result = compute_sample_metrics(target_future, forecast, target_history=target_history)

    # mse / mae still carry the sample
    assert "mse" in result
    assert "mae" in result
    # mase is exposed as None (not silently absent) WITH a reason
    assert result.get("mase") is None
    assert mase_unavailable_reason(result) == "flat_history"


def test_single_row_history_reason_is_distinct():
    """A single-row history also yields no mase, with its own reason."""
    result = compute_sample_metrics([[12.0]], [[11.0]], target_history=[[10.0]])

    assert result.get("mase") is None
    assert mase_unavailable_reason(result) == "no_history_diffs"


def test_normal_history_has_no_unavailable_reason():
    result = compute_sample_metrics([[16.0], [18.0]], [[16.0], [17.0]], target_history=[[10.0], [12.0], [14.0]])

    assert result["mase"] is not None
    assert mase_unavailable_reason(result) is None


def _patch_flat_history(session: Session) -> None:
    """Flatten every series point so all sample histories are stationary."""
    from app.models.series_point import SeriesPoint
    from sqlmodel import select

    for point in session.exec(select(SeriesPoint)).all():
        point.values_json = {key: 5.0 for key in point.values_json}
        session.add(point)
    session.commit()


def test_report_surfaces_mase_unavailable_for_flat_unit(tmp_path):
    from tests.run_helpers import create_loaded_track_with_models

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        _patch_flat_history(session)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        report = generate_run_report(session, run.benchmarking_run_id, tmp_path / "runtime")
        payload = json.loads((tmp_path / "runtime" / "reports" / f"{run.benchmarking_run_id}.json").read_text())

    assert report.status == "ready"
    unit_entry = payload["model_metrics"][0]
    # unit still succeeded on mse/mae, but mase is reported unavailable WITH a reason
    assert unit_entry["status"] == "succeeded"
    assert unit_entry["metrics"].get("mase") is None
    assert unit_entry.get("mase_unavailable_reason") == "flat_history"
