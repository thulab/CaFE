import pytest
from sqlmodel import Session

from app.core.errors import ApiError
from app.core.config import get_settings
from app.services.run_executor import create_benchmarking_run
from tests.run_helpers import create_loaded_track_with_models


def test_create_run_rejects_model_when_context_is_shorter_than_min_input(app):
    with Session(app.state.engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, get_settings().runtime_dir, model_count=1)
        model = models[0]
        model.forecast_limits = {"min_input_length": 16, "max_output_length": 720, "max_target_count": 1, "max_covariate_count": 0}
        session.add(model)
        session.commit()

        with pytest.raises(ApiError) as exc:
            create_benchmarking_run(session, track.track_id, [model.model_id])

    assert exc.value.error_code == "model_window_unsupported"
    assert exc.value.details["reasons"][model.model_id] == ["min_input_length"]


def test_create_run_rejects_model_when_horizon_exceeds_max_output(app):
    with Session(app.state.engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, get_settings().runtime_dir, model_count=1)
        model = models[0]
        model.forecast_limits = {"max_output_length": 2, "max_target_count": 1, "max_covariate_count": 0}
        session.add(model)
        session.commit()

        with pytest.raises(ApiError) as exc:
            create_benchmarking_run(session, track.track_id, [model.model_id])

    assert exc.value.error_code == "model_window_unsupported"
    assert exc.value.details["reasons"][model.model_id] == ["max_output_length"]
