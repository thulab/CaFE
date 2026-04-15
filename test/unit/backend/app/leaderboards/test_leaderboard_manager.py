from __future__ import annotations

import unittest

from backend.app.datasets.domain import BatchGenerationRequest, TrackKind
from backend.app.datasets.manager import DataManager
from backend.app.leaderboards.manager import LeaderboardManager
from backend.app.models.domain import HuggingFaceModelRegistrationRequest
from backend.app.models.manager import ModelManager
from backend.app.tasks.domain import TaskRunRequest
from backend.app.tasks.manager import TaskManager
from test.support.helpers import FakeHuggingFaceRunner, temporary_runtime_dir


class LeaderboardManagerTest(unittest.TestCase):
    def test_overall_leaderboard_aggregates_track_rankings(self) -> None:
        with temporary_runtime_dir(prefix="leaderboard-") as runtime_root:
            data_manager = DataManager(runtime_root)
            model_manager = ModelManager(runtime_root)
            model_manager.huggingface_runner_factory = FakeHuggingFaceRunner
            task_manager = TaskManager(runtime_root, data_manager=data_manager, model_manager=model_manager)
            leaderboard_manager = LeaderboardManager(
                runtime_root,
                data_manager=data_manager,
                model_manager=model_manager,
                task_manager=task_manager,
            )

            custom_model = model_manager.register_huggingface_model(
                HuggingFaceModelRegistrationRequest(
                    repo_id="org/custom-forecast-model",
                    name="Custom Forecast Model",
                    manual="custom",
                    task="chronos-2",
                )
            )

            batch_a = data_manager.generate_batch(
                BatchGenerationRequest(track=TrackKind.FORECAST_ACCURACY, sample_count=1, context_length=24, horizon=8, seed=3)
            )
            batch_b = data_manager.generate_batch(
                BatchGenerationRequest(track=TrackKind.NOISE_ROBUSTNESS, sample_count=1, context_length=24, horizon=8, seed=4)
            )

            task_manager.run_task(TaskRunRequest(model_id="amazon-chronos-2", batch_id=batch_a.batch_id))
            task_manager.run_task(TaskRunRequest(model_id=custom_model.model_id, batch_id=batch_a.batch_id))
            task_manager.run_task(TaskRunRequest(model_id="amazon-chronos-2", batch_id=batch_b.batch_id))

            forecast_board = leaderboard_manager.track_leaderboard(TrackKind.FORECAST_ACCURACY)
            overall_board = leaderboard_manager.overall_leaderboard()

        self.assertEqual(forecast_board[0].rank, 1)
        self.assertEqual(forecast_board[0].metric_id, "mse")
        self.assertEqual(forecast_board[0].track_variant_id, "univariate_forecast.clean")
        self.assertEqual(forecast_board[0].metric_value, forecast_board[0].metric_snapshot["mse"])
        builtin_row = next(row for row in overall_board if row.model_id == "amazon-chronos-2")
        custom_row = next(row for row in overall_board if row.model_id == custom_model.model_id)
        self.assertEqual(builtin_row.metric_id, "mse")
        self.assertGreaterEqual(builtin_row.covered_tracks, custom_row.covered_tracks)
        self.assertLessEqual(builtin_row.rank, custom_row.rank)
