from __future__ import annotations

import unittest

from backend.app.data_management.domain import BatchGenerationRequest, TrackKind
from backend.app.data_management.manager import DataManager
from backend.app.leaderboard_management.manager import LeaderboardManager
from backend.app.model_management.domain import ModelAdapter, ModelRegistrationRequest
from backend.app.model_management.manager import ModelManager
from backend.app.task_management.domain import TaskRunRequest
from backend.app.task_management.manager import TaskManager
from test.support.helpers import temporary_runtime_dir


class LeaderboardManagerTest(unittest.TestCase):
    def test_overall_leaderboard_aggregates_track_rankings(self) -> None:
        with temporary_runtime_dir(prefix="leaderboard-") as runtime_root:
            data_manager = DataManager(runtime_root)
            model_manager = ModelManager(runtime_root)
            task_manager = TaskManager(runtime_root, data_manager=data_manager, model_manager=model_manager)
            leaderboard_manager = LeaderboardManager(
                runtime_root,
                data_manager=data_manager,
                model_manager=model_manager,
                task_manager=task_manager,
            )

            model_manager.register_model(
                ModelRegistrationRequest(
                    model_id="custom-recent-mean",
                    name="Custom Recent Mean",
                    adapter=ModelAdapter.RECENT_MEAN,
                    manual="custom",
                )
            )

            batch_a = data_manager.generate_batch(
                BatchGenerationRequest(track=TrackKind.FORECAST_ACCURACY, sample_count=1, context_length=24, horizon=8, seed=3)
            )
            batch_b = data_manager.generate_batch(
                BatchGenerationRequest(track=TrackKind.NOISE_ROBUSTNESS, sample_count=1, context_length=24, horizon=8, seed=4)
            )

            task_manager.run_task(TaskRunRequest(model_id="seasonal-naive-stub", batch_id=batch_a.batch_id))
            task_manager.run_task(TaskRunRequest(model_id="custom-recent-mean", batch_id=batch_a.batch_id))
            task_manager.run_task(TaskRunRequest(model_id="seasonal-naive-stub", batch_id=batch_b.batch_id))

            forecast_board = leaderboard_manager.track_leaderboard(TrackKind.FORECAST_ACCURACY)
            overall_board = leaderboard_manager.overall_leaderboard()

        self.assertEqual(forecast_board[0].rank, 1)
        seasonal_row = next(row for row in overall_board if row.model_id == "seasonal-naive-stub")
        custom_row = next(row for row in overall_board if row.model_id == "custom-recent-mean")
        self.assertGreaterEqual(seasonal_row.covered_tracks, custom_row.covered_tracks)
        self.assertLessEqual(seasonal_row.rank, custom_row.rank)
