from __future__ import annotations

import unittest
from datetime import datetime, timezone

from backend.app.datasets.domain import TrackKind
from backend.app.leaderboards.domain import (
    AdminDashboardOverview,
    BatchSummary,
    LeaderboardEntry,
    OverallLeaderboardEntry,
    TaskSummary,
    TrackLeaderboard,
    UserDashboardOverview,
)
from backend.app.tasks.domain import TaskStatus


class LeaderboardEntryTest(unittest.TestCase):
    def test_create_entry(self) -> None:
        entry = LeaderboardEntry(
            rank=1,
            task_id="task-1",
            model_id="model-1",
            model_name="Model 1",
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            track_variant_id="forecast_accuracy.clean",
            track_label="Forecast Accuracy",
            metric_id="mse",
            metric_value=0.1,
        )
        self.assertEqual(entry.rank, 1)
        self.assertEqual(entry.metric_value, 0.1)

    def test_composite_score_from_snapshot(self) -> None:
        entry = LeaderboardEntry(
            rank=1,
            task_id="task-1",
            model_id="model-1",
            model_name="Model 1",
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            track_variant_id="forecast_accuracy.clean",
            track_label="Forecast Accuracy",
            metric_id="mse",
            metric_value=0.1,
            metric_snapshot={"composite_score": 95.0, "mse": 0.1, "smape": 0.2, "latency_ms": 10.0},
        )
        self.assertEqual(entry.composite_score, 95.0)
        self.assertEqual(entry.mse, 0.1)
        self.assertEqual(entry.smape, 0.2)
        self.assertEqual(entry.mean_latency_ms, 10.0)

    def test_composite_score_none_when_missing(self) -> None:
        entry = LeaderboardEntry(
            rank=1,
            task_id="task-1",
            model_id="model-1",
            model_name="Model 1",
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            track_variant_id="forecast_accuracy.clean",
            track_label="Forecast Accuracy",
            metric_id="mse",
            metric_value=0.1,
        )
        self.assertIsNone(entry.composite_score)


class BatchSummaryTest(unittest.TestCase):
    def test_create_batch_summary(self) -> None:
        summary = BatchSummary(
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            track_variant_id="forecast_accuracy.clean",
            policy="equal",
            created_at=datetime.now(timezone.utc),
            sample_count=100,
            context_length=96,
            horizon=24,
            validation_passed=True,
        )
        self.assertEqual(summary.batch_id, "batch-1")
        self.assertEqual(summary.sample_count, 100)
        self.assertTrue(summary.validation_passed)


class TaskSummaryTest(unittest.TestCase):
    def test_create_task_summary(self) -> None:
        summary = TaskSummary(
            task_id="task-1",
            model_id="model-1",
            model_name="Model 1",
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            track_variant_id="forecast_accuracy.clean",
            status=TaskStatus.SUCCEEDED,
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(summary.task_id, "task-1")
        self.assertEqual(summary.status, TaskStatus.SUCCEEDED)

    def test_task_summary_with_metrics(self) -> None:
        summary = TaskSummary(
            task_id="task-1",
            model_id="model-1",
            model_name="Model 1",
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            status=TaskStatus.SUCCEEDED,
            created_at=datetime.now(timezone.utc),
            primary_metric_id="mse",
            primary_metric_value=0.1,
            composite_score=95.0,
        )
        self.assertEqual(summary.primary_metric_value, 0.1)
        self.assertEqual(summary.composite_score, 95.0)


class TrackLeaderboardTest(unittest.TestCase):
    def test_create_track_leaderboard(self) -> None:
        leaderboard = TrackLeaderboard(
            track="forecast_accuracy.clean",
            track_label="Forecast Accuracy",
            metric_id="mse",
            ranking_strategy="mean",
            entries=[],
        )
        self.assertEqual(leaderboard.track, "forecast_accuracy.clean")
        self.assertEqual(leaderboard.metric_id, "mse")


class OverallLeaderboardEntryTest(unittest.TestCase):
    def test_create_overall_entry(self) -> None:
        entry = OverallLeaderboardEntry(
            rank=1,
            model_id="model-1",
            model_name="Model 1",
            metric_id="mse",
            rank_sum=4,
            covered_tracks=4,
            mean_metric_value=0.15,
            track_ranks={"track1": 1, "track2": 1, "track3": 1, "track4": 1},
            track_values={"track1": 0.1, "track2": 0.15, "track3": 0.2, "track4": 0.1},
        )
        self.assertEqual(entry.rank, 1)
        self.assertEqual(entry.covered_tracks, 4)
        self.assertEqual(entry.mean_composite_score, 0.15)
        self.assertEqual(entry.track_scores["track1"], 0.1)


class UserDashboardOverviewTest(unittest.TestCase):
    def test_create_user_overview(self) -> None:
        overview = UserDashboardOverview(
            tracks=[],
            models=[],
            overall_leaderboard_strategy="rank_sum",
            overall_metric_id="mse",
            overall_leaderboard=[],
            track_leaderboards=[],
        )
        self.assertEqual(overview.overall_metric_id, "mse")
        self.assertEqual(overview.overall_leaderboard_strategy, "rank_sum")


class AdminDashboardOverviewTest(unittest.TestCase):
    def test_create_admin_overview(self) -> None:
        overview = AdminDashboardOverview(
            tracks=[],
            models=[],
            batches=[],
            recent_tasks=[],
            overall_leaderboard_strategy="rank_sum",
            overall_metric_id="mse",
            leaderboard=[],
        )
        self.assertEqual(overview.overall_metric_id, "mse")
        self.assertEqual(len(overview.batches), 0)
        self.assertEqual(len(overview.recent_tasks), 0)


if __name__ == "__main__":
    unittest.main()