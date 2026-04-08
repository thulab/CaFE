from __future__ import annotations

from pathlib import Path

from .config import AppSettings, get_settings
from .data_management.manager import DataManager
from .data_management.domain import TrackKind
from .errors import BenchmarkError, NotFoundError
from .leaderboard_management.domain import (
    AdminDashboardOverview,
    BatchSummary,
    TaskSummary,
    TrackLeaderboard,
    UserDashboardOverview,
)
from .leaderboard_management.manager import LeaderboardManager
from .model_management.domain import ModelRecord
from .model_management.manager import ExecutionResult, ModelManager
from .storage import FileRepository
from .task_management.manager import TaskManager


class BenchmarkEngine:
    def __init__(self, runtime_root: Path, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = FileRepository(runtime_root)
        self.data_manager = DataManager(runtime_root=runtime_root, settings=self.settings, repository=self.repo)
        self.model_manager = ModelManager(runtime_root=runtime_root, settings=self.settings, repository=self.repo)
        self.task_manager = TaskManager(
            runtime_root=runtime_root,
            data_manager=self.data_manager,
            model_manager=self.model_manager,
            settings=self.settings,
            repository=self.repo,
        )
        self.leaderboard_manager = LeaderboardManager(
            runtime_root=runtime_root,
            data_manager=self.data_manager,
            model_manager=self.model_manager,
            task_manager=self.task_manager,
            settings=self.settings,
            repository=self.repo,
        )

    @property
    def huggingface_runner_factory(self):
        return self.model_manager.huggingface_runner_factory

    @huggingface_runner_factory.setter
    def huggingface_runner_factory(self, factory) -> None:
        self.model_manager.huggingface_runner_factory = factory

    def list_tracks(self):
        return self.data_manager.list_tracks()

    def list_models(self):
        return self.model_manager.list_models()

    def register_model(self, request):
        return self.model_manager.register_model(request)

    def register_huggingface_model(self, request):
        return self.model_manager.register_huggingface_model(request)

    def load_model(self, model_id: str):
        return self.model_manager.load_model(model_id)

    def list_batches(self):
        return self.data_manager.list_batches()

    def get_batch(self, batch_id: str):
        return self.data_manager.get_batch(batch_id)

    def generate_batch(self, request):
        return self.data_manager.generate_batch(request)

    def load_batch(self, request):
        return self.data_manager.load_batch(request)

    def list_tasks(self):
        return self.task_manager.list_tasks()

    def get_task(self, task_id: str):
        return self.task_manager.get_task(task_id)

    def get_report(self, report_id: str):
        return self.task_manager.get_report(report_id)

    def run_task(self, request):
        return self.task_manager.run_task(request)

    def leaderboard(self, track: str | None = None, metric_id: str = "mse"):
        return self.leaderboard_manager.leaderboard(track=track, metric_id=metric_id)

    def track_leaderboard(self, track: str, metric_id: str = "mse"):
        return self.leaderboard_manager.track_leaderboard(track, metric_id=metric_id)

    def overall_leaderboard(self, metric_id: str = "mse"):
        return self.leaderboard_manager.overall_leaderboard(metric_id=metric_id)

    def user_overview(self, metric_id: str = "mse") -> UserDashboardOverview:
        ui = self.settings.ui.dashboard
        overall = self.overall_leaderboard(metric_id=metric_id)
        track_boards = [
            TrackLeaderboard(
                track=track.track_variant_id,
                track_label=track.name,
                metric_id=metric_id,
                ranking_strategy=self.settings.benchmark.leaderboards.track_aggregation_strategy,
                entries=self.track_leaderboard(track.track_variant_id, metric_id=metric_id)[: ui.user_track_leaderboard_limit],
            )
            for track in self.list_tracks()
        ]
        return UserDashboardOverview(
            tracks=self.list_tracks(),
            models=self.list_models(),
            overall_leaderboard_strategy=self.settings.benchmark.leaderboards.overall_ranking_strategy,
            overall_metric_id=metric_id,
            overall_leaderboard=overall[: ui.user_leaderboard_limit],
            track_leaderboards=track_boards,
        )

    def admin_overview(self, metric_id: str = "mse") -> AdminDashboardOverview:
        ui = self.settings.ui.dashboard
        models = self.list_models()
        model_map = {model.model_id: model for model in models}
        overall = self.overall_leaderboard(metric_id=metric_id)
        return AdminDashboardOverview(
            tracks=self.list_tracks(),
            models=models,
            batches=[self._batch_summary(batch) for batch in self.list_batches()[: ui.admin_recent_batches_limit]],
            recent_tasks=[self._task_summary(task, model_map) for task in self.list_tasks()[: ui.admin_recent_tasks_limit]],
            overall_leaderboard_strategy=self.settings.benchmark.leaderboards.overall_ranking_strategy,
            overall_metric_id=metric_id,
            leaderboard=overall[: ui.admin_leaderboard_limit],
        )

    def overview(self, metric_id: str = "mse") -> AdminDashboardOverview:
        return self.admin_overview(metric_id=metric_id)

    def _batch_summary(self, batch) -> BatchSummary:
        return BatchSummary(
            batch_id=batch.batch_id,
            track=batch.track,
            track_variant_id=batch.track_variant_id,
            policy=batch.policy,
            created_at=batch.created_at,
            sample_count=batch.sample_count,
            context_length=batch.context_length,
            horizon=batch.horizon,
            validation_passed=batch.validation.passed,
        )

    def _task_summary(self, task, model_map: dict[str, ModelRecord]) -> TaskSummary:
        model = model_map.get(task.model_id)
        runtime_parameters = task.spec.model_runtime_parameters if task.spec is not None else {}
        evaluation_metrics = task.spec.evaluation_metrics if task.spec is not None else []
        return TaskSummary(
            task_id=task.task_id,
            model_id=task.model_id,
            model_name=model.name if model else task.model_id,
            batch_id=task.batch_id,
            track=task.track,
            track_variant_id=task.track_variant_id,
            status=task.status,
            created_at=task.created_at,
            primary_metric_id="mse" if task.metrics else None,
            primary_metric_value=task.metrics.mse if task.metrics else None,
            composite_score=task.metrics.composite_score if task.metrics else None,
            execution_repeat_count=task.spec.execution_repeat_count if task.spec else None,
            task_run_count=len(task.task_runs),
            report_id=task.report_id,
            error_message=task.error_message,
            model_runtime_parameters=runtime_parameters,
            evaluation_metrics=evaluation_metrics,
        )


__all__ = ["BenchmarkEngine", "BenchmarkError", "ExecutionResult", "NotFoundError"]
