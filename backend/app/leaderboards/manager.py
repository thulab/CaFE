from __future__ import annotations

from pathlib import Path

from ..config import AppSettings, get_settings
from ..errors import BenchmarkError
from ..models.manager import ModelManager
from ..storage import FileRepository
from ..tasks.domain import EvaluationTask, TaskStatus
from ..tasks.manager import TaskManager
from .domain import LeaderboardEntry, OverallLeaderboardEntry


class LeaderboardManager:
    def __init__(
        self,
        runtime_root: Path,
        data_manager,
        model_manager: ModelManager,
        task_manager: TaskManager,
        settings: AppSettings | None = None,
        repository: FileRepository | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repository or FileRepository(runtime_root)
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.task_manager = task_manager

    def leaderboard(self, track: str | None = None, metric_id: str = "mse") -> list[LeaderboardEntry] | list[OverallLeaderboardEntry]:
        if track is None:
            return self.overall_leaderboard(metric_id=metric_id)
        return self.track_leaderboard(track, metric_id=metric_id)

    def track_leaderboard(self, track: str, metric_id: str = "mse") -> list[LeaderboardEntry]:
        track_spec = self.data_manager.get_track_spec(track)
        models = {model.model_id: model for model in self.model_manager.list_models()}
        grouped: dict[str, list[EvaluationTask]] = {}
        for task in self.task_manager.list_tasks():
            if (
                task.status != TaskStatus.SUCCEEDED
                or task.metrics is None
                or task.track_variant_id != track_spec.track_variant_id
                or task.model_id not in models
            ):
                continue
            grouped.setdefault(task.model_id, []).append(task)

        selector = self._track_aggregation_selector(metric_id)
        entries: list[LeaderboardEntry] = []
        for model_id, tasks in grouped.items():
            selected = selector(tasks)
            model = models.get(model_id)
            entries.append(
                self._leaderboard_entry_from_task(
                    selected,
                    model_name=model.name if model else model_id,
                    metric_id=metric_id,
                    track_label=track_spec.name,
                )
            )

        entries.sort(key=lambda item: self._entry_sort_key(item))
        return [item.copy(update={"rank": index}) for index, item in enumerate(entries, start=1)]

    def overall_leaderboard(self, metric_id: str = "mse") -> list[OverallLeaderboardEntry]:
        strategy = self.settings.benchmark.leaderboards.overall_ranking_strategy
        track_specs = self.data_manager.list_tracks()
        track_boards = {track.track_variant_id: self.track_leaderboard(track.track_variant_id, metric_id=metric_id) for track in track_specs}
        if strategy == "rank_sum":
            return self._overall_leaderboard_rank_sum(track_boards, metric_id=metric_id)
        raise BenchmarkError(f"unsupported overall leaderboard strategy {strategy}")

    def _track_aggregation_selector(self, metric_id: str):
        strategy = self.settings.benchmark.leaderboards.track_aggregation_strategy
        if strategy == "best_composite_score" and metric_id == "composite_score":
            return lambda tasks: self._select_best_task(tasks, metric_id="composite_score")
        if strategy == "best_composite_score":
            return lambda tasks: self._select_best_task(tasks, metric_id=metric_id)
        raise BenchmarkError(f"unsupported track leaderboard strategy {strategy}")

    def _select_best_task(self, tasks: list[EvaluationTask], metric_id: str) -> EvaluationTask:
        direction = self._metric_direction(metric_id)
        return sorted(
            tasks,
            key=lambda item: self._task_metric_sort_key(item, metric_id=metric_id, direction=direction),
        )[0]

    def _leaderboard_entry_from_task(
        self,
        task: EvaluationTask,
        *,
        model_name: str,
        metric_id: str,
        track_label: str,
    ) -> LeaderboardEntry:
        if task.metrics is None:
            raise BenchmarkError(f"task {task.task_id} has no metrics")
        metric_value = self._metric_value_from_task(task, metric_id)
        return LeaderboardEntry(
            rank=0,
            task_id=task.task_id,
            model_id=task.model_id,
            model_name=model_name,
            batch_id=task.batch_id,
            track=task.track,
            track_variant_id=task.track_variant_id or task.track.value,
            track_label=track_label,
            metric_id=metric_id,
            metric_value=metric_value,
            metric_snapshot={
                "mse": task.metrics.mse,
                "mae": task.metrics.mae,
                "smape": task.metrics.smape,
                "mase": task.metrics.mase if task.metrics.mase is not None else 0.0,
                "relative_skill": task.metrics.relative_skill if task.metrics.relative_skill is not None else 0.0,
                "latency_ms": task.metrics.mean_latency_ms,
                "token_count": task.metrics.mean_token_count,
                "composite_score": task.metrics.composite_score,
            },
            sample_count=len(task.sample_outcomes),
        )

    def _overall_leaderboard_rank_sum(
        self,
        track_boards: dict[str, list[LeaderboardEntry]],
        *,
        metric_id: str,
    ) -> list[OverallLeaderboardEntry]:
        models = {model.model_id: model for model in self.model_manager.list_models()}
        track_order = [track.track_variant_id for track in self.data_manager.list_tracks()]
        penalty_policy = self.settings.benchmark.leaderboards.missing_track_rank_penalty
        rows: list[OverallLeaderboardEntry] = []

        for model_id in models:
            track_ranks: dict[str, int] = {}
            track_values: dict[str, float] = {}
            rank_sum = 0
            covered_tracks = 0

            for track_variant_id in track_order:
                entries = track_boards.get(track_variant_id, [])
                selected = next((entry for entry in entries if entry.model_id == model_id), None)
                if selected is not None:
                    rank_sum += selected.rank
                    covered_tracks += 1
                    track_ranks[track_variant_id] = selected.rank
                    track_values[track_variant_id] = selected.metric_value
                    continue
                penalty_rank = self._missing_track_penalty_rank(entries, penalty_policy)
                rank_sum += penalty_rank
                track_ranks[track_variant_id] = penalty_rank

            rows.append(
                OverallLeaderboardEntry(
                    rank=0,
                    model_id=model_id,
                    model_name=models[model_id].name,
                    metric_id=metric_id,
                    rank_sum=rank_sum,
                    covered_tracks=covered_tracks,
                    mean_metric_value=round(sum(track_values.values()) / covered_tracks, 6) if covered_tracks else 0.0,
                    track_ranks=track_ranks,
                    track_values=track_values,
                )
            )

        reverse_metric = self._metric_direction(metric_id) == "max"
        rows.sort(
            key=lambda item: (
                item.rank_sum,
                -item.covered_tracks,
                (-item.mean_metric_value if reverse_metric else item.mean_metric_value),
                item.model_name,
            )
        )
        return [item.copy(update={"rank": index}) for index, item in enumerate(rows, start=1)]

    def _missing_track_penalty_rank(self, entries: list[LeaderboardEntry], policy: str) -> int:
        if policy == "max_plus_one":
            return len(entries) + 1
        raise BenchmarkError(f"unsupported missing track rank penalty policy {policy}")

    def _metric_value_from_task(self, task: EvaluationTask, metric_id: str) -> float:
        if task.metrics is None:
            raise BenchmarkError(f"task {task.task_id} has no metrics")
        value_map = {
            "mse": task.metrics.mse,
            "mae": task.metrics.mae,
            "smape": task.metrics.smape,
            "mase": task.metrics.mase,
            "relative_skill": task.metrics.relative_skill,
            "latency_ms": task.metrics.mean_latency_ms,
            "token_count": task.metrics.mean_token_count,
            "composite_score": task.metrics.composite_score,
        }
        if metric_id not in value_map:
            raise BenchmarkError(f"unsupported leaderboard metric {metric_id}")
        if value_map[metric_id] is None:
            raise BenchmarkError(f"metric {metric_id} is not available for task {task.task_id}")
        return float(value_map[metric_id])

    def _metric_direction(self, metric_id: str) -> str:
        if metric_id in {"composite_score", "relative_skill"}:
            return "max"
        return "min"

    def _task_metric_sort_key(self, task: EvaluationTask, *, metric_id: str, direction: str):
        metric = self._metric_value_from_task(task, metric_id)
        primary = -metric if direction == "max" else metric
        mse = task.metrics.mse if task.metrics else float("inf")
        latency = task.metrics.mean_latency_ms if task.metrics else float("inf")
        created = task.created_at.timestamp()
        return (primary, mse, latency, created)

    def _entry_sort_key(self, entry: LeaderboardEntry):
        direction = self._metric_direction(entry.metric_id)
        primary = -entry.metric_value if direction == "max" else entry.metric_value
        mse = entry.metric_snapshot.get("mse", float("inf"))
        latency = entry.metric_snapshot.get("latency_ms", float("inf"))
        return (primary, mse, latency, entry.model_name)
