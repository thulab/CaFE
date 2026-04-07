from __future__ import annotations

from pathlib import Path

from ..config import AppSettings, get_settings
from ..data_management.manager import DataManager
from ..data_management.domain import TrackKind
from ..errors import BenchmarkError
from ..model_management.manager import ModelManager
from ..storage import FileRepository
from ..task_management.domain import EvaluationTask, TaskStatus
from ..task_management.manager import TaskManager
from .domain import LeaderboardEntry, OverallLeaderboardEntry


class LeaderboardManager:
    def __init__(
        self,
        runtime_root: Path,
        data_manager: DataManager,
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

    def leaderboard(self, track: TrackKind | None = None) -> list[LeaderboardEntry] | list[OverallLeaderboardEntry]:
        if track is None:
            return self.overall_leaderboard()
        return self.track_leaderboard(track)

    def track_leaderboard(self, track: TrackKind) -> list[LeaderboardEntry]:
        models = {model.model_id: model for model in self.model_manager.list_models()}
        grouped: dict[str, list[EvaluationTask]] = {}
        for task in self.task_manager.list_tasks():
            if task.status != TaskStatus.SUCCEEDED or task.metrics is None or task.track != track:
                continue
            grouped.setdefault(task.model_id, []).append(task)

        selector = self._track_aggregation_selector()
        entries: list[LeaderboardEntry] = []
        for model_id, tasks in grouped.items():
            selected = selector(tasks)
            model = models.get(model_id)
            entries.append(self._leaderboard_entry_from_task(selected, model_name=model.name if model else model_id))

        entries.sort(key=lambda item: (-item.composite_score, item.mse, item.mean_latency_ms, item.model_name))
        return [item.model_copy(update={"rank": index}) for index, item in enumerate(entries, start=1)]

    def overall_leaderboard(self) -> list[OverallLeaderboardEntry]:
        strategy = self.settings.benchmark.leaderboards.overall_ranking_strategy
        track_boards = {track.track: self.track_leaderboard(track.track) for track in self.data_manager.list_tracks()}
        if strategy == "rank_sum":
            return self._overall_leaderboard_rank_sum(track_boards)
        raise BenchmarkError(f"unsupported overall leaderboard strategy {strategy}")

    def _track_aggregation_selector(self):
        strategy = self.settings.benchmark.leaderboards.track_aggregation_strategy
        if strategy == "best_composite_score":
            return self._select_best_composite_task
        raise BenchmarkError(f"unsupported track leaderboard strategy {strategy}")

    def _select_best_composite_task(self, tasks: list[EvaluationTask]) -> EvaluationTask:
        return max(
            tasks,
            key=lambda item: (
                item.metrics.composite_score if item.metrics else float("-inf"),
                -(item.metrics.mse if item.metrics else float("inf")),
                item.created_at.timestamp(),
            ),
        )

    def _leaderboard_entry_from_task(self, task: EvaluationTask, model_name: str) -> LeaderboardEntry:
        if task.metrics is None:
            raise BenchmarkError(f"task {task.task_id} has no metrics")
        return LeaderboardEntry(
            rank=0,
            task_id=task.task_id,
            model_id=task.model_id,
            model_name=model_name,
            batch_id=task.batch_id,
            track=task.track,
            composite_score=task.metrics.composite_score,
            mse=task.metrics.mse,
            smape=task.metrics.smape,
            mean_latency_ms=task.metrics.mean_latency_ms,
        )

    def _overall_leaderboard_rank_sum(
        self,
        track_boards: dict[TrackKind, list[LeaderboardEntry]],
    ) -> list[OverallLeaderboardEntry]:
        models = {model.model_id: model for model in self.model_manager.list_models()}
        track_order = [track.track for track in self.data_manager.list_tracks()]
        penalty_policy = self.settings.benchmark.leaderboards.missing_track_rank_penalty
        model_ids = {task.model_id for task in self.task_manager.list_tasks()}
        model_ids.update(entry.model_id for entries in track_boards.values() for entry in entries)
        rows: list[OverallLeaderboardEntry] = []

        for model_id in model_ids:
            track_ranks: dict[str, int] = {}
            track_scores: dict[str, float] = {}
            rank_sum = 0
            covered_tracks = 0

            for track in track_order:
                entries = track_boards.get(track, [])
                selected = next((entry for entry in entries if entry.model_id == model_id), None)
                if selected is not None:
                    rank_sum += selected.rank
                    covered_tracks += 1
                    track_ranks[track.value] = selected.rank
                    track_scores[track.value] = selected.composite_score
                    continue
                penalty_rank = self._missing_track_penalty_rank(entries, penalty_policy)
                rank_sum += penalty_rank
                track_ranks[track.value] = penalty_rank

            rows.append(
                OverallLeaderboardEntry(
                    rank=0,
                    model_id=model_id,
                    model_name=models[model_id].name if model_id in models else model_id,
                    rank_sum=rank_sum,
                    covered_tracks=covered_tracks,
                    mean_composite_score=round(sum(track_scores.values()) / covered_tracks, 3) if covered_tracks else 0.0,
                    track_ranks=track_ranks,
                    track_scores=track_scores,
                )
            )

        rows.sort(key=lambda item: (item.rank_sum, -item.covered_tracks, -item.mean_composite_score, item.model_name))
        return [item.model_copy(update={"rank": index}) for index, item in enumerate(rows, start=1)]

    def _missing_track_penalty_rank(self, entries: list[LeaderboardEntry], policy: str) -> int:
        if policy == "max_plus_one":
            return len(entries) + 1
        raise BenchmarkError(f"unsupported missing track rank penalty policy {policy}")
