"""Lazy aggregate exports for application domain models."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AdminDashboardOverview": ("backend.app.leaderboards.domain", "AdminDashboardOverview"),
    "AggregatedMetrics": ("backend.app.tasks.domain", "AggregatedMetrics"),
    "BatchGenerationRequest": ("backend.app.datasets.domain", "BatchGenerationRequest"),
    "BatchSummary": ("backend.app.leaderboards.domain", "BatchSummary"),
    "BenchmarkReport": ("backend.app.tasks.domain", "BenchmarkReport"),
    "CsvBatchLoadRequest": ("backend.app.datasets.domain", "CsvBatchLoadRequest"),
    "DataProcessorConfig": ("backend.app.datasets.domain", "DataProcessorConfig"),
    "DataProcessorType": ("backend.app.datasets.domain", "DataProcessorType"),
    "DatasetBatch": ("backend.app.datasets.domain", "DatasetBatch"),
    "DatasetLoadRequest": ("backend.app.datasets.domain", "DatasetLoadRequest"),
    "DatasetSourceType": ("backend.app.datasets.domain", "DatasetSourceType"),
    "EvaluationTask": ("backend.app.tasks.domain", "EvaluationTask"),
    "HuggingFaceConfig": ("backend.app.models.domain", "HuggingFaceConfig"),
    "HuggingFaceModelRegistrationRequest": ("backend.app.models.domain", "HuggingFaceModelRegistrationRequest"),
    "HuggingFaceTask": ("backend.app.models.domain", "HuggingFaceTask"),
    "LeaderboardEntry": ("backend.app.leaderboards.domain", "LeaderboardEntry"),
    "ModelAdapter": ("backend.app.models.domain", "ModelAdapter"),
    "ModelRecord": ("backend.app.models.domain", "ModelRecord"),
    "ModelRegistrationRequest": ("backend.app.models.domain", "ModelRegistrationRequest"),
    "ModelRuntimeStatus": ("backend.app.models.domain", "ModelRuntimeStatus"),
    "OverallLeaderboardEntry": ("backend.app.leaderboards.domain", "OverallLeaderboardEntry"),
    "SampleOutcome": ("backend.app.tasks.domain", "SampleOutcome"),
    "SeriesSample": ("backend.app.datasets.domain", "SeriesSample"),
    "SeriesTruth": ("backend.app.datasets.domain", "SeriesTruth"),
    "TaskRunRequest": ("backend.app.tasks.domain", "TaskRunRequest"),
    "TaskRunRecord": ("backend.app.tasks.domain", "TaskRunRecord"),
    "TaskStatus": ("backend.app.tasks.domain", "TaskStatus"),
    "TaskSummary": ("backend.app.leaderboards.domain", "TaskSummary"),
    "TrackKind": ("backend.app.datasets.domain", "TrackKind"),
    "TrackLeaderboard": ("backend.app.leaderboards.domain", "TrackLeaderboard"),
    "TrackSpec": ("backend.app.datasets.domain", "TrackSpec"),
    "UserDashboardOverview": ("backend.app.leaderboards.domain", "UserDashboardOverview"),
    "ValidationReport": ("backend.app.datasets.domain", "ValidationReport"),
    "admin_batch_defaults": ("backend.app.domain.common", "admin_batch_defaults"),
    "user_submission_defaults": ("backend.app.domain.common", "user_submission_defaults"),
    "utc_now": ("backend.app.domain.common", "utc_now"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, symbol_name)
