"""Lazy aggregate exports for application domain models."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AdminDashboardOverview": ("backend.app.leaderboard_management.domain", "AdminDashboardOverview"),
    "AggregatedMetrics": ("backend.app.task_management.domain", "AggregatedMetrics"),
    "BatchGenerationRequest": ("backend.app.data_management.domain", "BatchGenerationRequest"),
    "BatchSummary": ("backend.app.leaderboard_management.domain", "BatchSummary"),
    "BenchmarkReport": ("backend.app.task_management.domain", "BenchmarkReport"),
    "CsvBatchLoadRequest": ("backend.app.data_management.domain", "CsvBatchLoadRequest"),
    "DataProcessorConfig": ("backend.app.data_management.domain", "DataProcessorConfig"),
    "DataProcessorType": ("backend.app.data_management.domain", "DataProcessorType"),
    "DatasetBatch": ("backend.app.data_management.domain", "DatasetBatch"),
    "DatasetLoadRequest": ("backend.app.data_management.domain", "DatasetLoadRequest"),
    "DatasetSourceType": ("backend.app.data_management.domain", "DatasetSourceType"),
    "EvaluationTask": ("backend.app.task_management.domain", "EvaluationTask"),
    "HuggingFaceConfig": ("backend.app.model_management.domain", "HuggingFaceConfig"),
    "HuggingFaceModelRegistrationRequest": ("backend.app.model_management.domain", "HuggingFaceModelRegistrationRequest"),
    "HuggingFaceTask": ("backend.app.model_management.domain", "HuggingFaceTask"),
    "LeaderboardEntry": ("backend.app.leaderboard_management.domain", "LeaderboardEntry"),
    "ModelAdapter": ("backend.app.model_management.domain", "ModelAdapter"),
    "ModelRecord": ("backend.app.model_management.domain", "ModelRecord"),
    "ModelRegistrationRequest": ("backend.app.model_management.domain", "ModelRegistrationRequest"),
    "ModelRuntimeStatus": ("backend.app.model_management.domain", "ModelRuntimeStatus"),
    "OverallLeaderboardEntry": ("backend.app.leaderboard_management.domain", "OverallLeaderboardEntry"),
    "SampleOutcome": ("backend.app.task_management.domain", "SampleOutcome"),
    "SeriesSample": ("backend.app.data_management.domain", "SeriesSample"),
    "SeriesTruth": ("backend.app.data_management.domain", "SeriesTruth"),
    "TaskRunRequest": ("backend.app.task_management.domain", "TaskRunRequest"),
    "TaskStatus": ("backend.app.task_management.domain", "TaskStatus"),
    "TaskSummary": ("backend.app.leaderboard_management.domain", "TaskSummary"),
    "TrackKind": ("backend.app.data_management.domain", "TrackKind"),
    "TrackLeaderboard": ("backend.app.leaderboard_management.domain", "TrackLeaderboard"),
    "TrackSpec": ("backend.app.data_management.domain", "TrackSpec"),
    "UserDashboardOverview": ("backend.app.leaderboard_management.domain", "UserDashboardOverview"),
    "ValidationReport": ("backend.app.data_management.domain", "ValidationReport"),
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
