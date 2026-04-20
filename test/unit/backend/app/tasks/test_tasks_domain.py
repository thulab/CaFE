from __future__ import annotations

import unittest

from backend.app.datasets.domain import TrackKind
from backend.app.tasks.domain import (
    AggregatedMetrics,
    DEFAULT_EXECUTION_REPEAT_COUNT,
    DEFAULT_EVALUATION_METRICS,
    EvaluationTask,
    SampleOutcome,
    TaskDatasetSpec,
    TaskRunRequest,
    TaskRunStatus,
    TaskSpec,
    TaskStatus,
)


class TaskStatusTest(unittest.TestCase):
    def test_status_values(self) -> None:
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.RUNNING.value, "running")
        self.assertEqual(TaskStatus.SUCCEEDED.value, "succeeded")
        self.assertEqual(TaskStatus.FAILED.value, "failed")


class TaskRunStatusTest(unittest.TestCase):
    def test_status_values(self) -> None:
        self.assertEqual(TaskRunStatus.QUEUED.value, "queued")
        self.assertEqual(TaskRunStatus.RUNNING.value, "running")
        self.assertEqual(TaskRunStatus.SUCCEEDED.value, "succeeded")
        self.assertEqual(TaskRunStatus.FAILED.value, "failed")


class SampleOutcomeTest(unittest.TestCase):
    def test_create_outcome(self) -> None:
        outcome = SampleOutcome(
            sample_id="sample-1",
            mse=0.1,
            mae=0.2,
            smape=0.3,
            latency_ms=10.0,
            token_count=100,
            prediction=[1.0, 2.0, 3.0],
        )
        self.assertEqual(outcome.sample_id, "sample-1")
        self.assertEqual(outcome.mse, 0.1)
        self.assertEqual(outcome.run_count, 1)

    def test_outcome_with_std(self) -> None:
        outcome = SampleOutcome(
            sample_id="sample-1",
            mse=0.1,
            mae=0.2,
            smape=0.3,
            latency_ms=10.0,
            token_count=100,
            prediction=[1.0, 2.0],
            run_count=3,
            mse_std=0.05,
            mae_std=0.03,
        )
        self.assertEqual(outcome.run_count, 3)
        self.assertEqual(outcome.mse_std, 0.05)


class AggregatedMetricsTest(unittest.TestCase):
    def test_create_metrics(self) -> None:
        metrics = AggregatedMetrics(
            mse=0.1,
            mae=0.2,
            smape=0.3,
            mean_latency_ms=10.0,
            mean_token_count=100.0,
            composite_score=95.0,
        )
        self.assertEqual(metrics.mse, 0.1)
        self.assertEqual(metrics.composite_score, 95.0)

    def test_stability_stats(self) -> None:
        metrics = AggregatedMetrics(
            mse=0.1,
            mae=0.2,
            smape=0.3,
            mean_latency_ms=10.0,
            mean_token_count=100.0,
            composite_score=95.0,
            stability_stats={"cv": 0.1, "iqr": 0.05},
        )
        self.assertEqual(metrics.stability_stats["cv"], 0.1)


class TaskDatasetSpecTest(unittest.TestCase):
    def test_create_dataset_spec(self) -> None:
        spec = TaskDatasetSpec(
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            sample_count=10,
        )
        self.assertEqual(spec.batch_id, "batch-1")
        self.assertEqual(spec.track, TrackKind.FORECAST_ACCURACY)
        self.assertEqual(spec.sample_count, 10)

    def test_sync_lengths(self) -> None:
        spec = TaskDatasetSpec(
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            sample_count=10,
            context_length=96,
            horizon=24,
        )
        self.assertEqual(spec.input_length, 96)
        self.assertEqual(spec.prediction_length, 24)
        self.assertEqual(spec.context_length, 96)
        self.assertEqual(spec.horizon, 24)

    def test_sync_lengths_with_input_length(self) -> None:
        spec = TaskDatasetSpec(
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            sample_count=10,
            input_length=96,
            prediction_length=24,
        )
        self.assertEqual(spec.context_length, 96)
        self.assertEqual(spec.horizon, 24)


class TaskSpecTest(unittest.TestCase):
    def test_create_task_spec(self) -> None:
        spec = TaskSpec(
            model_id="model-1",
            dataset=TaskDatasetSpec(
                batch_id="batch-1",
                track=TrackKind.FORECAST_ACCURACY,
                sample_count=10,
            ),
        )
        self.assertEqual(spec.model_id, "model-1")
        self.assertEqual(spec.execution_repeat_count, DEFAULT_EXECUTION_REPEAT_COUNT)
        self.assertEqual(spec.evaluation_metrics, DEFAULT_EVALUATION_METRICS)

    def test_custom_metrics_and_repeat(self) -> None:
        spec = TaskSpec(
            model_id="model-1",
            dataset=TaskDatasetSpec(
                batch_id="batch-1",
                track=TrackKind.FORECAST_ACCURACY,
                sample_count=10,
            ),
            evaluation_metrics=["mse", "mae"],
            execution_repeat_count=5,
        )
        self.assertEqual(spec.evaluation_metrics, ["mse", "mae"])
        self.assertEqual(spec.execution_repeat_count, 5)


class EvaluationTaskTest(unittest.TestCase):
    def test_create_evaluation_task(self) -> None:
        task = EvaluationTask(
            task_id="task-1",
            model_id="model-1",
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            status=TaskStatus.PENDING,
        )
        self.assertEqual(task.task_id, "task-1")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.metrics)

    def test_task_with_metrics(self) -> None:
        task = EvaluationTask(
            task_id="task-1",
            model_id="model-1",
            batch_id="batch-1",
            track=TrackKind.FORECAST_ACCURACY,
            status=TaskStatus.SUCCEEDED,
            metrics=AggregatedMetrics(
                mse=0.1,
                mae=0.2,
                smape=0.3,
                mean_latency_ms=10.0,
                mean_token_count=100.0,
                composite_score=95.0,
            ),
        )
        self.assertEqual(task.status, TaskStatus.SUCCEEDED)
        self.assertIsNotNone(task.metrics)
        self.assertEqual(task.metrics.composite_score, 95.0)


class TaskRunRequestTest(unittest.TestCase):
    def test_create_request(self) -> None:
        request = TaskRunRequest(
            model_id="model-1",
            batch_id="batch-1",
        )
        self.assertEqual(request.model_id, "model-1")
        self.assertEqual(request.batch_id, "batch-1")
        self.assertEqual(request.evaluation_metrics, DEFAULT_EVALUATION_METRICS)

    def test_custom_metrics_and_repeat(self) -> None:
        request = TaskRunRequest(
            model_id="model-1",
            batch_id="batch-1",
            evaluation_metrics=["mse"],
            execution_repeat_count=5,
        )
        self.assertEqual(request.evaluation_metrics, ["mse"])
        self.assertEqual(request.execution_repeat_count, 5)


if __name__ == "__main__":
    unittest.main()