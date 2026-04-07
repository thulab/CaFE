from __future__ import annotations

import unittest

from backend.app.data_management.domain import BatchGenerationRequest, TrackKind
from backend.app.data_management.manager import DataManager
from backend.app.model_management.manager import ModelManager
from backend.app.task_management.domain import TaskRunRequest, TaskStatus
from backend.app.task_management.manager import TaskManager
from test.support.helpers import FakeHuggingFaceRunner, temporary_runtime_dir


class TaskManagerTest(unittest.TestCase):
    def test_run_task_persists_report_and_metrics(self) -> None:
        with temporary_runtime_dir(prefix="task-manager-") as runtime_root:
            data_manager = DataManager(runtime_root)
            model_manager = ModelManager(runtime_root)
            model_manager.huggingface_runner_factory = FakeHuggingFaceRunner
            task_manager = TaskManager(runtime_root, data_manager=data_manager, model_manager=model_manager)
            batch = data_manager.generate_batch(
                BatchGenerationRequest(
                    track=TrackKind.COVARIATE_ROBUSTNESS,
                    sample_count=2,
                    context_length=24,
                    horizon=8,
                    seed=9,
                )
            )

            task = task_manager.run_task(
                TaskRunRequest(
                    model_id="amazon-chronos-2",
                    batch_id=batch.batch_id,
                    model_runtime_parameters={"batch_size": 2, "use_covariates": False},
                    evaluation_metrics=["mse", "latency_ms", "composite_score"],
                )
            )
            report = task_manager.get_report(task.report_id)

        self.assertEqual(task.status, TaskStatus.SUCCEEDED)
        self.assertIsNotNone(task.metrics)
        self.assertEqual(report.task_id, task.task_id)
        self.assertTrue(report.summary)
        self.assertGreaterEqual(len(task.sample_outcomes), 1)
        self.assertEqual(len(task.task_runs), 3)
        self.assertEqual(task.spec.execution_repeat_count, 3)
        self.assertEqual(task.sample_outcomes[0].run_count, 3)
        self.assertEqual(len(report.run_ids), 3)
        self.assertEqual(task.spec.model_runtime_parameters, {"batch_size": 2, "use_covariates": False})
        self.assertEqual(task.spec.evaluation_metrics, ["mse", "latency_ms", "composite_score"])

    def test_run_task_accepts_custom_repeat_count(self) -> None:
        with temporary_runtime_dir(prefix="task-manager-") as runtime_root:
            data_manager = DataManager(runtime_root)
            model_manager = ModelManager(runtime_root)
            model_manager.huggingface_runner_factory = FakeHuggingFaceRunner
            task_manager = TaskManager(runtime_root, data_manager=data_manager, model_manager=model_manager)
            batch = data_manager.generate_batch(
                BatchGenerationRequest(
                    track=TrackKind.FORECAST_ACCURACY,
                    sample_count=1,
                    context_length=12,
                    horizon=4,
                    seed=4,
                )
            )

            task = task_manager.run_task(
                TaskRunRequest(
                    model_id="amazon-chronos-2",
                    batch_id=batch.batch_id,
                    execution_repeat_count=2,
                )
            )

        self.assertEqual(task.spec.execution_repeat_count, 2)
        self.assertEqual(len(task.task_runs), 2)
