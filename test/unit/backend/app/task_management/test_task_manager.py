from __future__ import annotations

import unittest

from backend.app.data_management.domain import BatchGenerationRequest, TrackKind
from backend.app.data_management.manager import DataManager
from backend.app.model_management.manager import ModelManager
from backend.app.task_management.domain import TaskRunRequest, TaskStatus
from backend.app.task_management.manager import TaskManager
from test.support.helpers import temporary_runtime_dir


class TaskManagerTest(unittest.TestCase):
    def test_run_task_persists_report_and_metrics(self) -> None:
        with temporary_runtime_dir(prefix="task-manager-") as runtime_root:
            data_manager = DataManager(runtime_root)
            model_manager = ModelManager(runtime_root)
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
                TaskRunRequest(model_id="seasonal-naive-stub", batch_id=batch.batch_id)
            )
            report = task_manager.get_report(task.report_id)

        self.assertEqual(task.status, TaskStatus.SUCCEEDED)
        self.assertIsNotNone(task.metrics)
        self.assertEqual(report.task_id, task.task_id)
        self.assertTrue(report.summary)
        self.assertGreaterEqual(len(task.sample_outcomes), 1)
