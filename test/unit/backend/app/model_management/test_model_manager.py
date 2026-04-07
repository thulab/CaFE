from __future__ import annotations

import unittest

from backend.app.data_management.domain import TrackKind
from backend.app.errors import InternalBenchmarkError
from backend.app.model_management.domain import HuggingFaceModelRegistrationRequest, ModelRuntimeStatus
from backend.app.model_management.manager import ModelManager
from test.support.helpers import FakeHuggingFaceRunner, build_sample, temporary_runtime_dir


class BrokenHuggingFaceRunner(FakeHuggingFaceRunner):
    def load(self) -> None:
        raise RuntimeError("broken runner")


class ModelManagerTest(unittest.TestCase):
    def test_builtin_models_are_bootstrapped(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            model_ids = [model.model_id for model in manager.list_models()]

        self.assertIn("seasonal-naive-stub", model_ids)
        self.assertIn("recent-mean-stub", model_ids)
        self.assertIn("covariate-trap-stub", model_ids)

    def test_register_huggingface_model_normalizes_model_id(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            record = manager.register_huggingface_model(
                HuggingFaceModelRegistrationRequest(
                    repo_id="Org/Demo Forecast Model",
                    manual="demo",
                    task="chronos-2",
                )
            )

        self.assertEqual(record.model_id, "org-demo-forecast-model")
        self.assertEqual(record.runtime_status, ModelRuntimeStatus.REGISTERED)

    def test_load_model_marks_huggingface_failures(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            manager.huggingface_runner_factory = BrokenHuggingFaceRunner
            record = manager.register_huggingface_model(
                HuggingFaceModelRegistrationRequest(
                    repo_id="org/failure-model",
                    manual="demo",
                    task="chronos-2",
                )
            )

            with self.assertRaises(InternalBenchmarkError):
                manager.load_model(record.model_id)

            reloaded = manager.get_model(record.model_id)

        self.assertEqual(reloaded.runtime_status, ModelRuntimeStatus.LOAD_FAILED)
        self.assertEqual(reloaded.last_error, "broken runner")

    def test_execute_model_applies_cost_track_penalties(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            model = manager.get_model("recent-mean-stub")
            sample = build_sample(history=[1.0, 2.0, 4.0, 8.0], target=[16.0, 32.0])

            result = manager.execute_model(model, sample, TrackKind.COST_INTENSIVE)

        self.assertEqual(result.prediction, [3.75, 3.75])
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.token_count, 0)
