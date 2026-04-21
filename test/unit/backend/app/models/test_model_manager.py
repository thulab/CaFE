from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.app.datasets.domain import TrackKind
from backend.app.errors import InternalBenchmarkError
from backend.app.models.domain import HuggingFaceModelRegistrationRequest, ModelAdapter, ModelRuntimeStatus
from backend.app.models.manager import ModelManager
from test.support.helpers import FakeHuggingFaceRunner, build_sample, temporary_runtime_dir


class BrokenHuggingFaceRunner(FakeHuggingFaceRunner):
    def load(self) -> None:
        raise RuntimeError("broken runner")


class ModelManagerTest(unittest.TestCase):
    def test_builtin_models_are_bootstrapped(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            models = {model.model_id: model for model in manager.list_models()}

        self.assertEqual(
            set(models),
            {
                "amazon-chronos-2",
                "thuml-sundial-base-128m",
                "timesfm-2-5-200m",
                "chronos-bolt-base",
                "sundial-base-128m-v1",
                "moirai-moe-base",
                "lag-llama",
            },
        )
        self.assertEqual(models["amazon-chronos-2"].adapter, ModelAdapter.HUGGINGFACE_CHRONOS2)
        self.assertEqual(models["amazon-chronos-2"].huggingface.repo_id, "amazon/chronos-2")
        self.assertEqual(models["amazon-chronos-2"].spec.source.huggingface_url, "https://huggingface.co/amazon/chronos-2")
        self.assertIn("amazon-chronos-2", models["amazon-chronos-2"].spec.source.local_weight_path)
        self.assertEqual(
            [item.name for item in models["amazon-chronos-2"].spec.runtime_parameter_definitions],
            ["batch_size", "context_length", "use_covariates", "cross_learning", "max_output_patches"],
        )
        self.assertEqual(models["thuml-sundial-base-128m"].adapter, ModelAdapter.HUGGINGFACE_SUNDIAL)
        self.assertEqual(models["thuml-sundial-base-128m"].huggingface.task.value, "sundial")
        self.assertEqual(
            [item.name for item in models["thuml-sundial-base-128m"].spec.runtime_parameter_definitions],
            ["batch_size", "do_sample", "temperature", "top_p"],
        )
        self.assertEqual(models["timesfm-2-5-200m"].adapter, ModelAdapter.V1_TIMESFM_2_5_200M)
        self.assertEqual(models["timesfm-2-5-200m"].runtime_status, ModelRuntimeStatus.READY)
        self.assertIn("timesfm-2-5-200m", models["timesfm-2-5-200m"].spec.source.local_weight_path)

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
        self.assertEqual(record.spec.source.huggingface_url, "https://huggingface.co/Org/Demo Forecast Model")

    def test_register_huggingface_model_accepts_huggingface_url_and_infers_task(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            record = manager.register_huggingface_model(
                HuggingFaceModelRegistrationRequest(
                    huggingface_url="https://huggingface.co/amazon/chronos-2",
                    model_id="custom-chronos-url",
                    manual="demo",
                )
            )

        self.assertEqual(record.huggingface.repo_id, "amazon/chronos-2")
        self.assertEqual(record.huggingface.task.value, "chronos-2")
        self.assertEqual(record.adapter, ModelAdapter.HUGGINGFACE_CHRONOS2)

    def test_register_huggingface_model_maps_sundial_task_to_sundial_adapter(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            record = manager.register_huggingface_model(
                HuggingFaceModelRegistrationRequest(
                    huggingface_url="https://huggingface.co/thuml/sundial-base-128m",
                    model_id="custom-sundial",
                    manual="demo",
                )
            )

        self.assertEqual(record.adapter, ModelAdapter.HUGGINGFACE_SUNDIAL)
        self.assertEqual(record.huggingface.task.value, "sundial")

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

    def test_builtin_models_load_with_fallback_when_optional_dependencies_are_missing(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            with patch("backend.app.models.huggingface.importlib.util.find_spec") as find_spec:
                find_spec.side_effect = lambda name: object() if name == "transformers" else None
                chronos_model = manager.load_model("amazon-chronos-2")
                sundial_model = manager.load_model("thuml-sundial-base-128m")
                sample = build_sample(history=[1.0, 2.0, 3.0, 4.0], target=[5.0, 6.0])
                chronos_result = manager.execute_model(chronos_model, sample, TrackKind.FORECAST_ACCURACY)
                sundial_result = manager.execute_model(sundial_model, sample, TrackKind.FORECAST_ACCURACY)

        self.assertEqual(chronos_model.runtime_status, ModelRuntimeStatus.READY)
        self.assertEqual(sundial_model.runtime_status, ModelRuntimeStatus.READY)
        self.assertEqual(chronos_result.prediction, [3.0, 4.0])
        self.assertEqual(sundial_result.prediction, [2.5, 2.5])
        self.assertEqual(chronos_result.notes["decision"], "builtin_fallback_no_optional_dependencies")
        self.assertEqual(sundial_result.notes["decision"], "builtin_fallback_no_optional_dependencies")

    def test_custom_huggingface_model_reports_missing_dependency_install_hint(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            record = manager.register_huggingface_model(
                HuggingFaceModelRegistrationRequest(
                    repo_id="org/custom-sundial",
                    manual="demo",
                    task="sundial",
                )
            )
            with patch("backend.app.models.huggingface.importlib.util.find_spec") as find_spec:
                find_spec.side_effect = lambda name: object() if name == "transformers" else None
                with self.assertRaises(InternalBenchmarkError) as ctx:
                    manager.load_model(record.model_id)
            reloaded = manager.get_model(record.model_id)

        self.assertEqual(reloaded.runtime_status, ModelRuntimeStatus.LOAD_FAILED)
        self.assertIn("pip install -e .[huggingface]", str(ctx.exception))
        self.assertIn("Missing modules: torch", str(ctx.exception))

    def test_execute_model_applies_cost_track_penalties(self) -> None:
        with temporary_runtime_dir(prefix="model-manager-") as runtime_root:
            manager = ModelManager(runtime_root)
            manager.huggingface_runner_factory = FakeHuggingFaceRunner
            model = manager.get_model("amazon-chronos-2")
            sample = build_sample(history=[1.0, 2.0, 4.0, 8.0], target=[16.0, 32.0])

            result = manager.execute_model(model, sample, TrackKind.COST_INTENSIVE)
            reloaded = manager.get_model(model.model_id)

        self.assertEqual(result.prediction, [8.0, 8.0])
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.token_count, 0)
        self.assertEqual(reloaded.runtime_status, ModelRuntimeStatus.READY)
