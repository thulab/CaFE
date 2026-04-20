from __future__ import annotations

import unittest
from datetime import datetime, timezone

from backend.app.models.domain import (
    HuggingFaceConfig,
    HuggingFaceModelRegistrationRequest,
    HuggingFaceTask,
    ModelAdapter,
    ModelRegistrationRequest,
    build_huggingface_url,
    infer_huggingface_task,
    normalize_huggingface_repo_id,
    build_runtime_parameter_definitions,
    ModelRuntimeParameterDefinition,
    ParameterValueType,
)


class ModelAdapterTest(unittest.TestCase):
    def test_adapter_values(self) -> None:
        self.assertEqual(ModelAdapter.SEASONAL_NAIVE.value, "seasonal_naive")
        self.assertEqual(ModelAdapter.RECENT_MEAN.value, "recent_mean")
        self.assertEqual(ModelAdapter.COVARIATE_TRAP.value, "covariate_trap")
        self.assertEqual(ModelAdapter.HUGGINGFACE_TEXT_GENERATION.value, "huggingface_text_generation")
        self.assertEqual(ModelAdapter.HUGGINGFACE_CHRONOS2.value, "huggingface_chronos2")
        self.assertEqual(ModelAdapter.HUGGINGFACE_SUNDIAL.value, "huggingface_sundial")


class HuggingFaceTaskTest(unittest.TestCase):
    def test_task_values(self) -> None:
        self.assertEqual(HuggingFaceTask.TEXT_GENERATION.value, "text-generation")
        self.assertEqual(HuggingFaceTask.TEXT2TEXT_GENERATION.value, "text2text-generation")
        self.assertEqual(HuggingFaceTask.CHRONOS2.value, "chronos-2")
        self.assertEqual(HuggingFaceTask.SUNDIAL.value, "sundial")


class BuildHuggingFaceUrlTest(unittest.TestCase):
    def test_build_url(self) -> None:
        url = build_huggingface_url("amazon/chronos-2")
        self.assertEqual(url, "https://huggingface.co/amazon/chronos-2")


class NormalizeHuggingFaceRepoIdTest(unittest.TestCase):
    def test_normalize_simple(self) -> None:
        result = normalize_huggingface_repo_id("amazon/chronos-2")
        self.assertEqual(result, "amazon/chronos-2")

    def test_normalize_with_https(self) -> None:
        result = normalize_huggingface_repo_id("https://huggingface.co/amazon/chronos-2")
        self.assertEqual(result, "amazon/chronos-2")

    def test_normalize_with_www(self) -> None:
        result = normalize_huggingface_repo_id("https://www.huggingface.co/amazon/chronos-2")
        self.assertEqual(result, "amazon/chronos-2")

    def test_normalize_with_huggingface_prefix(self) -> None:
        result = normalize_huggingface_repo_id("huggingface.co/amazon/chronos-2")
        self.assertEqual(result, "amazon/chronos-2")

    def test_normalize_strips_whitespace(self) -> None:
        result = normalize_huggingface_repo_id("  amazon/chronos-2  ")
        self.assertEqual(result, "amazon/chronos-2")

    def test_normalize_empty_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_huggingface_repo_id("")
        self.assertIn("empty", str(ctx.exception))

    def test_normalize_invalid_host_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_huggingface_repo_id("https://invalid.host.com/model")
        self.assertIn("unsupported", str(ctx.exception))

    def test_normalize_too_short_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            normalize_huggingface_repo_id("https://huggingface.co/model")
        self.assertIn("invalid", str(ctx.exception))


class InferHuggingFaceTaskTest(unittest.TestCase):
    def test_infer_chronos2(self) -> None:
        result = infer_huggingface_task("amazon/chronos-2")
        self.assertEqual(result, HuggingFaceTask.CHRONOS2)

    def test_infer_chronos2_case_insensitive(self) -> None:
        result = infer_huggingface_task("AMAZON/CHRONOS-2")
        self.assertEqual(result, HuggingFaceTask.CHRONOS2)

    def test_infer_sundial(self) -> None:
        result = infer_huggingface_task("thuml/sundial-base-128m")
        self.assertEqual(result, HuggingFaceTask.SUNDIAL)

    def test_infer_text_generation_default(self) -> None:
        result = infer_huggingface_task("some/other-model")
        self.assertEqual(result, HuggingFaceTask.TEXT_GENERATION)


class BuildRuntimeParameterDefinitionsTest(unittest.TestCase):
    def test_build_for_text_generation(self) -> None:
        params = build_runtime_parameter_definitions(HuggingFaceTask.TEXT_GENERATION)
        param_names = [p.name for p in params]
        self.assertIn("batch_size", param_names)
        self.assertIn("max_new_tokens", param_names)
        self.assertIn("do_sample", param_names)
        self.assertIn("temperature", param_names)
        self.assertIn("top_p", param_names)

    def test_build_for_chronos2(self) -> None:
        params = build_runtime_parameter_definitions(HuggingFaceTask.CHRONOS2)
        param_names = [p.name for p in params]
        self.assertIn("batch_size", param_names)
        self.assertIn("context_length", param_names)
        self.assertIn("use_covariates", param_names)
        self.assertIn("cross_learning", param_names)
        self.assertIn("max_output_patches", param_names)

    def test_build_for_sundial(self) -> None:
        params = build_runtime_parameter_definitions(HuggingFaceTask.SUNDIAL)
        param_names = [p.name for p in params]
        self.assertIn("batch_size", param_names)
        self.assertIn("do_sample", param_names)
        self.assertIn("temperature", param_names)
        self.assertIn("top_p", param_names)


class HuggingFaceConfigTest(unittest.TestCase):
    def test_default_values(self) -> None:
        config = HuggingFaceConfig(repo_id="test/model")
        self.assertEqual(config.repo_id, "test/model")
        self.assertEqual(config.task, HuggingFaceTask.TEXT_GENERATION)
        self.assertIsNone(config.revision)
        self.assertFalse(config.trust_remote_code)
        self.assertEqual(config.max_new_tokens, 128)

    def test_with_chronos2_task(self) -> None:
        config = HuggingFaceConfig(
            repo_id="amazon/chronos-2",
            task=HuggingFaceTask.CHRONOS2,
        )
        self.assertEqual(config.task, HuggingFaceTask.CHRONOS2)


class ModelRegistrationRequestTest(unittest.TestCase):
    def test_create_request(self) -> None:
        request = ModelRegistrationRequest(
            model_id="my-model",
            name="My Model",
            adapter=ModelAdapter.SEASONAL_NAIVE,
            manual="Test model",
        )
        self.assertEqual(request.model_id, "my-model")
        self.assertEqual(request.name, "My Model")
        self.assertEqual(request.adapter, ModelAdapter.SEASONAL_NAIVE)
        self.assertEqual(request.source_type, "uploaded_manual")


class HuggingFaceModelRegistrationRequestTest(unittest.TestCase):
    def test_create_request(self) -> None:
        request = HuggingFaceModelRegistrationRequest(
            repo_id="amazon/chronos-2",
            name="Chronos 2",
            manual="Amazon Chronos 2 model",
        )
        self.assertEqual(request.repo_id, "amazon/chronos-2")
        self.assertEqual(request.name, "Chronos 2")
        self.assertIn("forecast", request.capabilities)
        self.assertIn("huggingface", request.capabilities)


if __name__ == "__main__":
    unittest.main()