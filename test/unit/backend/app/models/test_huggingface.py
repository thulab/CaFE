from __future__ import annotations

import unittest

from backend.app.datasets.domain import SeriesSample, SeriesTruth, TrackKind
from backend.app.models.domain import HuggingFaceConfig, HuggingFaceTask
from backend.app.models.huggingface import (
    HuggingFaceForecast,
    HuggingFaceRunnerError,
    HuggingFaceModelRunner,
    BuiltinChronos2FallbackRunner,
    BuiltinSundialFallbackRunner,
    _missing_dependency_names,
    _supports_builtin_dependency_fallback,
    _format_missing_dependency_message,
)
from test.support.helpers import build_sample


class HuggingFaceForecastTest(unittest.TestCase):
    def test_create_forecast(self) -> None:
        forecast = HuggingFaceForecast(
            prediction=[1.0, 2.0, 3.0],
            latency_ms=10.5,
            token_count=100,
            notes={"model": "test"},
        )
        self.assertEqual(forecast.prediction, [1.0, 2.0, 3.0])
        self.assertEqual(forecast.latency_ms, 10.5)
        self.assertEqual(forecast.token_count, 100)
        self.assertEqual(forecast.notes["model"], "test")


class HuggingFaceRunnerErrorTest(unittest.TestCase):
    def test_error_inherits_from_runtime_error(self) -> None:
        error = HuggingFaceRunnerError("test error")
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "test error")


class MissingDependencyNamesTest(unittest.TestCase):
    def test_missing_dependencies_returns_list(self) -> None:
        result = _missing_dependency_names("nonexistent_module_abc123")
        self.assertIsInstance(result, list)
        self.assertIn("nonexistent_module_abc123", result)

    def test_all_present_returns_empty(self) -> None:
        result = _missing_dependency_names("unittest")
        self.assertEqual(result, [])


class SupportsBuiltinDependencyFallbackTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        result = _supports_builtin_dependency_fallback(
            repo_id="amazon/chronos-2",
            expected_repo_id="amazon/chronos-2",
        )
        self.assertTrue(result)

    def test_case_insensitive(self) -> None:
        result = _supports_builtin_dependency_fallback(
            repo_id="AMAZON/CHRONOS-2",
            expected_repo_id="amazon/chronos-2",
        )
        self.assertTrue(result)

    def test_trailing_slash(self) -> None:
        result = _supports_builtin_dependency_fallback(
            repo_id="amazon/chronos-2/",
            expected_repo_id="amazon/chronos-2",
        )
        self.assertTrue(result)

    def test_no_match(self) -> None:
        result = _supports_builtin_dependency_fallback(
            repo_id="other/model",
            expected_repo_id="amazon/chronos-2",
        )
        self.assertFalse(result)


class FormatMissingDependencyMessageTest(unittest.TestCase):
    def test_format_message(self) -> None:
        msg = _format_missing_dependency_message(
            model_label="TestModel",
            missing=["transformers", "torch"],
            packages=["transformers", "torch"],
        )
        self.assertIn("TestModel", msg)
        self.assertIn("transformers", msg)
        self.assertIn("torch", msg)


class BuiltinChronos2FallbackRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HuggingFaceConfig(
            repo_id="amazon/chronos-2",
            task=HuggingFaceTask.CHRONOS2,
        )
        self.runner = BuiltinChronos2FallbackRunner(self.config)

    def test_load_does_nothing(self) -> None:
        self.runner.load()

    def test_forecast_batch_returns_forecasts(self) -> None:
        sample = build_sample(history=[1.0, 2.0, 3.0, 4.0], target=[5.0, 6.0])
        forecasts = self.runner.forecast_batch([sample], track=TrackKind.FORECAST_ACCURACY)
        self.assertEqual(len(forecasts), 1)
        self.assertIsInstance(forecasts[0], HuggingFaceForecast)

    def test_predict_uses_dominant_period(self) -> None:
        sample = build_sample(
            history=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            target=[9.0, 10.0],
            truth=SeriesTruth(
                trend_type="linear",
                periods=[4, 8],
                dominant_period=4,
                amplitude_mode="stable",
                phase_shift=False,
                noise_level=0.1,
                difficulty="easy",
            ),
        )
        forecast = self.runner._predict(sample)
        self.assertEqual(len(forecast), 2)


class BuiltinSundialFallbackRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HuggingFaceConfig(
            repo_id="thuml/sundial-base-128m",
            task=HuggingFaceTask.SUNDIAL,
        )
        self.runner = BuiltinSundialFallbackRunner(self.config)

    def test_load_does_nothing(self) -> None:
        self.runner.load()

    def test_forecast_batch_returns_forecasts(self) -> None:
        sample = build_sample(history=[1.0, 2.0, 3.0, 4.0], target=[5.0, 6.0])
        forecasts = self.runner.forecast_batch([sample], track=TrackKind.FORECAST_ACCURACY)
        self.assertEqual(len(forecasts), 1)
        self.assertIsInstance(forecasts[0], HuggingFaceForecast)

    def test_predict_uses_recent_mean(self) -> None:
        sample = build_sample(history=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], target=[9.0, 10.0])
        forecast = self.runner._predict(sample)
        self.assertEqual(len(forecast), 2)
        mean_val = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8
        self.assertAlmostEqual(forecast[0], mean_val, places=3)


class HuggingFaceModelRunnerTest(unittest.TestCase):
    def test_delegate_for_chronos2_without_deps_uses_fallback(self) -> None:
        config = HuggingFaceConfig(
            repo_id="amazon/chronos-2",
            task=HuggingFaceTask.CHRONOS2,
        )
        runner = HuggingFaceModelRunner(config)
        delegate = runner._delegate()
        self.assertIsInstance(delegate, BuiltinChronos2FallbackRunner)

    def test_delegate_for_sundial_without_deps_uses_fallback(self) -> None:
        config = HuggingFaceConfig(
            repo_id="thuml/sundial-base-128m",
            task=HuggingFaceTask.SUNDIAL,
        )
        runner = HuggingFaceModelRunner(config)
        delegate = runner._delegate()
        self.assertIsInstance(delegate, BuiltinSundialFallbackRunner)


if __name__ == "__main__":
    unittest.main()