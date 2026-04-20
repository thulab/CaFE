from __future__ import annotations

import random
import unittest

from backend.app.config import AppSettings
from backend.app.datasets.domain import NoiseMode, TrackKind, TrackTemplateKind, TrackSpec
from backend.app.datasets.synthetic import SyntheticDatasetGenerator


def _make_test_settings() -> AppSettings:
    return AppSettings.model_validate({
        "system": {
            "runtime": {
                "root": "/tmp/test",
                "backend_pid_file": "p",
                "frontend_pid_file": "f",
                "backend_log_file": "b.log",
                "frontend_log_file": "f.log",
            },
            "healthcheck": {"attempts": 1, "interval_seconds": 1.0, "timeout_seconds": 10.0},
            "shutdown": {"grace_attempts": 1, "interval_seconds": 1.0},
        },
        "service": {
            "backend": {"host": "localhost", "port": 8000},
            "frontend": {"host": "localhost", "port": 5000, "get_timeout_seconds": 10.0, "post_timeout_seconds": 10.0},
        },
        "ui": {
            "dashboard": {
                "user_leaderboard_limit": 10,
                "user_track_leaderboard_limit": 5,
                "admin_recent_batches_limit": 20,
                "admin_recent_tasks_limit": 20,
                "admin_leaderboard_limit": 20,
            },
            "user_model_submission": {
                "huggingface_url": "",
                "name": "",
                "manual": "",
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": 1.0,
                "batch_size": 1,
                "load_retries": 1,
                "load_retry_backoff_seconds": 1.0,
                "do_sample": False,
                "trust_remote_code": False,
                "use_covariates": True,
                "cross_learning": False,
                "recommended_profile_label": "",
            },
            "admin_batch_generation": {
                "track": "forecast_accuracy",
                "sample_count": 10,
                "context_length": 96,
                "horizon": 24,
                "seed": 42,
                "min_sample_count": 1,
                "min_context_length": 8,
                "min_horizon": 1,
            },
        },
        "benchmark": {
            "tracks": {},
            "builtin_models": [],
            "synthetic_generation": {
                "max_generation_attempts": 10,
                "phase_shift_probability": 0.5,
                "context_length_period_threshold": 100,
                "default_short_periods": [4, 8, 12],
                "default_long_periods": [12, 24, 48],
                "covariate_robustness_periods": [6, 12, 24],
                "noise_robustness_periods": [8, 16, 32],
                "cost_intensive_periods": [24, 48],
                "amplitude_modes": ["stable", "slow_drift", "mid_spike"],
                "trend_types": ["linear", "piecewise_linear", "smooth_curve"],
                "difficulties": ["easy", "medium", "hard"],
                "amplitude_base": 1.0,
                "slow_drift_strength": 0.1,
                "mid_spike_multiplier": 2.0,
                "phase_shift_radians": 1.5708,
                "covariate_helpful_scale": 0.5,
                "covariate_helpful_noise_divisor": 2.0,
                "covariate_distractor_count": 2,
                "covariate_distractor_period_choices": [6, 12],
                "covariate_distractor_amplitude": 0.1,
                "covariate_distractor_noise_std": 0.05,
                "noise_history_multiplier": 1.5,
                "noise_probe_std": 0.1,
                "calendar_signal_period": 24,
                "load_signal_trend_scale": 0.01,
                "trend_linear_scale": 0.01,
                "trend_piecewise_first_ratio": 0.33,
                "trend_piecewise_first_slope": 0.02,
                "trend_piecewise_second_ratio": 0.66,
                "trend_piecewise_second_base": 0.5,
                "trend_piecewise_second_slope": 0.01,
                "trend_piecewise_third_base": 0.7,
                "trend_piecewise_third_slope": 0.005,
                "trend_smooth_base": 0.0,
                "trend_smooth_slope": 0.01,
                "trend_smooth_wave": 0.2,
                "noise_base_levels": {"forecast_accuracy": 0.1, "noise_robustness": 0.35, "covariate_robustness": 0.2, "cost_intensive": 0.15},
                "difficulty_factors": {"easy": 1.0, "medium": 1.5, "hard": 2.0},
            },
            "huggingface": {
                "text_generation_history_limit": 50,
                "text_generation_covariate_limit": 5,
                "text_generation_covariate_value_limit": 20,
            },
            "scoring": {
                "base_tokens": 100,
                "history_token_divisor": 4,
                "covariate_token_weight": 1,
                "token_per_horizon": 10,
                "cost_track_latency_multiplier": 1.5,
                "cost_track_token_bonus": 50,
                "noise_track_latency_multiplier": 0.8,
                "composite_base": 100.0,
                "composite_mse_offset": 0.01,
                "composite_latency_penalty": 0.001,
                "composite_token_penalty": 0.0001,
            },
            "stub_models": {
                "seasonal_naive": {"latency_base": 0.5, "latency_per_horizon": 0.01, "latency_per_covariate": 0.005},
                "recent_mean": {"window": 8, "latency_base": 0.3, "latency_per_horizon": 0.005},
                "covariate_trap": {"latency_base": 0.4, "latency_per_horizon": 0.008, "latency_per_covariate": 0.01},
            },
            "reporting": {"bad_case_count": 5, "strength_mse_threshold": 0.05, "strength_latency_ms_threshold": 100.0, "risk_token_threshold": 1000},
            "leaderboards": {"track_aggregation_strategy": "best_composite_score", "overall_ranking_strategy": "rank_sum", "missing_track_rank_penalty": "max_rank"},
        },
        "data_inference": {
            "trend_linear_threshold_per_step": 0.01,
            "difficulty_medium_amplitude": 5.0,
            "difficulty_hard_amplitude": 10.0,
            "cost_period_floor": 12,
            "cost_period_cap": 48,
            "future_known_covariates_exact": ["calendar_signal"],
            "future_known_covariates_prefixes": ["load_"],
        },
        "validation": {"low_variance_min_range": 0.01},
    })


class SyntheticDatasetGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _make_test_settings()
        self.generator = SyntheticDatasetGenerator(self.settings)

    def _make_track_spec(self, track: TrackKind = TrackKind.FORECAST_ACCURACY, noise_mode: NoiseMode = NoiseMode.CLEAN) -> TrackSpec:
        from backend.app.datasets.domain import ExecutionConstraint
        return TrackSpec(
            track=track,
            track_variant_id=f"{track.value}.{noise_mode.value}",
            track_template_kind=TrackTemplateKind.UNIVARIATE_FORECAST,
            noise_mode=noise_mode,
            execution_constraint=ExecutionConstraint.PER_CHANNEL_UNIVARIATE,
            name=track.value,
            description="test",
            fairness_policy="equal",
            default_context_length=96,
            default_horizon=24,
            suggested_sample_count=100,
            input_channels=["target"],
            target_channels=["target"],
            future_known_channels=[],
            knobs=[],
            aliases=[],
        )

    def test_generate_sample_returns_series_sample(self) -> None:
        rng = random.Random(42)
        track_spec = self._make_track_spec()
        sample = self.generator.generate_sample(
            rng=rng,
            sample_id="test-sample",
            track_spec=track_spec,
            input_length=24,
            prediction_length=8,
        )

        self.assertEqual(sample.sample_id, "test-sample")
        self.assertEqual(len(sample.history), 24)
        self.assertEqual(len(sample.target), 8)
        self.assertIsNotNone(sample.truth)

    def test_generate_sample_clean_mode_no_noise_added(self) -> None:
        rng = random.Random(42)
        track_spec = self._make_track_spec(noise_mode=NoiseMode.CLEAN)
        sample = self.generator.generate_sample(
            rng=rng,
            sample_id="clean-sample",
            track_spec=track_spec,
            input_length=24,
            prediction_length=8,
        )

        self.assertEqual(len(sample.history), 24)
        self.assertEqual(len(sample.target), 8)
        self.assertGreater(sample.truth.noise_level, 0)

    def test_generate_sample_noisy_mode_applies_noise(self) -> None:
        rng = random.Random(42)
        track_spec = self._make_track_spec(noise_mode=NoiseMode.NOISY)
        sample = self.generator.generate_sample(
            rng=rng,
            sample_id="noisy-sample",
            track_spec=track_spec,
            input_length=24,
            prediction_length=8,
        )

        self.assertEqual(len(sample.history), 24)
        self.assertEqual(len(sample.target), 8)
        self.assertGreater(sample.truth.noise_level, 0.1)

    def test_generate_sample_trend_value_linear(self) -> None:
        trend_value = self.generator._trend_value(step=50, total_length=100, trend_type="linear")
        self.assertGreater(trend_value, 0)

    def test_generate_sample_trend_value_smooth_curve(self) -> None:
        trend_value = self.generator._trend_value(step=50, total_length=100, trend_type="smooth_curve")
        self.assertIsNotNone(trend_value)

    def test_generate_sample_trend_value_piecewise_linear(self) -> None:
        trend_value = self.generator._trend_value(step=50, total_length=100, trend_type="piecewise_linear")
        self.assertIsNotNone(trend_value)

    def test_noise_level_for_track(self) -> None:
        level = self.generator._noise_level_for("forecast_accuracy", "easy")
        self.assertAlmostEqual(level, 0.1, places=3)

        level_hard = self.generator._noise_level_for("forecast_accuracy", "hard")
        self.assertAlmostEqual(level_hard, 0.2, places=3)

        level_noise_track = self.generator._noise_level_for("noise_robustness", "easy")
        self.assertAlmostEqual(level_noise_track, 0.35, places=3)


if __name__ == "__main__":
    unittest.main()