from __future__ import annotations

import unittest
from pathlib import Path

from backend.app.config import AppSettings
from backend.app.datasets.domain import BatchGenerationRequest, TrackKind
from backend.app.services import BenchmarkEngine
from test.support.helpers import temporary_runtime_dir


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
            "tracks": {
                "forecast_accuracy": {
                    "name": "Forecast Accuracy",
                    "description": "Standard zero-shot forecasting",
                    "fairness_policy": "equal",
                    "default_context_length": 96,
                    "default_horizon": 24,
                    "suggested_sample_count": 100,
                    "knobs": [],
                },
                "noise_robustness": {
                    "name": "Noise Robustness",
                    "description": "Noise robustness evaluation",
                    "fairness_policy": "equal",
                    "default_context_length": 96,
                    "default_horizon": 24,
                    "suggested_sample_count": 100,
                    "knobs": [],
                },
                "covariate_robustness": {
                    "name": "Covariate Robustness",
                    "description": "Covariate robustness evaluation",
                    "fairness_policy": "equal",
                    "default_context_length": 96,
                    "default_horizon": 24,
                    "suggested_sample_count": 100,
                    "knobs": [],
                },
                "cost_intensive": {
                    "name": "Cost Intensive",
                    "description": "Long context cost evaluation",
                    "fairness_policy": "equal",
                    "default_context_length": 192,
                    "default_horizon": 48,
                    "suggested_sample_count": 50,
                    "knobs": [],
                },
            },
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


class BenchmarkEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _make_test_settings()

    def test_init_creates_all_managers(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            self.assertIsNotNone(engine.data_manager)
            self.assertIsNotNone(engine.model_manager)
            self.assertIsNotNone(engine.task_manager)
            self.assertIsNotNone(engine.leaderboard_manager)

    def test_list_tracks(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            tracks = engine.list_tracks()
            self.assertIsInstance(tracks, list)
            self.assertGreater(len(tracks), 0)

    def test_list_models(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            models = engine.list_models()
            self.assertIsInstance(models, list)

    def test_generate_batch(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            batch = engine.generate_batch(
                BatchGenerationRequest(
                    track=TrackKind.FORECAST_ACCURACY,
                    sample_count=2,
                    context_length=24,
                    horizon=8,
                    seed=5,
                )
            )
            self.assertEqual(batch.sample_count, 2)
            self.assertEqual(batch.track, TrackKind.FORECAST_ACCURACY)

    def test_list_batches(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            batches = engine.list_batches()
            self.assertIsInstance(batches, list)

    def test_get_batch_after_generate(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            batch = engine.generate_batch(
                BatchGenerationRequest(
                    track=TrackKind.FORECAST_ACCURACY,
                    sample_count=2,
                    context_length=24,
                    horizon=8,
                    seed=5,
                )
            )
            retrieved = engine.get_batch(batch.batch_id)
            self.assertEqual(retrieved.batch_id, batch.batch_id)

    def test_leaderboard_returns_track_leaderboard(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            tracks = engine.list_tracks()
            if tracks:
                track_id = tracks[0].track_variant_id
                lb = engine.leaderboard(track=track_id, metric_id="mse")
                self.assertIsInstance(lb, list)

    def test_overview_returns_admin_dashboard(self) -> None:
        with temporary_runtime_dir(prefix="engine-test-") as runtime_root:
            engine = BenchmarkEngine(runtime_root, settings=self.settings)
            overview = engine.overview(metric_id="mse")
            self.assertIsNotNone(overview)
            self.assertEqual(overview.overall_metric_id, "mse")


if __name__ == "__main__":
    unittest.main()