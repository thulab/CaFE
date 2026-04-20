from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from backend.app.config import (
    AppSettings,
    TrackSpecConfig,
    default_conf_path,
    default_repo_root,
    infer_difficulty,
    infer_periods_for_track,
    infer_trend_type,
    is_future_known_covariate,
    lookup_key,
    track_value,
)


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings.model_validate({
            "system": {
                "runtime": {
                    "root": "/tmp/test_runtime",
                    "backend_pid_file": "backend.pid",
                    "frontend_pid_file": "frontend.pid",
                    "backend_log_file": "backend.log",
                    "frontend_log_file": "frontend.log",
                },
                "healthcheck": {
                    "attempts": 5,
                    "interval_seconds": 1.0,
                    "timeout_seconds": 30.0,
                },
                "shutdown": {
                    "grace_attempts": 3,
                    "interval_seconds": 2.0,
                },
            },
            "service": {
                "backend": {"host": "localhost", "port": 8000, "reload": False},
                "frontend": {
                    "host": "localhost",
                    "port": 5000,
                    "debug": False,
                    "backend_base_url": "",
                    "get_timeout_seconds": 10.0,
                    "post_timeout_seconds": 30.0,
                },
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
                    "huggingface_url": "https://huggingface.co",
                    "name": "User Model",
                    "model_id": "",
                    "revision": "",
                    "manual": "user model",
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "device_map": "",
                    "torch_dtype": "",
                    "attn_implementation": "",
                    "batch_size": 1,
                    "context_length": None,
                    "max_output_patches": None,
                    "load_retries": 1,
                    "load_retry_backoff_seconds": 1.0,
                    "do_sample": False,
                    "trust_remote_code": False,
                    "use_covariates": True,
                    "cross_learning": False,
                    "recommended_profile_label": "default",
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
                    }
                },
                "builtin_models": [],
                "synthetic_generation": {
                    "max_generation_attempts": 10,
                    "phase_shift_probability": 0.2,
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
                    "noise_base_levels": {"forecast_accuracy": 0.1, "noise_robustness": 0.35},
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
                    "seasonal_naive": {
                        "latency_base": 0.5,
                        "latency_per_horizon": 0.01,
                        "latency_per_covariate": 0.005,
                    },
                    "recent_mean": {
                        "window": 8,
                        "latency_base": 0.3,
                        "latency_per_horizon": 0.005,
                    },
                    "covariate_trap": {
                        "latency_base": 0.4,
                        "latency_per_horizon": 0.008,
                        "latency_per_covariate": 0.01,
                    },
                },
                "reporting": {
                    "bad_case_count": 5,
                    "strength_mse_threshold": 0.05,
                    "strength_latency_ms_threshold": 100.0,
                    "risk_token_threshold": 1000,
                },
                "leaderboards": {
                    "track_aggregation_strategy": "mean",
                    "overall_ranking_strategy": "rank_sum",
                    "missing_track_rank_penalty": "max_rank",
                },
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
            "validation": {
                "low_variance_min_range": 0.01,
            },
        })

    def test_settings_resolve_path_absolute(self) -> None:
        path = Path("/absolute/path")
        resolved = self.settings.resolve_path(path)
        self.assertEqual(resolved, path)

    def test_settings_resolve_path_relative(self) -> None:
        path = Path("relative/path")
        resolved = self.settings.resolve_path(path, repo_root=Path("/custom/root"))
        self.assertEqual(resolved, Path("/custom/root/relative/path"))

    def test_settings_runtime_root(self) -> None:
        root = self.settings.runtime_root()
        self.assertEqual(root, Path("/tmp/test_runtime"))

    def test_settings_backend_url(self) -> None:
        url = self.settings.backend_url()
        self.assertEqual(url, "http://localhost:8000")

    def test_settings_frontend_url(self) -> None:
        url = self.settings.frontend_url()
        self.assertEqual(url, "http://localhost:5000")

    def test_settings_frontend_backend_base_url_empty(self) -> None:
        url = self.settings.frontend_backend_base_url()
        self.assertEqual(url, "http://localhost:8000")

    def test_track_value_with_enum(self) -> None:
        class FakeTrack:
            value = "test_value"

        self.assertEqual(track_value(FakeTrack()), "test_value")

    def test_track_value_with_string(self) -> None:
        self.assertEqual(track_value("direct_string"), "direct_string")

    def test_lookup_key_nested_model(self) -> None:
        result = lookup_key(self.settings, "system.runtime.root")
        self.assertEqual(result, "/tmp/test_runtime")

    def test_lookup_key_with_dict(self) -> None:
        data = {"a": {"b": {"c": 42}}}
        result = lookup_key(data, "a.b.c")
        self.assertEqual(result, 42)

    def test_lookup_key_simple_attribute(self) -> None:
        result = lookup_key(self.settings.system, "runtime.root")
        self.assertEqual(result, "/tmp/test_runtime")

    def test_user_model_submission_properties(self) -> None:
        submission = self.settings.ui.user_model_submission
        self.assertEqual(submission.max_output_patches_value, "")
        self.assertEqual(submission.context_length_value, "")


class InferPeriodsForTrackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings.model_validate({
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
                    "phase_shift_probability": 0.2,
                    "context_length_period_threshold": 100,
                    "default_short_periods": [4, 8, 12],
                    "default_long_periods": [12, 24, 48],
                    "covariate_robustness_periods": [6, 12, 24],
                    "noise_robustness_periods": [8, 16, 32],
                    "cost_intensive_periods": [24, 48],
                    "amplitude_modes": ["stable"],
                    "trend_types": ["linear"],
                    "difficulties": ["easy"],
                    "amplitude_base": 1.0,
                    "slow_drift_strength": 0.0,
                    "mid_spike_multiplier": 1.0,
                    "phase_shift_radians": 0.0,
                    "covariate_helpful_scale": 0.0,
                    "covariate_helpful_noise_divisor": 1.0,
                    "covariate_distractor_count": 0,
                    "covariate_distractor_period_choices": [],
                    "covariate_distractor_amplitude": 0.0,
                    "covariate_distractor_noise_std": 0.0,
                    "noise_history_multiplier": 1.0,
                    "noise_probe_std": 0.0,
                    "calendar_signal_period": 24,
                    "load_signal_trend_scale": 0.0,
                    "trend_linear_scale": 0.0,
                    "trend_piecewise_first_ratio": 0.0,
                    "trend_piecewise_first_slope": 0.0,
                    "trend_piecewise_second_ratio": 0.0,
                    "trend_piecewise_second_base": 0.0,
                    "trend_piecewise_second_slope": 0.0,
                    "trend_piecewise_third_base": 0.0,
                    "trend_piecewise_third_slope": 0.0,
                    "trend_smooth_base": 0.0,
                    "trend_smooth_slope": 0.0,
                    "trend_smooth_wave": 0.0,
                    "noise_base_levels": {},
                    "difficulty_factors": {},
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
                    "cost_track_latency_multiplier": 1.0,
                    "cost_track_token_bonus": 0,
                    "noise_track_latency_multiplier": 1.0,
                    "composite_base": 100.0,
                    "composite_mse_offset": 0.01,
                    "composite_latency_penalty": 0.001,
                    "composite_token_penalty": 0.0001,
                },
                "stub_models": {
                    "seasonal_naive": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                    "recent_mean": {"window": 8, "latency_base": 0.0, "latency_per_horizon": 0.0},
                    "covariate_trap": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                },
                "reporting": {"bad_case_count": 5, "strength_mse_threshold": 0.05, "strength_latency_ms_threshold": 100.0, "risk_token_threshold": 1000},
                "leaderboards": {"track_aggregation_strategy": "mean", "overall_ranking_strategy": "rank_sum", "missing_track_rank_penalty": "max_rank"},
            },
            "data_inference": {
                "trend_linear_threshold_per_step": 0.01,
                "difficulty_medium_amplitude": 5.0,
                "difficulty_hard_amplitude": 10.0,
                "cost_period_floor": 12,
                "cost_period_cap": 48,
                "future_known_covariates_exact": [],
                "future_known_covariates_prefixes": [],
            },
            "validation": {"low_variance_min_range": 0.01},
        })

    def test_infer_periods_cost_intensive(self) -> None:
        result = infer_periods_for_track("cost_intensive", 200, settings=self.settings)
        self.assertEqual(len(result), 3)

    def test_infer_periods_noise_robustness(self) -> None:
        result = infer_periods_for_track("noise_robustness", 50, settings=self.settings)
        self.assertEqual(result, [8, 16, 32])

    def test_infer_periods_covariate_robustness(self) -> None:
        result = infer_periods_for_track("covariate_robustness", 50, settings=self.settings)
        self.assertEqual(result, [6, 12, 24])

    def test_infer_periods_short_context(self) -> None:
        result = infer_periods_for_track("forecast_accuracy", 50, settings=self.settings)
        self.assertEqual(result, [4, 8, 12])

    def test_infer_periods_long_context(self) -> None:
        result = infer_periods_for_track("forecast_accuracy", 150, settings=self.settings)
        self.assertEqual(result, [12, 24, 48])


class InferTrendTypeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings.model_validate({
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
                    "phase_shift_probability": 0.2,
                    "context_length_period_threshold": 100,
                    "default_short_periods": [4, 8, 12],
                    "default_long_periods": [12, 24, 48],
                    "covariate_robustness_periods": [6, 12, 24],
                    "noise_robustness_periods": [8, 16, 32],
                    "cost_intensive_periods": [24, 48],
                    "amplitude_modes": ["stable"],
                    "trend_types": ["linear"],
                    "difficulties": ["easy"],
                    "amplitude_base": 1.0,
                    "slow_drift_strength": 0.0,
                    "mid_spike_multiplier": 1.0,
                    "phase_shift_radians": 0.0,
                    "covariate_helpful_scale": 0.0,
                    "covariate_helpful_noise_divisor": 1.0,
                    "covariate_distractor_count": 0,
                    "covariate_distractor_period_choices": [],
                    "covariate_distractor_amplitude": 0.0,
                    "covariate_distractor_noise_std": 0.0,
                    "noise_history_multiplier": 1.0,
                    "noise_probe_std": 0.0,
                    "calendar_signal_period": 24,
                    "load_signal_trend_scale": 0.0,
                    "trend_linear_scale": 0.0,
                    "trend_piecewise_first_ratio": 0.0,
                    "trend_piecewise_first_slope": 0.0,
                    "trend_piecewise_second_ratio": 0.0,
                    "trend_piecewise_second_base": 0.0,
                    "trend_piecewise_second_slope": 0.0,
                    "trend_piecewise_third_base": 0.0,
                    "trend_piecewise_third_slope": 0.0,
                    "trend_smooth_base": 0.0,
                    "trend_smooth_slope": 0.0,
                    "trend_smooth_wave": 0.0,
                    "noise_base_levels": {},
                    "difficulty_factors": {},
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
                    "cost_track_latency_multiplier": 1.0,
                    "cost_track_token_bonus": 0,
                    "noise_track_latency_multiplier": 1.0,
                    "composite_base": 100.0,
                    "composite_mse_offset": 0.01,
                    "composite_latency_penalty": 0.001,
                    "composite_token_penalty": 0.0001,
                },
                "stub_models": {
                    "seasonal_naive": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                    "recent_mean": {"window": 8, "latency_base": 0.0, "latency_per_horizon": 0.0},
                    "covariate_trap": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                },
                "reporting": {"bad_case_count": 5, "strength_mse_threshold": 0.05, "strength_latency_ms_threshold": 100.0, "risk_token_threshold": 1000},
                "leaderboards": {"track_aggregation_strategy": "mean", "overall_ranking_strategy": "rank_sum", "missing_track_rank_penalty": "max_rank"},
            },
            "data_inference": {
                "trend_linear_threshold_per_step": 0.01,
                "difficulty_medium_amplitude": 5.0,
                "difficulty_hard_amplitude": 10.0,
                "cost_period_floor": 12,
                "cost_period_cap": 48,
                "future_known_covariates_exact": [],
                "future_known_covariates_prefixes": [],
            },
            "validation": {"low_variance_min_range": 0.01},
        })

    def test_infer_trend_type_linear(self) -> None:
        series = [float(i) for i in range(100)]
        result = infer_trend_type(series, settings=self.settings)
        self.assertEqual(result, "linear")

    def test_infer_trend_type_smooth_curve(self) -> None:
        import math
        series = [math.sin(i * 0.1) for i in range(100)]
        result = infer_trend_type(series, settings=self.settings)
        self.assertEqual(result, "smooth_curve")


class InferDifficultyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings.model_validate({
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
                    "phase_shift_probability": 0.2,
                    "context_length_period_threshold": 100,
                    "default_short_periods": [4, 8, 12],
                    "default_long_periods": [12, 24, 48],
                    "covariate_robustness_periods": [6, 12, 24],
                    "noise_robustness_periods": [8, 16, 32],
                    "cost_intensive_periods": [24, 48],
                    "amplitude_modes": ["stable"],
                    "trend_types": ["linear"],
                    "difficulties": ["easy"],
                    "amplitude_base": 1.0,
                    "slow_drift_strength": 0.0,
                    "mid_spike_multiplier": 1.0,
                    "phase_shift_radians": 0.0,
                    "covariate_helpful_scale": 0.0,
                    "covariate_helpful_noise_divisor": 1.0,
                    "covariate_distractor_count": 0,
                    "covariate_distractor_period_choices": [],
                    "covariate_distractor_amplitude": 0.0,
                    "covariate_distractor_noise_std": 0.0,
                    "noise_history_multiplier": 1.0,
                    "noise_probe_std": 0.0,
                    "calendar_signal_period": 24,
                    "load_signal_trend_scale": 0.0,
                    "trend_linear_scale": 0.0,
                    "trend_piecewise_first_ratio": 0.0,
                    "trend_piecewise_first_slope": 0.0,
                    "trend_piecewise_second_ratio": 0.0,
                    "trend_piecewise_second_base": 0.0,
                    "trend_piecewise_second_slope": 0.0,
                    "trend_piecewise_third_base": 0.0,
                    "trend_piecewise_third_slope": 0.0,
                    "trend_smooth_base": 0.0,
                    "trend_smooth_slope": 0.0,
                    "trend_smooth_wave": 0.0,
                    "noise_base_levels": {},
                    "difficulty_factors": {},
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
                    "cost_track_latency_multiplier": 1.0,
                    "cost_track_token_bonus": 0,
                    "noise_track_latency_multiplier": 1.0,
                    "composite_base": 100.0,
                    "composite_mse_offset": 0.01,
                    "composite_latency_penalty": 0.001,
                    "composite_token_penalty": 0.0001,
                },
                "stub_models": {
                    "seasonal_naive": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                    "recent_mean": {"window": 8, "latency_base": 0.0, "latency_per_horizon": 0.0},
                    "covariate_trap": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                },
                "reporting": {"bad_case_count": 5, "strength_mse_threshold": 0.05, "strength_latency_ms_threshold": 100.0, "risk_token_threshold": 1000},
                "leaderboards": {"track_aggregation_strategy": "mean", "overall_ranking_strategy": "rank_sum", "missing_track_rank_penalty": "max_rank"},
            },
            "data_inference": {
                "trend_linear_threshold_per_step": 0.01,
                "difficulty_medium_amplitude": 5.0,
                "difficulty_hard_amplitude": 10.0,
                "cost_period_floor": 12,
                "cost_period_cap": 48,
                "future_known_covariates_exact": [],
                "future_known_covariates_prefixes": [],
            },
            "validation": {"low_variance_min_range": 0.01},
        })

    def test_infer_difficulty_easy(self) -> None:
        series = [1.0, 1.1, 1.2, 1.0, 1.1]
        result = infer_difficulty(series, settings=self.settings)
        self.assertEqual(result, "easy")

    def test_infer_difficulty_medium(self) -> None:
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
        result = infer_difficulty(series, settings=self.settings)
        self.assertEqual(result, "medium")

    def test_infer_difficulty_hard(self) -> None:
        series = [1.0, 5.0, 2.0, 8.0, 3.0, 12.0]
        result = infer_difficulty(series, settings=self.settings)
        self.assertEqual(result, "hard")


class FutureKnownCovariatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings.model_validate({
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
                    "phase_shift_probability": 0.2,
                    "context_length_period_threshold": 100,
                    "default_short_periods": [4, 8, 12],
                    "default_long_periods": [12, 24, 48],
                    "covariate_robustness_periods": [6, 12, 24],
                    "noise_robustness_periods": [8, 16, 32],
                    "cost_intensive_periods": [24, 48],
                    "amplitude_modes": ["stable"],
                    "trend_types": ["linear"],
                    "difficulties": ["easy"],
                    "amplitude_base": 1.0,
                    "slow_drift_strength": 0.0,
                    "mid_spike_multiplier": 1.0,
                    "phase_shift_radians": 0.0,
                    "covariate_helpful_scale": 0.0,
                    "covariate_helpful_noise_divisor": 1.0,
                    "covariate_distractor_count": 0,
                    "covariate_distractor_period_choices": [],
                    "covariate_distractor_amplitude": 0.0,
                    "covariate_distractor_noise_std": 0.0,
                    "noise_history_multiplier": 1.0,
                    "noise_probe_std": 0.0,
                    "calendar_signal_period": 24,
                    "load_signal_trend_scale": 0.0,
                    "trend_linear_scale": 0.0,
                    "trend_piecewise_first_ratio": 0.0,
                    "trend_piecewise_first_slope": 0.0,
                    "trend_piecewise_second_ratio": 0.0,
                    "trend_piecewise_second_base": 0.0,
                    "trend_piecewise_second_slope": 0.0,
                    "trend_piecewise_third_base": 0.0,
                    "trend_piecewise_third_slope": 0.0,
                    "trend_smooth_base": 0.0,
                    "trend_smooth_slope": 0.0,
                    "trend_smooth_wave": 0.0,
                    "noise_base_levels": {},
                    "difficulty_factors": {},
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
                    "cost_track_latency_multiplier": 1.0,
                    "cost_track_token_bonus": 0,
                    "noise_track_latency_multiplier": 1.0,
                    "composite_base": 100.0,
                    "composite_mse_offset": 0.01,
                    "composite_latency_penalty": 0.001,
                    "composite_token_penalty": 0.0001,
                },
                "stub_models": {
                    "seasonal_naive": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                    "recent_mean": {"window": 8, "latency_base": 0.0, "latency_per_horizon": 0.0},
                    "covariate_trap": {"latency_base": 0.0, "latency_per_horizon": 0.0, "latency_per_covariate": 0.0},
                },
                "reporting": {"bad_case_count": 5, "strength_mse_threshold": 0.05, "strength_latency_ms_threshold": 100.0, "risk_token_threshold": 1000},
                "leaderboards": {"track_aggregation_strategy": "mean", "overall_ranking_strategy": "rank_sum", "missing_track_rank_penalty": "max_rank"},
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

    def test_future_known_covariates_exact_match(self) -> None:
        result = self.settings.data_inference.future_known_covariates_exact
        self.assertIn("calendar_signal", result)

    def test_future_known_covariates_prefix_match(self) -> None:
        from backend.app.config import future_known_covariates
        result = future_known_covariates(["calendar_signal", "load_power", "other"], settings=self.settings)
        self.assertIn("calendar_signal", result)
        self.assertIn("load_power", result)
        self.assertNotIn("other", result)

    def test_is_future_known_covariate(self) -> None:
        self.assertTrue(is_future_known_covariate("calendar_signal", settings=self.settings))
        self.assertTrue(is_future_known_covariate("load_power", settings=self.settings))
        self.assertFalse(is_future_known_covariate("random_name", settings=self.settings))


if __name__ == "__main__":
    unittest.main()