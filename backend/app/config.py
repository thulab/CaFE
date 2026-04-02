from __future__ import annotations

import os
import sys
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    root: str
    backend_pid_file: str
    frontend_pid_file: str
    backend_log_file: str
    frontend_log_file: str


class HealthcheckConfig(BaseModel):
    attempts: int
    interval_seconds: float
    timeout_seconds: float


class ShutdownConfig(BaseModel):
    grace_attempts: int
    interval_seconds: float


class SystemConfig(BaseModel):
    runtime: RuntimeConfig
    healthcheck: HealthcheckConfig
    shutdown: ShutdownConfig


class BackendServiceConfig(BaseModel):
    host: str
    port: int
    reload: bool = False


class FrontendServiceConfig(BaseModel):
    host: str
    port: int
    debug: bool = False
    backend_base_url: str = ""
    get_timeout_seconds: float
    post_timeout_seconds: float


class ServiceConfig(BaseModel):
    backend: BackendServiceConfig
    frontend: FrontendServiceConfig


class DashboardUiConfig(BaseModel):
    user_leaderboard_limit: int
    user_track_leaderboard_limit: int
    admin_recent_batches_limit: int
    admin_recent_tasks_limit: int
    admin_leaderboard_limit: int


class UserModelSubmissionConfig(BaseModel):
    repo_id: str
    name: str
    model_id: str = ""
    task: str
    revision: str = ""
    manual: str
    max_new_tokens: int
    temperature: float
    top_p: float
    device_map: str = ""
    torch_dtype: str = ""
    attn_implementation: str = ""
    batch_size: int
    context_length: int | None = None
    max_output_patches: int | None = None
    load_retries: int
    load_retry_backoff_seconds: float
    do_sample: bool
    trust_remote_code: bool
    use_covariates: bool
    cross_learning: bool
    recommended_profile_label: str

    @property
    def max_output_patches_value(self) -> str:
        return "" if not self.max_output_patches else str(self.max_output_patches)

    @property
    def context_length_value(self) -> str:
        return "" if self.context_length is None else str(self.context_length)


class AdminBatchGenerationConfig(BaseModel):
    track: str
    sample_count: int
    context_length: int
    horizon: int
    seed: int
    min_sample_count: int
    min_context_length: int
    min_horizon: int


class UiConfig(BaseModel):
    dashboard: DashboardUiConfig
    user_model_submission: UserModelSubmissionConfig
    admin_batch_generation: AdminBatchGenerationConfig


class TrackSpecConfig(BaseModel):
    name: str
    description: str
    fairness_policy: str
    default_context_length: int
    default_horizon: int
    suggested_sample_count: int
    knobs: list[str] = Field(default_factory=list)


class BuiltinModelConfig(BaseModel):
    model_id: str
    name: str
    adapter: str
    source_type: str
    manual: str
    capabilities: list[str] = Field(default_factory=list)


class SyntheticGenerationConfig(BaseModel):
    max_generation_attempts: int
    phase_shift_probability: float
    context_length_period_threshold: int
    default_short_periods: list[int]
    default_long_periods: list[int]
    covariate_robustness_periods: list[int]
    noise_robustness_periods: list[int]
    cost_intensive_periods: list[int]
    amplitude_modes: list[str]
    trend_types: list[str]
    difficulties: list[str]
    amplitude_base: float
    slow_drift_strength: float
    mid_spike_multiplier: float
    phase_shift_radians: float
    covariate_helpful_scale: float
    covariate_helpful_noise_divisor: float
    covariate_distractor_count: int
    covariate_distractor_period_choices: list[int]
    covariate_distractor_amplitude: float
    covariate_distractor_noise_std: float
    noise_history_multiplier: float
    noise_probe_std: float
    calendar_signal_period: int
    load_signal_trend_scale: float
    trend_linear_scale: float
    trend_piecewise_first_ratio: float
    trend_piecewise_first_slope: float
    trend_piecewise_second_ratio: float
    trend_piecewise_second_base: float
    trend_piecewise_second_slope: float
    trend_piecewise_third_base: float
    trend_piecewise_third_slope: float
    trend_smooth_base: float
    trend_smooth_slope: float
    trend_smooth_wave: float
    noise_base_levels: dict[str, float]
    difficulty_factors: dict[str, float]


class HuggingFaceRuntimeConfig(BaseModel):
    text_generation_history_limit: int
    text_generation_covariate_limit: int
    text_generation_covariate_value_limit: int


class ScoringConfig(BaseModel):
    base_tokens: int
    history_token_divisor: int
    covariate_token_weight: int
    token_per_horizon: int
    cost_track_latency_multiplier: float
    cost_track_token_bonus: int
    noise_track_latency_multiplier: float
    composite_base: float
    composite_mse_offset: float
    composite_latency_penalty: float
    composite_token_penalty: float


class SeasonalNaiveStubConfig(BaseModel):
    latency_base: float
    latency_per_horizon: float
    latency_per_covariate: float


class RecentMeanStubConfig(BaseModel):
    window: int
    latency_base: float
    latency_per_horizon: float


class CovariateTrapStubConfig(BaseModel):
    latency_base: float
    latency_per_horizon: float
    latency_per_covariate: float


class StubModelsConfig(BaseModel):
    seasonal_naive: SeasonalNaiveStubConfig
    recent_mean: RecentMeanStubConfig
    covariate_trap: CovariateTrapStubConfig


class ReportingConfig(BaseModel):
    bad_case_count: int
    strength_mse_threshold: float
    strength_latency_ms_threshold: float
    risk_token_threshold: int


class LeaderboardConfig(BaseModel):
    track_aggregation_strategy: str
    overall_ranking_strategy: str
    missing_track_rank_penalty: str


class BenchmarkConfig(BaseModel):
    tracks: dict[str, TrackSpecConfig]
    builtin_models: list[BuiltinModelConfig] = Field(default_factory=list)
    synthetic_generation: SyntheticGenerationConfig
    huggingface: HuggingFaceRuntimeConfig
    scoring: ScoringConfig
    stub_models: StubModelsConfig
    reporting: ReportingConfig
    leaderboards: LeaderboardConfig


class DataInferenceConfig(BaseModel):
    trend_linear_threshold_per_step: float
    difficulty_medium_amplitude: float
    difficulty_hard_amplitude: float
    cost_period_floor: int
    cost_period_cap: int
    future_known_covariates_exact: list[str] = Field(default_factory=list)
    future_known_covariates_prefixes: list[str] = Field(default_factory=list)


class ValidationConfig(BaseModel):
    low_variance_min_range: float


class AppSettings(BaseModel):
    system: SystemConfig
    service: ServiceConfig
    ui: UiConfig
    benchmark: BenchmarkConfig
    data_inference: DataInferenceConfig
    validation: ValidationConfig

    def resolve_path(self, value: str | Path, repo_root: Path | None = None) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (repo_root or default_repo_root()) / path

    def runtime_root(self, repo_root: Path | None = None) -> Path:
        return self.resolve_path(self.system.runtime.root, repo_root=repo_root)

    def system_runtime_dir(self, repo_root: Path | None = None) -> Path:
        return self.runtime_root(repo_root=repo_root) / "system"

    def backend_url(self) -> str:
        return f"http://{self.service.backend.host}:{self.service.backend.port}"

    def frontend_url(self) -> str:
        return f"http://{self.service.frontend.host}:{self.service.frontend.port}"

    def frontend_backend_base_url(self) -> str:
        configured = self.service.frontend.backend_base_url.strip()
        return configured or self.backend_url()


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_conf_path(repo_root: Path | None = None) -> Path:
    env_path = os.environ.get("TSBENCHMARK_CONF")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (repo_root or default_repo_root()) / "conf" / "system.toml"


@lru_cache(maxsize=None)
def _load_settings_cached(conf_path: str) -> AppSettings:
    path = Path(conf_path)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return AppSettings.model_validate(payload)


def load_settings(conf_path: str | Path | None = None) -> AppSettings:
    path = Path(conf_path).expanduser().resolve() if conf_path is not None else default_conf_path()
    return _load_settings_cached(str(path))


def get_settings() -> AppSettings:
    return load_settings()


def infer_periods_for_track(track: object, length: int, settings: AppSettings | None = None) -> list[int]:
    cfg = (settings or get_settings()).benchmark.synthetic_generation
    key = track_value(track)
    if key == "cost_intensive":
        floor = max((settings or get_settings()).data_inference.cost_period_floor, 1)
        cap = max((settings or get_settings()).data_inference.cost_period_cap, floor)
        dynamic = min(cap, max(length // 2, floor))
        return [cfg.cost_intensive_periods[0], cfg.cost_intensive_periods[1], dynamic]
    if key == "noise_robustness":
        return list(cfg.noise_robustness_periods)
    if key == "covariate_robustness":
        return list(cfg.covariate_robustness_periods)
    if length >= cfg.context_length_period_threshold:
        return list(cfg.default_long_periods)
    return list(cfg.default_short_periods)


def infer_trend_type(series: list[float], settings: AppSettings | None = None) -> str:
    cfg = (settings or get_settings()).data_inference
    diffs = [series[index] - series[index - 1] for index in range(1, len(series))]
    return "linear" if abs(sum(diffs)) > len(series) * cfg.trend_linear_threshold_per_step else "smooth_curve"


def infer_difficulty(series: list[float], settings: AppSettings | None = None) -> str:
    cfg = (settings or get_settings()).data_inference
    amplitude = max(series) - min(series)
    if amplitude > cfg.difficulty_hard_amplitude:
        return "hard"
    if amplitude > cfg.difficulty_medium_amplitude:
        return "medium"
    return "easy"


def future_known_covariates(names: list[str], settings: AppSettings | None = None) -> list[str]:
    cfg = (settings or get_settings()).data_inference
    return [
        name
        for name in names
        if name in cfg.future_known_covariates_exact
        or any(name.startswith(prefix) for prefix in cfg.future_known_covariates_prefixes)
    ]


def is_future_known_covariate(name: str, settings: AppSettings | None = None) -> bool:
    return name in set(future_known_covariates([name], settings=settings))


def track_value(track: object) -> str:
    return str(getattr(track, "value", track))


def lookup_key(data: Any, dotted_key: str) -> Any:
    current = data
    for part in dotted_key.split("."):
        if isinstance(current, BaseModel):
            current = getattr(current, part)
        elif isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "get":
        print("usage: python -m backend.app.config get <dotted.key> [conf_path]", file=sys.stderr)
        return 1

    key = argv[2]
    conf_path = argv[3] if len(argv) > 3 else None
    value = lookup_key(load_settings(conf_path), key)
    if isinstance(value, bool):
        print("true" if value else "false")
    elif isinstance(value, (list, dict, BaseModel)):
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        import json

        print(json.dumps(value, ensure_ascii=False))
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
