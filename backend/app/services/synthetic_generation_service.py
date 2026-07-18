from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlmodel import Session

from app.core.errors import ApiError
from app.core.ids import new_id
from app.core.time import utc_now
from app.models.dataset import DatasetManifest, Shard
from app.services.dataset_load_service import SampleWindow
from app.services.dataset_reader import DatasetReadResult
from app.services.sample_store import SampleStore
from app.services.series_store import SeriesStore
from app.services.synthetic_capability_contrast import (
    evaluate_capability_contrast,
)
from app.services.synthetic_feature_gate import (
    evaluate_feature_support_gate,
    matching_calibrated_buckets as matching_feature_gate_buckets,
)
from app.services.synthetic_generator_conditioning import (
    GeneratorConditioning,
    matching_generator_profiles,
    resolve_generator_conditioning,
    select_balanced_profile_id,
)
from app.services.synthetic_near_distance_gate import evaluate_near_distance_gate, matching_calibrated_buckets


@dataclass(frozen=True)
class SyntheticCapability:
    capability_id: str
    label: str
    description: str
    label_zh: str
    description_zh: str
    task_type: str
    target_dim_mode: str
    covariate_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    name: str
    capabilities: list[str]
    context_length: int
    horizon: int
    sample_count: int
    intensity: int
    season_length: int
    target_dim: int
    seed: int
    frequency: str = "h"
    anchor_profile_ids: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SeasonalityResolution:
    season_length: int
    source: str
    profile_ids: tuple[str, ...]
    candidate_periods: tuple[int, ...]
    requested_frequency: str


SYNTHETIC_CAPABILITIES: tuple[SyntheticCapability, ...] = (
    SyntheticCapability(
        "trend",
        "Trend",
        "Single-target series with controllable trend over a non-periodic persistent background.",
        "趋势",
        "在非周期持续性背景上带有可控趋势的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "multi_seasonal",
        "Multi-seasonal",
        "Single-target series with multiple overlapping seasonal periods.",
        "多季节性",
        "带有多重叠加季节周期的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "regime_switching",
        "Predictable regime switching",
        "Single-target series with a recurring, history-observable regime schedule.",
        "可预测状态切换",
        "带有可从历史识别的重复状态切换规律的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "time_varying_seasonality",
        "Time-varying seasonality",
        "Single-target series with smoothly modulated seasonal amplitude and phase.",
        "时变季节性",
        "季节振幅和相位按平滑规律变化的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "nonlinear_persistence",
        "Nonlinear multi-lag persistence",
        "Single-target stable dynamics with short, seasonal, and nonlinear lag dependence.",
        "非线性多滞后持久性",
        "同时依赖短滞后、季节滞后和非线性滞后项的稳定单目标动态。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "predictable_intermittency",
        "Predictable intermittency",
        "Single-target sparse pulses with a history-observable non-uniform interval motif.",
        "可预测间歇性",
        "带有可从历史识别的非等间隔时钟和稀疏脉冲的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "common_factor",
        "Common factor",
        "Multiple targets driven by a shared non-periodic dynamic factor.",
        "公共因子",
        "由共享非周期动态因子驱动的多目标序列。",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "hierarchical_coherence",
        "Hierarchical coherence",
        "Multiple targets with an exact parent-child additive structure.",
        "层级一致性",
        "带有父子加总结构的多目标序列。",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "covariate_response",
        "Known-future covariate response",
        "Targets whose future depends on known weather and event covariates.",
        "协变量响应",
        "未来走势依赖已知天气和事件协变量的目标序列。",
        "covariate_forecast",
        "covariate",
        ("weather", "event"),
    ),
)

CAPABILITIES_BY_ID: dict[str, SyntheticCapability] = {
    capability.capability_id: capability for capability in SYNTHETIC_CAPABILITIES
}

PAPER_GENERATOR_VERSION = "capts-paper-v2"
PAPER_UNIVARIATE_CAPABILITY_IDS: tuple[str, ...] = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)
PAPER_STRUCTURED_CAPABILITY_IDS: tuple[str, ...] = (
    "common_factor",
    "hierarchical_coherence",
    "covariate_response",
)
PAPER_CAPABILITY_IDS = PAPER_UNIVARIATE_CAPABILITY_IDS + PAPER_STRUCTURED_CAPABILITY_IDS

PREDICTABILITY_CONTRACTS: dict[str, str] = {
    "trend": "one trend law is observed in context and continued through the forecast horizon",
    "multi_seasonal": "every forecast-period component completes at least two cycles in context",
    "time_varying_seasonality": "amplitude and phase follow a smooth modulation observed in context",
    "regime_switching": "a regular alternating regime schedule has at least two historical and one future switch",
    "nonlinear_persistence": "all recurrence lags are observed and the recurrence is coefficient-stable",
    "predictable_intermittency": "every transition in a repeating non-uniform interval motif is exposed in history and at least one pulse occurs in the future",
    "common_factor": "a non-periodic shared dynamic factor and fixed channel loadings continue across the forecast boundary",
    "hierarchical_coherence": "bottom-level component laws continue and the parent is their exact sum",
    "covariate_response": "future exogenous values are supplied and include an event after historical effect examples",
}

INTENSITY_FEATURE_DIRECTIONS: dict[str, dict[str, str]] = {
    "trend": {
        "trend_strength": "increase",
        "slope_abs": "increase",
    },
    "multi_seasonal": {"multi_period_score": "increase"},
    "time_varying_seasonality": {
        "seasonal_amplitude_modulation": "increase",
        "seasonal_phase_variation": "increase",
    },
    "regime_switching": {
        "regime_clock_history_incremental_r2": "increase",
        "change_point_shift_energy": "increase",
        "level_shift_strength": "increase",
    },
    "nonlinear_persistence": {
        "nonlinear_multi_lag_gain": "increase",
        "nonlinear_conditional_gain": "increase",
    },
    "predictable_intermittency": {
        "burst_rate": "increase",
        "spike_rate": "increase",
        "outlier_rate": "increase",
    },
    "common_factor": {
        "pca_top1_explained": "increase",
        "effective_factor_rank": "decrease",
        "avg_abs_target_corr": "increase",
    },
    "hierarchical_coherence": {"hierarchy_child_heterogeneity": "increase"},
    "covariate_response": {
        "covariate_incremental_r2": "increase",
        "future_abs_covariate_target_corr": "increase",
        "event_lift_abs": "increase",
    },
}

MOCK_ANCHOR = {
    "anchor_mode": "preselected_profile_conditioned",
    "anchor_source_uri": "synthetic-anchor://public/profile-conditioned-v2",
}

ANCHOR_FEATURE_QUANTILES: dict[str, dict[str, dict[str, float]]] = {
    "m4_hourly_daily_168ctx": {
        "trend_strength": {"p05": 0.0000, "p50": 0.1659, "p95": 0.7714},
        "seasonal_strength": {"p05": 0.5768, "p50": 0.9129, "p95": 0.9961},
        "acf_abs_mean": {"p05": 0.2515, "p50": 0.5201, "p95": 0.5574},
        "slope_abs": {"p05": 0.0136, "p50": 0.1264, "p95": 0.3543},
        "curvature_abs": {"p05": 0.0042, "p50": 0.0711, "p95": 0.6756},
        "noise_ratio": {"p05": 0.0038, "p50": 0.0821, "p95": 0.3871},
        "spike_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.1257},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0260},
        "multi_period_score": {"p05": 0.0084, "p50": 0.0415, "p95": 0.1769},
        "seasonal_drift_score": {"p05": 0.0058, "p50": 0.0761, "p95": 0.4297},
        "seasonal_amplitude_cv": {"p05": 0.4202, "p50": 0.4860, "p95": 0.7334},
        "change_point_shift_energy": {"p05": 0.3035, "p50": 0.5072, "p95": 0.9798},
        "level_shift_strength": {"p05": 0.3115, "p50": 0.5132, "p95": 0.9885},
        "volatility_shift_strength": {"p05": 0.0662, "p50": 0.1619, "p95": 0.6469},
        "nonlinear_lag1_gain": {"p05": 0.0000, "p50": 0.0003, "p95": 0.0538},
        "burst_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0104},
    },
    "us_births_weekly": {
        "trend_strength": {"p05": 0.0109, "p50": 0.2459, "p95": 0.2938},
        "seasonal_strength": {"p05": 0.6238, "p50": 0.7040, "p95": 0.8309},
        "acf_abs_mean": {"p05": 0.2318, "p50": 0.2674, "p95": 0.3569},
        "slope_abs": {"p05": 0.0500, "p50": 0.2102, "p95": 0.3606},
        "curvature_abs": {"p05": 0.1165, "p50": 0.3674, "p95": 0.5142},
        "noise_ratio": {"p05": 0.1609, "p50": 0.2693, "p95": 0.3567},
        "spike_rate": {"p05": 0.0000, "p50": 0.0038, "p95": 0.0716},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0000},
        "multi_period_score": {"p05": 0.1856, "p50": 0.2036, "p95": 0.2313},
        "seasonal_drift_score": {"p05": 0.3837, "p50": 0.4704, "p95": 0.5572},
        "seasonal_amplitude_cv": {"p05": 0.5126, "p50": 0.5295, "p95": 0.6719},
        "change_point_shift_energy": {"p05": 0.4210, "p50": 0.6002, "p95": 0.8802},
        "level_shift_strength": {"p05": 0.4304, "p50": 0.6061, "p95": 0.8829},
        "volatility_shift_strength": {"p05": 0.1131, "p50": 0.2028, "p95": 0.3690},
        "nonlinear_lag1_gain": {"p05": 0.0225, "p50": 0.0405, "p95": 0.0766},
        "burst_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0000},
    },
    "electricity_hourly_daily_168ctx": {
        "trend_strength": {"p05": 0.0000, "p50": 0.0427, "p95": 0.4120},
        "seasonal_strength": {"p05": 0.3058, "p50": 0.9161, "p95": 0.9783},
        "acf_abs_mean": {"p05": 0.2386, "p50": 0.4420, "p95": 0.5151},
        "slope_abs": {"p05": 0.0109, "p50": 0.0946, "p95": 0.3537},
        "curvature_abs": {"p05": 0.0085, "p50": 0.1117, "p95": 0.7900},
        "noise_ratio": {"p05": 0.0212, "p50": 0.0811, "p95": 0.5517},
        "spike_rate": {"p05": 0.0000, "p50": 0.0419, "p95": 0.1728},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0990},
        "multi_period_score": {"p05": 0.0328, "p50": 0.1297, "p95": 0.3989},
        "seasonal_drift_score": {"p05": 0.0309, "p50": 0.0946, "p95": 0.3236},
        "seasonal_amplitude_cv": {"p05": 0.3123, "p50": 0.4962, "p95": 0.8362},
        "change_point_shift_energy": {"p05": 0.2651, "p50": 0.4609, "p95": 1.0847},
        "level_shift_strength": {"p05": 0.2711, "p50": 0.4720, "p95": 1.0920},
        "volatility_shift_strength": {"p05": 0.1000, "p50": 0.2160, "p95": 0.6584},
        "nonlinear_lag1_gain": {"p05": 0.0001, "p50": 0.0028, "p95": 0.0290},
        "burst_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0052},
    },
    "electricity_hourly_panel_168ctx": {
        "trend_strength": {"p05": 0.0000, "p50": 0.0761, "p95": 0.3433},
        "seasonal_strength": {"p05": 0.4506, "p50": 0.9103, "p95": 0.9701},
        "acf_abs_mean": {"p05": 0.2892, "p50": 0.4422, "p95": 0.4977},
        "slope_abs": {"p05": 0.0298, "p50": 0.1041, "p95": 0.3491},
        "curvature_abs": {"p05": 0.0299, "p50": 0.1261, "p95": 0.7159},
        "noise_ratio": {"p05": 0.0294, "p50": 0.0835, "p95": 0.4616},
        "spike_rate": {"p05": 0.0017, "p50": 0.0489, "p95": 0.1501},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0903},
        "multi_period_score": {"p05": 0.0343, "p50": 0.1316, "p95": 0.4245},
        "seasonal_drift_score": {"p05": 0.0259, "p50": 0.0782, "p95": 0.2428},
        "seasonal_amplitude_cv": {"p05": 0.3319, "p50": 0.4859, "p95": 0.8413},
        "change_point_shift_energy": {"p05": 0.2713, "p50": 0.4507, "p95": 1.0756},
        "level_shift_strength": {"p05": 0.2784, "p50": 0.4591, "p95": 1.0842},
        "volatility_shift_strength": {"p05": 0.1008, "p50": 0.1960, "p95": 0.5319},
        "nonlinear_lag1_gain": {"p05": 0.0001, "p50": 0.0017, "p95": 0.0195},
        "burst_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0000},
        "avg_abs_target_corr": {"p05": 0.2903, "p50": 0.8478, "p95": 0.9484},
        "pca_top1_explained": {"p05": 0.6727, "p50": 0.9627, "p95": 0.9988},
        "pca_top2_explained": {"p05": 0.9512, "p50": 0.9954, "p95": 1.0},
        "effective_factor_rank": {"p05": 1.0097, "p50": 1.1915, "p95": 2.0619},
        "lead_lag_peak_abs": {"p05": 0.5567, "p50": 0.8952, "p95": 0.9538},
    },
    "traffic_hourly_daily_168ctx": {
        "trend_strength": {"p05": 0.0052, "p50": 0.0816, "p95": 0.2665},
        "seasonal_strength": {"p05": 0.4688, "p50": 0.7156, "p95": 0.9024},
        "acf_abs_mean": {"p05": 0.1957, "p50": 0.3335, "p95": 0.4807},
        "slope_abs": {"p05": 0.0097, "p50": 0.1010, "p95": 0.4206},
        "curvature_abs": {"p05": 0.0231, "p50": 0.2742, "p95": 1.0373},
        "noise_ratio": {"p05": 0.0964, "p50": 0.2754, "p95": 0.5009},
        "spike_rate": {"p05": 0.0000, "p50": 0.0681, "p95": 0.1571},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0052, "p95": 0.0938},
        "multi_period_score": {"p05": 0.0322, "p50": 0.0979, "p95": 0.1940},
        "seasonal_drift_score": {"p05": 0.0736, "p50": 0.2189, "p95": 0.7112},
        "seasonal_amplitude_cv": {"p05": 0.4436, "p50": 0.6379, "p95": 1.0029},
        "change_point_shift_energy": {"p05": 0.2848, "p50": 0.4681, "p95": 0.8373},
        "level_shift_strength": {"p05": 0.2908, "p50": 0.4738, "p95": 0.8434},
        "volatility_shift_strength": {"p05": 0.1376, "p50": 0.4322, "p95": 0.9407},
        "nonlinear_lag1_gain": {"p05": 0.0011, "p50": 0.0210, "p95": 0.1148},
        "burst_rate": {"p05": 0.0000, "p50": 0.0052, "p95": 0.0938},
    },
    "traffic_hourly_panel_168ctx": {
        "trend_strength": {"p05": 0.0252, "p50": 0.0851, "p95": 0.2136},
        "seasonal_strength": {"p05": 0.5522, "p50": 0.7111, "p95": 0.8366},
        "acf_abs_mean": {"p05": 0.2469, "p50": 0.3340, "p95": 0.4264},
        "slope_abs": {"p05": 0.0306, "p50": 0.1208, "p95": 0.3277},
        "curvature_abs": {"p05": 0.0802, "p50": 0.3175, "p95": 0.8355},
        "noise_ratio": {"p05": 0.1600, "p50": 0.2794, "p95": 0.4233},
        "spike_rate": {"p05": 0.0227, "p50": 0.0716, "p95": 0.1239},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0191, "p95": 0.0625},
        "multi_period_score": {"p05": 0.0452, "p50": 0.1043, "p95": 0.1891},
        "seasonal_drift_score": {"p05": 0.0765, "p50": 0.1993, "p95": 0.4495},
        "seasonal_amplitude_cv": {"p05": 0.4636, "p50": 0.5863, "p95": 0.8228},
        "change_point_shift_energy": {"p05": 0.3040, "p50": 0.4888, "p95": 0.7730},
        "level_shift_strength": {"p05": 0.3110, "p50": 0.4977, "p95": 0.7848},
        "volatility_shift_strength": {"p05": 0.1631, "p50": 0.3554, "p95": 0.6623},
        "nonlinear_lag1_gain": {"p05": 0.0027, "p50": 0.0157, "p95": 0.0534},
        "burst_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0312},
        "avg_abs_target_corr": {"p05": 0.3484, "p50": 0.6290, "p95": 0.8637},
        "pca_top1_explained": {"p05": 0.6206, "p50": 0.8179, "p95": 0.9450},
        "pca_top2_explained": {"p05": 0.9148, "p50": 0.9718, "p95": 0.9959},
        "effective_factor_rank": {"p05": 1.2655, "p50": 1.7376, "p95": 2.2662},
        "lead_lag_peak_abs": {"p05": 0.6158, "p50": 0.7959, "p95": 0.9208},
    },
    "m5_daily_covariate_365ctx_28h": {
        "trend_strength": {"p05": 0.0000, "p50": 0.0198, "p95": 0.2457},
        "seasonal_strength": {"p05": 0.0000, "p50": 0.0172, "p95": 0.0739},
        "noise_ratio": {"p05": 0.0000, "p50": 0.9245, "p95": 0.9841},
        "spike_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0435},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0280},
        "burst_rate": {"p05": 0.0000, "p50": 0.0127, "p95": 0.0636},
        "avg_abs_covariate_target_corr": {"p05": 0.0000, "p50": 0.0239, "p95": 0.1143},
        "future_abs_covariate_target_corr": {"p05": 0.0000, "p50": 0.0568, "p95": 0.1652},
        "event_lift_abs": {"p05": 0.0000, "p50": 0.0869, "p95": 1.1189},
    },
    "m5_daily_hierarchy_365ctx_28h": {
        "trend_strength": {"p05": 0.0190, "p50": 0.1364, "p95": 0.3821},
        "seasonal_strength": {"p05": 0.0914, "p50": 0.3339, "p95": 0.7295},
        "noise_ratio": {"p05": 0.2407, "p50": 0.5802, "p95": 0.8422},
        "spike_rate": {"p05": 0.0009, "p50": 0.0043, "p95": 0.0119},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0042, "p95": 0.0110},
        "burst_rate": {"p05": 0.0000, "p50": 0.0025, "p95": 0.0076},
        "hierarchy_residual_mean_abs": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0000},
        "avg_abs_target_corr": {"p05": 0.3933, "p50": 0.6495, "p95": 0.9062},
        "pca_top1_explained": {"p05": 0.9399, "p50": 0.9849, "p95": 0.9945},
        "effective_factor_rank": {"p05": 1.0347, "p50": 1.0813, "p95": 1.2551},
    },
    "gefcom2014_load_hourly_covariate_168ctx_24h": {
        "trend_strength": {"p05": 0.0000, "p50": 0.3329, "p95": 0.7326},
        "seasonal_strength": {"p05": 0.3133, "p50": 0.6767, "p95": 0.9476},
        "noise_ratio": {"p05": 0.0482, "p50": 0.2460, "p95": 0.5727},
        "spike_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0105},
        "outlier_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0104},
        "burst_rate": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0052},
        "avg_abs_covariate_target_corr": {"p05": 0.2304, "p50": 0.7441, "p95": 0.8598},
        "future_abs_covariate_target_corr": {"p05": 0.1734, "p50": 0.7103, "p95": 0.9030},
        "event_lift_abs": {"p05": 0.0000, "p50": 0.0000, "p95": 0.0000},
    },
}

ANCHOR_PROFILE_BUCKETS: dict[str, dict[str, Any]] = {
    "m4_hourly_daily_96ctx": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 2000,
    },
    "m4_hourly_daily_168ctx": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 2000,
    },
    "m4_hourly_weekly": {
        "frequency": "h",
        "season_length": 168,
        "significant_periods": (168,),
        "window_count": 1000,
    },
    "electricity_hourly_daily_168ctx": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 2000,
    },
    "electricity_hourly_daily_2048ctx_24h": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 600,
    },
    "electricity_hourly_panel_168ctx": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 2000,
    },
    "traffic_hourly_daily_168ctx": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 2000,
    },
    "traffic_hourly_panel_168ctx": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 2000,
    },
    "m5_daily_covariate_365ctx_28h": {
        "frequency": "d",
        "season_length": 7,
        "significant_periods": (7,),
        "window_count": 2000,
    },
    "m5_daily_hierarchy_365ctx_28h": {
        "frequency": "d",
        "season_length": 7,
        "significant_periods": (7,),
        "window_count": 1000,
    },
    "gefcom2014_load_hourly_covariate_168ctx_24h": {
        "frequency": "h",
        "season_length": 24,
        "significant_periods": (24,),
        "window_count": 2000,
    },
    "us_births_weekly": {
        "frequency": "d",
        "season_length": 7,
        "significant_periods": (7,),
        "window_count": 20,
    },
    "us_births_annual_diagnostic": {
        "frequency": "d",
        "season_length": 365,
        "significant_periods": (365,),
        "window_count": 20,
    },
}

TARGET_FEATURES_BY_CAPABILITY: dict[str, tuple[str, ...]] = {
    "trend": ("trend_strength", "slope_abs", "curvature_abs"),
    "multi_seasonal": ("multi_period_score",),
    "time_varying_seasonality": (
        "seasonal_amplitude_modulation",
        "seasonal_phase_variation",
    ),
    "regime_switching": (
        "regime_clock_history_incremental_r2",
        "change_point_shift_energy",
        "level_shift_strength",
    ),
    "nonlinear_persistence": (
        "nonlinear_multi_lag_gain",
        "nonlinear_conditional_gain",
    ),
    "predictable_intermittency": ("burst_rate", "spike_rate", "outlier_rate"),
    "common_factor": ("pca_top1_explained", "effective_factor_rank", "avg_abs_target_corr"),
    "hierarchical_coherence": ("hierarchy_child_heterogeneity",),
    "covariate_response": (
        "covariate_incremental_r2",
        "future_abs_covariate_target_corr",
        "event_lift_abs",
    ),
}

CONTROL_FEATURES_BY_CAPABILITY: dict[str, tuple[str, ...]] = {
    # Fixed-period seasonal strength and decomposition noise are not valid
    # nuisance controls for an isolated trend probe: requiring either one
    # would force a periodic carrier back into every trend sample.  Tail
    # behavior remains controlled online; broader fidelity is audited offline.
    "trend": ("outlier_rate", "spike_rate"),
    # A second periodic component is residualized as noise by the single-period
    # profile extractor, making noise_ratio target-coupled for this capability.
    "multi_seasonal": ("trend_strength", "outlier_rate", "spike_rate"),
    # Smooth amplitude/phase modulation is mechanically scored as residual
    # ``noise_ratio`` by the profile extractor, so noise_ratio is target-coupled
    # here and cannot serve as a nuisance control.
    "time_varying_seasonality": ("trend_strength", "outlier_rate", "spike_rate"),
    # Recurring level switches mechanically create or suppress all residual
    # spike/outlier summaries after context standardization.  There is no
    # independent observable nuisance among the current feature family, so the
    # feature-support artifact records an explicit no-control contract.  Real
    # parameter support, construction predictability, canonical dose, and the
    # near-distance gate remain mandatory.
    "regime_switching": (),
    # Recurrence strength mechanically changes the variance attributed to the
    # seasonal and residual components, so neither is a valid nuisance control.
    "nonlinear_persistence": ("trend_strength", "outlier_rate", "spike_rate"),
    # A recurring pulse clock is itself recovered as seasonal signal and its
    # phase/period also projects onto the trend smoother.  It therefore changes
    # every available decomposition, spike, and outlier summary.  The feature
    # artifact records the same explicit no-control contract as regime
    # switching; train-conditioned nuisance parameters, construction, absolute
    # spike dose, and near-distance remain mandatory.
    "predictable_intermittency": (),
    # Signal-to-noise ratios are mechanically changed by factor/effect strength,
    # so they are target-coupled rather than valid nuisance controls here.
    "common_factor": ("trend_strength", "outlier_rate", "spike_rate"),
    "hierarchical_coherence": ("hierarchy_residual_mean_abs", "outlier_rate", "spike_rate"),
    "covariate_response": (
        "covariate_residual_acf_abs_mean",
        "covariate_residual_outlier_rate",
        "covariate_residual_spike_rate",
    ),
}


def synthetic_capability_catalog() -> list[dict[str, Any]]:
    return [
        {
            "capability_id": capability.capability_id,
            "label": capability.label,
            "description": capability.description,
            "label_i18n": {"en-US": capability.label, "zh-CN": capability.label_zh},
            "description_i18n": {"en-US": capability.description, "zh-CN": capability.description_zh},
            "task_type": capability.task_type,
            "target_dim_mode": capability.target_dim_mode,
            "covariate_columns": list(capability.covariate_columns),
            "paper_included": True,
            "paper_track": (
                "univariate"
                if capability.capability_id in PAPER_UNIVARIATE_CAPABILITY_IDS
                else "structured"
            ),
            "generator_version": PAPER_GENERATOR_VERSION,
            "predictability_contract": PREDICTABILITY_CONTRACTS[capability.capability_id],
            "intensity_features": INTENSITY_FEATURE_DIRECTIONS[capability.capability_id],
            "default_params": _default_params_for_capability(capability),
            "limits": {
                "context_length": {"min": 16, "max": 2048},
                "horizon": {"min": 1, "max": 512},
                "sample_count": {"min": 1, "max": 1000},
                "intensity": {"min": 1, "max": 5},
                "target_dim": {"min": 1, "max": 16},
            },
        }
        for capability in SYNTHETIC_CAPABILITIES
    ]


def _default_params_for_capability(capability: SyntheticCapability) -> dict[str, int | str]:
    context_length = 365 if capability.capability_id == "hierarchical_coherence" else 168
    horizon = 28 if capability.capability_id == "hierarchical_coherence" else 24
    frequency = "d" if capability.capability_id == "hierarchical_coherence" else "h"
    return {
        "context_length": context_length,
        "horizon": horizon,
        "sample_count": 32,
        "intensity": 3,
        "target_dim": 3 if capability.target_dim_mode == "multi" else 1,
        "frequency": frequency,
    }


def generate_synthetic_shards(
    session: Session,
    runtime_dir: Path,
    config: SyntheticGenerationConfig,
) -> list[Shard]:
    _validate_config(config)
    capabilities = _capabilities(config.capabilities)
    generation_id = new_id()
    source_uri = f"synthetic://{generation_id}"
    storage_dir = Path(runtime_dir) / "synthetic"
    storage_dir.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest(
        name=config.name,
        domain="synthetic",
        source_type="synthetic",
        source_uri=source_uri,
        file_format="synthetic",
        time_column="time",
        frequency=config.frequency,
        status="loaded",
    )
    session.add(manifest)
    session.flush()

    shards: list[Shard] = []
    try:
        for capability in capabilities:
            shard = _generate_capability_shard(session, storage_dir, manifest, source_uri, capability, config, generation_id)
            shards.append(shard)
        manifest.updated_at = utc_now()
        session.add(manifest)
        session.commit()
        for shard in shards:
            session.refresh(shard)
        return shards
    except Exception:
        session.rollback()
        raise


def _validate_config(config: SyntheticGenerationConfig) -> None:
    if not config.capabilities:
        raise ApiError("synthetic_capability_required", "at least one synthetic capability must be selected")
    if len(config.capabilities) != len(set(config.capabilities)):
        raise ApiError("synthetic_capability_duplicate", "synthetic capabilities must be distinct")
    positive_fields = {
        "context_length": config.context_length,
        "horizon": config.horizon,
        "sample_count": config.sample_count,
        "target_dim": config.target_dim,
    }
    invalid = {name: value for name, value in positive_fields.items() if int(value) <= 0}
    if invalid:
        raise ApiError("synthetic_config_invalid", "synthetic generation parameters must be positive", invalid)
    if config.context_length < 16:
        raise ApiError("synthetic_context_too_short", "context_length must be at least 16")
    if config.context_length > 2048 or config.horizon > 512 or config.sample_count > 1000 or config.target_dim > 16:
        raise ApiError(
            "synthetic_config_too_large",
            "synthetic generation request exceeds MVP limits",
            {
                "context_length_max": 2048,
                "horizon_max": 512,
                "sample_count_max": 1000,
                "target_dim_max": 16,
            },
        )
    if not 1 <= config.intensity <= 5:
        raise ApiError("synthetic_intensity_invalid", "intensity must be between 1 and 5")
    unknown_anchor_capabilities = sorted(set(config.anchor_profile_ids) - set(config.capabilities))
    if unknown_anchor_capabilities:
        raise ApiError(
            "synthetic_anchor_capability_invalid",
            "anchor_profile_ids may only contain selected capabilities",
            {"capability_ids": unknown_anchor_capabilities},
        )
    empty_anchor_profiles = sorted(
        capability_id
        for capability_id, profile_id in config.anchor_profile_ids.items()
        if not str(profile_id).strip()
    )
    if empty_anchor_profiles:
        raise ApiError(
            "synthetic_anchor_profile_invalid",
            "anchor profile ids must be non-empty",
            {"capability_ids": empty_anchor_profiles},
        )


def _capabilities(capability_ids: list[str]) -> list[SyntheticCapability]:
    missing = [capability_id for capability_id in capability_ids if capability_id not in CAPABILITIES_BY_ID]
    if missing:
        raise ApiError("synthetic_capability_unknown", "unknown synthetic capability", {"capability_ids": missing}, 404)
    return [CAPABILITIES_BY_ID[capability_id] for capability_id in capability_ids]


def _generate_capability_shard(
    session: Session,
    storage_dir: Path,
    manifest: DatasetManifest,
    source_uri: str,
    capability: SyntheticCapability,
    config: SyntheticGenerationConfig,
    generation_id: str,
) -> Shard:
    context = int(config.context_length)
    horizon = int(config.horizon)
    sample_length = context + horizon
    target_dim = _target_dim_for_capability(capability, config.target_dim)
    near_distance_buckets = _require_near_distance_calibration(capability, context, horizon, target_dim)
    feature_gate_buckets = _require_feature_gate_calibration(capability, context, horizon, target_dim)
    generator_profiles = _require_generator_conditioning_calibration(
        capability,
        context,
        horizon,
        target_dim,
        config.frequency,
    )
    anchor_profile_candidates = tuple(
        profile_id
        for profile_id in generator_profiles
        if profile_id in set(near_distance_buckets) and profile_id in set(feature_gate_buckets)
    )
    if not anchor_profile_candidates:
        raise ApiError(
            "synthetic_anchor_profile_not_calibrated",
            "no real profile has generator, feature-support, and near-distance calibration for this request",
            {
                "capability_id": capability.capability_id,
                "generator_profiles": generator_profiles,
                "feature_gate_profiles": feature_gate_buckets,
                "near_distance_profiles": near_distance_buckets,
            },
        )
    requested_anchor_profile = config.anchor_profile_ids.get(capability.capability_id)
    if requested_anchor_profile is not None and requested_anchor_profile not in anchor_profile_candidates:
        raise ApiError(
            "synthetic_anchor_profile_invalid",
            "requested anchor profile is not calibrated for this capability and window",
            {
                "capability_id": capability.capability_id,
                "profile_id": requested_anchor_profile,
                "available_profile_ids": list(anchor_profile_candidates),
            },
        )
    selected_profile_pool = (
        (requested_anchor_profile,) if requested_anchor_profile is not None else anchor_profile_candidates
    )
    target_columns = [f"target_{index}" for index in range(target_dim)]
    covariate_columns = list(capability.covariate_columns)
    columns = [*target_columns, *covariate_columns]
    base_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    delta = _frequency_delta(config.frequency)
    profile_conditionings = {
        profile_id: _resolve_required_generator_conditioning(
            capability.capability_id,
            profile_id,
            context,
            horizon,
            target_dim,
        )
        for profile_id in selected_profile_pool
    }
    season_lengths = tuple(
        sorted({conditioning.season_length for conditioning in profile_conditionings.values()})
    )

    all_timestamps: list[datetime] = []
    all_values: list[list[float]] = []
    windows: list[SampleWindow] = []
    sample_metadata: list[dict[str, Any]] = []
    for sample_index in range(config.sample_count):
        sample_seed = _seed_for(config.seed, capability.capability_id, sample_index)
        anchor_profile_id = select_balanced_profile_id(
            selected_profile_pool,
            capability_id=capability.capability_id,
            seed=config.seed,
            sample_index=sample_index,
        )
        conditioning = profile_conditionings[anchor_profile_id]
        target, latent_params, covariates, realized_features = _generate_accepted_sample_values(
            capability.capability_id,
            sample_length,
            context,
            target_dim,
            conditioning.season_length,
            config.intensity,
            sample_seed,
            anchor_profile_id=anchor_profile_id,
            generator_conditioning=conditioning,
        )
        if covariates is not None and covariates.size:
            values = np.concatenate([target, covariates], axis=1)
        else:
            values = target

        row_start = sample_index * sample_length
        row_end = row_start + sample_length - 1
        windows.append(
            SampleWindow(
                source_row_start=row_start,
                source_row_end=row_end,
                context_start=row_start,
                context_end=row_start + context - 1,
                horizon_start=row_start + context,
                horizon_end=row_end,
                context_length=context,
                horizon=horizon,
            )
        )
        all_timestamps.extend(base_start + delta * row for row in range(row_start, row_end + 1))
        all_values.extend(values.astype(float).tolist())
        sample_metadata.append(
            {
                "schema_version": "synthetic_sample_metadata.v1",
                "generator_version": PAPER_GENERATOR_VERSION,
                "capability_id": capability.capability_id,
                "capability_label": capability.label,
                "intensity": config.intensity,
                "difficulty": config.intensity,
                "seed": config.seed,
                "sample_seed": sample_seed,
                "anchor_profile_id": anchor_profile_id,
                "anchor_profiles": list(selected_profile_pool),
                "anchor_profile_selection": (
                    "explicit" if requested_anchor_profile is not None else "balanced_uniform"
                ),
                "canonical_scale_id": conditioning.canonical_scale_id,
                "canonical_scale_fingerprint": conditioning.canonical_scale_fingerprint,
                "canonical_target_feature": conditioning.canonical_target_feature,
                "canonical_target_strength": conditioning.canonical_target_values[
                    config.intensity - 1
                ],
                "local_real_percentile": conditioning.local_real_percentiles[
                    config.intensity - 1
                ],
                "season_length": conditioning.season_length,
                "requested_season_length": config.season_length,
                "season_length_source": "preselected_anchor_profile",
                "season_length_candidates": list(season_lengths),
                "season_length_profiles": list(selected_profile_pool),
                "latent_params": latent_params,
                "realized_features": realized_features,
                **MOCK_ANCHOR,
            }
        )

    canonical_conditioning = next(iter(profile_conditionings.values()))
    generation_config = {
        "schema_version": "synthetic_generation.v1",
        "generator_version": PAPER_GENERATOR_VERSION,
        "generation_id": generation_id,
        "capability_id": capability.capability_id,
        "capability_label": capability.label,
        "task_type": capability.task_type,
        "intensity": config.intensity,
        "difficulty": config.intensity,
        "intensity_definition": (
            "capability-global canonical realized strength; not a required monotonic model-error difficulty"
        ),
        "canonical_scale_id": canonical_conditioning.canonical_scale_id,
        "canonical_scale_fingerprint": canonical_conditioning.canonical_scale_fingerprint,
        "canonical_target_feature": canonical_conditioning.canonical_target_feature,
        "canonical_target_strength": canonical_conditioning.canonical_target_values[
            config.intensity - 1
        ],
        "seed": config.seed,
        "context_length": context,
        "horizon": horizon,
        "sample_count": config.sample_count,
        "season_length": season_lengths[0] if len(season_lengths) == 1 else None,
        "requested_season_length": config.season_length,
        "season_length_source": "preselected_anchor_profile",
        "season_length_candidates": list(season_lengths),
        "season_length_profiles": list(selected_profile_pool),
        "target_dim": target_dim,
        "requested_target_dim": config.target_dim,
        "covariate_columns": covariate_columns,
        "near_distance_calibration_buckets": near_distance_buckets,
        "feature_gate_calibration_buckets": feature_gate_buckets,
        "generator_conditioning_profiles": list(selected_profile_pool),
        "anchor_profiles": list(selected_profile_pool),
        "anchor_profile_selection": (
            "explicit" if requested_anchor_profile is not None else "balanced_uniform"
        ),
        "frequency": config.frequency,
        **MOCK_ANCHOR,
    }
    shard = Shard(
        name=_shard_name(config.name, capability, len(config.capabilities) > 1),
        shard_type="synthetic",
        capability_type=capability.capability_id,
        dataset_manifest_id=manifest.dataset_manifest_id,
        source_uri=source_uri,
        storage_uri=str(storage_dir / f"{generation_id}-{capability.capability_id}.json"),
        time_range_start=all_timestamps[0].isoformat(),
        time_range_end=all_timestamps[-1].isoformat(),
        row_count=len(all_values),
        target_columns=target_columns,
        target_dim=target_dim,
        covariate_columns=covariate_columns,
        covariate_dim=len(covariate_columns),
        frequency=config.frequency,
        context_length=context,
        horizon=horizon,
        stride=horizon,
        sample_count=config.sample_count,
        generation_config=generation_config,
        status="ready",
    )
    session.add(shard)
    session.flush()

    read_result = DatasetReadResult(
        columns=["time", *columns],
        rows=[{"time": timestamp.isoformat()} for timestamp in all_timestamps],
        timestamps=all_timestamps,
        target_columns=target_columns,
        covariate_columns=covariate_columns,
        values=all_values,
        frequency=config.frequency,
        encoding="synthetic",
        delimiter=",",
    )
    SeriesStore().write(session, shard.shard_id, all_timestamps, columns, all_values)
    sample_indexes = SampleStore().write_samples(shard.shard_id, windows, target_columns, covariate_columns, read_result)
    for sample_index, metadata in zip(sample_indexes, sample_metadata, strict=True):
        sample_index.sample_metadata = metadata
        session.add(sample_index)
    _write_generation_manifest(Path(shard.storage_uri), generation_config, sample_metadata)
    session.add(shard)
    return shard


def _shard_name(base_name: str, capability: SyntheticCapability, multi_capability: bool) -> str:
    clean = base_name.strip() or "Synthetic test cases"
    if multi_capability:
        return f"{clean} - {capability.label}"
    return clean


def _target_dim_for_capability(capability: SyntheticCapability, requested: int) -> int:
    if capability.target_dim_mode == "fixed_1":
        return 1
    if capability.target_dim_mode == "multi":
        return max(3, int(requested))
    if capability.target_dim_mode == "covariate":
        return 1
    return max(1, int(requested))


def _require_near_distance_calibration(
    capability: SyntheticCapability,
    context_length: int,
    horizon: int,
    target_dim: int,
) -> list[str]:
    profile_ids = _profile_ids_for_capability(capability.capability_id)
    buckets = matching_calibrated_buckets(
        profile_ids=profile_ids,
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
    )
    if buckets:
        return [str(bucket["profile_id"]) for bucket in buckets]
    raise ApiError(
        "synthetic_near_distance_not_calibrated",
        "synthetic near-distance gate has no calibrated real bucket for this request",
        {
            "capability_id": capability.capability_id,
            "profile_ids": list(profile_ids),
            "context_length": int(context_length),
            "horizon": int(horizon),
            "target_dim": int(target_dim),
        },
    )


def _require_feature_gate_calibration(
    capability: SyntheticCapability,
    context_length: int,
    horizon: int,
    target_dim: int,
) -> list[str]:
    profile_ids = _profile_ids_for_capability(capability.capability_id)
    buckets = matching_feature_gate_buckets(
        capability_id=capability.capability_id,
        profile_ids=profile_ids,
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
    )
    if buckets:
        return [str(bucket["profile_id"]) for bucket in buckets]
    raise ApiError(
        "synthetic_feature_gate_not_calibrated",
        "synthetic feature-support gate has no calibrated real bucket for this request",
        {
            "capability_id": capability.capability_id,
            "profile_ids": list(profile_ids),
            "context_length": int(context_length),
            "horizon": int(horizon),
            "target_dim": int(target_dim),
        },
    )


def _require_generator_conditioning_calibration(
    capability: SyntheticCapability,
    context_length: int,
    horizon: int,
    target_dim: int,
    frequency: str,
) -> list[str]:
    profile_ids = _profile_ids_for_capability(capability.capability_id)
    profiles = matching_generator_profiles(
        capability_id=capability.capability_id,
        profile_ids=profile_ids,
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
        frequency=frequency,
    )
    if profiles:
        return [str(profile["profile_id"]) for profile in profiles]
    raise ApiError(
        "synthetic_generator_not_calibrated",
        "synthetic generator has no profile-conditioned calibration for this request",
        {
            "capability_id": capability.capability_id,
            "profile_ids": list(profile_ids),
            "context_length": int(context_length),
            "horizon": int(horizon),
            "target_dim": int(target_dim),
            "frequency": frequency,
        },
    )


def _resolve_required_generator_conditioning(
    capability_id: str,
    profile_id: str,
    context_length: int,
    horizon: int,
    target_dim: int,
) -> GeneratorConditioning:
    conditioning = resolve_generator_conditioning(
        capability_id=capability_id,
        profile_id=profile_id,
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
    )
    if conditioning is None:
        raise ApiError(
            "synthetic_generator_not_calibrated",
            "synthetic generator conditioning is missing for the selected real profile",
            {
                "capability_id": capability_id,
                "profile_id": profile_id,
                "context_length": int(context_length),
                "horizon": int(horizon),
                "target_dim": int(target_dim),
            },
        )
    return conditioning


def _seed_for(seed: int, capability_id: str, sample_index: int) -> int:
    payload = f"{seed}:{capability_id}:{sample_index}".encode("utf-8")
    return int(hashlib.blake2s(payload, digest_size=8).hexdigest(), 16) % (2**32 - 1)


def _attempt_seed(sample_seed: int, attempt: int) -> int:
    return (int(sample_seed) + int(attempt) * 104729) % (2**32 - 1)


def _frequency_delta(frequency: str) -> timedelta:
    value = (frequency or "h").strip().lower()
    aliases = {
        "h": timedelta(hours=1),
        "hour": timedelta(hours=1),
        "hourly": timedelta(hours=1),
        "d": timedelta(days=1),
        "day": timedelta(days=1),
        "daily": timedelta(days=1),
        "min": timedelta(minutes=1),
        "minute": timedelta(minutes=1),
    }
    if value in aliases:
        return aliases[value]
    for suffix, unit in (("min", "minutes"), ("m", "minutes"), ("h", "hours"), ("d", "days")):
        if value.endswith(suffix):
            number = value[: -len(suffix)]
            if number.isdigit() and int(number) > 0:
                return timedelta(**{unit: int(number)})
    return timedelta(hours=1)


HOURLY_UNIVARIATE_PROFILE_IDS = (
    "m4_hourly_daily_168ctx",
    "electricity_hourly_daily_168ctx",
    "traffic_hourly_daily_168ctx",
    "electricity_hourly_daily_2048ctx_24h",
)
HOURLY_PANEL_PROFILE_IDS = (
    "electricity_hourly_panel_168ctx",
    "traffic_hourly_panel_168ctx",
)
COVARIATE_PROFILE_IDS = (
    "m5_daily_covariate_365ctx_28h",
    "gefcom2014_load_hourly_covariate_168ctx_24h",
)
HIERARCHY_PROFILE_IDS = ("m5_daily_hierarchy_365ctx_28h",)
ACCEPTANCE_PROFILE_GROUPS: dict[str, tuple[str, ...]] = {
    "hourly_univariate_envelope_168ctx": HOURLY_UNIVARIATE_PROFILE_IDS,
    "hourly_panel_envelope_168ctx": HOURLY_PANEL_PROFILE_IDS,
    "known_future_covariate_envelope_v1": COVARIATE_PROFILE_IDS,
    "m5_hierarchy_envelope_365ctx_28h": HIERARCHY_PROFILE_IDS,
    "synthetic_structural_guard_v1": (),
}
ACCEPTANCE_PROFILE_BY_CAPABILITY: dict[str, str] = {
    "trend": "hourly_univariate_envelope_168ctx",
    "multi_seasonal": "hourly_univariate_envelope_168ctx",
    "time_varying_seasonality": "hourly_univariate_envelope_168ctx",
    "regime_switching": "hourly_univariate_envelope_168ctx",
    "nonlinear_persistence": "hourly_univariate_envelope_168ctx",
    "predictable_intermittency": "hourly_univariate_envelope_168ctx",
    "common_factor": "hourly_panel_envelope_168ctx",
    "hierarchical_coherence": "m5_hierarchy_envelope_365ctx_28h",
    "covariate_response": "known_future_covariate_envelope_v1",
}


def _resolve_seasonality(capability_id: str, *, requested_frequency: str, seed: int) -> SeasonalityResolution:
    profile_ids = _profile_ids_for_capability(capability_id)
    frequency = _canonical_frequency(requested_frequency)
    frequency_matched = tuple(
        profile_id
        for profile_id in profile_ids
        if _canonical_frequency(str(ANCHOR_PROFILE_BUCKETS.get(profile_id, {}).get("frequency", ""))) == frequency
    )
    candidate_profile_ids = frequency_matched or profile_ids
    period_weights = _period_weights(candidate_profile_ids)
    if not period_weights:
        fallback = _default_periods_for_frequency(requested_frequency)
        return SeasonalityResolution(
            season_length=fallback[0],
            source="frequency_default",
            profile_ids=(),
            candidate_periods=fallback,
            requested_frequency=requested_frequency,
        )

    periods = tuple(sorted(period_weights))
    if len(periods) == 1:
        source = "profile_bucket"
        selected = periods[0]
    else:
        source = "significant_period_sample"
        rng = np.random.default_rng(seed)
        weights = np.asarray([period_weights[period] for period in periods], dtype=float)
        weights = weights / np.sum(weights)
        selected = int(rng.choice(np.asarray(periods, dtype=int), p=weights))
    return SeasonalityResolution(
        season_length=int(selected),
        source=source,
        profile_ids=candidate_profile_ids,
        candidate_periods=periods,
        requested_frequency=requested_frequency,
    )


def _profile_ids_for_capability(capability_id: str) -> tuple[str, ...]:
    profile_id = ACCEPTANCE_PROFILE_BY_CAPABILITY.get(capability_id)
    if profile_id is None:
        return ()
    return ACCEPTANCE_PROFILE_GROUPS.get(profile_id, (profile_id,))


def _period_weights(profile_ids: tuple[str, ...]) -> dict[int, float]:
    weights: dict[int, float] = {}
    for profile_id in profile_ids:
        bucket = ANCHOR_PROFILE_BUCKETS.get(profile_id, {})
        periods = tuple(int(period) for period in bucket.get("significant_periods", ()) if int(period) >= 4)
        if not periods:
            season_length = bucket.get("season_length")
            periods = (int(season_length),) if season_length is not None and int(season_length) >= 4 else ()
        if not periods:
            continue
        profile_weight = float(bucket.get("window_count", 1) or 1) / len(periods)
        for period in periods:
            weights[period] = weights.get(period, 0.0) + profile_weight
    return weights


def _canonical_frequency(frequency: str) -> str:
    value = (frequency or "h").strip().lower()
    aliases = {
        "h": "h",
        "hour": "h",
        "hourly": "h",
        "d": "d",
        "day": "d",
        "daily": "d",
        "min": "1min",
        "minute": "1min",
    }
    if value in aliases:
        return aliases[value]
    for suffix in ("min", "m", "h", "d"):
        if value.endswith(suffix):
            number = value[: -len(suffix)]
            if number.isdigit() and int(number) > 0:
                normalized_suffix = "min" if suffix in {"min", "m"} else suffix
                return f"{int(number)}{normalized_suffix}"
    return value


def _default_periods_for_frequency(frequency: str) -> tuple[int, ...]:
    value = _canonical_frequency(frequency)
    if value == "h":
        return (24,)
    if value == "d":
        return (7,)
    if value.endswith("min"):
        minutes = int(value.removesuffix("min"))
        if minutes > 0:
            return (max(4, int(round(24 * 60 / minutes))),)
    return (24,)


BOUNDED_ACCEPTANCE_FEATURES = {
    "trend_strength",
    "seasonal_strength",
    "noise_ratio",
    "outlier_rate",
    "spike_rate",
    "multi_period_score",
    "burst_rate",
    "diff_spike_rate",
    "avg_abs_target_corr",
    "pca_top1_explained",
    "pca_top2_explained",
    "lead_lag_peak_abs",
    "avg_abs_covariate_target_corr",
    "future_abs_covariate_target_corr",
}

# Legacy pilot-cap tables are retained only so archived pre-paper experiment
# scripts remain readable. The online capts-paper-v2 generation path does not call
# these tables or `_accept_synthetic_features`; it uses synthetic_feature_gate.


def _cap_from_profiles(
    feature: str,
    profile_ids: tuple[str, ...],
    *,
    multiplier: float = 1.5,
    default: float | None = None,
) -> float:
    p95_values = [
        ANCHOR_FEATURE_QUANTILES[profile_id][feature]["p95"]
        for profile_id in profile_ids
        if feature in ANCHOR_FEATURE_QUANTILES.get(profile_id, {})
    ]
    if not p95_values:
        if default is None:
            raise KeyError(f"feature {feature!r} is not present in profiles {profile_ids!r}")
        return float(default)
    cap = max(p95_values) * multiplier
    if feature in BOUNDED_ACCEPTANCE_FEATURES:
        cap = min(cap, 1.0)
    return float(cap)


def _caps_from_profiles(
    profile_ids: tuple[str, ...],
    features: tuple[str, ...],
    *,
    multiplier: float = 1.5,
) -> dict[str, float]:
    return {feature: _cap_from_profiles(feature, profile_ids, multiplier=multiplier) for feature in features}


PILOT_ACCEPTANCE_CAPS: dict[str, dict[str, float]] = {
    "trend": {
        **_caps_from_profiles(
            HOURLY_UNIVARIATE_PROFILE_IDS,
            ("trend_strength", "slope_abs", "curvature_abs", "noise_ratio", "spike_rate"),
        ),
    },
    "multi_seasonal": {
        **_caps_from_profiles(
            HOURLY_UNIVARIATE_PROFILE_IDS,
            ("trend_strength", "multi_period_score", "seasonal_strength", "noise_ratio", "spike_rate"),
        ),
    },
    "time_varying_seasonality": {
        **_caps_from_profiles(
            HOURLY_UNIVARIATE_PROFILE_IDS,
            ("seasonal_drift_score", "seasonal_amplitude_cv", "noise_ratio", "spike_rate"),
        ),
    },
    "regime_switching": {
        **_caps_from_profiles(
            HOURLY_UNIVARIATE_PROFILE_IDS,
            ("change_point_shift_energy", "level_shift_strength", "volatility_shift_strength", "spike_rate"),
            multiplier=2.5,
        ),
    },
    "nonlinear_persistence": {
        **_caps_from_profiles(
            HOURLY_UNIVARIATE_PROFILE_IDS,
            ("nonlinear_lag1_gain", "acf_abs_mean", "spike_rate"),
        ),
        "noise_ratio": 1.0,
    },
    "predictable_intermittency": {
        **_caps_from_profiles(
            HOURLY_UNIVARIATE_PROFILE_IDS,
            ("burst_rate", "spike_rate", "outlier_rate", "trend_strength", "seasonal_strength"),
        ),
        "noise_ratio": 1.0,
    },
    "common_factor": {
        **_caps_from_profiles(
            HOURLY_PANEL_PROFILE_IDS,
            ("pca_top1_explained", "effective_factor_rank", "avg_abs_target_corr", "spike_rate"),
        ),
        "noise_ratio": 0.9,
    },
    "hierarchical_coherence": {
        "hierarchy_residual_mean_abs": max(_cap_from_profiles("hierarchy_residual_mean_abs", HIERARCHY_PROFILE_IDS), 1e-6),
        "noise_ratio": _cap_from_profiles("noise_ratio", HIERARCHY_PROFILE_IDS),
        "avg_abs_target_corr": _cap_from_profiles("avg_abs_target_corr", HIERARCHY_PROFILE_IDS),
    },
    "covariate_response": {
        **_caps_from_profiles(
            COVARIATE_PROFILE_IDS,
            ("avg_abs_covariate_target_corr", "future_abs_covariate_target_corr", "noise_ratio", "spike_rate"),
        ),
        "event_lift_abs": _cap_from_profiles("event_lift_abs", COVARIATE_PROFILE_IDS, multiplier=5.0),
    },
}
PILOT_ACCEPTANCE_MINS: dict[str, dict[str, float]] = {
    "multi_seasonal": {"seasonal_strength": 0.55},
}


def _generate_accepted_sample_values(
    capability_id: str,
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    sample_seed: int,
    *,
    anchor_profile_id: str | None = None,
    generator_conditioning: GeneratorConditioning | None = None,
    generator_conditioning_artifact: dict[str, Any] | None = None,
    feature_gate_artifact: dict[str, Any] | None = None,
    near_distance_artifact: dict[str, Any] | None = None,
    acceptance_profile_ids: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None, dict[str, float]]:
    capability = CAPABILITIES_BY_ID.get(capability_id)
    horizon = length - context_length
    near_distance_profile_ids: tuple[str, ...] = ()
    if capability is not None:
        requested_profile_ids = (
            tuple(acceptance_profile_ids)
            if acceptance_profile_ids is not None
            else _profile_ids_for_capability(capability_id)
        )
        near_distance_buckets = matching_calibrated_buckets(
            profile_ids=requested_profile_ids,
            context_length=context_length,
            horizon=horizon,
            target_dim=target_dim,
            artifact=near_distance_artifact,
        )
        if not near_distance_buckets:
            raise ApiError(
                "synthetic_near_distance_not_calibrated",
                "synthetic near-distance gate has no calibrated real bucket for this request",
                {
                    "capability_id": capability_id,
                    "profile_ids": list(requested_profile_ids),
                    "context_length": int(context_length),
                    "horizon": int(horizon),
                    "target_dim": int(target_dim),
                },
            )
        near_distance_profile_ids = tuple(
            str(bucket["profile_id"]) for bucket in near_distance_buckets
        )
        feature_buckets = matching_feature_gate_buckets(
            capability_id=capability_id,
            profile_ids=requested_profile_ids,
            context_length=context_length,
            horizon=horizon,
            target_dim=target_dim,
            artifact=feature_gate_artifact,
        )
        if not feature_buckets:
            raise ApiError(
                "synthetic_feature_gate_not_calibrated",
                "synthetic feature-support gate has no calibrated real bucket for this request",
                {
                    "capability_id": capability_id,
                    "profile_ids": list(requested_profile_ids),
                    "context_length": int(context_length),
                    "horizon": int(horizon),
                    "target_dim": int(target_dim),
                },
            )
        feature_profile_ids = tuple(str(bucket["profile_id"]) for bucket in feature_buckets)
        generator_profiles = matching_generator_profiles(
            capability_id=capability_id,
            profile_ids=requested_profile_ids,
            context_length=context_length,
            horizon=horizon,
            target_dim=target_dim,
            artifact=generator_conditioning_artifact,
        )
        generator_profile_ids = tuple(str(profile["profile_id"]) for profile in generator_profiles)
        calibrated_profile_ids = tuple(
            profile_id
            for profile_id in generator_profile_ids
            if profile_id in set(feature_profile_ids) and profile_id in set(near_distance_profile_ids)
        )
        if not calibrated_profile_ids:
            raise ApiError(
                "synthetic_anchor_profile_not_calibrated",
                "no profile has all generator and post-generation calibrations for this request",
                {
                    "capability_id": capability_id,
                    "context_length": int(context_length),
                    "horizon": int(horizon),
                    "target_dim": int(target_dim),
                },
            )
        if anchor_profile_id is None:
            anchor_profile_id = select_balanced_profile_id(
                calibrated_profile_ids,
                capability_id=capability_id,
                seed=sample_seed,
                sample_index=0,
            )
        elif anchor_profile_id not in calibrated_profile_ids:
            raise ApiError(
                "synthetic_anchor_profile_invalid",
                "selected anchor profile is not fully calibrated for this request",
                {
                    "capability_id": capability_id,
                    "profile_id": anchor_profile_id,
                    "available_profile_ids": list(calibrated_profile_ids),
                },
            )
        if generator_conditioning is None:
            generator_conditioning = _resolve_required_generator_conditioning(
                capability_id,
                anchor_profile_id,
                context_length,
                horizon,
                target_dim,
            )
    if generator_conditioning is not None:
        if anchor_profile_id is not None and generator_conditioning.profile_id != anchor_profile_id:
            raise ValueError("generator conditioning profile does not match anchor_profile_id")
        anchor_profile_id = generator_conditioning.profile_id
        season_length = generator_conditioning.season_length
    max_attempts = 32
    last_result: tuple[np.ndarray, dict[str, Any], np.ndarray | None, dict[str, float]] | None = None
    for attempt in range(max_attempts):
        rng = np.random.default_rng(_attempt_seed(sample_seed, attempt))
        target, latent_params, covariates = _generate_sample_values(
            capability_id,
            length,
            context_length,
            target_dim,
            season_length,
            intensity,
            rng,
            generator_conditioning=generator_conditioning,
        )
        raw_target = np.asarray(target, dtype=float)
        raw_covariates = (
            np.asarray(covariates, dtype=float)
            if covariates is not None
            else None
        )
        predictability = latent_params.get("predictability", {})
        predictability_accepted = bool(predictability.get("construction_validated"))
        if not predictability_accepted:
            raise ApiError(
                "synthetic_predictability_contract_failed",
                "synthetic configuration does not provide enough observable structure for the capability",
                {
                    "capability_id": capability_id,
                    "context_length": int(context_length),
                    "horizon": int(length - context_length),
                    "season_length": int(season_length),
                    "predictability": predictability,
                },
            )
        target = (
            _standardize_hierarchy_by_context(target, context_length)
            if capability_id == "hierarchical_coherence"
            else _standardize_by_context(target, context_length)
        )
        if covariates is not None and covariates.size:
            covariates = _normalize_covariates(covariates, context_length)
        realized_features = _realized_features(
            target,
            covariates,
            season_length,
            context_length,
        )
        if capability_id == "regime_switching":
            realized_features["regime_clock_history_incremental_r2"] = (
                _regime_clock_history_incremental_r2(
                    target,
                    context_length=context_length,
                    season_length=season_length,
                    cut_points=latent_params["cut_points"],
                    dwell_length=int(latent_params["dwell_length"]),
                )
            )
        feature_gate = evaluate_feature_support_gate(
            capability_id=capability_id,
            features=realized_features,
            profile_ids=(anchor_profile_id,) if anchor_profile_id is not None else (),
            context_length=context_length,
            horizon=horizon,
            target_dim=int(target.shape[1]),
            artifact=feature_gate_artifact,
        )
        feature_accepted = bool(feature_gate["accepted"])
        failed_features = list(feature_gate.get("failed_features", []))
        near_distance = evaluate_near_distance_gate(
            target=target,
            features=realized_features,
            profile_ids=near_distance_profile_ids or _profile_ids_for_capability(capability_id),
            context_length=context_length,
            horizon=horizon,
            artifact=near_distance_artifact,
        )
        capability_contrast = evaluate_capability_contrast(
            capability_id=capability_id,
            target=raw_target,
            context_length=context_length,
            season_length=season_length,
            intensity=intensity,
            latent_params=latent_params,
            covariates=raw_covariates,
            evaluation_scale="generator_raw",
        )
        accepted = bool(feature_accepted and near_distance["accepted"])
        validation = _validation_summary(
            capability_id,
            realized_features,
            feature_gate,
            near_distance,
            predictability,
            capability_contrast,
            anchor_profile_id,
        )
        failed_gates = []
        if not feature_accepted:
            failed_gates.append("feature_support")
        if not near_distance["accepted"]:
            failed_gates.append("near_distance")
        latent_params = {
            **latent_params,
            "intensity": int(intensity),
            "intensity_definition": (
                "capability-global canonical realized strength; model-error monotonicity is evaluated separately"
            ),
            "acceptance": {
                "accepted": bool(accepted),
                "attempts": attempt + 1,
                "failed_gates": failed_gates,
                "failed_features": failed_features,
                "profile": anchor_profile_id,
                "profile_group": _acceptance_profile_id(capability_id),
                "profile_selection_stage": "pre_generation",
                "validation": validation,
            },
        }
        last_result = (target, latent_params, covariates, realized_features)
        if accepted:
            return last_result
    assert last_result is not None
    _target, latent_params, _covariates, _features = last_result
    raise ApiError(
        "synthetic_acceptance_failed",
        "synthetic sample failed post-generation acceptance after maximum attempts",
        {
            "capability_id": capability_id,
            "intensity": int(intensity),
            "attempts": max_attempts,
            "failed_gates": latent_params.get("acceptance", {}).get("failed_gates", []),
            "failed_features": latent_params.get("acceptance", {}).get("failed_features", []),
            "feature_gate_status": latent_params.get("acceptance", {}).get("validation", {}).get("feature_gate", {}).get("status"),
            "feature_gate_profile_id": latent_params.get("acceptance", {}).get("validation", {}).get("feature_gate", {}).get("matched_profile_id"),
            "anchor_profile_id": anchor_profile_id,
            "feature_gate_score": latent_params.get("acceptance", {}).get("validation", {}).get("feature_gate", {}).get("score"),
            "feature_gate_threshold": latent_params.get("acceptance", {}).get("validation", {}).get("feature_gate", {}).get("threshold"),
            "near_distance_status": latent_params.get("acceptance", {}).get("validation", {}).get("near_distance_gate", {}).get("status"),
        },
    )


def _accept_synthetic_features(capability_id: str, features: dict[str, float]) -> tuple[bool, list[str]]:
    """Evaluate the obsolete one-sided pilot caps for archived experiments only."""
    caps = PILOT_ACCEPTANCE_CAPS.get(capability_id)
    if not caps:
        return True, []
    failed: list[str] = []
    for feature, cap in caps.items():
        value = features.get(feature)
        if value is not None and np.isfinite(value) and value > cap:
            failed.append(feature)
    for feature, floor in PILOT_ACCEPTANCE_MINS.get(capability_id, {}).items():
        value = features.get(feature)
        if value is not None and np.isfinite(value) and value < floor and feature not in failed:
            failed.append(feature)
    return not failed, failed


def _validation_summary(
    capability_id: str,
    features: dict[str, float],
    feature_gate: dict[str, Any],
    near_distance: dict[str, Any],
    predictability: dict[str, Any],
    capability_contrast: dict[str, Any],
    anchor_profile_id: str | None,
) -> dict[str, Any]:
    target_features = TARGET_FEATURES_BY_CAPABILITY.get(capability_id, ())
    return {
        "schema_version": "synthetic_post_generation_validation.v4",
        "anchor_profile_group": _acceptance_profile_id(capability_id),
        "anchor_profile_id": anchor_profile_id,
        "anchor_profile_selection_stage": "pre_generation",
        "target_features": {
            feature: float(features[feature])
            for feature in target_features
            if feature in features and np.isfinite(features[feature])
        },
        "control_features": feature_gate.get("control_features", {}),
        "feature_gate": feature_gate,
        "near_distance_gate": near_distance,
        "predictability_gate": {
            "accepted": bool(predictability.get("construction_validated")),
            "contract": predictability.get("contract"),
            "evidence": predictability.get("evidence", {}),
        },
        "capability_contrast": capability_contrast,
        "capability_contrast_qualification": (
            "offline seed-bank aggregate; never used to select individual futures"
        ),
        "novelty_check": "online_dcr_nndr_gate",
        "distribution_check": "offline_control_feature_mmd_swd_required",
    }


def _acceptance_profile_id(capability_id: str) -> str | None:
    return ACCEPTANCE_PROFILE_BY_CAPABILITY.get(capability_id)


def _control_feature_bounds(capability_id: str) -> dict[str, dict[str, float]]:
    profile_id = _acceptance_profile_id(capability_id)
    if profile_id is None:
        return {}
    profile_ids = ACCEPTANCE_PROFILE_GROUPS.get(profile_id)
    if profile_ids is None:
        profile_ids = (profile_id,)
    if not profile_ids:
        return {}
    return _profile_envelope(profile_ids)


def _profile_envelope(profile_ids: tuple[str, ...]) -> dict[str, dict[str, float]]:
    features = {
        feature
        for profile_id in profile_ids
        for feature in ANCHOR_FEATURE_QUANTILES.get(profile_id, {})
    }
    envelope: dict[str, dict[str, float]] = {}
    for feature in features:
        rows = [
            ANCHOR_FEATURE_QUANTILES[profile_id][feature]
            for profile_id in profile_ids
            if feature in ANCHOR_FEATURE_QUANTILES.get(profile_id, {})
        ]
        envelope[feature] = {
            "p05": min(float(row.get("p05", float("-inf"))) for row in rows),
            "p50": float(np.median([float(row.get("p50", row.get("p95", 0.0))) for row in rows])),
            "p95": max(float(row.get("p95", float("inf"))) for row in rows),
        }
    return envelope


def _generate_sample_values(
    capability_id: str,
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    *,
    generator_conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    if capability_id == "trend":
        result = _generate_trend(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "multi_seasonal":
        result = _generate_multi_seasonal(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "regime_switching":
        result = _generate_regime_switching(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "time_varying_seasonality":
        result = _generate_time_varying_seasonality(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "nonlinear_persistence":
        result = _generate_nonlinear_persistence(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "predictable_intermittency":
        result = _generate_predictable_intermittency(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "common_factor":
        result = _generate_common_factor(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "hierarchical_coherence":
        result = _generate_hierarchical_coherence(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    elif capability_id == "covariate_response":
        result = _generate_covariate_response(length, context_length, target_dim, season_length, intensity, rng, generator_conditioning)
    else:
        raise ApiError("synthetic_capability_unknown", "unknown synthetic capability", {"capability_id": capability_id}, 404)
    values, metadata, covariates = result
    if generator_conditioning is not None:
        metadata = {
            **metadata,
            "anchor_profile": generator_conditioning.profile_id,
            "generator_conditioning": generator_conditioning.metadata(intensity),
        }
    return values, metadata, covariates


def _base_features(
    length: int,
    context_length: int,
    season_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(length, dtype=float)
    period = max(4, int(season_length))
    seasonal = np.sin(2 * np.pi * t / period)
    slow = np.cos(2 * np.pi * t / max(8, period * 4))
    periods_from_forecast_origin = (t - max(0, context_length - 1)) / period
    return seasonal, slow, periods_from_forecast_origin


def _paper_generator_metadata(
    capability_id: str,
    *,
    validated: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generator_version": PAPER_GENERATOR_VERSION,
        "predictability": {
            "contract": PREDICTABILITY_CONTRACTS[capability_id],
            "construction_validated": bool(validated),
            "evidence": evidence,
        },
    }


def _conditioned_lambda(intensity: int, conditioning: GeneratorConditioning | None) -> float:
    if conditioning is None:
        return _intensity_lambda(intensity)
    return conditioning.lambda_for(intensity)


def _conditioned_parameter(
    conditioning: GeneratorConditioning | None,
    name: str,
    default: float,
) -> float:
    if conditioning is None:
        return float(default)
    return float(conditioning.parameters.get(name, default))


def _conditioned_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    base_scale: float,
    conditioning: GeneratorConditioning | None,
) -> np.ndarray:
    scale = base_scale * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0)
    degrees_of_freedom = _conditioned_parameter(
        conditioning,
        "noise_degrees_of_freedom",
        0.0,
    )
    if degrees_of_freedom > 2.05:
        standardized = rng.standard_t(degrees_of_freedom, size=shape)
        standardized /= np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0))
    else:
        standardized = rng.normal(0.0, 1.0, size=shape)
    return scale * standardized


def _stable_nonperiodic_process(
    length: int,
    context_length: int,
    target_dim: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None,
    *,
    amplitude: float,
    innovation_scale: float = 1.0,
    slow_root_base: float = 0.955,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a profile-conditioned, forecastable process without a fixed period.

    The two real roots define a stable AR(2) law.  The process is standardized
    using history only, so extending the horizon cannot alter an existing
    prefix.  A child RNG isolates length-dependent innovation draws from all
    later generator parameter draws.
    """

    process_seed = int(rng.integers(0, 2**32 - 1))
    process_rng = np.random.default_rng(process_seed)
    residual_phi = _conditioned_parameter(conditioning, "residual_ar_phi", 0.0)
    if slow_root_base < 0.90:
        slow_root = float(
            np.clip(slow_root_base + 0.10 * residual_phi, 0.82, 0.94)
        )
    else:
        slow_root = float(
            np.clip(slow_root_base + 0.04 * residual_phi, 0.94, 0.985)
        )
    fast_roots = process_rng.uniform(0.15, 0.32, size=target_dim)
    phi_1 = slow_root + fast_roots
    phi_2 = -(slow_root * fast_roots)
    warmup = 64
    innovations = _conditioned_noise(
        process_rng,
        (warmup + length, target_dim),
        innovation_scale,
        conditioning,
    )
    state = np.zeros_like(innovations)
    state[0] = innovations[0]
    state[1] = phi_1 * state[0] + innovations[1]
    for index in range(2, warmup + length):
        state[index] = (
            phi_1 * state[index - 1]
            + phi_2 * state[index - 2]
            + innovations[index]
        )
    values = state[warmup:]
    history = values[:context_length]
    center = np.mean(history, axis=0, keepdims=True)
    scale = np.std(history, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    noise_multiplier = _conditioned_parameter(
        conditioning,
        "noise_scale_multiplier",
        1.0,
    )
    realized_amplitude = float(amplitude * np.sqrt(noise_multiplier))
    values = realized_amplitude * (values - center) / scale
    return values, {
        "law": "stable_ar2_nonperiodic",
        "slow_root": slow_root,
        "fast_root_mean": float(np.mean(fast_roots)),
        "phi_1_mean": float(np.mean(phi_1)),
        "phi_2_mean": float(np.mean(phi_2)),
        "amplitude": realized_amplitude,
        "innovation_distribution": (
            "student_t"
            if _conditioned_parameter(
                conditioning,
                "noise_degrees_of_freedom",
                0.0,
            )
            > 2.05
            else "gaussian"
        ),
    }


def _background_trend(
    time_axis: np.ndarray,
    target_dim: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None,
) -> tuple[np.ndarray, np.ndarray]:
    scale = _conditioned_parameter(conditioning, "background_trend_scale", 0.0)
    if scale <= 0.0:
        slopes = np.zeros(target_dim, dtype=float)
    else:
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=target_dim)
        slopes = signs * scale * rng.uniform(0.75, 1.25, size=target_dim)
    return time_axis[:, None] * slopes[None, :], slopes


def _generate_trend(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    _, _, time_axis = _base_features(length, context_length, season_length)
    trend_direction = rng.choice(np.asarray([-1.0, 1.0]), size=target_dim)
    shape_scale = rng.uniform(0.9, 1.1, size=target_dim)
    trend_scale = structure_scale * (0.002 + 0.098 * lam)
    slope = trend_direction * trend_scale * shape_scale
    curvature = trend_direction * trend_scale * 0.06 * shape_scale
    values = time_axis[:, None] * slope + (time_axis[:, None] ** 2) * curvature
    background, background_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        target_dim,
        rng,
        conditioning,
        amplitude=0.12,
        innovation_scale=0.30,
        slow_root_base=0.84,
    )
    values += background
    values += _conditioned_noise(rng, (length, target_dim), 0.08, conditioning)
    noise_scale = 0.08 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0)
    return (
        values,
        {
            **_paper_generator_metadata(
                "trend",
                validated=context_length >= 32,
                evidence={
                    "context_observation_count": int(context_length),
                    "forecast_law": "same_polynomial_coefficients",
                    "background_law": "stable_ar2_nonperiodic",
                },
            ),
            "anchor_profile": "m4_hourly_daily_168ctx",
            "trend_scale": float(trend_scale),
            "slope_by_target": [float(value) for value in slope],
            "curvature_by_target": [float(value) for value in curvature],
            "slope_mean": float(np.mean(slope)),
            "slope_abs_mean": float(np.mean(np.abs(slope))),
            "curvature_mean": float(np.mean(curvature)),
            "curvature_abs_mean": float(np.mean(np.abs(curvature))),
            "background_process": background_metadata,
            "noise_scale": float(noise_scale),
        },
        None,
    )


def _generate_multi_seasonal(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    t = np.arange(length, dtype=float)
    primary_period = max(4, season_length)
    secondary_period = primary_period * 2
    tertiary_period = max(4, primary_period // 2)
    values = np.zeros((length, target_dim))
    period_amplitudes: list[dict[str, float]] = []

    def add_period(period: int, amplitude: np.ndarray) -> None:
        nonlocal values
        phase = rng.uniform(0, 2 * np.pi, size=target_dim)
        values += amplitude[None, :] * np.sin(2 * np.pi * t[:, None] / period + phase[None, :])
        period_amplitudes.append({"period": float(period), "amplitude_mean": float(np.mean(amplitude))})

    seasonal_multiplier = _conditioned_parameter(conditioning, "seasonal_amplitude_multiplier", 1.0)
    amp = seasonal_multiplier * rng.uniform(0.9, 1.1, size=target_dim)
    add_period(primary_period, amp)
    additional_period_strength = structure_scale * (0.10 + 0.70 * lam)
    amp = additional_period_strength * rng.uniform(0.8, 1.2, size=target_dim)
    add_period(secondary_period, amp)
    amp = 0.35 * additional_period_strength * rng.uniform(0.8, 1.2, size=target_dim)
    add_period(tertiary_period, amp)
    slow_period = max(primary_period * 7, primary_period + 1)
    slow_phase = rng.uniform(0, 2 * np.pi, size=target_dim)
    slow_amplitude = 0.05 * _conditioned_parameter(conditioning, "slow_amplitude_multiplier", 1.0)
    values += slow_amplitude * np.cos(2 * np.pi * t[:, None] / slow_period + slow_phase[None, :])
    _, _, time_axis = _base_features(length, context_length, season_length)
    background, background_slopes = _background_trend(time_axis, target_dim, rng, conditioning)
    values += background
    noise_scale = 0.08 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0)
    values += _conditioned_noise(rng, values.shape, 0.08, conditioning)
    return (
        values,
        {
            **_paper_generator_metadata(
                "multi_seasonal",
                validated=context_length >= 2 * secondary_period,
                evidence={
                    "longest_period": int(secondary_period),
                    "longest_period_cycles_in_context": float(context_length / secondary_period),
                },
            ),
            "anchor_profile": "m4_hourly_daily_168ctx",
            "periods": [int(item["period"]) for item in period_amplitudes],
            "period_amplitudes": period_amplitudes,
            "additional_period_strength": float(additional_period_strength),
            "background_slope_abs_mean": float(np.mean(np.abs(background_slopes))),
            "noise_scale": float(noise_scale),
        },
        None,
    )


def _generate_regime_switching(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    _, _, time_axis = _base_features(length, context_length, season_length)
    background, background_slopes = _background_trend(time_axis, target_dim, rng, conditioning)
    dwell_length = max(4, min(2 * max(4, season_length), max(4, context_length // 3)))
    future_switch_at = context_length + max(1, int(season_length) // 2)
    first_switch = future_switch_at
    while first_switch - dwell_length > 0:
        first_switch -= dwell_length
    cut_points = list(range(first_switch, length, dwell_length))
    cut_points = [int(point) for point in cut_points if 0 < point < length]

    state = np.zeros(length, dtype=float)
    boundaries = [0, *cut_points, length]
    initial_sign = float(rng.choice(np.asarray([-1.0, 1.0])))
    for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state[start:end] = initial_sign * (-1.0 if segment % 2 else 1.0)

    regime_strength = structure_scale * (0.25 + 0.85 * lam)
    channel_scale = rng.uniform(0.9, 1.1, size=target_dim)
    values = regime_strength * state[:, None] * channel_scale[None, :]
    stochastic_background, background_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        target_dim,
        rng,
        conditioning,
        amplitude=0.22,
        innovation_scale=0.25,
        slow_root_base=0.84,
    )
    values += stochastic_background
    values += background
    values += _conditioned_noise(rng, (length, target_dim), 0.08, conditioning)
    historical_switches = [point for point in cut_points if point < context_length]
    future_switches = [point for point in cut_points if point >= context_length]
    validated = len(historical_switches) >= 2 and len(future_switches) >= 1
    return (
        values,
        {
            **_paper_generator_metadata(
                "regime_switching",
                validated=validated,
                evidence={
                    "historical_switch_count": len(historical_switches),
                    "future_switch_count": len(future_switches),
                    "constant_dwell_length": int(dwell_length),
                    "alternating_state_order": True,
                },
            ),
            "switch_count": len(cut_points),
            "cut_points": cut_points,
            "forecast_switch": int(bool(future_switches)),
            "future_switch_at": int(future_switches[0]) if future_switches else None,
            "dwell_length": int(dwell_length),
            "regime_strength": float(regime_strength),
            "background_process": background_metadata,
            "background_slope_abs_mean": float(np.mean(np.abs(background_slopes))),
            "noise_scale": 0.08 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0),
        },
        None,
    )


def _generate_time_varying_seasonality(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    t = np.arange(length, dtype=float)
    primary_period = max(4, season_length)
    modulation_period = primary_period * 4
    modulation_phase = rng.uniform(0, 2 * np.pi, size=target_dim)
    carrier_phase = rng.uniform(0, 2 * np.pi, size=target_dim)
    modulation_strength = structure_scale * (0.10 + 0.90 * lam)
    amplitude_depth = 0.06 + 0.34 * modulation_strength
    phase_depth_cycles = 0.01 + 0.07 * modulation_strength
    modulation = np.sin(2 * np.pi * t[:, None] / modulation_period + modulation_phase[None, :])
    amplitude = 1.0 + amplitude_depth * modulation
    phase_modulation = 2 * np.pi * phase_depth_cycles * modulation
    seasonal_multiplier = _conditioned_parameter(conditioning, "seasonal_amplitude_multiplier", 1.0)
    values = seasonal_multiplier * amplitude * np.sin(
        2 * np.pi * t[:, None] / primary_period
        + carrier_phase[None, :]
        + phase_modulation
    )
    residue_amplitude = 0.15 * _conditioned_parameter(conditioning, "slow_amplitude_multiplier", 1.0)
    values += residue_amplitude * np.cos(
        2 * np.pi * t[:, None] / (primary_period * 2)
        + carrier_phase[None, :] / 2
    )
    _, _, time_axis = _base_features(length, context_length, season_length)
    background, background_slopes = _background_trend(time_axis, target_dim, rng, conditioning)
    values += background
    noise_scale = 0.08 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0)
    values += _conditioned_noise(rng, (length, target_dim), 0.08, conditioning)
    return (
        values,
        {
            **_paper_generator_metadata(
                "time_varying_seasonality",
                validated=context_length >= modulation_period,
                evidence={
                    "modulation_period": int(modulation_period),
                    "modulation_cycles_in_context": float(context_length / modulation_period),
                    "forecast_modulation_law": "periodic_continuation",
                },
            ),
            "modulation_strength": float(modulation_strength),
            "amplitude_depth": float(amplitude_depth),
            "amplitude_delta_mean": float(2 * amplitude_depth),
            "phase_modulation_depth_cycles": float(phase_depth_cycles),
            "background_slope_abs_mean": float(np.mean(np.abs(background_slopes))),
            "noise_scale": float(noise_scale),
        },
        None,
    )


def _generate_nonlinear_persistence(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    _, _, time_axis = _base_features(length, context_length, season_length)
    background, background_slopes = _background_trend(time_axis, target_dim, rng, conditioning)
    stochastic_background, background_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        target_dim,
        rng,
        conditioning,
        amplitude=0.03,
        innovation_scale=0.25,
        slow_root_base=0.84,
    )
    seasonal_lag = max(4, int(season_length))
    nonlinear_lag = max(2, seasonal_lag // 2)
    dependency_strength = float(
        np.clip(structure_scale * (0.02 + 0.98 * lam), 0.0, 1.0)
    )
    ar_phi = 0.10
    transform_version = int(
        round(
            _conditioned_parameter(
                conditioning,
                "nonlinear_transform_version",
                2.0,
            )
        )
    )
    if transform_version >= 2:
        seasonal_memory = 0.05 * dependency_strength
        nonlinear_strength = 0.75 * dependency_strength
        nonlinear_frequency = 1.10
        stability_bound = (
            ar_phi
            + seasonal_memory
            + nonlinear_frequency * nonlinear_strength
        )
        warmup_scale = 1.00
        innovation_scale = 0.12
    else:
        seasonal_memory = 0.25 * dependency_strength
        nonlinear_strength = 0.30 * dependency_strength
        nonlinear_frequency = 2.0
        stability_bound = ar_phi + seasonal_memory + 2.0 * nonlinear_strength
        warmup_scale = 0.30
        innovation_scale = 0.06
    burn_in_steps = max(256, 8 * seasonal_lag)
    recurrence_length = burn_in_steps + length
    state = np.zeros((recurrence_length, target_dim))
    warmup = min(recurrence_length, seasonal_lag)
    state[:warmup] = rng.normal(
        0.0,
        warmup_scale,
        size=(warmup, target_dim),
    )
    for idx in range(warmup, recurrence_length):
        nonlinear_response = (
            np.sin(
                nonlinear_frequency * state[idx - nonlinear_lag]
            )
            if transform_version < 2
            else np.sin(
                nonlinear_frequency * state[idx - nonlinear_lag]
            )
            ** 2
            - 0.25
        )
        state[idx] = (
            ar_phi * state[idx - 1]
            + seasonal_memory * state[idx - seasonal_lag]
            + nonlinear_strength * nonlinear_response
            + _conditioned_noise(
                rng,
                (target_dim,),
                innovation_scale,
                conditioning,
            )
        )
    state = state[burn_in_steps : burn_in_steps + length]
    recurrence_amplitude = 1.0 + 2.0 * dependency_strength
    values = recurrence_amplitude * state + stochastic_background
    values += background
    validated = context_length >= 2 * seasonal_lag and stability_bound < 1.0
    return (
        values,
        {
            **_paper_generator_metadata(
                "nonlinear_persistence",
                validated=validated,
                evidence={
                    "maximum_lag": int(seasonal_lag),
                    "maximum_lag_observations_in_context": float(context_length / seasonal_lag),
                    "coefficient_stability_bound": float(stability_bound),
                    "transform_version": int(transform_version),
                    "burn_in_steps": int(burn_in_steps),
                },
            ),
            "dependency_strength": float(dependency_strength),
            "ar_phi": float(ar_phi),
            "seasonal_lag": int(seasonal_lag),
            "seasonal_memory": float(seasonal_memory),
            "nonlinear_lag": int(nonlinear_lag),
            "nonlinear_strength": float(nonlinear_strength),
            "nonlinear_frequency": float(nonlinear_frequency),
            "nonlinear_transform": (
                "sin_squared_centered"
                if transform_version >= 2
                else "sin"
            ),
            "burn_in_steps": int(burn_in_steps),
            "recurrence_amplitude": float(recurrence_amplitude),
            "background_process": background_metadata,
            "background_slope_abs_mean": float(np.mean(np.abs(background_slopes))),
            "noise_scale": innovation_scale
            * _conditioned_parameter(
                conditioning,
                "noise_scale_multiplier",
                1.0,
            ),
        },
        None,
    )


def _generate_predictable_intermittency(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    _, _, time_axis = _base_features(length, context_length, season_length)
    nominal_period = max(4, int(season_length))
    interval_pattern = (
        max(4, int(round(0.75 * nominal_period))),
        max(4, int(round(1.00 * nominal_period))),
        max(4, int(round(1.25 * nominal_period))),
    )
    forecast_center = context_length + max(1, int(season_length) // 2)
    pulse_centers = [int(forecast_center)]
    cursor = int(forecast_center)
    interval_index = -1
    while cursor - interval_pattern[interval_index % len(interval_pattern)] >= 0:
        cursor -= interval_pattern[interval_index % len(interval_pattern)]
        pulse_centers.append(int(cursor))
        interval_index -= 1
    cursor = int(forecast_center)
    interval_index = 0
    while cursor + interval_pattern[interval_index % len(interval_pattern)] < length:
        cursor += interval_pattern[interval_index % len(interval_pattern)]
        pulse_centers.append(int(cursor))
        interval_index += 1
    pulse_centers = sorted(
        center for center in pulse_centers if 0 <= center < length
    )
    pulse_width = max(0.65, nominal_period / 40.0)
    t = np.arange(length, dtype=float)
    pulse_shape = np.zeros(length, dtype=float)
    for center in pulse_centers:
        pulse_shape += np.exp(-0.5 * ((t - center) / pulse_width) ** 2)
    pulse_strength = structure_scale * (0.35 + 1.25 * lam)
    channel_scale = rng.uniform(0.9, 1.1, size=target_dim)
    values = pulse_strength * pulse_shape[:, None] * channel_scale[None, :]
    stochastic_background, background_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        target_dim,
        rng,
        conditioning,
        amplitude=0.18,
        innovation_scale=0.25,
        slow_root_base=0.84,
    )
    values += stochastic_background
    background, background_slopes = _background_trend(time_axis, target_dim, rng, conditioning)
    values += background
    values += _conditioned_noise(rng, (length, target_dim), 0.05, conditioning)
    historical_centers = [center for center in pulse_centers if center < context_length]
    future_centers = [center for center in pulse_centers if center >= context_length]
    validated = (
        len(historical_centers) >= len(interval_pattern) + 1
        and len(future_centers) >= 1
    )
    return (
        values,
        {
            **_paper_generator_metadata(
                "predictable_intermittency",
                validated=validated,
                evidence={
                    "historical_pulse_count": len(historical_centers),
                    "future_pulse_count": len(future_centers),
                    "interval_pattern": [
                        int(value) for value in interval_pattern
                    ],
                    "interval_pattern_repetitions_in_context": float(
                        len(historical_centers) / len(interval_pattern)
                    ),
                },
            ),
            "pulse_period": int(nominal_period),
            "pulse_interval_pattern": [
                int(value) for value in interval_pattern
            ],
            "pulse_centers": pulse_centers,
            "pulse_width": float(pulse_width),
            "pulse_strength": float(pulse_strength),
            "burst_count": len(pulse_centers),
            "background_process": background_metadata,
            "background_slope_abs_mean": float(np.mean(np.abs(background_slopes))),
            "noise_scale": 0.05 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0),
        },
        None,
    )


def _generate_common_factor(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    _, _, time_axis = _base_features(length, context_length, season_length)
    shared_factor_values, factor_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        1,
        rng,
        conditioning,
        amplitude=1.0,
        innovation_scale=0.18,
    )
    shared_factor = shared_factor_values[:, 0]
    shared_strength = structure_scale * (0.15 + 1.05 * lam)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=target_dim)
    loadings = signs * rng.uniform(0.8, 1.2, size=target_dim)
    values = shared_strength * shared_factor[:, None] * loadings[None, :]
    local_amplitude = 0.45 * _conditioned_parameter(conditioning, "local_amplitude_multiplier", 1.0)
    local_components, local_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        target_dim,
        rng,
        conditioning,
        amplitude=local_amplitude,
        innovation_scale=0.30,
    )
    values += local_components
    background, background_slopes = _background_trend(time_axis, target_dim, rng, conditioning)
    values += background
    values += _conditioned_noise(rng, (length, target_dim), 0.08, conditioning)
    validated = target_dim >= 3 and context_length >= 32
    return (
        values,
        {
            **_paper_generator_metadata(
                "common_factor",
                validated=validated,
                evidence={
                    "target_dim": int(target_dim),
                    "shared_factor_law": "stable_ar2_nonperiodic",
                    "loadings_constant_across_boundary": True,
                },
            ),
            "factor_rank": 1,
            "shared_factor_strength": float(shared_strength),
            "shared_factor_process": factor_metadata,
            "local_process": local_metadata,
            "local_amplitude": float(local_amplitude),
            "background_slope_abs_mean": float(np.mean(np.abs(background_slopes))),
            "noise_scale": 0.08 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0),
            "loadings": [float(value) for value in loadings],
        },
        None,
    )


def _generate_hierarchical_coherence(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    child_count = max(2, target_dim - 1)
    heterogeneity_strength = structure_scale * (0.10 + 0.70 * lam)
    shared_values, shared_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        1,
        rng,
        conditioning,
        amplitude=0.55,
        innovation_scale=0.20,
    )
    shared_component = shared_values[:, 0]
    local_components, local_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        child_count,
        rng,
        conditioning,
        amplitude=0.55
        * _conditioned_parameter(
            conditioning,
            "local_amplitude_multiplier",
            1.0,
        ),
        innovation_scale=0.24,
    )
    local_components -= np.mean(local_components, axis=1, keepdims=True)
    common_noise_seed, idiosyncratic_noise_seed = rng.integers(0, 2**32 - 1, size=2)
    common_noise_rng = np.random.default_rng(int(common_noise_seed))
    idiosyncratic_noise_rng = np.random.default_rng(int(idiosyncratic_noise_seed))
    idiosyncratic_noise = _conditioned_noise(
        idiosyncratic_noise_rng,
        (length, child_count),
        0.04,
        conditioning,
    )
    idiosyncratic_noise -= np.mean(idiosyncratic_noise, axis=1, keepdims=True)
    common_noise = _conditioned_noise(common_noise_rng, (length,), 0.03, conditioning)
    children = (
        (shared_component + common_noise)[:, None] / child_count
        + heterogeneity_strength * local_components
        + idiosyncratic_noise
    )
    parent = np.sum(children, axis=1, keepdims=True)
    values = np.concatenate([parent, children], axis=1)
    validated = target_dim >= 3 and context_length >= 32
    return (
        values,
        {
            **_paper_generator_metadata(
                "hierarchical_coherence",
                validated=validated,
                evidence={
                    "target_dim": int(target_dim),
                    "future_only_shock_count": 0,
                    "component_laws_constant_across_boundary": True,
                },
            ),
            "hierarchy": "target_0=sum(target_1:)",
            "child_count": int(child_count),
            "heterogeneity_strength": float(heterogeneity_strength),
            "shared_process": shared_metadata,
            "local_process": local_metadata,
            "noise_scale": 0.04 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0),
            "coherence_residual_mean_abs": float(np.mean(np.abs(parent[:, 0] - np.sum(children, axis=1)))),
        },
        None,
    )


def _generate_covariate_response(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    lam = _conditioned_lambda(intensity, conditioning)
    structure_scale = _conditioned_parameter(conditioning, "structure_scale", 1.0)
    target_noise_seed = int(rng.integers(0, 2**32 - 1))
    target_noise_rng = np.random.default_rng(int(target_noise_seed))
    effect_strength = structure_scale * (0.25 + 0.95 * lam)
    weather_sign = rng.choice(np.asarray([-1.0, 1.0]), size=target_dim)
    beta_weather = weather_sign * effect_strength * rng.uniform(0.6, 1.0, size=target_dim)
    beta_event = effect_strength * rng.uniform(0.9, 1.3, size=target_dim)
    weather_values, weather_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        1,
        rng,
        conditioning,
        amplitude=1.0,
        innovation_scale=0.20,
    )
    weather = weather_values[:, 0]
    event = np.zeros(length)
    event_width = max(2, min(4, max(2, season_length // 8)))
    history_starts = [
        max(1, context_length // 4),
        max(1, context_length // 2),
        max(1, (3 * context_length) // 4),
    ]
    history_starts = sorted({min(context_length - event_width, start) for start in history_starts})
    future_start = context_length + max(1, int(season_length) // 2) - event_width // 2
    event_starts = [*history_starts, int(future_start)]
    for start in event_starts:
        event[int(start) : min(length, int(start) + event_width)] = 1.0
    covariates = np.stack([weather, event], axis=1)
    _, _, time_axis = _base_features(length, context_length, season_length)
    base_trend_scale = _conditioned_parameter(conditioning, "background_trend_scale", 0.01)
    baseline, baseline_metadata = _stable_nonperiodic_process(
        length,
        context_length,
        target_dim,
        rng,
        conditioning,
        amplitude=0.24,
        innovation_scale=0.24,
    )
    values = baseline + base_trend_scale * time_axis[:, None]
    values = values + weather[:, None] * beta_weather + event[:, None] * beta_event
    innovations = _conditioned_noise(
        target_noise_rng,
        (length, target_dim),
        0.08,
        conditioning,
    )
    residual_ar_phi = _conditioned_parameter(conditioning, "residual_ar_phi", 0.0)
    residual = np.array(innovations, copy=True)
    for index in range(1, length):
        residual[index] += residual_ar_phi * residual[index - 1]
    values += residual
    validated = len(history_starts) >= 2 and context_length <= future_start < length
    return (
        values,
        {
            **_paper_generator_metadata(
                "covariate_response",
                validated=validated,
                evidence={
                    "historical_event_count": len(history_starts),
                    "future_event_count": 1,
                    "future_covariates_supplied": True,
                },
            ),
            "future_covariate_dim": 2,
            "effect_strength": float(effect_strength),
            "weather_process": weather_metadata,
            "baseline_process": baseline_metadata,
            "weather_effect_by_target": [
                float(value) for value in beta_weather
            ],
            "event_effect_by_target": [
                float(value) for value in beta_event
            ],
            "weather_effect_mean": float(np.mean(np.abs(beta_weather))),
            "event_effect_mean": float(np.mean(np.abs(beta_event))),
            "event_starts": [int(start) for start in event_starts],
            "event_width": int(event_width),
            "future_event_start": int(future_start),
            "residual_ar_phi": float(residual_ar_phi),
            "noise_scale": 0.08 * _conditioned_parameter(conditioning, "noise_scale_multiplier", 1.0),
        },
        covariates,
    )


def _intensity_lambda(intensity: int) -> float:
    return (int(intensity) - 1) / 4


def _difficulty_lambda(difficulty: int) -> float:
    return _intensity_lambda(difficulty)


def _standardize_by_context(values: np.ndarray, context_length: int) -> np.ndarray:
    context = values[:context_length]
    mean = context.mean(axis=0, keepdims=True)
    std = context.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (values - mean) / std


def _standardize_hierarchy_by_context(values: np.ndarray, context_length: int) -> np.ndarray:
    context = values[:context_length]
    mean = context.mean(axis=0, keepdims=True)
    centered = values - mean
    scale = float(np.std(context[:, 0]))
    if scale <= 1e-6:
        scale = float(np.mean(np.std(context, axis=0)))
    if scale <= 1e-6:
        scale = 1.0
    return centered / scale


def _normalize_covariates(covariates: np.ndarray, context_length: int) -> np.ndarray:
    normalized = covariates.copy()
    for index in range(normalized.shape[1]):
        column = normalized[:context_length, index]
        if set(np.unique(normalized[:, index])).issubset({0.0, 1.0}):
            continue
        mean = float(column.mean())
        std = float(column.std()) or 1.0
        normalized[:, index] = (normalized[:, index] - mean) / std
    return normalized


def _realized_features(
    target: np.ndarray,
    covariates: np.ndarray | None,
    season_length: int,
    context_length: int,
) -> dict[str, float]:
    features = {
        "target_mean_abs": float(np.mean(np.abs(target))),
        "target_std_mean": float(np.mean(target.std(axis=0))),
        "target_max_abs": float(np.max(np.abs(target))),
    }
    features.update(_target_profile_features(target, season_length))
    features.update(_structural_univariate_features(np.mean(target, axis=1), season_length))
    if target.shape[1] > 1:
        features.update(_multivariate_profile_features(target))
    if target.shape[1] > 2:
        hierarchy_residual = target[:, 0] - np.sum(target[:, 1:], axis=1)
        features["hierarchy_residual_mean_abs"] = float(np.mean(np.abs(hierarchy_residual)))
        children = target[:, 1:]
        features["hierarchy_child_heterogeneity"] = float(
            np.mean(np.std(children, axis=1))
        )
        child_magnitude = float(np.mean(np.sum(np.abs(children), axis=1)))
        features["hierarchy_aggregation_ratio"] = (
            float(np.mean(np.abs(target[:, 0]))) / child_magnitude
            if child_magnitude > 1e-12
            else 0.0
        )
    if covariates is not None and covariates.size:
        features.update(
            _covariate_profile_features(
                target,
                covariates,
                context_length,
                season_length,
            )
        )
    return features


def _structural_univariate_features(values: np.ndarray, season_length: int) -> dict[str, float]:
    y = _robust_scale(np.asarray(values, dtype=float))
    n = y.size
    if n < 12:
        return {}
    min_seg = max(6, min(24, n // 8))
    std_all = float(np.std(y)) or 1.0
    cuts = np.arange(min_seg, n - min_seg, dtype=int)
    if cuts.size:
        prefix = np.concatenate([[0.0], np.cumsum(y)])
        prefix_sq = np.concatenate([[0.0], np.cumsum(y * y)])
        total = prefix[-1]
        total_sq = prefix_sq[-1]
        left_count = cuts.astype(float)
        right_count = (n - cuts).astype(float)
        left_sum = prefix[cuts]
        right_sum = total - left_sum
        left_mean = left_sum / left_count
        right_mean = right_sum / right_count
        left_var = np.maximum(prefix_sq[cuts] / left_count - left_mean * left_mean, 0.0)
        right_var = np.maximum((total_sq - prefix_sq[cuts]) / right_count - right_mean * right_mean, 0.0)
        level_scores_arr = np.abs(left_mean - right_mean) / std_all
        volatility_scores_arr = np.abs(np.sqrt(left_var) - np.sqrt(right_var)) / std_all
        level_scores = level_scores_arr.tolist()
        volatility_scores = volatility_scores_arr.tolist()
    else:
        level_scores = []
        volatility_scores = []
    seasonal_profile = _phase_profile(y, season_length)
    half = max(1, n // 2)
    seasonal_left = _phase_profile(y[:half], season_length)
    seasonal_right = _phase_profile(y[half:], season_length)
    modulation_features = _seasonal_modulation_features(y, season_length)
    diff = np.diff(y)
    return {
        "level_shift_strength": float(max(level_scores)) if level_scores else 0.0,
        "volatility_shift_strength": float(max(volatility_scores)) if volatility_scores else 0.0,
        "change_point_shift_energy": float(np.mean(sorted(level_scores, reverse=True)[:3])) if level_scores else 0.0,
        "burst_rate": float(np.mean(np.abs(y) > 3.0)),
        "diff_spike_rate": float(np.mean(np.abs(_robust_scale(diff)) > 3.0)) if diff.size else 0.0,
        "multi_period_score": _multi_period_score(y, season_length),
        "seasonal_drift_score": float(np.mean(np.abs(seasonal_left - seasonal_right))) if seasonal_left.size and seasonal_right.size else 0.0,
        "seasonal_amplitude_cv": float(np.std(np.abs(seasonal_profile)) / (np.mean(np.abs(seasonal_profile)) + 1e-9)) if seasonal_profile.size else 0.0,
        "nonlinear_lag1_gain": _nonlinear_lag1_gain(y),
        "nonlinear_multi_lag_gain": _nonlinear_multi_lag_gain(y, season_length),
        "nonlinear_conditional_gain": _nonlinear_conditional_gain(
            y,
            season_length,
        ),
        **modulation_features,
    }


def _regime_clock_history_incremental_r2(
    target: np.ndarray,
    *,
    context_length: int,
    season_length: int,
    cut_points: list[int] | tuple[int, ...],
    dwell_length: int,
) -> float:
    """Measure the history-only gain of the generator's recurring state clock.

    The nuisance model contains a linear trend, ordinary seasonal harmonics,
    and clock-period harmonics.  The returned value is therefore the extra
    explanatory power of the discrete state itself, rather than a generic
    change-point score that can also be large for smooth stochastic drift.
    """

    values = np.asarray(target, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2
        or context_length < 8
        or context_length > len(values)
        or dwell_length <= 0
    ):
        return 0.0

    state = np.ones(len(values), dtype=float)
    boundaries = [
        0,
        *sorted(
            {
                int(point)
                for point in cut_points
                if 0 < int(point) < len(values)
            }
        ),
        len(values),
    ]
    for segment_index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        state[start:end] = 1.0 if segment_index % 2 == 0 else -1.0

    time = np.arange(len(values), dtype=float)
    normalized_time = (time - (len(values) - 1) / 2.0) / max(
        len(values) - 1,
        1,
    )
    columns = [np.ones(len(values)), normalized_time]
    seasonal_period = max(4, int(season_length))
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * time / seasonal_period
        columns.extend([np.sin(angle), np.cos(angle)])
    clock_period = max(4, 2 * int(dwell_length))
    for harmonic in (1, 2, 3, 4, 5):
        angle = 2.0 * np.pi * harmonic * time / clock_period
        columns.extend([np.sin(angle), np.cos(angle)])
    baseline = np.column_stack(columns)[:context_length]
    state_history = state[:context_length]

    gains: list[float] = []
    for channel in range(values.shape[1]):
        history = values[:context_length, channel]
        baseline_coefficients = np.linalg.lstsq(
            baseline,
            history,
            rcond=None,
        )[0]
        baseline_sse = float(
            np.sum((history - baseline @ baseline_coefficients) ** 2)
        )
        full_design = np.column_stack([baseline, state_history])
        full_coefficients = np.linalg.lstsq(
            full_design,
            history,
            rcond=None,
        )[0]
        full_sse = float(
            np.sum((history - full_design @ full_coefficients) ** 2)
        )
        gains.append(
            max(0.0, (baseline_sse - full_sse) / max(baseline_sse, 1e-9))
        )
    return float(np.median(gains))


def _seasonal_modulation_features(values: np.ndarray, season_length: int) -> dict[str, float]:
    period = max(4, int(season_length))
    cycle_count = values.size // period
    if cycle_count < 3:
        return {
            "seasonal_amplitude_modulation": 0.0,
            "seasonal_phase_variation": 0.0,
        }
    phase_index = np.arange(period, dtype=float)
    sine = np.sin(2 * np.pi * phase_index / period)
    cosine = np.cos(2 * np.pi * phase_index / period)
    amplitudes: list[float] = []
    phases: list[float] = []
    for cycle in range(cycle_count):
        segment = values[cycle * period : (cycle + 1) * period]
        centered = segment - float(np.mean(segment))
        sine_coefficient = 2.0 * float(np.mean(centered * sine))
        cosine_coefficient = 2.0 * float(np.mean(centered * cosine))
        amplitudes.append(float(np.hypot(sine_coefficient, cosine_coefficient)))
        phases.append(float(np.arctan2(cosine_coefficient, sine_coefficient)))
    amplitude_array = np.asarray(amplitudes, dtype=float)
    phase_array = np.unwrap(np.asarray(phases, dtype=float))
    return {
        "seasonal_amplitude_modulation": float(
            np.std(amplitude_array) / (np.mean(amplitude_array) + 1e-9)
        ),
        "seasonal_phase_variation": float(np.std(phase_array) / np.pi),
    }


def _multivariate_profile_features(target: np.ndarray) -> dict[str, float]:
    centered = target - np.mean(target, axis=0, keepdims=True)
    corr_values = [
        abs(_safe_corr(target[:, left], target[:, right]))
        for left in range(target.shape[1])
        for right in range(target.shape[1])
        if left != right
    ]
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular**2
    total = float(np.sum(variance))
    explained = variance / total if total > 1e-12 else np.zeros_like(variance)
    entropy = -float(np.sum([value * np.log(value) for value in explained if value > 1e-12]))
    return {
        "avg_abs_target_corr": float(np.mean(corr_values)) if corr_values else 0.0,
        "pca_top1_explained": float(explained[0]) if explained.size else 0.0,
        "pca_top2_explained": float(np.sum(explained[:2])) if explained.size else 0.0,
        "effective_factor_rank": float(np.exp(entropy)) if explained.size else 0.0,
        "lead_lag_peak_abs": _lead_lag_peak_abs(target),
    }


def _covariate_profile_features(
    target: np.ndarray,
    covariates: np.ndarray,
    context_length: int,
    season_length: int,
) -> dict[str, float]:
    scores: list[float] = []
    future_scores: list[float] = []
    for cov_idx in range(covariates.shape[1]):
        for target_idx in range(target.shape[1]):
            corr_value = _safe_corr(covariates[:, cov_idx], target[:, target_idx])
            if np.isfinite(corr_value):
                scores.append(abs(float(corr_value)))
            if context_length < len(target):
                future_corr = _safe_corr(covariates[context_length:, cov_idx], target[context_length:, target_idx])
                if np.isfinite(future_corr):
                    future_scores.append(abs(float(future_corr)))
    event_lifts: list[float] = []
    for cov_idx in range(covariates.shape[1]):
        column = covariates[:, cov_idx]
        unique = np.unique(column)
        if unique.size <= 3 and np.any(column > 0):
            active = column > 0
            inactive = ~active
            if active.any() and inactive.any():
                event_lifts.append(abs(float(np.mean(target[active]) - np.mean(target[inactive]))))
    t = np.arange(target.shape[0], dtype=float)
    period = max(4, int(season_length))
    baseline_design = np.column_stack(
        [
            np.ones(target.shape[0], dtype=float),
            np.sin(2 * np.pi * t / period),
            np.cos(2 * np.pi * t / period),
        ]
    )
    covariate_design = np.column_stack([baseline_design, covariates])
    incremental_scores = [
        max(
            0.0,
            _r2(target[:, target_idx], covariate_design)
            - _r2(target[:, target_idx], baseline_design),
        )
        for target_idx in range(target.shape[1])
    ]
    residual_acf_scores: list[float] = []
    residual_outlier_scores: list[float] = []
    residual_spike_scores: list[float] = []
    for target_idx in range(target.shape[1]):
        residual = _linear_residual(target[:, target_idx], covariate_design)
        residual_acf_scores.append(
            _mean_abs_autocorrelation(
                _robust_scale(residual),
                max_lag=min(10, max(1, residual.size // 4)),
            )
        )
        residual_outlier_scores.append(_outlier_rate(residual))
        residual_spike_scores.append(_spike_rate(residual))
    return {
        "avg_abs_covariate_target_corr": float(np.mean(scores)) if scores else 0.0,
        "future_abs_covariate_target_corr": float(np.mean(future_scores)) if future_scores else 0.0,
        "event_lift_abs": float(np.mean(event_lifts)) if event_lifts else 0.0,
        "covariate_incremental_r2": float(np.mean(incremental_scores)) if incremental_scores else 0.0,
        "covariate_residual_acf_abs_mean": float(np.mean(residual_acf_scores)) if residual_acf_scores else 0.0,
        "covariate_residual_outlier_rate": float(np.mean(residual_outlier_scores)) if residual_outlier_scores else 0.0,
        "covariate_residual_spike_rate": float(np.mean(residual_spike_scores)) if residual_spike_scores else 0.0,
    }


def _target_profile_features(target: np.ndarray, season_length: int) -> dict[str, float]:
    per_target = [_single_target_profile(target[:, index], season_length) for index in range(target.shape[1])]
    names = {
        name
        for row in per_target
        for name, value in row.items()
        if np.isfinite(value)
    }
    return {
        name: float(np.mean([row[name] for row in per_target if name in row and np.isfinite(row[name])]))
        for name in sorted(names)
    }


def _single_target_profile(values: np.ndarray, season_length: int) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return {}
    scaled = _robust_scale(y)
    trend = _polynomial_trend(scaled)
    seasonal = _seasonal_by_phase(scaled - trend, season_length)
    residual = scaled - trend - seasonal
    total_var = _safe_var(scaled)
    return {
        "trend_strength": _strength(residual, trend + residual),
        "seasonal_strength": _strength(residual, seasonal + residual),
        "slope_abs": abs(_polyfit_coeff(scaled, degree=2, coeff_index=1)),
        "curvature_abs": abs(_polyfit_coeff(scaled, degree=2, coeff_index=0)),
        "noise_ratio": float(np.clip(_safe_var(residual) / total_var, 0.0, 1.0)) if total_var > 0 else 0.0,
        "acf1": _autocorrelation(scaled, 1),
        "acf_abs_mean": _mean_abs_autocorrelation(scaled, max_lag=min(10, max(1, y.size // 4))),
        "outlier_rate": _outlier_rate(scaled),
        "spike_rate": _spike_rate(scaled),
    }


def _phase_profile(values: np.ndarray, season_length: int) -> np.ndarray:
    if season_length < 2 or values.size < season_length:
        return np.asarray([], dtype=float)
    period = int(season_length)
    phases = np.arange(values.size) % period
    profile = np.asarray(
        [
            float(np.mean(values[phases == phase])) if np.any(phases == phase) else 0.0
            for phase in range(period)
        ],
        dtype=float,
    )
    return profile - float(np.mean(profile))


def _multi_period_score(values: np.ndarray, season_length: int) -> float:
    if values.size < 8:
        return 0.0
    centered = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if spectrum.size <= 2:
        return 0.0
    spectrum[0] = 0.0
    total = float(np.sum(spectrum))
    if total <= 1e-12:
        return 0.0
    primary_index = int(round(values.size / max(2, season_length)))
    exclude = {idx for idx in range(max(1, primary_index - 1), min(spectrum.size, primary_index + 2))}
    secondary = np.asarray([value for idx, value in enumerate(spectrum) if idx not in exclude and idx > 0], dtype=float)
    return float(np.max(secondary) / total) if secondary.size else 0.0


def _nonlinear_lag1_gain(values: np.ndarray) -> float:
    if values.size < 8:
        return 0.0
    x = values[:-1]
    y = values[1:]
    linear = np.column_stack([np.ones_like(x), x])
    nonlinear = np.column_stack([np.ones_like(x), x, x**2, np.sin(x)])
    return max(0.0, _r2(y, nonlinear) - _r2(y, linear))


def _nonlinear_multi_lag_gain(values: np.ndarray, season_length: int) -> float:
    seasonal_lag = max(4, int(season_length))
    nonlinear_lag = max(2, seasonal_lag // 2)
    start = max(seasonal_lag, nonlinear_lag, 1)
    if values.size - start < 8:
        return 0.0
    target = values[start:]
    lag1 = values[start - 1 : -1]
    lag_seasonal = values[: values.size - seasonal_lag]
    if lag_seasonal.size > target.size:
        lag_seasonal = lag_seasonal[-target.size :]
    lag_nonlinear = values[start - nonlinear_lag : values.size - nonlinear_lag]
    linear = np.column_stack([np.ones_like(target), lag1])
    nonlinear = np.column_stack(
        [
            np.ones_like(target),
            lag1,
            lag_seasonal,
            np.sin(2.0 * lag_nonlinear),
        ]
    )
    return max(0.0, _r2(target, nonlinear) - _r2(target, linear))


def _nonlinear_conditional_gain(values: np.ndarray, season_length: int) -> float:
    """Bias-corrected nonlinear-lag gain after linear lag conditioning."""

    seasonal_lag = max(4, int(season_length))
    nonlinear_lag = max(2, seasonal_lag // 2)
    start = max(seasonal_lag, nonlinear_lag, 1)
    if values.size - start < 8:
        return 0.0
    target = values[start:]
    lag1 = values[start - 1 : -1]
    lag_seasonal = values[: values.size - seasonal_lag]
    if lag_seasonal.size > target.size:
        lag_seasonal = lag_seasonal[-target.size :]
    lag_nonlinear = values[
        start - nonlinear_lag : values.size - nonlinear_lag
    ]
    linear = np.column_stack(
        [
            np.ones_like(target),
            lag1,
            lag_seasonal,
            lag_nonlinear,
        ]
    )
    nonlinear = np.column_stack(
        [
            linear,
            np.sin(1.1 * lag_nonlinear) ** 2,
        ]
    )
    return float(
        _adjusted_r2(target, nonlinear) - _adjusted_r2(target, linear)
    )


def _r2(y: np.ndarray, design: np.ndarray) -> float:
    try:
        coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    fitted = design @ coeffs
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - fitted) ** 2) / denom)


def _adjusted_r2(y: np.ndarray, design: np.ndarray) -> float:
    """Return adjusted R² using the effective design rank."""

    observations = int(len(y))
    try:
        predictor_count = max(int(np.linalg.matrix_rank(design)) - 1, 0)
    except np.linalg.LinAlgError:
        return 0.0
    residual_degrees_of_freedom = observations - predictor_count - 1
    if observations <= 1 or residual_degrees_of_freedom <= 0:
        return 0.0
    raw_r2 = _r2(y, design)
    return float(
        1.0
        - (1.0 - raw_r2)
        * (observations - 1)
        / residual_degrees_of_freedom
    )


def _linear_residual(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    try:
        coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.asarray(y, dtype=float) - float(np.mean(y))
    return np.asarray(y, dtype=float) - design @ coeffs


def _lead_lag_peak_abs(values: np.ndarray, max_lag: int = 12) -> float:
    if values.shape[1] < 2:
        return 0.0
    peaks: list[float] = []
    lag_limit = min(max_lag, max(1, values.shape[0] // 4))
    for left in range(values.shape[1]):
        for right in range(values.shape[1]):
            if left == right:
                continue
            for lag in range(1, lag_limit + 1):
                peaks.append(abs(_safe_corr(values[:-lag, left], values[lag:, right])))
    finite = [value for value in peaks if np.isfinite(value)]
    return float(max(finite)) if finite else 0.0


def _robust_scale(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-9:
        return (values - median) / iqr
    std = float(np.std(values))
    if std > 1e-9:
        return (values - median) / std
    return values - median


def _polynomial_trend(values: np.ndarray) -> np.ndarray:
    if values.size < 4:
        return np.full_like(values, float(np.mean(values)))
    t = np.linspace(-1.0, 1.0, values.size)
    coeffs = np.polyfit(t, values, min(2, values.size - 1))
    return np.polyval(coeffs, t)


def _seasonal_by_phase(values: np.ndarray, season_length: int) -> np.ndarray:
    if season_length < 2 or values.size < season_length * 2:
        return np.zeros_like(values)
    seasonal = np.zeros_like(values)
    phases = np.arange(values.size) % int(season_length)
    for phase in range(int(season_length)):
        mask = phases == phase
        if mask.any():
            seasonal[mask] = float(np.mean(values[mask]))
    return seasonal - float(np.mean(seasonal))


def _strength(residual: np.ndarray, residual_plus_component: np.ndarray) -> float:
    denom = _safe_var(residual_plus_component)
    if denom <= 0:
        return 0.0
    return float(np.clip(1.0 - _safe_var(residual) / denom, 0.0, 1.0))


def _safe_var(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    value = float(np.var(values))
    return value if np.isfinite(value) else 0.0


def _polyfit_coeff(values: np.ndarray, *, degree: int, coeff_index: int) -> float:
    if values.size <= degree:
        return 0.0
    t = np.linspace(-1.0, 1.0, values.size)
    return float(np.polyfit(t, values, degree)[coeff_index])


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or values.size <= lag:
        return 0.0
    a = values[:-lag] - float(np.mean(values[:-lag]))
    b = values[lag:] - float(np.mean(values[lag:]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_abs_autocorrelation(values: np.ndarray, max_lag: int) -> float:
    if max_lag <= 0:
        return 0.0
    values_by_lag = [abs(_autocorrelation(values, lag)) for lag in range(1, max_lag + 1)]
    return float(np.mean(values_by_lag)) if values_by_lag else 0.0


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size != right.size or left.size < 3:
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _outlier_rate(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-9:
        return 0.0
    robust_z = 0.6745 * np.abs(values - median) / mad
    return float(np.mean(robust_z > 4.0))


def _spike_rate(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    diff = np.diff(values)
    median = float(np.median(diff))
    mad = float(np.median(np.abs(diff - median)))
    if mad <= 1e-9:
        return 0.0
    robust_z = 0.6745 * np.abs(diff - median) / mad
    return float(np.mean(robust_z > 4.0))


def _write_generation_manifest(path: Path, generation_config: dict[str, Any], sample_metadata: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic_shard_manifest.v1",
                "generation_config": generation_config,
                "sample_count": len(sample_metadata),
                "sample_metadata_preview": sample_metadata[:20],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
