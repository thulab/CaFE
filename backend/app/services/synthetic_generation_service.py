from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


SYNTHETIC_CAPABILITIES: tuple[SyntheticCapability, ...] = (
    SyntheticCapability(
        "trend",
        "Trend",
        "Single-target series with controllable trend and seasonal residue.",
        "趋势",
        "带有可控趋势和季节残差的单目标序列。",
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
        "Regime switching",
        "Single-target series with level and volatility changes.",
        "状态切换",
        "带有水平和波动率切换的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "time_varying_seasonality",
        "Time-varying seasonality",
        "Single-target series with drifting seasonal amplitude and phase.",
        "时变季节性",
        "带有季节振幅和相位漂移的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "long_memory_nonlinear",
        "Long-memory nonlinear",
        "Single-target autoregressive dynamics with nonlinear carry-over.",
        "长记忆非线性",
        "带有非线性延续效应的单目标自回归动态。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "intermittent_heteroskedastic",
        "Intermittent heteroskedastic",
        "Single-target sparse bursts with changing noise scale.",
        "间歇异方差",
        "带有稀疏突发和变化噪声尺度的单目标序列。",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "common_factor",
        "Common factor",
        "Multiple targets driven by shared latent factors.",
        "公共因子",
        "由共享潜在因子驱动的多目标序列。",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "lead_lag_coupling",
        "Lead-lag coupling",
        "Multiple targets with lagged cross-channel dependencies.",
        "Lead-lag 耦合",
        "带有滞后跨通道依赖的多目标序列。",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "coherent_regime_shift",
        "Coherent regime shift",
        "Multiple targets that shift together across regimes.",
        "协同状态切换",
        "多个目标在同一状态变化中协同切换的序列。",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "hierarchical_coherence",
        "Hierarchical coherence",
        "Multiple targets with parent-child additive structure.",
        "层级一致性",
        "带有父子加总结构的多目标序列。",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "covariate_response",
        "Covariate response",
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

MOCK_ANCHOR = {
    "anchor_mode": "profile_calibrated",
    "anchor_source_uri": "synthetic-anchor://public/multi-profile-v1",
    "anchor_profiles": [
        "m4_hourly_daily_96ctx",
        "m4_hourly_daily_168ctx",
        "m4_hourly_weekly",
        "electricity_hourly_daily_168ctx",
        "electricity_hourly_panel_168ctx",
        "traffic_hourly_daily_168ctx",
        "traffic_hourly_panel_168ctx",
        "us_births_weekly",
        "us_births_annual_diagnostic",
    ],
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
}

TARGET_FEATURES_BY_CAPABILITY: dict[str, tuple[str, ...]] = {
    "trend": ("trend_strength", "slope_abs", "curvature_abs"),
    "multi_seasonal": ("multi_period_score", "seasonal_strength"),
    "time_varying_seasonality": ("seasonal_drift_score", "seasonal_amplitude_cv"),
    "regime_switching": ("change_point_shift_energy", "level_shift_strength"),
    "long_memory_nonlinear": ("nonlinear_lag1_gain",),
    "intermittent_heteroskedastic": ("burst_rate", "spike_rate", "outlier_rate", "noise_ratio"),
    "common_factor": ("pca_top1_explained", "effective_factor_rank"),
    "lead_lag_coupling": ("lead_lag_peak_abs",),
    "coherent_regime_shift": ("level_shift_strength", "avg_abs_target_corr"),
    "hierarchical_coherence": ("hierarchy_residual_mean_abs",),
    "covariate_response": ("avg_abs_covariate_target_corr", "future_abs_covariate_target_corr", "event_lift_abs"),
}

CONTROL_FEATURES_BY_CAPABILITY: dict[str, tuple[str, ...]] = {
    "trend": ("seasonal_strength", "noise_ratio", "spike_rate"),
    "multi_seasonal": ("trend_strength", "noise_ratio", "spike_rate"),
    "time_varying_seasonality": ("trend_strength", "noise_ratio", "spike_rate"),
    "regime_switching": ("seasonal_strength", "spike_rate"),
    "long_memory_nonlinear": ("seasonal_strength", "noise_ratio", "spike_rate"),
    "intermittent_heteroskedastic": ("trend_strength", "seasonal_strength"),
    "common_factor": ("noise_ratio", "spike_rate"),
    "lead_lag_coupling": ("noise_ratio", "spike_rate"),
    "coherent_regime_shift": ("noise_ratio", "spike_rate"),
    "hierarchical_coherence": ("hierarchy_residual_mean_abs", "noise_ratio"),
    "covariate_response": ("trend_strength", "seasonal_strength", "noise_ratio", "spike_rate"),
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
            "default_params": {
                "context_length": 60,
                "horizon": 16,
                "sample_count": 32,
                "intensity": 3,
                "season_length": 24,
                "target_dim": 3 if capability.target_dim_mode in {"multi", "covariate"} else 1,
                "frequency": "h",
            },
            "limits": {
                "context_length": {"min": 16, "max": 2048},
                "horizon": {"min": 1, "max": 512},
                "sample_count": {"min": 1, "max": 1000},
                "intensity": {"min": 1, "max": 5},
                "season_length": {"min": 4, "max": 512},
                "target_dim": {"min": 1, "max": 16},
            },
        }
        for capability in SYNTHETIC_CAPABILITIES
    ]


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
        "season_length": config.season_length,
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
    target_columns = [f"target_{index}" for index in range(target_dim)]
    covariate_columns = list(capability.covariate_columns)
    columns = [*target_columns, *covariate_columns]
    base_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    delta = _frequency_delta(config.frequency)

    all_timestamps: list[datetime] = []
    all_values: list[list[float]] = []
    windows: list[SampleWindow] = []
    sample_metadata: list[dict[str, Any]] = []
    for sample_index in range(config.sample_count):
        sample_seed = _seed_for(config.seed, capability.capability_id, sample_index)
        target, latent_params, covariates, realized_features = _generate_accepted_sample_values(
            capability.capability_id,
            sample_length,
            context,
            target_dim,
            config.season_length,
            config.intensity,
            sample_seed,
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
                "capability_id": capability.capability_id,
                "capability_label": capability.label,
                "intensity": config.intensity,
                "difficulty": config.intensity,
                "seed": config.seed,
                "sample_seed": sample_seed,
                "latent_params": latent_params,
                "realized_features": realized_features,
                **MOCK_ANCHOR,
            }
        )

    generation_config = {
        "schema_version": "synthetic_generation.v1",
        "generation_id": generation_id,
        "capability_id": capability.capability_id,
        "capability_label": capability.label,
        "task_type": capability.task_type,
        "intensity": config.intensity,
        "difficulty": config.intensity,
        "intensity_definition": "target temporal structure strength; not a required monotonic model-error difficulty",
        "seed": config.seed,
        "context_length": context,
        "horizon": horizon,
        "sample_count": config.sample_count,
        "season_length": config.season_length,
        "target_dim": target_dim,
        "requested_target_dim": config.target_dim,
        "covariate_columns": covariate_columns,
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
        return max(2, int(requested))
    return max(1, int(requested))


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
)
HOURLY_PANEL_PROFILE_IDS = (
    "electricity_hourly_panel_168ctx",
    "traffic_hourly_panel_168ctx",
)
ACCEPTANCE_PROFILE_GROUPS: dict[str, tuple[str, ...]] = {
    "hourly_univariate_envelope_168ctx": HOURLY_UNIVARIATE_PROFILE_IDS,
    "hourly_panel_envelope_168ctx": HOURLY_PANEL_PROFILE_IDS,
    "synthetic_structural_guard_v1": (),
}
ACCEPTANCE_PROFILE_BY_CAPABILITY: dict[str, str] = {
    "trend": "hourly_univariate_envelope_168ctx",
    "multi_seasonal": "hourly_univariate_envelope_168ctx",
    "time_varying_seasonality": "hourly_univariate_envelope_168ctx",
    "regime_switching": "hourly_univariate_envelope_168ctx",
    "long_memory_nonlinear": "hourly_univariate_envelope_168ctx",
    "intermittent_heteroskedastic": "hourly_univariate_envelope_168ctx",
    "common_factor": "hourly_panel_envelope_168ctx",
    "lead_lag_coupling": "hourly_panel_envelope_168ctx",
    "coherent_regime_shift": "hourly_panel_envelope_168ctx",
    "hierarchical_coherence": "synthetic_structural_guard_v1",
    "covariate_response": "synthetic_structural_guard_v1",
}
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
    "long_memory_nonlinear": {
        **_caps_from_profiles(
            HOURLY_UNIVARIATE_PROFILE_IDS,
            ("nonlinear_lag1_gain", "acf_abs_mean", "spike_rate"),
        ),
        "noise_ratio": 1.0,
    },
    "intermittent_heteroskedastic": {
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
    "lead_lag_coupling": {
        **_caps_from_profiles(
            HOURLY_PANEL_PROFILE_IDS,
            ("lead_lag_peak_abs", "avg_abs_target_corr", "spike_rate"),
        ),
        "noise_ratio": 0.9,
    },
    "coherent_regime_shift": {
        **_caps_from_profiles(
            HOURLY_PANEL_PROFILE_IDS,
            ("level_shift_strength", "avg_abs_target_corr", "spike_rate"),
            multiplier=2.5,
        ),
        "noise_ratio": 0.9,
    },
    "hierarchical_coherence": {
        "hierarchy_residual_mean_abs": 1e-6,
        "noise_ratio": 0.9,
    },
    "covariate_response": {
        "avg_abs_covariate_target_corr": 1.0,
        "future_abs_covariate_target_corr": 1.0,
        "event_lift_abs": 5.0,
        "noise_ratio": 0.95,
        **_caps_from_profiles(HOURLY_UNIVARIATE_PROFILE_IDS, ("spike_rate",)),
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
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None, dict[str, float]]:
    max_attempts = 12 if capability_id in PILOT_ACCEPTANCE_CAPS else 1
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
        )
        target = (
            _standardize_hierarchy_by_context(target, context_length)
            if capability_id == "hierarchical_coherence"
            else _standardize_by_context(target, context_length)
        )
        if covariates is not None and covariates.size:
            covariates = _normalize_covariates(covariates, context_length)
        realized_features = _realized_features(target, covariates, season_length, context_length)
        accepted, failed_features = _accept_synthetic_features(capability_id, realized_features)
        validation = _validation_summary(capability_id, realized_features)
        latent_params = {
            **latent_params,
            "intensity": int(intensity),
            "intensity_definition": "target temporal structure strength; model-error monotonicity is evaluated separately",
            "acceptance": {
                "accepted": bool(accepted),
                "attempts": attempt + 1,
                "failed_features": failed_features,
                "profile": _acceptance_profile_id(capability_id),
                "validation": validation,
            },
        }
        last_result = (target, latent_params, covariates, realized_features)
        if accepted:
            return last_result
    assert last_result is not None
    return last_result


def _accept_synthetic_features(capability_id: str, features: dict[str, float]) -> tuple[bool, list[str]]:
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


def _validation_summary(capability_id: str, features: dict[str, float]) -> dict[str, Any]:
    target_features = TARGET_FEATURES_BY_CAPABILITY.get(capability_id, ())
    control_features = CONTROL_FEATURES_BY_CAPABILITY.get(capability_id, ())
    control_bounds = _control_feature_bounds(capability_id)
    control_results: dict[str, dict[str, float | bool]] = {}
    for feature in control_features:
        value = features.get(feature)
        bounds = control_bounds.get(feature)
        if value is None or bounds is None or not np.isfinite(value):
            continue
        lower = float(bounds.get("p05", float("-inf")))
        upper = float(bounds.get("p95", float("inf")))
        control_results[feature] = {
            "value": float(value),
            "p05": lower,
            "p95": upper,
            "inside_anchor_range": bool(lower <= float(value) <= upper),
        }
    return {
        "schema_version": "synthetic_post_generation_validation.v1",
        "anchor_profile_id": _acceptance_profile_id(capability_id),
        "target_features": {
            feature: float(features[feature])
            for feature in target_features
            if feature in features and np.isfinite(features[feature])
        },
        "control_features": control_results,
        "novelty_check": "offline_dcr_nndr_required",
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
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    if capability_id == "trend":
        return _generate_trend(length, target_dim, season_length, intensity, rng)
    if capability_id == "multi_seasonal":
        return _generate_multi_seasonal(length, target_dim, season_length, intensity, rng)
    if capability_id == "regime_switching":
        return _generate_regime_switching(length, context_length, target_dim, season_length, intensity, rng)
    if capability_id == "time_varying_seasonality":
        return _generate_time_varying_seasonality(length, target_dim, season_length, intensity, rng)
    if capability_id == "long_memory_nonlinear":
        return _generate_long_memory_nonlinear(length, target_dim, season_length, intensity, rng)
    if capability_id == "intermittent_heteroskedastic":
        return _generate_intermittent_heteroskedastic(length, target_dim, season_length, intensity, rng)
    if capability_id == "common_factor":
        return _generate_common_factor(length, target_dim, season_length, intensity, rng)
    if capability_id == "lead_lag_coupling":
        return _generate_lead_lag_coupling(length, target_dim, season_length, intensity, rng)
    if capability_id == "coherent_regime_shift":
        return _generate_coherent_regime_shift(length, context_length, target_dim, season_length, intensity, rng)
    if capability_id == "hierarchical_coherence":
        return _generate_hierarchical_coherence(length, target_dim, season_length, intensity, rng)
    if capability_id == "covariate_response":
        return _generate_covariate_response(length, target_dim, season_length, intensity, rng)
    raise ApiError("synthetic_capability_unknown", "unknown synthetic capability", {"capability_id": capability_id}, 404)


def _base_features(length: int, season_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(length, dtype=float)
    seasonal = np.sin(2 * np.pi * t / max(4, season_length))
    slow = np.cos(2 * np.pi * t / max(8, season_length * 4))
    trend = np.linspace(-1.0, 1.0, length)
    return seasonal, slow, trend


def _generate_trend(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    seasonal, slow, trend = _base_features(length, season_length)
    slope_direction = rng.choice(np.asarray([-1.0, 1.0]), size=target_dim)
    curvature_direction = rng.choice(np.asarray([-1.0, 1.0]), size=target_dim)
    slope = slope_direction * (0.02 + 0.18 * lam) * rng.uniform(0.8, 1.2, size=target_dim)
    curvature = curvature_direction * (0.02 + 0.16 * lam) * rng.uniform(0.5, 1.1, size=target_dim)
    seasonal_amp = max(0.12, 0.45 - 0.10 * lam)
    noise_scale = max(0.04, 0.12 - 0.04 * lam)
    values = trend[:, None] * slope + ((trend[:, None] ** 2) - 0.35) * curvature
    values += seasonal_amp * seasonal[:, None] + 0.08 * slow[:, None]
    values += rng.normal(0.0, noise_scale, size=(length, target_dim))
    return (
        values,
        {
            "generator_version": "v2-pilot",
            "anchor_profile": "m4_hourly_daily_168ctx",
            "slope_mean": float(np.mean(slope)),
            "slope_abs_mean": float(np.mean(np.abs(slope))),
            "curvature_mean": float(np.mean(curvature)),
            "curvature_abs_mean": float(np.mean(np.abs(curvature))),
            "seasonal_amplitude": float(seasonal_amp),
            "noise_scale": float(noise_scale),
        },
        None,
    )


def _generate_multi_seasonal(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    t = np.arange(length, dtype=float)
    primary_period = max(4, season_length)
    secondary_period = max(primary_period + 1, primary_period * 2)
    tertiary_period = max(4, primary_period // 2)
    values = np.zeros((length, target_dim))
    period_amplitudes: list[dict[str, float]] = []

    def add_period(period: int, amplitude: np.ndarray) -> None:
        nonlocal values
        phase = rng.uniform(0, 2 * np.pi, size=target_dim)
        values += amplitude[None, :] * np.sin(2 * np.pi * t[:, None] / period + phase[None, :])
        period_amplitudes.append({"period": float(period), "amplitude_mean": float(np.mean(amplitude))})

    amp = rng.uniform(0.9, 1.1, size=target_dim)
    add_period(primary_period, amp)
    if lam > 0:
        amp = 0.7 * lam * rng.uniform(0.8, 1.2, size=target_dim)
        add_period(secondary_period, amp)
    if lam > 0.5:
        amp = 0.3 * ((lam - 0.5) * 2.0) * rng.uniform(0.8, 1.2, size=target_dim)
        add_period(tertiary_period, amp)
    slow_period = max(primary_period * 7, primary_period + 1)
    slow_phase = rng.uniform(0, 2 * np.pi, size=target_dim)
    values += 0.05 * np.cos(2 * np.pi * t[:, None] / slow_period + slow_phase[None, :])
    noise_scale = max(0.04, 0.10 - 0.03 * lam)
    values += rng.normal(0.0, noise_scale, size=values.shape)
    return (
        values,
        {
            "generator_version": "v2-pilot",
            "anchor_profile": "m4_hourly_daily_168ctx",
            "periods": [int(item["period"]) for item in period_amplitudes],
            "period_amplitudes": period_amplitudes,
            "secondary_amplitude_ratio": float(lam),
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
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    seasonal, slow, _ = _base_features(length, season_length)
    switch_count = max(1, int(round(1 + lam * 4)))
    random_pool = np.arange(4, max(5, length - 4))
    cut_points = set(
        rng.choice(random_pool, size=min(switch_count, max(1, length - 8), len(random_pool)), replace=False).tolist()
    )
    if context_length < length:
        cut_points.add(int(rng.integers(context_length, length)))
    cut_points = sorted(point for point in cut_points if 0 < point < length)
    levels = rng.normal(0.0, 0.8 + 0.6 * lam, size=(len(cut_points) + 1, target_dim))
    values = np.zeros((length, target_dim))
    boundaries = [0, *cut_points, length]
    for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        scale = 0.08 + 0.06 * segment + 0.08 * lam
        values[start:end] = levels[segment] + 0.35 * seasonal[start:end, None] + 0.18 * slow[start:end, None]
        values[start:end] += rng.normal(0.0, scale, size=(end - start, target_dim))
    return values, {"switch_count": len(cut_points), "cut_points": cut_points, "forecast_switch": int(any(point >= context_length for point in cut_points))}, None


def _generate_time_varying_seasonality(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    t = np.arange(length, dtype=float)
    primary_period = max(4, season_length)
    slow_period = max(primary_period * 5, primary_period + 1)
    drift = t / max(1, length - 1)
    phase_drift = (0.05 + 0.45 * lam) * (drift ** 1.35) * 2 * np.pi
    amplitude_start = 0.55 + 0.1 * rng.random(target_dim)
    amplitude_delta = (0.2 + 1.0 * lam) * rng.uniform(0.8, 1.2, size=target_dim)
    amplitude = amplitude_start[None, :] + amplitude_delta[None, :] * drift[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=target_dim)
    values = amplitude * np.sin(2 * np.pi * t[:, None] / primary_period + phase[None, :] + phase_drift[:, None])
    values += 0.18 * np.cos(2 * np.pi * t[:, None] / slow_period + phase[None, :] / 2)
    values += 0.08 * drift[:, None]
    noise_scale = max(0.04, 0.11 - 0.03 * lam)
    values += rng.normal(0.0, noise_scale, size=(length, target_dim))
    return (
        values,
        {
            "generator_version": "v2-pilot",
            "amplitude_delta_mean": float(np.mean(amplitude_delta)),
            "phase_drift_cycles": float(np.max(phase_drift) / (2 * np.pi)),
            "noise_scale": float(noise_scale),
        },
        None,
    )


def _generate_long_memory_nonlinear(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    seasonal, slow, _ = _base_features(length, season_length)
    values = np.zeros((length, target_dim))
    phi = 0.65 + 0.22 * lam
    nonlinear = 0.15 + 0.35 * lam
    values[0] = rng.normal(0.0, 0.3, size=target_dim)
    for idx in range(1, length):
        values[idx] = (
            phi * values[idx - 1]
            + nonlinear * np.sin(values[idx - 1])
            + 0.18 * seasonal[idx]
            + 0.08 * slow[idx]
            + rng.normal(0.0, 0.08 + 0.08 * lam, size=target_dim)
        )
    return values, {"ar_phi": float(phi), "nonlinear_strength": float(nonlinear)}, None


def _generate_intermittent_heteroskedastic(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    seasonal, _, trend = _base_features(length, season_length)
    event_prob = 0.04 + 0.1 * lam
    bursts = rng.random((length, target_dim)) < event_prob
    burst_size = rng.gamma(shape=1.5 + lam, scale=0.8 + lam, size=(length, target_dim)) * bursts
    volatility = 0.08 + 0.22 * lam * (np.sin(np.arange(length) / max(3, season_length / 3)) + 1.3)
    values = 0.15 * trend[:, None] + 0.25 * seasonal[:, None] + burst_size
    values += rng.normal(0.0, volatility[:, None], size=(length, target_dim))
    return values, {"event_probability": float(event_prob), "burst_count": int(bursts.sum())}, None


def _generate_common_factor(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    seasonal, slow, trend = _base_features(length, season_length)
    ar = np.zeros(length)
    for idx in range(1, length):
        ar[idx] = 0.82 * ar[idx - 1] + rng.normal(0.0, 0.2 + 0.1 * lam)
    rank = min(target_dim, 2 + int(lam >= 0.5))
    factors = np.vstack([seasonal, slow, trend, ar][:rank]).T
    loadings = rng.normal(0.0, 1.0, size=(target_dim, rank))
    values = factors @ loadings.T + rng.normal(0.0, 0.12 + 0.12 * lam, size=(length, target_dim))
    return values, {"factor_rank": rank, "noise_scale": float(0.12 + 0.12 * lam)}, None


def _generate_lead_lag_coupling(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    max_lag = max(1, min(max(2, season_length // 3), 8 + int(10 * lam)))
    values, _, _ = _generate_common_factor(length + max_lag, target_dim, season_length, intensity, rng)
    weights = rng.uniform(0.15, 0.45 + 0.25 * lam, size=target_dim)
    lags = rng.integers(1, max_lag + 1, size=target_dim)
    coupled = values.copy()
    for channel in range(1, target_dim):
        leader = (channel - 1) % target_dim
        lag = int(lags[channel])
        coupled[lag:, channel] += weights[channel] * values[:-lag, leader]
    coupled = coupled[max_lag:]
    return coupled, {"max_lag": int(max_lag), "coupling_strength_mean": float(np.mean(weights))}, None


def _generate_coherent_regime_shift(
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    seasonal, slow, trend = _base_features(length, season_length)
    shift_low = min(max(1, int(context_length)), length - 1)
    shift_at = int(rng.integers(shift_low, length)) if shift_low < length else length - 1
    direction = rng.normal(0.0, 0.8 + 1.2 * lam, size=target_dim)
    common = 0.5 * seasonal + 0.2 * slow + 0.15 * trend
    values = common[:, None] + rng.normal(0.0, 0.1 + 0.08 * lam, size=(length, target_dim))
    values[shift_at:] += direction
    return values, {"shift_at": shift_at, "shift_norm": float(np.linalg.norm(direction))}, None


def _generate_hierarchical_coherence(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _intensity_lambda(intensity)
    seasonal, slow, trend = _base_features(length, season_length)
    t = np.arange(length, dtype=float)
    child_count = max(2, target_dim - 1)
    children = np.zeros((length, child_count))
    for child in range(child_count):
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.45, 0.75) * (1.0 + 0.25 * lam)
        local_period = max(4, season_length + int((child - child_count / 2) * max(1, season_length // 6)))
        children[:, child] = (
            amplitude * np.sin(2 * np.pi * t / local_period + phase)
            + (0.18 + 0.12 * lam) * slow
            + rng.normal(0.0, 0.06 + 0.03 * lam, size=length)
        )
    shock_count = int(round(lam * 3))
    for _ in range(shock_count):
        start = int(rng.integers(max(2, season_length // 2), max(3, length - 2)))
        width = int(rng.integers(2, max(3, min(10, season_length))))
        direction = rng.normal(0.0, 0.2 + 0.35 * lam, size=child_count)
        children[start : min(length, start + width)] += direction[None, :]
    children += (0.03 + 0.08 * lam) * trend[:, None] * rng.uniform(0.7, 1.3, size=child_count)
    parent = np.sum(children, axis=1, keepdims=True)
    values = np.concatenate([parent, children], axis=1)
    if values.shape[1] > target_dim:
        values = values[:, :target_dim]
    return (
        values,
        {
            "generator_version": "v2-pilot",
            "hierarchy": "target_0=sum(target_1:)",
            "child_count": int(child_count),
            "shock_count": int(shock_count),
            "coherence_residual_mean_abs": float(np.mean(np.abs(parent[:, 0] - np.sum(children, axis=1)))),
        },
        None,
    )


def _generate_covariate_response(
    length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    lam = _intensity_lambda(intensity)
    t = np.arange(length, dtype=float)
    weather = np.sin(2 * np.pi * t / max(8, season_length * 2)) + rng.normal(0.0, 0.08, size=length)
    event = np.zeros(length)
    start_pool = np.arange(max(2, season_length // 2), max(3, length - 2))
    if len(start_pool):
        event_count = max(1, min(len(start_pool), int(round(1 + lam * 3))))
        for start in rng.choice(start_pool, size=event_count, replace=False):
            width = int(rng.integers(2, max(3, min(10, season_length))))
            event[int(start) : min(length, int(start) + width)] = 1.0
    covariates = np.stack([weather, event], axis=1)
    seasonal, slow, trend = _base_features(length, season_length)
    beta_weather = rng.uniform(-0.5, 0.8, size=target_dim) * (0.7 + lam)
    beta_event = rng.uniform(0.4, 1.6, size=target_dim) * (0.8 + lam)
    values = 0.35 * seasonal[:, None] + 0.18 * slow[:, None] + 0.1 * trend[:, None]
    values = values + weather[:, None] * beta_weather + event[:, None] * beta_event
    values += rng.normal(0.0, 0.1 + 0.08 * lam, size=(length, target_dim))
    return (
        values,
        {
            "future_covariate_dim": 2,
            "weather_effect_mean": float(np.mean(np.abs(beta_weather))),
            "event_effect_mean": float(np.mean(np.abs(beta_event))),
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
    if covariates is not None and covariates.size:
        features.update(_covariate_profile_features(target, covariates, context_length))
    return features


def _structural_univariate_features(values: np.ndarray, season_length: int) -> dict[str, float]:
    y = _robust_scale(np.asarray(values, dtype=float))
    n = y.size
    if n < 12:
        return {}
    min_seg = max(6, min(24, n // 8))
    level_scores: list[float] = []
    volatility_scores: list[float] = []
    std_all = float(np.std(y)) or 1.0
    for cut in range(min_seg, n - min_seg):
        left = y[:cut]
        right = y[cut:]
        level_scores.append(abs(float(np.mean(left) - np.mean(right))) / std_all)
        volatility_scores.append(abs(float(np.std(left) - np.std(right))) / std_all)
    seasonal_profile = _phase_profile(y, season_length)
    half = max(1, n // 2)
    seasonal_left = _phase_profile(y[:half], season_length)
    seasonal_right = _phase_profile(y[half:], season_length)
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


def _covariate_profile_features(target: np.ndarray, covariates: np.ndarray, context_length: int) -> dict[str, float]:
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
    return {
        "avg_abs_covariate_target_corr": float(np.mean(scores)) if scores else 0.0,
        "future_abs_covariate_target_corr": float(np.mean(future_scores)) if future_scores else 0.0,
        "event_lift_abs": float(np.mean(event_lifts)) if event_lifts else 0.0,
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
