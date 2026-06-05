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
    difficulty: int
    season_length: int
    target_dim: int
    seed: int
    frequency: str = "h"


SYNTHETIC_CAPABILITIES: tuple[SyntheticCapability, ...] = (
    SyntheticCapability(
        "trend",
        "Trend",
        "Single-target series with controllable trend and seasonal residue.",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "multi_seasonal",
        "Multi-seasonal",
        "Single-target series with multiple overlapping seasonal periods.",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "regime_switching",
        "Regime switching",
        "Single-target series with level and volatility changes.",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "long_memory_nonlinear",
        "Long-memory nonlinear",
        "Single-target autoregressive dynamics with nonlinear carry-over.",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "intermittent_heteroskedastic",
        "Intermittent heteroskedastic",
        "Single-target sparse bursts with changing noise scale.",
        "univariate_forecast",
        "fixed_1",
    ),
    SyntheticCapability(
        "common_factor",
        "Common factor",
        "Multiple targets driven by shared latent factors.",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "lead_lag_coupling",
        "Lead-lag coupling",
        "Multiple targets with lagged cross-channel dependencies.",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "coherent_regime_shift",
        "Coherent regime shift",
        "Multiple targets that shift together across regimes.",
        "multivariate_forecast",
        "multi",
    ),
    SyntheticCapability(
        "covariate_response",
        "Covariate response",
        "Targets whose future depends on known weather and event covariates.",
        "covariate_forecast",
        "covariate",
        ("weather", "event"),
    ),
)

CAPABILITIES_BY_ID: dict[str, SyntheticCapability] = {
    capability.capability_id: capability for capability in SYNTHETIC_CAPABILITIES
}

MOCK_ANCHOR = {
    "anchor_mode": "fixed_mock",
    "anchor_source_uri": "synthetic-anchor://builtin/mock-v1",
}


def synthetic_capability_catalog() -> list[dict[str, Any]]:
    return [
        {
            "capability_id": capability.capability_id,
            "label": capability.label,
            "description": capability.description,
            "task_type": capability.task_type,
            "target_dim_mode": capability.target_dim_mode,
            "covariate_columns": list(capability.covariate_columns),
            "default_params": {
                "context_length": 60,
                "horizon": 16,
                "sample_count": 32,
                "difficulty": 3,
                "season_length": 24,
                "target_dim": 3 if capability.target_dim_mode in {"multi", "covariate"} else 1,
                "frequency": "h",
            },
            "limits": {
                "context_length": {"min": 8, "max": 2048},
                "horizon": {"min": 1, "max": 512},
                "sample_count": {"min": 1, "max": 1000},
                "difficulty": {"min": 1, "max": 5},
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
    if config.context_length < 8:
        raise ApiError("synthetic_context_too_short", "context_length must be at least 8")
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
    if not 1 <= config.difficulty <= 5:
        raise ApiError("synthetic_difficulty_invalid", "difficulty must be between 1 and 5")


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
        rng = np.random.default_rng(_seed_for(config.seed, capability.capability_id, sample_index))
        target, latent_params, covariates = _generate_sample_values(
            capability.capability_id,
            sample_length,
            target_dim,
            config.season_length,
            config.difficulty,
            rng,
        )
        target = _standardize_by_context(target, context)
        if covariates is not None and covariates.size:
            covariates = _normalize_covariates(covariates, context)
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
                "difficulty": config.difficulty,
                "seed": config.seed,
                "sample_seed": _seed_for(config.seed, capability.capability_id, sample_index),
                "latent_params": latent_params,
                "realized_features": _realized_features(target, covariates),
                **MOCK_ANCHOR,
            }
        )

    generation_config = {
        "schema_version": "synthetic_generation.v1",
        "generation_id": generation_id,
        "capability_id": capability.capability_id,
        "capability_label": capability.label,
        "task_type": capability.task_type,
        "difficulty": config.difficulty,
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


def _generate_sample_values(
    capability_id: str,
    length: int,
    target_dim: int,
    season_length: int,
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    if capability_id == "trend":
        return _generate_trend(length, target_dim, season_length, difficulty, rng)
    if capability_id == "multi_seasonal":
        return _generate_multi_seasonal(length, target_dim, season_length, difficulty, rng)
    if capability_id == "regime_switching":
        return _generate_regime_switching(length, target_dim, season_length, difficulty, rng)
    if capability_id == "long_memory_nonlinear":
        return _generate_long_memory_nonlinear(length, target_dim, season_length, difficulty, rng)
    if capability_id == "intermittent_heteroskedastic":
        return _generate_intermittent_heteroskedastic(length, target_dim, season_length, difficulty, rng)
    if capability_id == "common_factor":
        return _generate_common_factor(length, target_dim, season_length, difficulty, rng)
    if capability_id == "lead_lag_coupling":
        return _generate_lead_lag_coupling(length, target_dim, season_length, difficulty, rng)
    if capability_id == "coherent_regime_shift":
        return _generate_coherent_regime_shift(length, target_dim, season_length, difficulty, rng)
    if capability_id == "covariate_response":
        return _generate_covariate_response(length, target_dim, season_length, difficulty, rng)
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
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
    seasonal, slow, trend = _base_features(length, season_length)
    slope = rng.uniform(-1.2, 1.2, size=target_dim) * (0.6 + lam)
    curvature = rng.uniform(-0.7, 0.7, size=target_dim) * lam
    values = trend[:, None] * slope + (trend[:, None] ** 2) * curvature
    values += (0.25 + 0.2 * lam) * seasonal[:, None] + 0.12 * slow[:, None]
    values += rng.normal(0.0, 0.08 + 0.08 * lam, size=(length, target_dim))
    return values, {"slope_mean": float(np.mean(slope)), "curvature_mean": float(np.mean(curvature))}, None


def _generate_multi_seasonal(
    length: int,
    target_dim: int,
    season_length: int,
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
    t = np.arange(length, dtype=float)
    periods = [max(4, season_length), max(5, season_length // 2), max(8, season_length * 2)]
    values = np.zeros((length, target_dim))
    amplitudes = []
    for period in periods:
        amp = rng.uniform(0.2, 0.8 + 0.4 * lam, size=target_dim)
        phase = rng.uniform(0, 2 * np.pi, size=target_dim)
        values += amp[None, :] * np.sin(2 * np.pi * t[:, None] / period + phase[None, :])
        amplitudes.append(float(np.mean(amp)))
    values += rng.normal(0.0, 0.08 + 0.09 * lam, size=values.shape)
    return values, {"periods": periods, "amplitude_mean": float(np.mean(amplitudes))}, None


def _generate_regime_switching(
    length: int,
    target_dim: int,
    season_length: int,
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
    seasonal, slow, _ = _base_features(length, season_length)
    switch_count = max(1, int(round(1 + lam * 4)))
    cut_points = sorted(rng.choice(np.arange(4, max(5, length - 4)), size=min(switch_count, max(1, length - 8)), replace=False))
    levels = rng.normal(0.0, 0.8 + 0.6 * lam, size=(len(cut_points) + 1, target_dim))
    values = np.zeros((length, target_dim))
    boundaries = [0, *cut_points, length]
    for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=True)):
        scale = 0.08 + 0.06 * segment + 0.08 * lam
        values[start:end] = levels[segment] + 0.35 * seasonal[start:end, None] + 0.18 * slow[start:end, None]
        values[start:end] += rng.normal(0.0, scale, size=(end - start, target_dim))
    return values, {"switch_count": len(cut_points), "cut_points": cut_points}, None


def _generate_long_memory_nonlinear(
    length: int,
    target_dim: int,
    season_length: int,
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
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
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
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
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
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
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
    max_lag = max(1, min(max(2, season_length // 3), 8 + int(10 * lam)))
    values, _, _ = _generate_common_factor(length + max_lag, target_dim, season_length, difficulty, rng)
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
    target_dim: int,
    season_length: int,
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], None]:
    lam = _difficulty_lambda(difficulty)
    seasonal, slow, trend = _base_features(length, season_length)
    shift_at = int(rng.integers(max(3, length // 3), max(4, 2 * length // 3)))
    direction = rng.normal(0.0, 0.8 + 1.2 * lam, size=target_dim)
    common = 0.5 * seasonal + 0.2 * slow + 0.15 * trend
    values = common[:, None] + rng.normal(0.0, 0.1 + 0.08 * lam, size=(length, target_dim))
    values[shift_at:] += direction
    return values, {"shift_at": shift_at, "shift_norm": float(np.linalg.norm(direction))}, None


def _generate_covariate_response(
    length: int,
    target_dim: int,
    season_length: int,
    difficulty: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    lam = _difficulty_lambda(difficulty)
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


def _difficulty_lambda(difficulty: int) -> float:
    return (int(difficulty) - 1) / 4


def _standardize_by_context(values: np.ndarray, context_length: int) -> np.ndarray:
    context = values[:context_length]
    mean = context.mean(axis=0, keepdims=True)
    std = context.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (values - mean) / std


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


def _realized_features(target: np.ndarray, covariates: np.ndarray | None) -> dict[str, float]:
    features = {
        "target_mean_abs": float(np.mean(np.abs(target))),
        "target_std_mean": float(np.mean(target.std(axis=0))),
        "target_max_abs": float(np.max(np.abs(target))),
    }
    if target.shape[1] > 1:
        corr = np.nan_to_num(np.corrcoef(target.T), nan=0.0)
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        features["avg_abs_target_corr"] = float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0
    if covariates is not None and covariates.size:
        scores: list[float] = []
        for cov_idx in range(covariates.shape[1]):
            for target_idx in range(target.shape[1]):
                corr_value = np.corrcoef(covariates[:, cov_idx], target[:, target_idx])[0, 1]
                if np.isfinite(corr_value):
                    scores.append(abs(float(corr_value)))
        features["avg_abs_covariate_target_corr"] = float(np.mean(scores)) if scores else 0.0
    return features


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
