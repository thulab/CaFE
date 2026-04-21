from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FamilyOutput:
    values: np.ndarray
    latent_params: dict[str, float]


def _noise_scale(anchor_features: dict[str, float], control_lambda: float) -> float:
    base = 0.08 + 0.6 * anchor_features.get("spectral_entropy", 0.5)
    return float(base * (0.8 + control_lambda))


def generate_trend(length: int, season_length: int, control_lambda: float, rng: np.random.Generator, anchor_features: dict[str, float]) -> FamilyOutput:
    t = np.arange(length, dtype=float)
    slope = rng.uniform(-0.015, 0.03) * (1 + 2.8 * control_lambda)
    changepoint = int(length * rng.uniform(0.45, 0.8))
    secondary = slope * rng.uniform(-1.4, -0.2)
    trend = slope * t
    trend[changepoint:] += secondary * np.arange(length - changepoint)
    damp = np.exp(-np.maximum(0, t - changepoint / 2) / max(12, season_length))
    seasonal = 0.2 * np.sin(2 * np.pi * t / max(season_length, 4))
    noise = rng.normal(0.0, _noise_scale(anchor_features, control_lambda), size=length)
    values = 2.0 + trend * damp + seasonal + noise
    return FamilyOutput(values=values, latent_params={"slope": slope, "secondary_slope": secondary, "changepoint": changepoint})


def generate_multi_seasonal(length: int, season_length: int, control_lambda: float, rng: np.random.Generator, anchor_features: dict[str, float]) -> FamilyOutput:
    t = np.arange(length, dtype=float)
    s1 = max(4, season_length)
    s2 = int(np.clip(round(s1 * rng.uniform(1.5, 3.5)), s1 + 1, 96))
    amp_drift = 1 + control_lambda * np.sin(2 * np.pi * t / max(length // 2, 8))
    phase_drift = control_lambda * np.pi * np.linspace(0, 1, length)
    signal = amp_drift * np.sin(2 * np.pi * t / s1 + phase_drift) + 0.65 * np.cos(2 * np.pi * t / s2)
    noise = rng.normal(0.0, _noise_scale(anchor_features, control_lambda) * 0.85, size=length)
    values = signal + noise
    return FamilyOutput(values=values, latent_params={"season_length_1": s1, "season_length_2": s2, "phase_drift": float(phase_drift[-1])})


def generate_regime_switching(length: int, season_length: int, control_lambda: float, rng: np.random.Generator, anchor_features: dict[str, float]) -> FamilyOutput:
    values = np.zeros(length)
    state = 0
    switch_prob = float(np.clip(0.015 + control_lambda * 0.08, 0.01, 0.2))
    params = [(0.15, -0.6, 0.25), (0.86, 1.0 + control_lambda, 0.4 + 0.4 * control_lambda)]
    switches = 0
    for idx in range(1, length):
        if rng.random() < switch_prob:
            state = 1 - state
            switches += 1
        phi, mu, sigma = params[state]
        values[idx] = mu + phi * values[idx - 1] + rng.normal(0.0, sigma)
    if season_length > 1:
        values += 0.15 * np.sin(2 * np.pi * np.arange(length) / season_length)
    return FamilyOutput(values=values, latent_params={"switch_prob": switch_prob, "switches": switches})


def generate_long_memory_nonlinear(length: int, season_length: int, control_lambda: float, rng: np.random.Generator, anchor_features: dict[str, float]) -> FamilyOutput:
    delay = int(np.clip(round(max(8, season_length) * (0.7 + control_lambda)), 8, 48))
    beta = 0.16 + 0.08 * control_lambda
    gamma = 0.08 + 0.02 * control_lambda
    power = int(np.clip(round(6 + 4 * control_lambda), 5, 12))
    values = np.ones(length + delay + 1) * 1.2
    for idx in range(delay, len(values) - 1):
        delayed = values[idx - delay]
        drift = beta * delayed / (1 + delayed**power) - gamma * values[idx]
        values[idx + 1] = values[idx] + drift + rng.normal(0.0, _noise_scale(anchor_features, control_lambda) * 0.15)
    values = values[delay + 1 :]
    values += 0.05 * np.sin(2 * np.pi * np.arange(length) / max(season_length, 4))
    return FamilyOutput(values=values, latent_params={"delay": delay, "beta": beta, "gamma": gamma, "power": power})


def generate_intermittent_heteroskedastic(length: int, season_length: int, control_lambda: float, rng: np.random.Generator, anchor_features: dict[str, float]) -> FamilyOutput:
    zero_prob = float(np.clip(0.45 + 0.4 * control_lambda, 0.3, 0.92))
    burst_scale = 1.5 + 4.0 * control_lambda
    demand = rng.poisson(1.2, size=length).astype(float)
    demand[rng.random(length) < zero_prob] = 0.0
    vol = np.ones(length) * 0.2
    eps = rng.normal(0.0, 1.0, size=length)
    for idx in range(1, length):
        vol[idx] = 0.05 + 0.22 * abs(eps[idx - 1]) + (0.55 + 0.25 * control_lambda) * vol[idx - 1]
    bursts = rng.random(length) < (0.03 + 0.08 * control_lambda)
    values = demand + bursts * rng.gamma(shape=2.0, scale=burst_scale, size=length) + eps * vol
    if season_length > 1:
        values += 0.2 * np.maximum(0, np.sin(2 * np.pi * np.arange(length) / season_length))
    return FamilyOutput(values=values, latent_params={"zero_prob": zero_prob, "burst_scale": burst_scale})


FAMILY_GENERATORS = {
    "trend": generate_trend,
    "multi_seasonal": generate_multi_seasonal,
    "regime_switching": generate_regime_switching,
    "long_memory_nonlinear": generate_long_memory_nonlinear,
    "intermittent_heteroskedastic": generate_intermittent_heteroskedastic,
}

