from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "synthetic_capability_qualification.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "synthetic_capability_qualification",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_regime_clock_requires_a_history_learnable_schedule_that_continues_in_future():
    module = load_module()
    context_length = 420
    horizon = 60
    season_length = 60
    time = np.arange(context_length + horizon, dtype=float)
    state = (np.mod(time, season_length) >= season_length / 2).astype(float)
    noise = np.random.default_rng(7).normal(scale=0.08, size=len(time))
    target = 0.25 * np.sin(2 * math.pi * time / season_length) + 2.0 * state + noise

    recurring = module.regime_clock_features(
        target,
        context_length=context_length,
        season_length=season_length,
    )
    broken_future = module.regime_clock_features(
        np.concatenate([target[:context_length], np.full(horizon, target[:context_length].mean())]),
        context_length=context_length,
        season_length=season_length,
    )

    assert recurring["qualified"] is True
    assert recurring["selected_period"] == season_length
    assert broken_future["qualified"] is False
    assert broken_future["future_state_coverage"] < recurring["future_state_coverage"]


def test_regime_clock_rejects_one_off_changes_and_sparse_pulses():
    module = load_module()
    context_length = 420
    horizon = 60
    season_length = 60
    length = context_length + horizon
    time = np.arange(length)
    one_off_change = np.concatenate([np.zeros(300), np.ones(length - 300)])
    sparse_pulses = (np.mod(time, season_length) == 0).astype(float) * 5.0
    sparse_pulses += np.random.default_rng(11).normal(scale=0.05, size=length)

    change_audit = module.regime_clock_features(
        one_off_change,
        context_length=context_length,
        season_length=season_length,
    )
    pulse_audit = module.regime_clock_features(
        sparse_pulses,
        context_length=context_length,
        season_length=season_length,
    )

    assert change_audit["qualified"] is False
    assert pulse_audit["qualified"] is False
    assert pulse_audit["context_absolute_skew"] > module.REGIME_CONTEXT_ABSOLUTE_SKEW_MAX
