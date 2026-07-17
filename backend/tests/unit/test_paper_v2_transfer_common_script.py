from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "paper_v2_transfer_common.py"


def load_module():
    spec = importlib.util.spec_from_file_location("paper_v2_transfer_common", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transfer_profile_universe_is_unique_and_excludes_development_families() -> None:
    module = load_module()
    profiles = module.TRANSFER_PROFILE_SPECS

    assert len(profiles) == 9
    assert len({profile.profile_id for profile in profiles}) == len(profiles)
    assert {profile.frequency for profile in profiles} == {"h"}
    assert all(
        profile.synthetic_capabilities == module.PAPER_UNIVARIATE_CAPABILITY_IDS
        for profile in profiles
    )
    assert not {
        "hospital",
        "jena_weather",
        "bizitobs_l2c",
        "electricity",
        "m4_hourly",
        "traffic",
    } & {profile.family_id for profile in profiles}


def test_transfer_profile_shapes_and_measurement_horizon_are_frozen() -> None:
    module = load_module()

    for profile in module.TRANSFER_PROFILE_SPECS:
        assert (
            profile.context_length,
            profile.horizon,
            profile.season_length,
            profile.feature_measurement_horizon,
        ) == (504, 48, 24, 24)


def test_impute_observed_window_uses_only_values_inside_window() -> None:
    module = load_module()
    values = np.asarray([np.nan, 1.0, np.nan, 3.0, np.nan])

    imputed, observed_fraction = module.impute_observed_window(
        values,
        minimum_observed_fraction=0.4,
    )

    assert observed_fraction == pytest.approx(0.4)
    assert imputed == pytest.approx([1.0, 1.0, 2.0, 3.0, 3.0])
    rejected, _ = module.impute_observed_window(values)
    assert rejected is None


def test_primary_feature_maps_continuously_to_canonical_intensity() -> None:
    module = load_module()
    targets = [0.1, 0.2, 0.4, 0.7, 1.0]

    assert module.primary_feature_intensity_coordinate(0.05, targets) == 1.0
    assert module.primary_feature_intensity_coordinate(1.2, targets) == 5.0
    assert module.primary_feature_intensity_coordinate(0.3, targets) == pytest.approx(
        2.5
    )
