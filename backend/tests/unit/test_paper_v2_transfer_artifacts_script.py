from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).parents[3] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "build_paper_v2_transfer_artifacts.py"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_paper_v2_transfer_artifacts",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_measurement_rows_use_one_period_not_full_horizon(
    monkeypatch,
) -> None:
    module = load_module()
    spec = module.transfer_profile_specs(["gift_solar_h_504ctx_48h"])[0]
    observed_shapes: list[tuple[int, int]] = []

    def realized(target, covariates, season_length, context_length):
        observed_shapes.append(target.shape)
        assert covariates is None
        assert season_length == 24
        assert context_length == 504
        return {"marker": float(target[-1, 0])}

    monkeypatch.setattr(module, "_realized_features", realized)
    target = np.arange(552, dtype=float)[:, None]

    measured = module.canonical_measurement_rows(
        [{"target": target, "covariates": None, "features": {"old": 1.0}}],
        spec,
    )

    assert observed_shapes == [(528, 1)]
    assert measured[0]["features"] == {"marker": 527.0}
    assert measured[0]["target"].shape == (552, 1)


def test_sample_evenly_is_deterministic_and_keeps_endpoints() -> None:
    module = load_module()
    assert module.sample_evenly(list(range(10)), 4) == [0, 3, 6, 9]
