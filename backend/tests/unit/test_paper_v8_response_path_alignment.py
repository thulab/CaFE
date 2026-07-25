from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_common():
    path = SCRIPT_DIR / "paper_v8_pipeline_common.py"
    spec = importlib.util.spec_from_file_location(
        "paper_v8_pipeline_common_alignment_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_response_curve_uses_formal_logical_seed_anchor_bank(monkeypatch):
    common = load_common()
    dataset = common.resolve_dataset("gift_electricity_h")
    anchors = [{"marker": index} for index in range(4)]
    selected: list[int] = []

    def anchor_for_seed(
        values,
        *,
        dataset_id,
        capability_id,
        seed_index,
    ):
        assert values is anchors
        assert dataset_id == dataset.dataset_id
        assert capability_id == "trend"
        selected.append(seed_index)
        return values[seed_index]

    def calibration_member(
        _dataset,
        anchor,
        *,
        capability_id,
        family_role,
        lambda_value,
        calibration_seed_index,
    ):
        assert capability_id == "trend"
        assert family_role == "primary"
        assert anchor["marker"] == calibration_seed_index
        return {
            common.PRIMARY_TARGET_FEATURE[capability_id]: lambda_value
        }, {}

    monkeypatch.setattr(common, "anchor_for_seed", anchor_for_seed)
    monkeypatch.setattr(
        common,
        "generate_calibration_member",
        calibration_member,
    )

    _grid, _response, audit = common.monotone_response_curve(
        dataset,
        anchors,
        capability_id="trend",
        family_role="primary",
        calibration_seed_count=4,
    )

    assert selected == [0, 1, 2, 3]
    assert audit["path_anchor_policy"] == "formal_logical_seed_hash_v1"
    assert audit["path_rng_policy"] == "formal_generation_path_v1"


def test_calibration_member_reuses_formal_generation_path_rng(monkeypatch):
    common = load_common()
    dataset = common.resolve_dataset("gift_electricity_h")
    observed: dict[str, float] = {}

    monkeypatch.setattr(
        common,
        "derive_deterministic_parameters",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr(
        common,
        "build_conditioning",
        lambda *args, **kwargs: SimpleNamespace(
            target_dim=1,
            season_length=24,
        ),
    )

    def generate_sample(
        capability_id,
        length,
        context,
        target_dim,
        season_length,
        intensity,
        rng,
        **kwargs,
    ):
        del (
            capability_id,
            context,
            season_length,
            intensity,
            kwargs,
        )
        observed["first_random"] = float(rng.random())
        return np.zeros((length, target_dim)), {}, None

    monkeypatch.setattr(
        common,
        "generate_deterministic_sample",
        generate_sample,
    )
    monkeypatch.setattr(
        common,
        "standardize_generated_sample",
        lambda _capability, target, covariates, **kwargs: (
            target,
            covariates,
        ),
    )
    monkeypatch.setattr(
        common,
        "measured_features",
        lambda *args, **kwargs: {"curvature_abs": 0.1},
    )
    seed_index = 7

    common.generate_calibration_member(
        dataset,
        {
            "features": {},
            "feature_period": 24,
            "frequency": "H",
        },
        capability_id="trend",
        family_role="primary",
        lambda_value=0.5,
        calibration_seed_index=seed_index,
    )

    expected_rng = np.random.default_rng(
        common.stable_seed(
            dataset.dataset_id,
            "trend",
            seed_index,
            "generation-path",
            base=common.GENERATION_PATH_SEED,
        )
    )
    assert observed["first_random"] == float(expected_rng.random())


def test_inverse_branch_excludes_raw_foldbacks_instead_of_enveloping_them():
    common = load_common()
    grid = np.arange(7, dtype=float) * 0.05
    response = np.asarray(
        [
            0.00042867889306655305,
            0.000426572021469826,
            0.00042796865025135356,
            0.000433542821524335,
            0.0004361920429278368,
            0.0004241030018258315,
            0.00043587856418443247,
        ]
    )

    start, end, audit = common.raw_increasing_response_branch(
        grid,
        response,
        support_index=6,
    )

    assert (start, end) == (1, 4)
    assert audit["selected_point_count"] == 4
    selected = response[start : end + 1]
    assert np.all(np.diff(selected) > 0.0)
    assert selected.tolist() != np.maximum.accumulate(response).tolist()


def test_selected_response_gate_rejects_hidden_inverse_reversal():
    common = load_common()

    assert common.selected_response_hard_failure_reasons(
        [0.0004287, 0.0004267, 0.0004297, 0.0004348, 0.0004362]
    ) == ["selected_response_not_monotone"]
    assert common.selected_response_hard_failure_reasons(
        [0.0004266, 0.0004289, 0.0004315, 0.0004340, 0.0004362]
    ) == []
