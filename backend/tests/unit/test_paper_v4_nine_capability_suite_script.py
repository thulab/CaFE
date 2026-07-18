from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "build_paper_v4_nine_capability_suite.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location(
        "build_paper_v4_nine_capability_suite",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_all_tasks_use_four_lookbacks_and_h48(module) -> None:
    assert module.CONTEXT_LENGTHS == (96, 168, 336, 504)
    assert module.HORIZON == 48
    assert len(module.ALL_CAPABILITY_IDS) == 9
    assert module.TASK_DESIGNS["hierarchy"].season_length == 7


def test_real_paired_views_share_exact_future(module) -> None:
    length = module.MAX_CONTEXT_LENGTH + module.MASTER_LOADER_HORIZON
    target = np.column_stack(
        [
            np.arange(length, dtype=float),
            2.0 * np.arange(length, dtype=float),
        ]
    )
    raw_future_shapes = []
    standardized_futures = []
    for context_length in module.CONTEXT_LENGTHS:
        view, _ = module.paired_view(
            target,
            None,
            context_length=context_length,
            hierarchy=None,
        )
        assert view.shape == (context_length + module.HORIZON, 2)
        raw_future_shapes.append(target[module.MAX_CONTEXT_LENGTH : module.MAX_CONTEXT_LENGTH + module.HORIZON].shape)
        standardized_futures.append(view[-module.HORIZON :])
    assert raw_future_shapes == [(48, 2)] * 4
    # Different lookback normalizations may alter values, but not future indices.
    assert not np.array_equal(standardized_futures[0], standardized_futures[-1])


def test_hierarchy_view_preserves_additivity(module) -> None:
    length = module.MAX_CONTEXT_LENGTH + module.MASTER_LOADER_HORIZON
    child_a = np.sin(np.arange(length) / 7.0) + 3
    child_b = np.cos(np.arange(length) / 5.0) + 4
    target = np.column_stack([child_a + child_b, child_a, child_b])
    for context_length in module.CONTEXT_LENGTHS:
        view, _ = module.paired_view(
            target,
            None,
            context_length=context_length,
            hierarchy="additive_first",
        )
        assert np.max(np.abs(view[:, 0] - view[:, 1] - view[:, 2])) < 1e-10


def test_mapping_assigns_all_nine_capabilities(module) -> None:
    assert {
        module.task_id_for_capability(capability_id)
        for capability_id in module.ALL_CAPABILITY_IDS
    } == {"univariate", "common_factor", "hierarchy", "covariate"}
