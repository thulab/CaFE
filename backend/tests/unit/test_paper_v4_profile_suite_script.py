from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).parents[3] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "build_paper_v4_profile_suite.py"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_paper_v4_profile_suite",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_suite_is_unique_hourly_and_domain_diverse() -> None:
    module = load_module()
    sources = module.SOURCE_SPECS

    assert len(sources) == 13
    assert len({source.source_id for source in sources}) == len(sources)
    assert len({source.family_id for source in sources}) == 11
    assert len({source.domain for source in sources}) == 5
    assert {source.frequency for source in sources} == {"h"}


def test_nested_views_share_exact_future() -> None:
    module = load_module()
    master = np.arange(
        module.MAX_CONTEXT_LENGTH + module.HORIZON,
        dtype=float,
    )

    views = {
        length: module.nested_view(master, length)
        for length in module.CONTEXT_LENGTHS
    }

    for length, view in views.items():
        assert view.shape == (length + module.HORIZON,)
        assert view[-module.HORIZON :] == pytest.approx(
            master[-module.HORIZON :]
        )
    assert views[96][0] == 408
    assert views[168][0] == 336
    assert views[336][0] == 168
    assert views[504][0] == 0


def test_family_balancing_does_not_double_weight_sibling_configs() -> None:
    module = load_module()
    rows = [
        {"family_id": "A", "source_id": "A1"},
        {"family_id": "A", "source_id": "A1"},
        {"family_id": "A", "source_id": "A2"},
        {"family_id": "B", "source_id": "B1"},
        {"family_id": "B", "source_id": "B1"},
        {"family_id": "B", "source_id": "B1"},
    ]

    weights = module.source_balanced_weights(rows)

    assert float(weights.sum()) == pytest.approx(1.0)
    assert float(weights[:3].sum()) == pytest.approx(0.5)
    assert float(weights[3:].sum()) == pytest.approx(0.5)
    assert float(weights[:2].sum()) == pytest.approx(0.25)
    assert float(weights[2]) == pytest.approx(0.25)


def test_weighted_quantile_uses_positive_finite_mass() -> None:
    module = load_module()

    result = module.weighted_quantile(
        [0.0, 10.0, float("nan")],
        [0.5, 0.5, 1.0],
        levels=[0.25, 0.5, 0.75],
    )

    assert result == pytest.approx([0.0, 5.0, 10.0])


def test_candidate_selection_prioritizes_series_coverage() -> None:
    module = load_module()
    values = np.arange(100, dtype=float)
    candidates = [
        (f"series-{series}", f"item-{series}", 0, values, start, 100)
        for series in range(7)
        for start in range(20)
    ]

    selected = module.select_series_balanced_candidates(candidates, 14)

    counts = {
        series_id: sum(candidate[0] == series_id for candidate in selected)
        for series_id in {candidate[0] for candidate in selected}
    }
    assert len(selected) == 14
    assert len(counts) == 7
    assert set(counts.values()) == {2}
    assert {candidate[4] for candidate in selected} == {0, 19}
