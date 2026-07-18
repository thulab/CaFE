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


def test_dataset_suite_is_unique_hourly_and_domain_diverse() -> None:
    module = load_module()
    datasets = module.DATASET_SPECS

    assert len(datasets) == 13
    assert len({dataset.dataset_id for dataset in datasets}) == len(datasets)
    assert len({dataset.domain for dataset in datasets}) == 5
    assert {dataset.frequency for dataset in datasets} == {"h"}
    assert all(not hasattr(dataset, "family_id") for dataset in datasets)


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


def test_build_suite_emits_only_dataset_local_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    datasets = (
        module.DatasetSpec("dataset_a", "Dataset A", "Domain A", "fake", "a"),
        module.DatasetSpec("dataset_b", "Dataset B", "Domain B", "fake", "b"),
    )

    def fake_build_profile_rows(dataset, **_kwargs):
        feature_value = 1.0 if dataset.dataset_id == "dataset_a" else 100.0
        rows = [
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "domain": dataset.domain,
                "context_length": context_length,
                "full__trend_strength": feature_value,
            }
            for context_length in module.CONTEXT_LENGTHS
        ]
        return rows, {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.dataset_name,
            "domain": dataset.domain,
            "paired_master_window_count": 1,
        }

    monkeypatch.setattr(module, "build_profile_rows", fake_build_profile_rows)
    suite, rows, inventory = module.build_suite(
        datasets=datasets,
        gift_eval_dir=Path("/unused"),
        data_dir=Path("/unused"),
        max_windows=30,
    )

    assert suite["schema_version"].endswith(".v2")
    assert suite["selection"]["dataset_count"] == 2
    assert "family_count" not in suite["selection"]
    assert "global_profiles" not in suite
    assert "sources" not in suite
    assert len(suite["profiles"]) == 8
    assert {row["dataset_id"] for row in rows} == {"dataset_a", "dataset_b"}
    assert {item["dataset_id"] for item in inventory} == {
        "dataset_a",
        "dataset_b",
    }
    assert (
        suite["profiles"]["dataset_a__L96_H48"]["features"][
            "full__trend_strength"
        ]["p50"]
        == pytest.approx(1.0)
    )
    assert (
        suite["profiles"]["dataset_b__L96_H48"]["features"][
            "full__trend_strength"
        ]["p50"]
        == pytest.approx(100.0)
    )


def test_selected_dataset_specs_rejects_unknown_ids() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="unknown profile datasets"):
        module.selected_dataset_specs(["missing"])

    selected = module.selected_dataset_specs(
        ["gift_ett2_h", "gift_ett1_h"]
    )
    assert [dataset.dataset_id for dataset in selected] == [
        "gift_ett2_h",
        "gift_ett1_h",
    ]


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
