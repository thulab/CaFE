from __future__ import annotations

import pytest

from cafe.data.fev_qualification import select_qualification_configs


def test_select_configs_filters_tasks_and_assets_in_source_order():
    tasks = [
        {"config_id": "a", "task_index": 0},
        {"config_id": "b", "task_index": 1},
        {"config_id": "c", "task_index": 2},
    ]
    files = [
        {"configs": ["a"], "path": "a.parquet"},
        {"configs": ["b"], "path": "b.parquet"},
        {"configs": ["c"], "path": "c.parquet"},
    ]

    selected_tasks, selected_files = select_qualification_configs(
        tasks,
        files,
        ["c", "a"],
    )

    assert [row["config_id"] for row in selected_tasks] == ["a", "c"]
    assert [row["configs"][0] for row in selected_files] == ["a", "c"]


def test_select_configs_rejects_unknown_ids():
    with pytest.raises(ValueError, match="unknown FEV config IDs"):
        select_qualification_configs(
            [{"config_id": "a"}],
            [{"configs": ["a"]}],
            ["missing"],
        )
