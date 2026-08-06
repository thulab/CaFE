from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pa_parquet

from cafe import protocol
from cafe.data.fev_qualification import discover_categorical_levels
from cafe.data.fev_qualification import qualification_matrix_csv
from cafe.data.fev_qualification import summarize_qualification


def test_discover_categorical_levels_counts_values_and_nulls(
    tmp_path: Path,
):
    parquet_path = tmp_path / "categories.parquet"
    pa_parquet.write_table(
        pa.table(
            {
                "holiday": pa.array(
                    [["none", "national", None], ["none", "local"]],
                    type=pa.list_(pa.string()),
                ),
                "event": pa.array(
                    [[False, True, None], [False, False]],
                    type=pa.list_(pa.bool_()),
                ),
            }
        ),
        parquet_path,
    )

    result = discover_categorical_levels(
        parquet_path,
        ["holiday", "event"],
    )

    assert result["holiday"] == {
        "levels": ["local", "national", "none"],
        "level_count": 3,
        "value_count": 5,
        "null_count": 1,
        "null_fraction": 0.2,
    }
    assert result["event"]["levels"] == ["False", "True"]
    assert result["event"]["null_count"] == 1


def test_qualification_summary_and_csv_use_data_level_statuses():
    statuses = {
        capability_id: "eligible" for capability_id in protocol.CAPABILITIES
    }
    statuses["hierarchical_coherence"] = (
        "eligible_via_existing_canonical_adapter"
    )
    row = {
        "task_index": 0,
        "task_view_id": "task-0",
        "config_id": "config-0",
        "frequency": "D",
        "target_count": 2,
        "known_dynamic_columns": ["known"],
        "native_record_count": 3,
        "minimum_length": 216,
        "median_length": 240.0,
        "maximum_length": 300,
        "stratum_count": 12,
        "accepted_anchor_count": 12,
        "target_nonfinite_count": 1,
        "target_nonfinite_fraction": 0.01,
        "known_covariate_nonfinite_count": 0,
        "known_covariate_nonfinite_fraction": 0.0,
        "categorical_known_columns": [],
        "existing_cafe_source_overlaps": [],
        "anchor_error": None,
        "capability_status": statuses,
    }

    summary = summarize_qualification([row])
    matrix = qualification_matrix_csv([row])

    assert summary["task_with_minimum_anchor_count"] == 1
    assert summary["eligible_capability_cells"] == len(protocol.CAPABILITIES)
    assert summary["task_with_target_missingness_count"] == 1
    assert "eligible_via_existing_canonical_adapter" in matrix
