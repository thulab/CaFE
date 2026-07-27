from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "prepare_paper_v8_chronos_finetune.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_v8_chronos", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_row(seed_index: int, *, counterfactual_member: int | None = None) -> dict:
    return {
        "sample_id": f"sample-{seed_index}-{counterfactual_member}",
        "master_sample_id": f"sample-{seed_index}-{counterfactual_member}",
        "paired_group_id": f"pair-{seed_index}",
        "dataset_id": "dataset",
        "capability_id": "trend",
        "intensity": 1,
        "seed_index": seed_index,
        "counterfactual_member": counterfactual_member,
        "target_dim": 1,
        "covariate_dim": 0,
        "covariate_column_names": [],
        "frequency": "h",
        "season_length": 24,
        "mase_scale_by_target": [0.5],
        "target": [[float(index)] for index in range(384)],
        "covariates": None,
    }


def test_compact_record_uses_fixed_l168_suffix() -> None:
    result = MODULE.compact_record(
        sample_row(7),
        split="A",
        source_experiment_id="experiment",
        source_protocol_sha256="abc",
        master_context_length=336,
        fixed_context_length=168,
        horizon=48,
    )

    assert len(result["target"]) == 216
    assert result["target"][0] == [168.0]
    assert result["target"][167] == [335.0]
    assert result["target"][168] == [336.0]
    assert result["target"][-1] == [383.0]


def test_validate_coverage_accepts_constant_counterfactual_multiplicity() -> None:
    rows = []
    for seed_index in range(64):
        rows.extend(
            [
                sample_row(seed_index, counterfactual_member=0),
                sample_row(seed_index, counterfactual_member=1),
            ]
        )

    audit = MODULE.validate_coverage(rows, seed_start=0, seed_count=64)

    assert audit["dataset_capability_intensity_groups"] == 1
    assert audit["per_seed_multiplicity_group_counts"] == {"2": 1}


def test_validate_coverage_rejects_missing_seed() -> None:
    rows = [sample_row(seed_index) for seed_index in range(63)]

    with pytest.raises(ValueError, match="seed coverage mismatch"):
        MODULE.validate_coverage(rows, seed_start=0, seed_count=64)
