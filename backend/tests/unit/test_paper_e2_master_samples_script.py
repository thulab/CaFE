from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_paper_e2_master_samples.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_paper_e2_master_samples",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate(module, intensity: int, *, accepted: bool) -> dict:
    target = np.full(
        (module.MAX_CONTEXT_LENGTH + module.HORIZON, 1),
        float(intensity),
    )
    return {
        "accepted": accepted,
        "construction_validated": True,
        "target": target,
        "covariates": None,
        "metadata": {
            "generator_version": module.PAPER_GENERATOR_VERSION,
            "generator_conditioning": {
                "target_feature": "trend_strength",
                "target_strength": float(intensity),
                "target_relative_level": (intensity - 1) / 4,
            },
        },
        "view_audits": [
            {
                "context_length": context_length,
                "profile_id": f"dataset__univariate__L{context_length}_H48",
                "passed": accepted,
            }
            for context_length in module.CONTEXT_LENGTHS
        ],
        "realized_features": {
            str(context_length): {
                "trend_strength": float(intensity),
            }
            for context_length in module.CONTEXT_LENGTHS
        },
    }


def test_joint_generation_retries_all_five_intensities_together(monkeypatch):
    module = load_module()
    calls: list[tuple[int, int]] = []
    accepted_seed = module._attempt_seed(41, 2)

    def fake_generate(**kwargs):
        intensity = int(kwargs["intensity"])
        attempt_seed = int(kwargs["attempt_seed"])
        calls.append((intensity, attempt_seed))
        return candidate(
            module,
            intensity,
            accepted=attempt_seed == accepted_seed,
        )

    monkeypatch.setattr(module, "generate_intensity_candidate", fake_generate)
    attempt, attempt_seed, candidates = module.generate_paired_group(
        capability_id="trend",
        sample_seed=41,
        profile={},
        conditioning={},
        feature_artifact={},
        near_artifact={},
        max_attempts=4,
    )

    assert attempt == 2
    assert attempt_seed == accepted_seed
    assert set(candidates) == set(module.INTENSITIES)
    assert len(calls) == 3 * len(module.INTENSITIES)
    for offset in range(0, len(calls), len(module.INTENSITIES)):
        batch = calls[offset : offset + len(module.INTENSITIES)]
        assert [item[0] for item in batch] == list(module.INTENSITIES)
        assert len({item[1] for item in batch}) == 1


def test_v7_defaults_freeze_five_by_sixty_four_and_two_blocks():
    module = load_module()

    assert module.DEFAULT_SUITE_DIR.parts[-2:] == (
        "v7",
        "01_nine_capability_suite",
    )
    assert module.DEFAULT_OUTPUT_DIR.parts[-2:] == (
        "v7",
        "E2_dynamic_stability",
    )
    assert len(module.DEFAULT_ROUND_SEEDS) == 5
    assert len(set(module.DEFAULT_ROUND_SEEDS)) == 5
    assert module.DEFAULT_SAMPLES_PER_ROUND == 64
    assert module.FORMAL_TOTAL_PER_INTENSITY == 320
    contract = module.formal_analysis_block_contract()
    assert contract["total_per_intensity"] == 320
    assert contract["block_size"] == 160
    assert contract["mutually_exclusive"] is True
    assert module.formal_pool_identity(3, 31) == {
        "pool_index": 159,
        "analysis_block_id": "A",
        "analysis_block_index": 159,
    }
    assert module.formal_pool_identity(3, 32) == {
        "pool_index": 160,
        "analysis_block_id": "B",
        "analysis_block_index": 0,
    }


def test_v7_formal_sampling_protocol_fails_closed():
    module = load_module()

    module.validate_sampling_protocol(
        round_seeds=module.DEFAULT_ROUND_SEEDS,
        samples_per_round=64,
        flat_batch_id=None,
        flat_batch_seed=None,
    )
    with pytest.raises(ValueError, match="five fixed round seeds"):
        module.validate_sampling_protocol(
            round_seeds=tuple(range(5)),
            samples_per_round=64,
            flat_batch_id=None,
            flat_batch_seed=None,
        )
    with pytest.raises(ValueError, match="64 samples"):
        module.validate_sampling_protocol(
            round_seeds=module.DEFAULT_ROUND_SEEDS,
            samples_per_round=32,
            flat_batch_id=None,
            flat_batch_seed=None,
        )
    module.validate_sampling_protocol(
        round_seeds=(17,),
        samples_per_round=160,
        flat_batch_id="legacy-B",
        flat_batch_seed=17,
    )


def test_master_rows_share_paired_identity_and_attempt_seed():
    module = load_module()
    cell = {
        "profile_id": "dataset__univariate__L504_H48",
        "dataset_id": "dataset",
        "task_id": "univariate",
        "capability_id": "trend",
    }
    profile = {
        "season_length": 24,
        "frequency": "h",
        "target_dim": 1,
        "hierarchy": None,
    }
    rows = [
        module.master_sample_row(
            cell=cell,
            profile=profile,
            intensity=intensity,
            round_index=1,
            round_seed=101,
            sample_index=3,
            sample_seed=17,
            attempt=2,
            attempt_seed=29,
            candidate=candidate(module, intensity, accepted=True),
        )
        for intensity in module.INTENSITIES
    ]

    assert len({row["paired_group_id"] for row in rows}) == 1
    assert len({row["paired_attempt_seed"] for row in rows}) == 1
    assert {row["intensity"] for row in rows} == set(module.INTENSITIES)
    assert {row["task_view_id"] for row in rows} == {
        "dataset::univariate"
    }
    assert all(
        row["context_lengths"] == list(module.CONTEXT_LENGTHS)
        for row in rows
    )
    assert all(row["horizon"] == 48 for row in rows)
    assert all(len(row["view_qualification"]) == 4 for row in rows)


def test_completed_shard_requires_all_intensities_and_four_views(tmp_path):
    module = load_module()
    cell = {
        "profile_id": "dataset__univariate__L504_H48",
        "dataset_id": "dataset",
        "task_id": "univariate",
        "capability_id": "trend",
    }
    profile = {
        "season_length": 24,
        "frequency": "h",
        "target_dim": 1,
        "hierarchy": None,
    }
    path = tmp_path / "cell.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for intensity in module.INTENSITIES:
            row = module.master_sample_row(
                cell=cell,
                profile=profile,
                intensity=intensity,
                round_index=1,
                round_seed=101,
                sample_index=0,
                sample_seed=17,
                attempt=0,
                attempt_seed=19,
                candidate=candidate(module, intensity, accepted=True),
            )
            handle.write(json.dumps(row) + "\n")

    audit = module.validate_complete_shard(
        path,
        cell=cell,
        expected=5,
    )

    assert audit["row_count"] == 5
    assert audit["paired_group_count"] == 1
    assert len(audit["sha256"]) == 64

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["view_qualification"] = rows[0]["view_qualification"][:-1]
    incomplete_views = tmp_path / "incomplete_views.jsonl"
    incomplete_views.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="qualified view set mismatch"):
        module.validate_complete_shard(
            incomplete_views,
            cell=cell,
            expected=5,
        )


def test_formal_shard_resume_validates_complete_disjoint_blocks(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    monkeypatch.setattr(module, "FORMAL_ROUND_COUNT", 2)
    monkeypatch.setattr(module, "DEFAULT_SAMPLES_PER_ROUND", 2)
    monkeypatch.setattr(module, "FORMAL_TOTAL_PER_INTENSITY", 4)
    monkeypatch.setattr(module, "FORMAL_ANALYSIS_BLOCK_SIZE", 2)
    cell = {
        "profile_id": "dataset__univariate__L504_H48",
        "dataset_id": "dataset",
        "task_id": "univariate",
        "capability_id": "trend",
    }
    profile = {
        "season_length": 24,
        "frequency": "h",
        "target_dim": 1,
        "hierarchy": None,
    }
    round_seeds = (101, 202)
    path = tmp_path / "formal.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for round_index, round_seed in enumerate(round_seeds, start=1):
            for sample_index in range(2):
                for intensity in module.INTENSITIES:
                    row = module.master_sample_row(
                        cell=cell,
                        profile=profile,
                        intensity=intensity,
                        round_index=round_index,
                        round_seed=round_seed,
                        sample_index=sample_index,
                        sample_seed=17,
                        attempt=0,
                        attempt_seed=19,
                        candidate=candidate(
                            module,
                            intensity,
                            accepted=True,
                        ),
                    )
                    handle.write(json.dumps(row) + "\n")

    audit = module.validate_complete_shard(
        path,
        cell=cell,
        expected=20,
        round_seeds=round_seeds,
        samples_per_round=2,
    )

    assert audit["paired_group_count"] == 4
    assert audit["analysis_block_group_counts"] == {"A": 2, "B": 2}
    assert audit["analysis_blocks_mutually_exclusive"] is True

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["analysis_block_id"] = "A"
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="analysis block identity mismatch"):
        module.validate_complete_shard(
            tampered,
            cell=cell,
            expected=20,
            round_seeds=round_seeds,
            samples_per_round=2,
        )


def test_select_cells_keeps_dataset_disjoint_generation_shards():
    module = load_module()
    cells = [
        {
            "dataset_id": dataset_id,
            "task_id": "univariate",
            "capability_id": capability_id,
        }
        for dataset_id, capability_id in (
            ("dataset_a", "trend"),
            ("dataset_a", "multi_seasonal"),
            ("dataset_b", "trend"),
        )
    ]

    selected = module.select_cells(cells, ("dataset_b",))

    assert selected == [cells[2]]
    with pytest.raises(ValueError, match="no supported cells"):
        module.select_cells(cells, ("missing_dataset",))


def test_flat_batch_rows_have_no_round_identity(tmp_path):
    module = load_module()
    cell = {
        "profile_id": "dataset__univariate__L504_H48",
        "dataset_id": "dataset",
        "task_id": "univariate",
        "capability_id": "trend",
    }
    profile = {
        "season_length": 24,
        "frequency": "h",
        "target_dim": 1,
        "hierarchy": None,
    }
    path = tmp_path / "flat.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for intensity in module.INTENSITIES:
            row = module.master_sample_row(
                cell=cell,
                profile=profile,
                intensity=intensity,
                round_index=None,
                round_seed=None,
                sample_index=0,
                sample_seed=17,
                attempt=0,
                attempt_seed=19,
                candidate=candidate(module, intensity, accepted=True),
                batch_id="B",
                batch_seed=2026072021,
            )
            assert "round_index" not in row
            assert "round_seed" not in row
            handle.write(json.dumps(row) + "\n")

    audit = module.validate_complete_shard(
        path,
        cell=cell,
        expected=5,
        flat_batch_id="B",
        flat_batch_seed=2026072021,
        samples_per_cell=1,
    )
    assert audit["paired_group_count"] == 1
