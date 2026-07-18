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
    / "generate_paper_v5_e2_master_samples.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_paper_v5_e2_master_samples",
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
