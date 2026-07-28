from __future__ import annotations

import json
from pathlib import Path

import pytest

from cafe import provenance


def test_experiment_identity_does_not_freeze_future_stage_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    first = provenance.initialize_experiment(
        root,
        experiment_id="continuation",
        created_at="2026-07-28T00:00:00Z",
    )
    second = provenance.initialize_experiment(
        root,
        experiment_id="continuation",
        created_at="2026-07-29T00:00:00Z",
    )

    assert first == second
    assert "protocol" not in first
    assert "git_revision" not in first


def test_stage_contracts_freeze_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "experiment"
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(
        provenance,
        "code_provenance",
        lambda _root: {
            "git_revision": "calibration-code",
            "git_dirty": False,
        },
    )
    calibration = provenance.ensure_stage_contract(
        root,
        stage="calibration",
        created_at="2026-07-28T00:00:00Z",
        repository_root=repository,
        config={"anchors": 32},
        upstream=[],
    )
    calibration_path = root / "stage_contracts" / "calibration.json"

    monkeypatch.setattr(
        provenance,
        "code_provenance",
        lambda _root: {
            "git_revision": "later-inference-code",
            "git_dirty": False,
        },
    )
    inference = provenance.ensure_stage_contract(
        root,
        stage="inference",
        created_at="2026-07-29T00:00:00Z",
        repository_root=repository,
        config={"models": ["model-a"]},
        upstream=provenance.upstream_records(
            [calibration_path],
            relative_to=root,
        ),
    )

    assert calibration["code"]["git_revision"] == "calibration-code"
    assert inference["code"]["git_revision"] == "later-inference-code"
    assert inference["upstream"][0]["sha256"] == provenance.file_sha256(
        calibration_path
    )


def test_stage_contract_rejects_redefinition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provenance,
        "code_provenance",
        lambda _root: {
            "git_revision": "same-code",
            "git_dirty": False,
        },
    )
    root = tmp_path / "experiment"
    repository = tmp_path / "repository"
    repository.mkdir()
    provenance.ensure_stage_contract(
        root,
        stage="generation",
        created_at="2026-07-28T00:00:00Z",
        repository_root=repository,
        config={"seed_count": 64},
        upstream=[],
    )

    with pytest.raises(ValueError, match="different stage contract"):
        provenance.ensure_stage_contract(
            root,
            stage="generation",
            created_at="2026-07-29T00:00:00Z",
            repository_root=repository,
            config={"seed_count": 128},
            upstream=[],
        )


def test_stage_contract_file_write_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "stage.json"
    value = {
        "schema_version": provenance.STAGE_CONTRACT_SCHEMA,
        "stage": "analysis",
    }
    provenance.write_json_once(path, value)

    assert json.loads(path.read_text(encoding="utf-8")) == value
    provenance.write_json_once(path, value)
