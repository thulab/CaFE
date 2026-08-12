from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cafe import provenance
from cafe import pipeline
from cafe.inference import runner as inference


def test_timer4_is_a_formal_default_model() -> None:
    assert "Timer-4.0" in inference.DEFAULT_MODELS
    assert "Timer-4.0" in inference.MODEL_EXECUTION_CONFIG
    assert inference.MODEL_EXECUTION_CONFIG["Timer-4.0"]["transport"] == "msgpack_bulk"


def test_pipeline_defines_the_service_api_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["cafe.pipeline"])

    args = pipeline.parse_args()

    assert args.api_prefix == "/ai/api/v1"


def test_pipeline_resolves_input_capabilities_from_the_live_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = live_input_mode_model(
        "Timer-4.0",
        max_targets=-1,
        max_history_covariates=-1,
        supports_future_covariates=True,
    )
    monkeypatch.setattr(
        inference,
        "health_catalog",
        lambda endpoint, api_prefix: (endpoint, {"Timer-4.0": model}),
    )
    args = SimpleNamespace(
        endpoints=["http://127.0.0.1:10810"],
        api_prefix="/ai/api/v1",
        models=["Timer-4.0"],
    )

    resolved = pipeline.resolve_stage_input_capabilities(args)

    assert resolved == {
        "Timer-4.0": inference.resolve_input_capability(model),
    }


def live_input_mode_model(
    model_id: str,
    *,
    max_targets: int,
    max_history_covariates: int,
    supports_future_covariates: bool,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "forecast_limits": {
            "min_input_length": 1,
            "max_input_length": 16_384,
            "max_output_length": 1_024,
            "max_future_covs_length": 1_024,
            "input_mode": {
                "max_target_count": max_targets,
                "max_history_covariate_count": max_history_covariates,
                "supports_future_covariates": supports_future_covariates,
                "max_static_covariate_count": 0,
            },
        },
    }


@pytest.mark.parametrize(
    (
        "model",
        "expected_target_mode",
        "expected_covariate_mode",
        "expected_request_count",
    ),
    [
        (
            live_input_mode_model(
                "Timer-4.0",
                max_targets=-1,
                max_history_covariates=-1,
                supports_future_covariates=True,
            ),
            "native_multivariate",
            "native",
            1,
        ),
        (
            live_input_mode_model(
                "Chronos-2",
                max_targets=-1,
                max_history_covariates=-1,
                supports_future_covariates=True,
            ),
            "native_multivariate",
            "native",
            1,
        ),
        (
            live_input_mode_model(
                "tirex2",
                max_targets=-1,
                max_history_covariates=-1,
                supports_future_covariates=True,
            ),
            "native_multivariate",
            "native",
            1,
        ),
        (
            live_input_mode_model(
                "toto2.0",
                max_targets=-1,
                max_history_covariates=0,
                supports_future_covariates=False,
            ),
            "native_multivariate",
            "omitted_unsupported",
            1,
        ),
        (
            live_input_mode_model(
                "timesfm2.5",
                max_targets=1,
                max_history_covariates=-1,
                supports_future_covariates=True,
            ),
            "independent_univariate",
            "native",
            5,
        ),
        (
            live_input_mode_model(
                "moirai2",
                max_targets=1,
                max_history_covariates=0,
                supports_future_covariates=False,
            ),
            "independent_univariate",
            "omitted_unsupported",
            5,
        ),
        (
            live_input_mode_model(
                "Timer-3.5",
                max_targets=1,
                max_history_covariates=0,
                supports_future_covariates=False,
            ),
            "independent_univariate",
            "omitted_unsupported",
            5,
        ),
    ],
)
def test_live_input_mode_drives_native_adaptation(
    model: dict[str, object],
    expected_target_mode: str,
    expected_covariate_mode: str,
    expected_request_count: int,
) -> None:
    sample = {
        "context_length": 168,
        "horizon": 48,
        "target_dim": 5,
        "covariate_dim": 2,
    }

    plan = inference.input_adaptation_plan(
        model,
        sample,
        policy_id=inference.INPUT_ADAPTATION_POLICY_ID,
    )

    assert plan is not None
    assert plan["target_mode"] == expected_target_mode
    assert plan["covariate_mode"] == expected_covariate_mode
    assert plan["target_request_count"] == expected_request_count
    capability = plan["resolved_input_capability"]
    assert capability["source_schema"] == "input_mode"
    assert capability["max_target_count"] in {None, 1}
    assert capability["max_history_covariate_count"] in {None, 0}


def test_legacy_input_capability_fallback_normalizes_unbounded_counts() -> None:
    model = {
        "forecast_limits": {
            "min_input_length": 1,
            "max_input_length": 2_880,
            "max_output_length": 720,
            "max_target_count": None,
            "max_covariate_count": -1,
            "max_future_covs_length": 720,
        }
    }

    capability = inference.resolve_input_capability(model)

    assert capability == {
        "schema_version": "cafe.resolved_input_capability.v1",
        "source_schema": "legacy_forecast_limits",
        "max_target_count": None,
        "max_history_covariate_count": None,
        "supports_future_covariates": True,
        "max_future_covariate_length": 720,
    }


def test_live_capability_must_match_frozen_inference_contract(
    tmp_path: Path,
) -> None:
    model = live_input_mode_model(
        "Timer-4.0",
        max_targets=-1,
        max_history_covariates=-1,
        supports_future_covariates=True,
    )
    capability = inference.resolve_input_capability(model)
    contract_path = tmp_path / "inference.json"
    contract_path.write_text(
        json.dumps(
            {
                "config": {
                    "resolved_model_input_capabilities": {
                        "Timer-4.0": capability,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = inference.validate_catalog_input_capabilities(
        {"http://127.0.0.1:10810": {"Timer-4.0": model}},
        ["Timer-4.0"],
        contract_path=contract_path,
    )

    assert resolved == {"Timer-4.0": capability}
    model["forecast_limits"]["input_mode"]["max_target_count"] = 1
    with pytest.raises(ValueError, match="changed after.*frozen"):
        inference.validate_catalog_input_capabilities(
            {"http://127.0.0.1:10810": {"Timer-4.0": model}},
            ["Timer-4.0"],
            contract_path=contract_path,
        )


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


def test_completed_v2_analysis_manifest_is_reusable(tmp_path: Path) -> None:
    dataset_id = "gift_electricity_h"
    shard_name = "seed_000000_000001"
    inference_dir = (
        tmp_path / dataset_id / "03_inference" / shard_name
    )
    analysis_dir = tmp_path / dataset_id / "04_analysis" / shard_name
    inference_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    inference_manifest_path = inference_dir / "inference_manifest.json"
    inference_manifest_path.write_text(
        json.dumps({"complete": True}),
        encoding="utf-8",
    )
    score_path = analysis_dir / "scores.json"
    score_path.write_text("[]\n", encoding="utf-8")
    analysis_manifest_path = analysis_dir / "analysis_manifest.json"
    analysis_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cafe.analysis_manifest.v2",
                "dataset_id": dataset_id,
                "models": ["naive_last"],
                "analysis_profile": "full",
                "inference_manifest_sha256": provenance.file_sha256(
                    inference_manifest_path
                ),
                "coverage": [
                    {
                        "model_id": "naive_last",
                        "missing_prediction_count": 0,
                    }
                ],
                "files": {
                    "scores": {
                        "path": str(score_path),
                        "bytes": score_path.stat().st_size,
                        "sha256": provenance.file_sha256(score_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert pipeline.reusable_analysis_manifest(
        tmp_path,
        dataset_id=dataset_id,
        seed_start=0,
        seed_count=1,
        models=["naive_last"],
    )
