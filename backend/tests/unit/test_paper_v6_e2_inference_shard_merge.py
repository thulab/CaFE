from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "merge_paper_v6_e2_inference_shards.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "merge_paper_v6_e2_inference_shards",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_shard(
    root: Path,
    *,
    model_id: str,
    base_url: str,
    identity: dict,
) -> Path:
    shard = root / model_id
    config = {
        **identity,
        "requested_models": [model_id],
        "service": {"base_url": base_url, "api_prefix": "/ai/api/v1"},
    }
    write_json(shard / "inference_config.json", config)
    write_json(
        shard / "inference_model_catalog.json",
        {
            "models": [
                {
                    "model_id": model_id,
                    "forecast_limits": {"max_target_count": None},
                }
            ]
        },
    )
    for directory, status_name in (
        ("predictions", "model_status.json"),
        ("real_source_predictions", "real_source_model_status.json"),
    ):
        path = shard / directory / f"{model_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"sample_id":"sample"}\n', encoding="utf-8")
        write_json(
            shard / status_name,
            {
                "models": {
                    model_id: {
                        "model_id": model_id,
                        "status": "complete",
                        "compatible_sample_count": 1,
                        "succeeded_count": 1,
                    }
                }
            },
        )
    return shard


def test_merge_shards_requires_complete_disjoint_model_coverage(tmp_path):
    module = load_module()
    output = tmp_path / "primary"
    identity = {
        "synthetic_input": {"sha256": "synthetic"},
        "real_source_input": {"sha256": "real"},
        "context_lengths": [96, 168, 336, 504],
        "horizon": 48,
    }
    write_json(
        output / "inference_config.json",
        {
            **identity,
            "requested_models": ["model_a", "model_b"],
        },
    )
    shards = [
        make_shard(
            tmp_path / "shards",
            model_id=model_id,
            base_url=f"http://host-{index}",
            identity=identity,
        )
        for index, model_id in enumerate(("model_a", "model_b"), start=1)
    ]

    result = module.merge_shards(output, shards)

    assert result["expected_models"] == ["model_a", "model_b"]
    assert set(
        module.read_json(output / "model_status.json")["models"]
    ) == {"model_a", "model_b"}
    assert (output / "predictions/model_a.jsonl").is_file()
    assert (output / "real_source_predictions/model_b.jsonl").is_file()
    assert [
        model["model_id"]
        for model in module.read_json(
            output / "inference_model_catalog.json"
        )["models"]
    ] == ["model_a", "model_b"]
    assert (
        result["shards"][0]["models"]["model_a"]["synthetic"][
            "failure_row_count"
        ]
        == 0
    )


def test_merge_identity_includes_input_adaptation_policy():
    module = load_module()
    base = {
        "synthetic_input": {"sha256": "synthetic"},
        "real_source_input": None,
        "context_lengths": [96, 168, 336, 504],
        "horizon": 48,
    }

    legacy = module.input_identity(base)
    adapted = module.input_identity(
        {
            **base,
            "input_adaptation_policy": {
                "policy_id": "paper-v7-input-adaptation-v1"
            },
        }
    )

    assert legacy != adapted
    assert adapted["input_adaptation_policy"]["policy_id"] == (
        "paper-v7-input-adaptation-v1"
    )


def test_merge_rejects_incomplete_adapted_child_request_coverage(tmp_path):
    module = load_module()
    predictions = tmp_path / "model.jsonl"
    predictions.write_text('{"sample_id":"sample"}\n', encoding="utf-8")
    status = {
        "status": "complete",
        "compatible_sample_count": 1,
        "succeeded_count": 1,
        "expected_original_view_count": 1,
        "unsupported_window_view_count": 0,
        "expected_http_request_count": 3,
        "successful_http_request_count": 2,
    }

    with pytest.raises(ValueError, match="HTTP request coverage"):
        module.validate_status(
            status,
            model_id="model",
            prediction_path=predictions,
        )
