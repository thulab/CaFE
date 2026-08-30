from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import msgpack
import numpy as np
import pytest

from cafe import core as protocol
from cafe.benchmark_extension import inference as inference_module
from cafe.benchmark_extension.inference import (
    _InputTokenLimiter,
    _batch_input_tokens,
    _batch_scheduling_tokens,
    _forecast_children_in_bounded_bulks,
    _iter_model_bulk_batches,
    _requested_execution_complete,
    _run_streaming_bulk_model,
    model_task_row,
)
from cafe.benchmark_extension.storage import iter_prediction_parquet
from cafe.benchmark_extension.distributed_worker import (
    _relocate_generation_files,
    _validate_model_protocol,
)
from cafe.inference.runner import (
    MODEL_EXECUTION_CONFIG,
    _bulk_request_content,
    input_adaptation_plan,
)


def _sample() -> dict:
    target = np.arange(250 * 3.0).reshape(250, 3)
    return {
        "schema_version": "cafe.benchmark_extension_sample.v1",
        "sample_id": "sample",
        "context_length": 200,
        "horizon": 50,
        "target_dim": 3,
        "covariate_dim": 0,
        "covariates": None,
        "frequency": "H",
        "target": target.tolist(),
    }


def test_moirai_long_context_bulk_rows_are_memory_bounded() -> None:
    config = inference_module.MODEL_INPUT_TOKEN_CONFIG["moirai2"]
    assert config["maximum_bulk_rows"] == 8
    assert config["native_multivariate_maximum_bulk_rows"] == 8


def test_expanded_children_are_split_at_the_transport_boundary(monkeypatch) -> None:
    observed_bulk_sizes = []

    async def fake_forecast(_client, *, children, **_kwargs):
        observed_bulk_sizes.append(len(children))
        values = np.asarray(
            [int(child["child_index"]) for child in children], dtype=np.float32
        )
        return {
            "forecasts": values[:, None, None],
            "attempts": 1,
            "elapsed_seconds": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        inference_module, "_forecast_bulk_with_retry", fake_forecast
    )
    result = asyncio.run(
        _forecast_children_in_bounded_bulks(
            object(),
            forecast_url="http://service/forecast/bulk",
            model_id="moirai2",
            children=[{"child_index": index} for index in range(20)],
            maximum_bulk_rows=8,
            max_attempts=3,
        )
    )

    assert observed_bulk_sizes == [8, 8, 4]
    assert result["bulk_request_count"] == 3
    assert result["attempts"] == 3
    np.testing.assert_array_equal(
        result["forecasts"][:, 0, 0], np.arange(20, dtype=np.float32)
    )


def test_distributed_worker_reuses_the_term_and_horizon_preflight() -> None:
    contexts = {
        "Timer-4.0": 8192,
        "Chronos-2": 8192,
        "Timer-3.5": 11520,
        "timesfm2.5": 15360,
        "moirai2": 16384,
        "toto2.0": 16384,
    }
    generation = {
        "config": {
            "term": "medium",
            "prediction_length": 480,
            "observed_covariate_availability": [],
            "source_distance_configuration": {"model_max_contexts": contexts},
        }
    }
    model = {
        "forecast_limits": {
            "max_input_length": 8192,
            "max_output_length": 960,
            "input_mode": {"supports_future_covariates": True},
        }
    }
    _validate_model_protocol(generation, "Timer-4.0", model)


def test_distributed_worker_relocates_hash_identical_contracts(tmp_path: Path) -> None:
    generation_dir = tmp_path / "01_generation"
    generation_dir.mkdir()
    contract = generation_dir / "treatment_contracts.parquet"
    contract.write_bytes(b"compact-contract")
    generation = {
        "files": {
            "capability_treatments": {
                "path": "/orchestrator/experiment/treatment_contracts.parquet",
                "sha256": protocol.file_sha256(contract),
            }
        }
    }
    relocated = _relocate_generation_files(generation, tmp_path)
    assert relocated["files"]["capability_treatments"]["path"] == str(
        contract.resolve()
    )
    assert generation["files"]["capability_treatments"]["path"].startswith(
        "/orchestrator/"
    )


def test_model_context_truncation_happens_after_full_history_treatment() -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {"max_input_length": 96, "min_input_length": 1},
    }
    row = model_task_row(_sample(), model)
    assert row["source_context_length"] == 200
    assert row["context_length"] == 96
    assert np.asarray(row["target"]).shape == (146, 3)
    np.testing.assert_array_equal(
        np.asarray(row["target"]),
        np.asarray(_sample()["target"])[104:],
    )


def test_model_context_accepts_an_exact_pre_materialized_suffix() -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {"max_input_length": 96, "min_input_length": 1},
    }
    sample = _sample()
    sample["materialized_history_start"] = 104
    sample["target"] = np.asarray(sample["target"])[104:]
    row = model_task_row(sample, model)
    assert row["source_context_length"] == 200
    assert row["context_length"] == 96
    np.testing.assert_array_equal(row["target"], sample["target"])


def test_distributed_worker_mapping_is_optional_and_exact() -> None:
    endpoints = ["http://host-a:10810", "http://host-b:10811"]
    assert inference_module.parse_distributed_workers(
        [], configured_endpoints=endpoints
    ) == {}
    assert inference_module.parse_distributed_workers(
        [
            "http://host-a:10810=host-a",
            "http://host-b:10811=local",
        ],
        configured_endpoints=endpoints,
    ) == {
        "http://host-a:10810": "host-a",
        "http://host-b:10811": "local",
    }
    with pytest.raises(ValueError, match="does not match"):
        inference_module.parse_distributed_workers(
            ["http://unknown:10810=worker"],
            configured_endpoints=endpoints,
        )


def test_distributed_worker_command_uses_an_explicit_python_prefix() -> None:
    assert Path(inference_module.sys.executable).is_file()
    args = argparse.Namespace(
        preprocess_workers=4,
        max_open_shape_groups=8,
        max_inflight_batches=16,
        max_inflight_mib=512,
        load_timeout_seconds=1800,
        forecast_timeout_seconds=1200,
        max_attempts=3,
        max_request_input_tokens=None,
        client_inflight_input_tokens=None,
        reuse_loaded_model=False,
        preserve_loaded_model=False,
    )
    command = inference_module._distributed_worker_command(
        python_prefix=["/data/xmy/CaFE/.venv/bin/python"],
        dataset_id="gift_fixture",
        output_root=Path("/data/xmy/CaFE/runtime/experiments/fixture"),
        gift_eval_dir=Path("/data/xmy/CaFE/data/gift-eval"),
        model_id="Timer-4.0",
        endpoint="http://127.0.0.1:10810",
        api_prefix="/ai/api/v1",
        devices="0,1,2,3",
        part_index=0,
        part_count=3,
        worker_output_dir=Path("/data/xmy/CaFE/runtime/worker"),
        args=args,
    )
    assert command[0] == "/data/xmy/CaFE/.venv/bin/python"
    assert command[1:4] == [
        "-m",
        "cafe.benchmark_extension.distributed_worker",
        "--dataset-id",
    ]


def test_remote_worker_prefix_selects_the_worker_source_tree() -> None:
    root = Path("/data/xmy/CaFE-worker")
    assert inference_module._remote_worker_python_prefix(root) == [
        "env",
        "PYTHONPATH=/data/xmy/CaFE-worker/src",
        "/data/xmy/CaFE-worker/.venv/bin/python",
    ]


def test_native_panel_is_preserved_in_generation_task() -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {"max_input_length": -1, "min_input_length": 1},
    }
    row = model_task_row(_sample(), model)
    assert row["target_dim"] == 3
    assert np.asarray(row["target"]).shape == (250, 3)


def test_inference_model_context_must_cover_generation_distance_contract() -> None:
    inference_module._validate_distance_context_contract(
        "tirex2",
        {"forecast_limits": {"max_input_length": 4096}},
        {"tirex2": 2048},
    )
    frozen = inference_module._model_with_context_contract(
        {"forecast_limits": {"max_input_length": 4096}}, 2048
    )
    assert inference_module._maximum_context(frozen) == 2048
    with pytest.raises(ValueError, match="contract requirement 2048"):
        inference_module._validate_distance_context_contract(
            "tirex2",
            {"forecast_limits": {"max_input_length": 1024}},
            {"tirex2": 2048},
        )


def test_medium_forecast_limit_preflight_rejects_tirex2() -> None:
    generation = {
        "config": {
            "prediction_length": 480,
            "observed_covariate_availability": [],
        }
    }
    with pytest.raises(ValueError, match="supports at most H=320"):
        inference_module._validate_forecast_limits(
            "tirex2",
            {
                "forecast_limits": {
                    "max_output_length": 320,
                    "max_future_covs_length": 320,
                    "input_mode": {"supports_future_covariates": True},
                }
            },
            generation,
        )
    inference_module._validate_forecast_limits(
        "Timer-4.0",
        {
            "forecast_limits": {
                "max_output_length": 960,
                "max_future_covs_length": 960,
                "input_mode": {"supports_future_covariates": True},
            }
        },
        generation,
    )


def test_model_major_partial_invocation_succeeds_before_global_completion() -> None:
    statuses = {"Timer-4.0": {"status": "complete"}}
    predictions = {"Timer-4.0": {"parts": [{"path": "part.parquet"}]}}
    assert _requested_execution_complete(["Timer-4.0"], statuses, predictions)
    assert not _requested_execution_complete(["Chronos-2"], statuses, predictions)


def test_zero_compatible_rows_are_a_completed_model_invocation() -> None:
    statuses = {"tirex2": {"status": "complete", "compatible_sample_count": 0}}
    predictions = {"tirex2": {"row_count": 0, "parts": []}}
    assert _requested_execution_complete(["tirex2"], statuses, predictions)


def test_four_card_replica_presets_honor_supported_contexts() -> None:
    assert {
        model_id: int(config["replicas_per_device"])
        for model_id, config in MODEL_EXECUTION_CONFIG.items()
        if model_id in inference_module.DEFAULT_MODELS
    } == {
        "Timer-4.0": 2,
        "Timer-3.5": 1,
        "Chronos-2": 1,
        "moirai2": 2,
        "toto2.0": 1,
        "timesfm2.5": 2,
        "tirex2": 2,
    }


def test_publication_v3_validation_remains_readable_for_inference(
    tmp_path: Path,
) -> None:
    generation_path = tmp_path / "01_generation" / "manifest.json"
    generation = {
        "schema_version": inference_module.GENERATION_SCHEMA,
        "config": {
            "pipeline_schema_version": inference_module.PIPELINE_SCHEMA,
        },
    }
    protocol.write_json(generation_path, generation)
    validation_path = tmp_path / "02_validation" / "report.json"
    legacy = {
        "schema_version": (
            inference_module.LEGACY_PUBLICATION_VALIDATION_SCHEMA
        ),
        "validation_policy": (
            inference_module.LEGACY_PUBLICATION_VALIDATION_POLICY
        ),
        "accepted": True,
        "generation_manifest_sha256": protocol.file_sha256(generation_path),
    }
    protocol.write_json(validation_path, legacy)

    loaded, loaded_generation_path, loaded_validation_path = (
        inference_module._validated_inputs(tmp_path)
    )
    assert loaded == generation
    assert loaded_generation_path == generation_path
    assert loaded_validation_path == validation_path

    legacy["validation_policy"] = "research_only"
    protocol.write_json(validation_path, legacy)
    with pytest.raises(ValueError, match="validation is not accepted"):
        inference_module._validated_inputs(tmp_path)


def test_group_row_limit_controls_input_adaptation_not_bulk_batch_rows() -> None:
    model = {
        "forecast_limits": {
            "max_input_length": 16,
            "min_input_length": 1,
            "max_output_length": 64,
            "max_group_rows": 4,
            "input_mode": {
                "max_target_count": 1,
                "max_history_covariate_count": 0,
                "supports_future_covariates": False,
            },
        }
    }
    summary = inference_module._adaptation_summary()
    batches = list(
        _iter_model_bulk_batches(
            (_sample() for _index in range(3)),
            model=model,
            execution={"task_batch_size": 1024},
            maximum_open_groups=2,
            maximum_buffered_bytes=1024 * 1024,
            summary=summary,
        )
    )
    # max_group_rows counts variables within one task, not bulk samples.
    assert [len(chunk) for _shard, _key, chunk in batches] == [3]

    sample = _sample()
    sample["context_length"] = 16
    sample["covariate_dim"] = 2
    sample["covariates"] = np.zeros((250, 2))
    model["forecast_limits"]["input_mode"][
        "max_history_covariate_count"
    ] = -1
    model["forecast_limits"]["input_mode"][
        "supports_future_covariates"
    ] = True
    plan = input_adaptation_plan(
        model,
        sample,
        policy_id=inference_module.INPUT_ADAPTATION_POLICY_ID,
    )
    assert plan is not None
    assert plan["target_mode"] == "independent_univariate"
    assert plan["covariate_mode"] == "native_known_future"

    model["forecast_limits"]["input_mode"]["max_target_count"] = -1
    native_plan = input_adaptation_plan(
        model,
        sample,
        policy_id=inference_module.INPUT_ADAPTATION_POLICY_ID,
    )
    assert native_plan is not None
    assert native_plan["target_mode"] == "native_multivariate"
    assert native_plan["covariate_mode"] == "omitted_unsupported"


def test_past_only_covariates_are_native_without_future_payload() -> None:
    sample = _sample()
    sample["context_length"] = 16
    sample["covariate_dim"] = 2
    sample["covariate_column_names"] = ["past_0", "past_1"]
    sample["covariate_availability"] = ["past_only", "past_only"]
    sample["future_covariate_visible"] = [False, False]
    sample["covariates"] = np.zeros((250, 2))
    model = {
        "forecast_limits": {
            "max_input_length": 16,
            "min_input_length": 1,
            "max_output_length": 64,
            "max_group_rows": 64,
            "input_mode": {
                "max_target_count": -1,
                "max_history_covariate_count": -1,
                "supports_history_covariates": True,
                "supports_future_covariates": False,
            },
        }
    }
    plan = input_adaptation_plan(
        model,
        sample,
        policy_id=inference_module.INPUT_ADAPTATION_POLICY_ID,
    )
    assert plan is not None
    assert plan["covariate_mode"] == "native_history_only"
    content, _shape, _horizon = _bulk_request_content("fixture", [sample])
    payload = msgpack.unpackb(content, raw=False)
    assert payload["history_covariates_shape"] == [1, 2, 16]
    assert "future_covariates" not in payload
    assert "future_covariates_shape" not in payload


def test_timesfm_omits_past_only_covariates_that_cannot_be_paired() -> None:
    sample = _sample()
    sample["context_length"] = 16
    sample["covariate_dim"] = 2
    sample["covariate_column_names"] = ["past_0", "past_1"]
    sample["covariate_availability"] = ["past_only", "past_only"]
    sample["future_covariate_visible"] = [False, False]
    sample["covariates"] = np.zeros((250, 2))
    model = {
        "model_id": "timesfm2.5",
        "forecast_limits": {
            "max_input_length": 16,
            "min_input_length": 1,
            "max_output_length": 64,
            "max_group_rows": 64,
            "input_mode": {
                "max_target_count": 1,
                "max_history_covariate_count": -1,
                "supports_future_covariates": True,
            },
        },
    }
    plan = input_adaptation_plan(
        model,
        sample,
        policy_id=inference_module.INPUT_ADAPTATION_POLICY_ID,
    )
    assert plan is not None
    assert plan["covariate_mode"] == "omitted_unsupported"
    assert plan["request_covariate_dim"] == 0
    assert plan["request_future_covariate_dim"] == 0
    assert plan["covariate_omission_reason"] == (
        "model_requires_history_and_future_covariates_together"
    )
    children = inference_module.adapted_request_samples(sample, plan)
    assert len(children) == sample["target_dim"]
    assert all(child["covariates"] is None for child in children)
    assert all(child["covariate_dim"] == 0 for child in children)
    content, _shape, _horizon = _bulk_request_content(
        "timesfm2.5", children
    )
    payload = msgpack.unpackb(content, raw=False)
    assert "history_covariates" not in payload
    assert "future_covariates" not in payload


def test_mixed_covariate_visibility_sends_only_known_future_columns() -> None:
    sample = _sample()
    sample["context_length"] = 200
    sample["covariate_dim"] = 2
    sample["covariate_column_names"] = ["past_0", "known_0"]
    sample["covariate_availability"] = ["past_only", "known_future"]
    sample["future_covariate_visible"] = [False, True]
    sample["covariates"] = np.column_stack(
        (np.arange(250.0), 1000.0 + np.arange(250.0))
    )

    content, _shape, _horizon = _bulk_request_content("fixture", [sample])
    payload = msgpack.unpackb(content, raw=False)
    assert payload["history_covariates_shape"] == [1, 2, 200]
    assert payload["future_covariates_shape"] == [1, 1, 50]
    assert payload["future_covariate_history_indices"] == [1]
    future = np.frombuffer(payload["future_covariates"], dtype=np.float32)
    np.testing.assert_array_equal(future, 1000.0 + np.arange(200.0, 250.0))

    model = {
        "model_id": "timesfm2.5",
        "forecast_limits": {
            "max_input_length": 200,
            "min_input_length": 1,
            "max_output_length": 64,
            "max_group_rows": 64,
            "input_mode": {
                "max_target_count": 1,
                "max_history_covariate_count": -1,
                "supports_future_covariates": True,
            },
        },
    }
    plan = input_adaptation_plan(
        model,
        sample,
        policy_id=inference_module.INPUT_ADAPTATION_POLICY_ID,
    )
    assert plan is not None
    assert plan["covariate_mode"] == "paired_known_future_only"
    assert plan["request_covariate_dim"] == 1
    assert plan["request_future_covariate_dim"] == 1
    assert plan["covariate_selection_reason"] == (
        "model_requires_paired_history_future_covariates"
    )
    children = inference_module.adapted_request_samples(sample, plan)
    assert len(children) == sample["target_dim"]
    assert all(child["covariate_dim"] == 1 for child in children)
    assert all(
        child["covariate_column_names"] == ["known_0"]
        for child in children
    )
    content, _shape, _horizon = _bulk_request_content(
        "timesfm2.5", children
    )
    payload = msgpack.unpackb(content, raw=False)
    assert payload["history_covariates_shape"] == [3, 1, 200]
    assert payload["future_covariates_shape"] == [3, 1, 50]
    assert payload["future_covariate_history_indices"] == [0]


def test_bulk_batches_shrink_to_request_input_token_budget() -> None:
    model = {
        "forecast_limits": {
            "max_input_length": 200,
            "min_input_length": 1,
            "max_output_length": 64,
            "max_group_rows": 64,
            "input_mode": {
                "max_target_count": -1,
                "max_history_covariate_count": 0,
                "supports_future_covariates": False,
            },
        }
    }
    summary = inference_module._adaptation_summary()
    batches = list(
        _iter_model_bulk_batches(
            (_sample() for _index in range(5)),
            model=model,
            execution={
                "task_batch_size": 64,
                "maximum_request_input_tokens": 1200,
            },
            maximum_open_groups=2,
            maximum_buffered_bytes=1024 * 1024,
            summary=summary,
        )
    )
    assert [len(chunk) for _shard, _key, chunk in batches] == [1, 1, 1, 1, 1]
    assert all(
        _batch_scheduling_tokens(
            chunk, output_horizon_token_multiplier=1.0
        )
        <= 1200
        for _shard, _key, chunk in batches
    )


def test_input_token_limiter_reduces_only_expensive_request_concurrency() -> None:
    async def exercise() -> int:
        limiter = _InputTokenLimiter(100)
        active = 0
        maximum = 0

        async def worker() -> None:
            nonlocal active, maximum
            await limiter.acquire(60)
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1
            await limiter.release(60)

        await asyncio.gather(*(worker() for _index in range(4)))
        return maximum

    assert asyncio.run(exercise()) == 1


def test_jena_like_panel_is_split_to_four_views_per_bulk() -> None:
    model = {
        "forecast_limits": {
            "max_input_length": 8192,
            "min_input_length": 1,
            "max_output_length": 48,
            "max_group_rows": 64,
            "input_mode": {
                "max_target_count": -1,
                "max_history_covariate_count": -1,
                "supports_future_covariates": True,
            },
        }
    }

    def samples():
        for index in range(5):
            yield {
                "schema_version": "fixture",
                "sample_id": f"jena-{index}",
                "source_shard_index": 0,
                "context_length": 8192,
                "horizon": 48,
                "target_dim": 21,
                "target_column_names": [f"target_{i}" for i in range(21)],
                "covariate_dim": 2,
                "covariate_column_names": ["cov_0", "cov_1"],
                "frequency": "10min",
                "target": np.zeros((8240, 21)),
                "covariates": np.zeros((8240, 2)),
            }

    summary = inference_module._adaptation_summary()
    batches = list(
        _iter_model_bulk_batches(
            samples(),
            model=model,
            execution={
                "task_batch_size": 192,
                "maximum_bulk_rows": 64,
                "maximum_request_input_tokens": (
                    inference_module.DEFAULT_MAX_REQUEST_INPUT_TOKENS
                ),
            },
            maximum_open_groups=2,
            maximum_buffered_bytes=1024**3,
            summary=summary,
        )
    )
    assert [len(chunk) for _shard, _key, chunk in batches] == [4, 1]
    assert _batch_input_tokens(batches[0][2]) == 754_048


def test_streaming_bulk_writes_atomic_source_shards_without_task_file(
    tmp_path,
    monkeypatch,
) -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {
            "max_input_length": 16,
            "min_input_length": 1,
            "max_output_length": 8,
            "input_mode": {
                "max_target_count": -1,
                "max_history_covariate_count": 0,
                "supports_future_covariates": False,
            },
        },
    }

    def samples():
        for index in range(2):
            yield {
                "schema_version": "fixture",
                "sample_id": f"sample-{index}",
                "source_shard_index": index,
                "context_length": 10,
                "horizon": 2,
                "target_dim": 1,
                "target_column_names": ["target_0"],
                "covariate_dim": 0,
                "covariate_column_names": [],
                "covariates": None,
                "frequency": "H",
                "target": np.arange(12.0)[:, None],
            }

    async def fake_forecast(
        _client,
        *,
        children,
        **_kwargs,
    ):
        return {
            "forecasts": np.zeros((len(children), 1, 2), dtype=np.float32),
            "attempts": 1,
            "elapsed_seconds": 0.01,
            "error": None,
        }

    monkeypatch.setattr(inference_module, "_forecast_bulk_with_retry", fake_forecast)
    summary, stats, parts = asyncio.run(
        _run_streaming_bulk_model(
            model_id="fixture",
            model=model,
            execution={"task_batch_size": 2, "http_concurrency": 2},
            endpoints=["http://endpoint-a", "http://endpoint-b"],
            api_prefix="/ai/api/v1",
            sample_factory=samples,
            prediction_dir=tmp_path / "predictions",
            failure_path=tmp_path / "failures.jsonl",
            forecast_timeout_seconds=30,
            max_attempts=1,
            maximum_open_groups=2,
            maximum_inflight_batches=1,
            maximum_inflight_bytes=1024 * 1024,
        )
    )
    assert summary["compatible_sample_count"] == 2
    assert stats["prediction_count"] == 2
    assert stats["endpoint_count"] == 2
    assert [row["source_shard_index"] for row in parts] == [0, 1]
    assert not (tmp_path / "tasks").exists()
    assert [
        row["sample_id"]
        for part in parts
        for row in iter_prediction_parquet(part["path"])
    ] == ["sample-0", "sample-1"]


def test_streaming_bulk_recovers_failed_request_after_shard_drain(
    tmp_path,
    monkeypatch,
) -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {
            "max_input_length": 16,
            "min_input_length": 1,
            "max_output_length": 8,
            "input_mode": {
                "max_target_count": -1,
                "max_history_covariate_count": 0,
                "supports_future_covariates": False,
            },
        },
    }

    def samples():
        yield {
            "schema_version": "fixture",
            "sample_id": "sample-0",
            "source_shard_index": 0,
            "context_length": 10,
            "horizon": 2,
            "target_dim": 1,
            "target_column_names": ["target_0"],
            "covariate_dim": 0,
            "covariate_column_names": [],
            "covariates": None,
            "frequency": "H",
            "target": np.arange(12.0)[:, None],
        }

    call_count = 0

    async def flaky_forecast(
        _client,
        *,
        children,
        **_kwargs,
    ):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "forecasts": None,
                "attempts": 3,
                "elapsed_seconds": 3.0,
                "error": "HTTP 503 admission exhausted",
            }
        return {
            "forecasts": np.zeros((len(children), 1, 2), dtype=np.float32),
            "attempts": 1,
            "elapsed_seconds": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        inference_module,
        "_forecast_bulk_with_retry",
        flaky_forecast,
    )
    summary, stats, parts = asyncio.run(
        _run_streaming_bulk_model(
            model_id="fixture",
            model=model,
            execution={"task_batch_size": 2, "http_concurrency": 1},
            endpoints=["http://endpoint-a"],
            api_prefix="/ai/api/v1",
            sample_factory=samples,
            prediction_dir=tmp_path / "predictions",
            failure_path=tmp_path / "failures.jsonl",
            forecast_timeout_seconds=30,
            max_attempts=3,
            maximum_open_groups=2,
            maximum_inflight_batches=1,
            maximum_inflight_bytes=1024 * 1024,
        )
    )
    assert summary["compatible_sample_count"] == 1
    assert stats["prediction_count"] == 1
    assert stats["failure_count"] == 0
    assert stats["attempt_count"] == 4
    assert stats["tail_retry_bulk_request_count"] == 1
    assert stats["tail_retry_recovered_view_count"] == 1
    assert len(parts) == 1
    assert (tmp_path / "failures.jsonl").read_text() == ""
