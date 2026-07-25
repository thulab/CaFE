from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import paper_sample_explorer as explorer  # noqa: E402


def test_server_defaults_to_requested_public_binding() -> None:
    args = explorer.parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 8766


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def indexed_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "E2_dynamic_stability"
    data_dir.mkdir()
    group_id = "dataset_a__univariate__capability_a__r1__s000"
    samples = []
    for intensity in range(1, 6):
        master_id = f"{group_id}__i{intensity}"
        samples.append(
            {
                "analysis_block_id": "A",
                "analysis_block_index": 0,
                "capability_id": "capability_a",
                "context_length": 4,
                "dataset_id": "dataset_a",
                "frequency": "h",
                "hierarchy": None,
                "horizon": 2,
                "intensity": intensity,
                "master_sample_id": master_id,
                "paired_group_id": group_id,
                "pool_index": 0,
                "realized_features_by_context": {
                    "2": {"feature_a": intensity + 0.2},
                    "4": {"feature_a": intensity + 0.4},
                },
                "round_index": 1,
                "sample_index": 0,
                "season_length": 2,
                "target": [
                    [float(intensity + offset), float(2 * intensity + offset)]
                    for offset in range(6)
                ],
                "target_dim": 2,
                "target_feature": "feature_a",
                "target_relative_level": (intensity - 1) / 4,
                "target_strength": intensity + 0.5,
            }
        )
    write_jsonl(data_dir / "samples.jsonl", samples)

    prediction_files = {}
    for model_id, filename, adjustment in (
        ("model-a", "model-a.jsonl", 0.25),
        ("model-b", "model-b.jsonl", 0.5),
        ("naive", "naive.jsonl", -0.25),
    ):
        rows = []
        for sample in samples:
            for context in (2, 4):
                _, actual = explorer.standardized_view(
                    sample["target"], context, 2, None
                )
                rows.append(
                    {
                        "context_length": context,
                        "forecast": [
                            [value + adjustment for value in target_row]
                            for target_row in actual
                        ],
                        "master_sample_id": sample["master_sample_id"],
                        "metrics": {"mae": abs(adjustment), "mase": 0.5},
                        "model_group": "baseline" if model_id == "naive" else "test",
                    }
                )
        path = data_dir / "predictions" / filename
        write_jsonl(path, list(reversed(rows)))
        prediction_files[model_id] = {"path": f"predictions/{filename}"}

    (data_dir / "generation_config.json").write_text(
        json.dumps({"context_lengths": [2, 4]}), encoding="utf-8"
    )
    (data_dir / "inference_manifest.json").write_text(
        json.dumps({"prediction_files": {"synthetic": prediction_files}}),
        encoding="utf-8",
    )
    score_rows = [
        "model_id,dataset_id,task_id,capability_id,intensity,mase_mean,"
        "mase_std,master_sample_count,score_policy,model_rank,"
        "compatible_model_count"
    ]
    for intensity in range(1, 6):
        score_rows.extend(
            [
                f"model-a,dataset_a,univariate,capability_a,{intensity},"
                f"{0.30 + intensity * 0.01},0.1,2,oracle_context,1,2",
                f"model-b,dataset_a,univariate,capability_a,{intensity},"
                f"{0.50 + intensity * 0.01},0.1,2,oracle_context,2,2",
            ]
        )
    (data_dir / "cell_full_pool_scores.csv").write_text(
        "\n".join(score_rows) + "\n", encoding="utf-8"
    )
    index_path = data_dir / "index.sqlite3"
    sources = explorer.ensure_index(
        data_dir, index_path, progress=lambda _message: None
    )
    assert len(sources) == 3
    return data_dir, index_path


def test_indexed_sample_returns_five_aligned_intensities(
    indexed_fixture: tuple[Path, Path],
) -> None:
    data_dir, index_path = indexed_fixture
    reader = explorer.SampleExplorer(data_dir, index_path)
    try:
        metadata = reader.meta()
        assert metadata["index"] == {
            "builtAt": metadata["index"]["builtAt"],
            "sampleCount": 5,
            "groupCount": 1,
            "predictionCount": 30,
        }
        assert metadata["contexts"] == [2, 4]
        assert [model["id"] for model in metadata["models"]] == [
            "naive",
            "model-a",
            "model-b",
        ]

        groups = reader.groups("dataset_a", "capability_a")
        assert len(groups) == 1
        payload = reader.sample(groups[0]["id"], 4)
    finally:
        reader.close()

    assert [row["intensity"] for row in payload["intensities"]] == [1, 2, 3, 4, 5]
    assert payload["targetColumns"] == ["target_0", "target_1"]
    first = payload["intensities"][0]
    assert len(first["history"]) == 4
    assert len(first["actual"]) == 2
    assert first["realizedFeature"] == pytest.approx(1.4)
    assert set(first["models"]) == {"naive", "model-a", "model-b"}
    assert first["models"]["model-a"]["metrics"]["mae"] == pytest.approx(0.25)
    assert first["models"]["model-a"]["forecast"][0][0] == pytest.approx(
        first["actual"][0][0] + 0.25
    )
    assert payload["missingPredictions"] == []
    ranking = payload["oracleContextRanking"]
    assert ranking["scorePolicy"] == "oracle_context"
    assert ranking["best"] == {
        "modelId": "model-a",
        "maseMean": pytest.approx(0.33),
        "sampleCount": 10,
        "intensityCount": 5,
        "rank": 1,
    }
    assert ranking["gapToRunnerUp"] == pytest.approx(0.2)


def test_context_restandardization_uses_only_visible_history() -> None:
    target = [[0.0], [10.0], [20.0], [30.0], [40.0], [50.0]]
    history, actual = explorer.standardized_view(target, 2, 2, None)
    assert history == [[-1.0], [1.0]]
    assert actual == [[3.0], [5.0]]


def test_v8_prediction_metrics_reject_target_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="target dimension mismatch"):
        explorer._prediction_metrics(
            actual=[[1.0, 2.0]],
            forecast=[[1.0]],
            mase_scale=1.0,
        )


def test_http_api_and_static_page(indexed_fixture: tuple[Path, Path]) -> None:
    data_dir, index_path = indexed_fixture
    reader = explorer.SampleExplorer(data_dir, index_path)
    server = explorer.ExplorerHTTPServer(("127.0.0.1", 0), reader)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base_url}/api/meta") as response:  # noqa: S310 - local test server.
            metadata = json.load(response)
        with urlopen(f"{base_url}/") as response:  # noqa: S310 - local test server.
            page = response.read().decode("utf-8")
            content_security_policy = response.headers["Content-Security-Policy"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        reader.close()

    assert metadata["index"]["groupCount"] == 1
    assert "TS Lens" in page
    assert "default-src 'self'" in content_security_policy


@pytest.fixture()
def v8_indexed_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict]:
    data_dir = tmp_path / "v8_formal_fixture"
    dataset_id = "gift_fixture_h"
    shard_name = "seed_000000_000001"
    inference_dir = data_dir / dataset_id / "03_inference" / shard_name
    tasks: list[dict] = []

    def add_group(
        capability_id: str,
        *,
        member: int | None = None,
    ) -> None:
        paired_group = (
            f"v8__{dataset_id}__{capability_id}__primary__seed000000"
        )
        for intensity in range(1, 6):
            member_suffix = f"__m{member}" if member is not None else ""
            master_id = (
                f"v8__{dataset_id}__{capability_id}__primary__"
                f"i{intensity}__seed000000{member_suffix}"
            )
            for context in (2, 4):
                target = [
                    [float(intensity + offset), float(2 * intensity + offset)]
                    for offset in range(context + 2)
                ]
                tasks.append(
                    {
                        "capability_id": capability_id,
                        "context_length": context,
                        "counterfactual_member": member,
                        "dataset_id": dataset_id,
                        "evaluation_table": "main",
                        "frequency": "H",
                        "generator_family_role": "primary",
                        "horizon": 2,
                        "intensity": intensity,
                        "mase_scale": 2.0,
                        "master_sample_id": master_id,
                        "paired_group_id": paired_group,
                        "realized_features": {
                            "feature_a": intensity + context / 10
                        },
                        "sample_id": f"{master_id}__L{context}",
                        "season_length": 2,
                        "seed_index": 0,
                        "target": target,
                        "target_dim": 2,
                        "target_feature": "feature_a",
                        "target_feature_value": intensity + context / 10,
                    }
                )

    add_group("trend")
    add_group("covariate_response", member=0)
    add_group("covariate_response", member=1)
    tasks.extend(
        [
            {
                **tasks[0],
                "evaluation_table": "observation_noise_robustness",
                "sample_id": "ignored-robustness__L2",
                "master_sample_id": "ignored-robustness",
            },
            {
                **tasks[0],
                "generator_family_role": "secondary",
                "sample_id": "ignored-secondary__L2",
                "master_sample_id": "ignored-secondary",
            },
        ]
    )
    task_path = inference_dir / "forecast_views.jsonl"
    write_jsonl(task_path, tasks)
    (inference_dir / "task_manifest.json").write_text(
        json.dumps({"task_file": {"path": str(task_path)}}),
        encoding="utf-8",
    )

    prediction_descriptors = []
    for model_id, adjustment in (("model-a", 0.25), ("model-b", 0.5)):
        prediction_path = (
            inference_dir
            / "model_shards"
            / model_id
            / "predictions"
            / f"{model_id}.jsonl"
        )
        predictions = []
        for task in tasks:
            actual = task["target"][task["context_length"] :]
            predictions.append(
                {
                    "forecast": [
                        [value + adjustment for value in row] for row in actual
                    ],
                    "input_adaptation": {
                        "target_mode": "native_multivariate"
                    },
                    "model_id": model_id,
                    "sample_id": task["sample_id"],
                }
            )
        write_jsonl(prediction_path, list(reversed(predictions)))
        prediction_descriptors.append(
            {"model_id": model_id, "path": str(prediction_path)}
        )
    (inference_dir / "inference_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "models": ["model-a", "model-b"],
                "predictions": {"files": prediction_descriptors},
            }
        ),
        encoding="utf-8",
    )

    score_dir = data_dir / dataset_id / "04_analysis" / shard_name
    score_dir.mkdir(parents=True)
    score_rows = []
    for capability_id in ("trend", "covariate_response"):
        for rank, (model_id, score) in enumerate(
            (("model-a", 0.25), ("model-b", 0.5)), start=1
        ):
            score_rows.append(
                {
                    "accuracy_rank": rank,
                    "accuracy_score": score,
                    "capability_id": capability_id,
                    "context_policy": "oracle_context",
                    "evaluation_table": "main",
                    "generator_family_role": "primary",
                    "intensities": [1, 2, 3, 4, 5],
                    "model_id": model_id,
                    "seed_count": 1,
                }
            )
    (score_dir / "scores.json").write_text(
        json.dumps({"scores": score_rows}), encoding="utf-8"
    )
    (data_dir / "distributed_analysis_manifest.json").write_text(
        json.dumps(
            {
                "datasets": [{"dataset_id": dataset_id}],
                "seed_count": 1,
                "seed_start": 0,
            }
        ),
        encoding="utf-8",
    )
    index_path = data_dir / "index.sqlite3"
    artifacts = explorer.ensure_v8_index(
        data_dir,
        index_path,
        progress=lambda _message: None,
    )
    return data_dir, index_path, artifacts


def test_v8_index_filters_main_primary_and_separates_members(
    v8_indexed_fixture: tuple[Path, Path, dict],
) -> None:
    data_dir, index_path, artifacts = v8_indexed_fixture
    reader = explorer.V8SampleExplorer(data_dir, index_path, artifacts)
    try:
        metadata = reader.meta()
        assert metadata["experiment"] == {
            "version": "v8",
            "id": "v8_formal_fixture",
            "shard": "seed_000000_000001",
            "sampleScope": "main/primary/clean",
        }
        assert metadata["index"] == {
            "builtAt": metadata["index"]["builtAt"],
            "sampleCount": 15,
            "groupCount": 3,
            "predictionCount": 60,
            "groupUnit": "seed groups",
        }
        assert metadata["contexts"] == [2, 4]
        assert [model["id"] for model in metadata["models"]] == [
            "model-a",
            "model-b",
        ]

        covariate_groups = reader.groups(
            "gift_fixture_h", "covariate_response"
        )
        assert [row["counterfactualMember"] for row in covariate_groups] == [
            0,
            1,
        ]
        trend_group = reader.groups("gift_fixture_h", "trend")[0]
        payload = reader.sample(trend_group["id"], 4)
    finally:
        reader.close()

    assert [row["intensity"] for row in payload["intensities"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    first = payload["intensities"][0]
    assert len(first["history"]) == 4
    assert len(first["actual"]) == 2
    assert first["targetStrength"] == pytest.approx(1.4)
    assert first["realizedFeature"] == pytest.approx(1.4)
    assert first["models"]["model-a"]["metrics"] == {
        "mae": pytest.approx(0.25),
        "mase": pytest.approx(0.125),
    }
    assert payload["missingPredictions"] == []
    assert payload["oracleContextRanking"]["best"]["modelId"] == "model-a"


def test_default_v8_parent_resolves_newest_completed_experiment(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "v8"
    older = parent / "older"
    newer = parent / "newer"
    older.mkdir(parents=True)
    newer.mkdir()
    (older / "model_major_inference_status.json").write_text(
        "{}", encoding="utf-8"
    )
    (newer / "distributed_analysis_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    assert explorer.resolve_data_dir(parent) == newer.resolve()
