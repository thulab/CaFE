from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe.benchmark_extension.generation import generate_dataset


def _fixture(tmp_path: Path, monkeypatch) -> Path:
    asset = tmp_path / "gift" / "fixture" / "H"
    asset.mkdir(parents=True)
    t = np.arange(800.0)
    target = 0.02 * t + np.sin(t / 10.0)
    table = pa.table(
        {
            "item_id": ["native-item"],
            "start": ["2020-01-01"],
            "freq": ["H"],
            "target": [target.tolist()],
        }
    )
    with pa.OSFile(str(asset / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    spec = protocol.DatasetSpec(
        "gift_fixture",
        "Fixture",
        "fixture/H",
        "fixture/H",
        "Test",
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    return tmp_path / "gift"


def test_generation_uses_all_official_instances_and_shared_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment" / "gift_fixture"
    manifest = generate_dataset(
        "gift_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=13,
        capability_ids=("trend", "regime_switching"),
        max_instances=None,
    )
    # ceil(.1 * 800 / 48) official windows, no random origin sampling.
    assert manifest["official_instance_count"] == 2
    assert manifest["files"]["official_baselines"]["row_count"] == 2
    treatments = list(
        protocol.iter_jsonl(
            dataset_root / "01_generation" / "capability_treatments.jsonl"
        )
    )
    assert len(treatments) == 2 * 2 * 5
    assert {row["capability_level"] for row in treatments} == {1, 2, 3, 4, 5}
    assert all(row["source_distance_gate"]["accepted"] for row in treatments)
    assert all(row["counterfactual_member"] == 1 for row in treatments)
    assert len({row["baseline_sample_id"] for row in treatments}) == 2


def test_augmentation_seed_changes_parameters_not_official_instances(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _fixture(tmp_path, monkeypatch)
    roots = [tmp_path / f"experiment_{seed}" / "gift_fixture" for seed in (1, 2)]
    rows = []
    for seed, root in zip((1, 2), roots, strict=True):
        generate_dataset(
            "gift_fixture",
            gift_eval_dir=gift_root,
            dataset_root=root,
            term="short",
            augmentation_seed=seed,
            capability_ids=("trend",),
            max_instances=1,
        )
        rows.append(
            list(
                protocol.iter_jsonl(
                    root / "01_generation" / "capability_treatments.jsonl"
                )
            )
        )
    assert {row["official_instance_id"] for row in rows[0]} == {
        row["official_instance_id"] for row in rows[1]
    }
    assert [row["sampled_coordinate"] for row in rows[0]] != [
        row["sampled_coordinate"] for row in rows[1]
    ]
