from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe.benchmark_extension.generation import (
    _replay_contract_instance,
    generate_dataset,
    iter_replay_contract_work_items,
    iter_replayed_samples,
)
from cafe.benchmark_extension.storage import (
    iter_compact_parquet,
    parquet_file_record,
    write_compact_parquet,
)
from cafe.benchmark_extension.validation import validate_generation


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    missing_final_label: bool = False,
) -> Path:
    asset = tmp_path / "gift" / "fixture" / "H"
    asset.mkdir(parents=True)
    t = np.arange(800.0)
    target = 0.02 * t + np.sin(t / 10.0)
    if missing_final_label:
        target[-1] = np.nan
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


def _panel_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    include_covariates: bool = True,
) -> Path:
    asset = tmp_path / "gift-panel" / "fixture-panel" / "H"
    asset.mkdir(parents=True)
    rng = np.random.default_rng(7)
    t = np.arange(800.0)
    driver = np.sin(t / 9.0) + 0.05 * rng.normal(size=t.size)
    panel = np.vstack(
        (
            driver,
            0.8 * np.roll(driver, 2) + 0.08 * rng.normal(size=t.size),
            -0.6 * driver + 0.08 * rng.normal(size=t.size),
        )
    )
    columns = {
        "item_id": ["native-panel"],
        "start": ["2020-01-01"],
        "freq": ["H"],
        "target": [panel.tolist()],
    }
    if include_covariates:
        columns["past_feat_dynamic_real"] = [
            np.vstack(
                (
                    np.sin(t / 13.0),
                    np.cos(t / 17.0),
                )
            ).tolist()
        ]
    table = pa.table(columns)
    with pa.OSFile(str(asset / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    spec = protocol.DatasetSpec(
        "gift_panel_fixture",
        "Panel Fixture",
        "fixture-panel/H",
        "fixture-panel/H",
        "Test",
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    return tmp_path / "gift-panel"


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
    assert manifest["config"]["source_distance_configuration"] == {
        "strength_reference": "full_official_history_macro_normalized_rms",
        "model_max_contexts": {
            "tirex2": 2048,
            "Timer-4.0": 8192,
            "Chronos-2": 8192,
            "Timer-3.5": 11520,
            "timesfm2.5": 15360,
            "moirai2": 16384,
            "toto2.0": 16384,
        },
        "minimum_model_context_macro_distance": 0.10,
        "maximum_model_context_macro_distance": 2.0,
        "maximum_model_context_channel_distance": 3.0,
    }
    assert manifest["unavailable_reason_count_by_capability"] == {
        "regime_switching": {},
        "trend": {},
    }
    treatments = list(iter_compact_parquet(dataset_root / "01_generation" / "treatment_contracts.parquet"))
    assert len(treatments) == 2 * 2 * 5
    assert {row["capability_level"] for row in treatments} == {1, 2, 3, 4, 5}
    assert all(row["source_distance_gate"]["accepted"] for row in treatments)
    assert all("mechanism_scoring_gate" in row for row in treatments)
    assert all(
        row["mechanism_scoring_gate"]["minimum_required_mase_rms"] == 0.05
        for row in treatments
    )
    assert all(row["counterfactual_member"] == 1 for row in treatments)
    assert len({row["baseline_sample_id"] for row in treatments}) == 2
    assert all("target" not in row and "covariates" not in row for row in treatments)


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
            list(iter_compact_parquet(root / "01_generation" / "treatment_contracts.parquet"))
        )
    assert {row["official_instance_id"] for row in rows[0]} == {
        row["official_instance_id"] for row in rows[1]
    }
    assert [row["sampled_coordinate"] for row in rows[0]] != [
        row["sampled_coordinate"] for row in rows[1]
    ]


def test_generation_excludes_incomplete_future_label_windows_and_audits_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _fixture(tmp_path, monkeypatch, missing_final_label=True)
    dataset_root = tmp_path / "experiment-complete-labels" / "gift_fixture"
    manifest = generate_dataset(
        "gift_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=13,
        capability_ids=("trend",),
        max_instances=None,
        workers=2,
    )
    assert manifest["official_instance_count"] == 1
    assert manifest["official_window_selection_audit"] == {
        "official_window_count": 2,
        "complete_future_label_count": 1,
        "partially_missing_future_label_count": 1,
        "fully_missing_future_label_count": 0,
    }
    assert manifest["config"]["future_label_eligibility_policy"] == (
        "require_every_horizon_target_cell_finite_v1"
    )
    baselines = list(
        iter_compact_parquet(
            dataset_root / "01_generation" / "official_instances.parquet"
        )
    )
    assert len(baselines) == 1
    assert baselines[0]["official_instance_id"].endswith("__w00__o704")
    report = validate_generation(
        dataset_root,
        gift_eval_dir=gift_root,
        mode="publication",
        workers=1,
    )
    assert report["accepted"]


def test_common_and_cross_emit_one_target_only_input_ablation_per_treatment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _panel_fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment-panel" / "gift_panel_fixture"
    manifest = generate_dataset(
        "gift_panel_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=19,
        capability_ids=("common_factor", "cross_series_dependence"),
        max_instances=1,
    )
    replayed = list(iter_replayed_samples(manifest, gift_eval_dir=gift_root))
    treatments = {
        row["sample_id"]: row
        for row in replayed
        if row["evaluation_table"] == "gift_eval_capability_treatment"
    }
    ablations = [
        row for row in replayed
        if row["evaluation_table"] == "gift_eval_capability_input_ablation"
    ]
    assert manifest["input_ablation_count"] == len(ablations) == 10
    assert len(treatments) == 10
    for row in ablations:
        source = treatments[row["input_ablation_source_sample_id"]]
        target = np.asarray(row["target"])
        source_target = np.asarray(source["target"])
        assessed = int(row["assessed_target_indices"][0])
        removed = [
            index for index in range(source_target.shape[1]) if index != assessed
        ]
        assert row["target_dim"] == 1
        assert row["ablation_target_indices"] == [0]
        assert row["ablated_input_indices"] == removed
        np.testing.assert_array_equal(
            target[:, 0], source_target[:, assessed]
        )
        np.testing.assert_array_equal(row["covariates"], source["covariates"])
        assert row["input_ablation_metadata"] == {
            "policy": "auxiliary_input_removal_v1",
            "mode": "assessed_target_only",
            "assessed_target_history_unchanged": True,
            "scored_future_unchanged": True,
            "kept_source_target_indices": [assessed],
            "removed_source_target_indices": removed,
        }


def test_covariate_impulse_ablation_removes_treated_covariate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _panel_fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment-covariate" / "gift_panel_fixture"
    manifest = generate_dataset(
        "gift_panel_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=31,
        capability_ids=("covariate_impulse_response",),
        max_instances=1,
    )
    replayed = list(iter_replayed_samples(manifest, gift_eval_dir=gift_root))
    treatments = {
        row["sample_id"]: row
        for row in replayed
        if row["evaluation_table"] == "gift_eval_capability_treatment"
    }
    ablations = [
        row
        for row in replayed
        if row["evaluation_table"] == "gift_eval_capability_input_ablation"
    ]
    assert len(treatments) == len(ablations) == 5
    for ablation in ablations:
        treatment = treatments[ablation["input_ablation_source_sample_id"]]
        np.testing.assert_array_equal(ablation["target"], treatment["target"])
        removed = int(ablation["ablated_input_indices"][0])
        kept = [
            index for index in range(int(treatment["covariate_dim"]))
            if index != removed
        ]
        assert ablation["ablation_target_indices"] == [
            int(ablation["assessed_target_indices"][0])
        ]
        assert ablation["covariate_dim"] == len(kept)
        np.testing.assert_array_equal(
            np.asarray(ablation["covariates"]),
            np.asarray(treatment["covariates"])[:, kept],
        )
        for field in (
            "covariate_column_names",
            "covariate_availability",
            "future_covariate_visible",
            "covariate_types",
        ):
            assert ablation[field] == [treatment[field][index] for index in kept]
        assert ablation["input_ablation_metadata"]["mode"] == (
            "remove_treated_covariate"
        )


def test_replay_handles_panel_ablations_without_native_covariates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _panel_fixture(
        tmp_path,
        monkeypatch,
        include_covariates=False,
    )
    dataset_root = tmp_path / "experiment-no-covariates" / "gift_panel_fixture"
    manifest = generate_dataset(
        "gift_panel_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=37,
        capability_ids=("common_factor",),
        max_instances=1,
    )

    replayed = list(iter_replayed_samples(manifest, gift_eval_dir=gift_root))
    ablations = [
        row
        for row in replayed
        if row["evaluation_table"] == "gift_eval_capability_input_ablation"
    ]
    assert len(ablations) == 5
    assert all(row["covariates"] is None for row in ablations)


def test_replay_keeps_legacy_shift_contract_compatibility(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _panel_fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment-legacy-replay" / "gift_panel_fixture"
    manifest = generate_dataset(
        "gift_panel_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=43,
        capability_ids=("common_factor",),
        max_instances=1,
    )
    instance, baseline, treatments, ablations = next(
        iter_replay_contract_work_items(manifest, gift_eval_dir=gift_root)
    )
    source = treatments[0]
    legacy = dict(ablations[0])
    legacy.update(
        {
            "target_dim": source["target_dim"],
            "target_column_names": source["target_column_names"],
            "future_observed_mask": source.get("future_observed_mask"),
            "input_ablation_metadata": {
                "policy": "deterministic_least_aligned_circular_shift_v1",
                "channel_audit": {
                    str(channel): {"circular_shift": 1}
                    for channel in legacy["ablated_input_indices"]
                },
            },
        }
    )
    replayed = _replay_contract_instance(
        instance, baseline, [source], [legacy]
    )
    treatment_target = np.asarray(replayed[1]["target"])
    legacy_target = np.asarray(replayed[2]["target"])
    context = int(legacy["context_length"])
    for channel in legacy["ablated_input_indices"]:
        np.testing.assert_array_equal(
            legacy_target[:context, channel],
            np.roll(treatment_target[:context, channel], 1),
        )


def test_compact_contract_replay_matches_every_stored_delta_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _panel_fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment-replay" / "gift_panel_fixture"
    manifest = generate_dataset(
        "gift_panel_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=23,
        capability_ids=(
            "trend",
            "multi_seasonal",
            "time_varying_seasonality",
            "regime_switching",
            "nonlinear_persistence",
            "predictable_intermittency",
            "common_factor",
            "cross_series_dependence",
            "covariate_impulse_response",
        ),
        max_instances=1,
    )
    replayed = list(
        iter_replayed_samples(
            manifest,
            gift_eval_dir=gift_root,
            replay_workers=4,
        )
    )
    baseline = next(
        row
        for row in replayed
        if row["evaluation_table"] == "gift_eval_official_baseline"
    )
    baseline_target = np.asarray(baseline["target"], dtype=float)
    treatments = [
        row
        for row in replayed
        if row["evaluation_table"] == "gift_eval_capability_treatment"
    ]
    assert treatments
    for row in treatments:
        target = np.asarray(row["target"], dtype=float)
        context = int(row["context_length"])
        assert isinstance(row["target"], np.ndarray)
        assert _array_sha256(target) == row["target_sha256"]
        assert _array_sha256(target[:context]) == row["history_sha256"]
        assert _array_sha256(target[context:]) == row["future_sha256"]
        assert (
            _array_sha256(target[:context] - baseline_target[:context])
            == row["history_delta_sha256"]
        )
        assert (
            _array_sha256(target[context:] - baseline_target[context:])
            == row["future_delta_sha256"]
        )
    report = validate_generation(dataset_root)
    assert report["accepted"]
    assert report["input_ablation_count"] == 15

    ablation_path = dataset_root / "01_generation" / "input_ablation_contracts.parquet"
    tampered = list(iter_compact_parquet(ablation_path))
    tampered[0]["input_ablation_metadata"]["kept_source_target_indices"] = [1]
    write_compact_parquet(ablation_path, tampered)
    manifest_path = dataset_root / "01_generation" / "manifest.json"
    changed_manifest = protocol.read_json(manifest_path)
    changed_manifest["files"]["input_ablations"] = {
        **parquet_file_record(ablation_path, row_count=len(tampered)),
    }
    protocol.write_json(manifest_path, changed_manifest)
    rejected = validate_generation(dataset_root, mode="publication", workers=1)
    assert not rejected["accepted"]
    assert any(
        row["scope"] == "input_ablations" for row in rejected["failures"]
    )


def test_model_context_replay_matches_full_replay_and_shard_partitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment-suffix" / "gift_fixture"
    manifest = generate_dataset(
        "gift_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=41,
        capability_ids=(
            "trend",
            "multi_seasonal",
            "time_varying_seasonality",
            "regime_switching",
            "predictable_intermittency",
        ),
        max_instances=None,
        shard_size=1,
    )
    full = {
        row["sample_id"]: row
        for row in iter_replayed_samples(manifest, gift_eval_dir=gift_root)
    }
    suffix = {
        row["sample_id"]: row
        for row in iter_replayed_samples(
            manifest,
            gift_eval_dir=gift_root,
            replay_workers=2,
            maximum_context=96,
        )
    }
    assert suffix.keys() == full.keys()
    for sample_id, row in suffix.items():
        source = full[sample_id]
        context = int(source["context_length"])
        start = max(0, context - 96)
        expected_target = np.vstack(
            (
                np.asarray(source["target"])[start:context],
                np.asarray(source["target"])[context:],
            )
        )
        np.testing.assert_allclose(row["target"], expected_target, atol=1e-12)
        if source.get("covariates") is not None:
            expected_covariates = np.vstack(
                (
                    np.asarray(source["covariates"])[start:context],
                    np.asarray(source["covariates"])[context:],
                )
            )
            np.testing.assert_allclose(
                row["covariates"], expected_covariates, atol=1e-12
            )
        assert row["materialized_history_start"] == start

    partitions = [
        {
            row["sample_id"]
            for row in iter_replayed_samples(
                manifest,
                gift_eval_dir=gift_root,
                source_shard_count=2,
                source_shard_index=index,
                maximum_context=96,
            )
        }
        for index in range(2)
    ]
    assert partitions[0]
    assert partitions[1]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0] | partitions[1] == set(full)
