from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe.benchmark_extension.gift_eval import (
    future_label_window_audit,
    iter_gift_eval_instances,
    official_forecast_origins,
    official_window_count,
    prediction_length,
)


def _write_arrow(
    path: Path,
    targets: list[np.ndarray],
    *,
    frequency: str,
    past_covariates: list[np.ndarray] | None = None,
) -> None:
    path.mkdir(parents=True)
    fields = {
        "item_id": [f"item_{index}" for index in range(len(targets))],
        "start": ["2020-01-01"] * len(targets),
        "freq": [frequency] * len(targets),
        "target": [values.tolist() for values in targets],
    }
    if past_covariates is not None:
        fields["past_feat_dynamic_real"] = [
            values.tolist() for values in past_covariates
        ]
    rows = pa.table(fields)
    with pa.OSFile(str(path / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa_ipc.new_stream(sink, rows.schema) as writer:
            writer.write_table(rows)


def test_official_short_term_window_formula() -> None:
    records = [("a", np.arange(1000.0)), ("b", np.arange(1200.0))]
    horizon = prediction_length("gift_electricity_h", "H", term="short")
    assert horizon == 48
    assert official_window_count("gift_electricity_h", records, horizon) == 3
    assert official_forecast_origins(
        1000,
        prediction_length_value=48,
        window_count=3,
    ) == (856, 904, 952)


def test_m4_has_one_official_window() -> None:
    records = [("a", np.arange(200.0))]
    horizon = prediction_length("gift_m4_hourly", "H", term="short")
    assert horizon == 48
    assert official_window_count("gift_m4_hourly", records, horizon) == 1


def test_native_multivariate_record_is_not_channel_expanded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset = tmp_path / "ett1" / "H"
    target = np.arange(7 * 600.0).reshape(7, 600)
    covariates = np.vstack(
        (np.sin(np.arange(600.0) / 12.0), np.cos(np.arange(600.0) / 12.0))
    )
    _write_arrow(
        asset,
        [target],
        frequency="H",
        past_covariates=[covariates],
    )
    spec = protocol.DatasetSpec(
        "gift_fixture",
        "Fixture",
        "ett1/H",
        "ett1/H",
        "Test",
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)

    rows = list(
        iter_gift_eval_instances(
            spec.dataset_id,
            tmp_path,
            max_instances=1,
        )
    )
    assert len(rows) == 1
    assert rows[0].target_dim == 7
    assert rows[0].history.shape == (504, 7)
    assert rows[0].future.shape == (48, 7)
    assert rows[0].history_covariates.shape == (504, 2)
    assert rows[0].covariate_availability == ("past_only", "past_only")
    assert rows[0].future_covariate_visible == (False, False)


def test_dataset_without_native_covariates_does_not_get_calendar_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset = tmp_path / "fixture"
    _write_arrow(asset, [np.arange(600.0)], frequency="H")
    spec = protocol.DatasetSpec(
        "gift_fixture_no_covariates",
        "Fixture",
        "fixture",
        "fixture",
        "Test",
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    instance = next(iter_gift_eval_instances(spec.dataset_id, tmp_path))
    assert instance.history_covariates.shape[1] == 0
    assert instance.covariate_column_names == ()


def test_history_imputation_never_reads_official_future(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset = tmp_path / "fixture"
    target = np.arange(600.0)
    target[10] = np.nan
    _write_arrow(asset, [target], frequency="H")
    spec = protocol.DatasetSpec(
        "gift_fixture",
        "Fixture",
        "fixture",
        "fixture",
        "Test",
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    first = next(iter_gift_eval_instances(spec.dataset_id, tmp_path))

    target[-48:] = 1e12
    replacement = tmp_path / "replacement"
    _write_arrow(replacement / "fixture", [target], frequency="H")
    second = next(iter_gift_eval_instances(spec.dataset_id, replacement))
    np.testing.assert_array_equal(first.history, second.history)


def test_forecast_window_with_any_missing_target_cell_is_excluded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset = tmp_path / "fixture"
    target = np.vstack((np.arange(800.0), np.arange(800.0) + 10.0))
    target[1, -1] = np.nan
    _write_arrow(asset, [target], frequency="H")
    spec = protocol.DatasetSpec(
        "gift_fixture_complete_labels",
        "Fixture",
        "fixture",
        "fixture",
        "Test",
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)

    audit = future_label_window_audit(
        target,
        prediction_length_value=48,
        window_count=2,
    )
    assert audit == {
        "official_window_count": 2,
        "complete_future_label_count": 1,
        "partially_missing_future_label_count": 1,
        "fully_missing_future_label_count": 0,
    }
    instances = list(iter_gift_eval_instances(spec.dataset_id, tmp_path))
    assert len(instances) == 1
    assert instances[0].window_index == 0
    assert bool(np.all(instances[0].future_observed_mask))
