from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe.benchmark_extension.gift_eval import (
    iter_gift_eval_instances,
    official_forecast_origins,
    official_window_count,
    prediction_length,
)


def _write_arrow(path: Path, targets: list[np.ndarray], *, frequency: str) -> None:
    path.mkdir(parents=True)
    rows = pa.table(
        {
            "item_id": [f"item_{index}" for index in range(len(targets))],
            "start": ["2020-01-01"] * len(targets),
            "freq": [frequency] * len(targets),
            "target": [values.tolist() for values in targets],
        }
    )
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
    _write_arrow(asset, [target], frequency="H")
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
