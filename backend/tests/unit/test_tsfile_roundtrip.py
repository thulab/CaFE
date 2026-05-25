"""把 TsFile 写入/读回 spike 固化成回归测试（plan Task B1.1）。

验证 spec §2 的可行性结论：CSV 值矩阵 → dataframe_to_tsfile(表模型) → 按行号切片读回，逐位一致。
"""
import pandas as pd
import pytest

tsfile = pytest.importorskip("tsfile")


def test_dataframe_to_tsfile_roundtrip(tmp_path):
    # 源数据：6 行，ms epoch 时间戳 + 单值列 + dataset_id 标签列。
    base_ms = 1_700_000_000_000
    step_ms = 3_600_000  # 1h
    values = [100.0, 101.5, 103.0, 104.2, 105.1, 106.0]
    df = pd.DataFrame(
        {
            "time": [base_ms + i * step_ms for i in range(len(values))],
            "value": values,
            "dataset_id": ["ds1"] * len(values),
        }
    )

    path = tmp_path / "shard.tsfile"
    tsfile.dataframe_to_tsfile(
        df,
        str(path),
        table_name="tsbench",
        time_column="time",
        tag_column=["dataset_id"],
    )

    with tsfile.TsFileDataFrame(str(path)) as read_df:
        series = read_df["tsbench.ds1.value"]
        window = series[2:5]

    assert [float(v) for v in window] == values[2:5]
