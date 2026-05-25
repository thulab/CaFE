import pandas as pd
import tsfile
from fastapi.testclient import TestClient

from app.main import create_app

_BASE_MS = 1_700_000_000_000  # 任意 ms epoch 基准


def _write_tsfile(path, devices=("dev1",), n=8):
    rows = []
    for device in devices:
        for i in range(n):
            rows.append(
                {"time": _BASE_MS + i * 3600_000, "target": 100.0 + i, "extra": 20.0 + i, "dataset_id": device}
            )
    tsfile.dataframe_to_tsfile(
        pd.DataFrame(rows), str(path), table_name="tsbench", time_column="time", tag_column=["dataset_id"]
    )


def test_sniff_headerless_csv_reports_has_header_false(tmp_path):
    """首行即数据（无表头）的 CSV，validation_summary.has_header 应为 False。"""
    client = TestClient(create_app())
    # 首行就是数据行：第一列可解析为时间，第二列可解析为数值。
    csv_text = "2026-01-01 00:00:00,10.0,a\n2026-01-01 01:00:00,11.0,b\n"
    file = (tmp_path / "headerless.csv")
    file.write_text(csv_text)

    with file.open("rb") as handle:
        resp = client.post("/dataset-manifests/upload", files={"file": ("headerless.csv", handle, "text/csv")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["validation_summary"]["has_header"] is False


def test_sniff_csv_infers_numeric_vs_string_per_column(tmp_path):
    """带表头的 CSV，inferred_type 应按列真正区分 numeric / string。"""
    client = TestClient(create_app())
    csv_text = "time,target,extra\n2026-01-01 00:00:00,10.0,a\n2026-01-01 01:00:00,11.0,b\n"
    file = (tmp_path / "withheader.csv")
    file.write_text(csv_text)

    with file.open("rb") as handle:
        resp = client.post("/dataset-manifests/upload", files={"file": ("withheader.csv", handle, "text/csv")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["validation_summary"]["has_header"] is True
    by_name = {column["name"]: column for column in body["columns"]}
    assert by_name["target"]["inferred_type"] == "numeric"
    assert by_name["extra"]["inferred_type"] == "string"


def test_sniff_tsfile_reports_device_and_measurements(tmp_path):
    """上传 .tsfile 时，sniff 应返回设备 / 物理量名与行数，且不崩溃。"""
    client = TestClient(create_app())
    path = tmp_path / "in.tsfile"
    _write_tsfile(path, n=8)

    with path.open("rb") as handle:
        resp = client.post("/dataset-manifests/upload", files={"file": ("in.tsfile", handle, "application/octet-stream")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["file_format"] == "tsfile"
    # 物理量名应作为列出现。
    column_names = {column["name"] for column in body["columns"]}
    assert {"target", "extra"} <= column_names
    # 设备名应被报告。
    assert body["device"] == "dev1"
    assert body["row_count_estimate"] == 8
