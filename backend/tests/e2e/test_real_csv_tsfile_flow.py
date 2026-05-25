"""真实 CSV 走新通路端到端（plan Task B7.3）：全列摄入 → TsFile → 指针切片 → ModelInput → MASE。

用仓库根 test/flow_template.csv（time,target,extra 三列全数值），验证：
- extra 列被摄入到 per-dataset TsFile（不再像旧 reader 那样被丢弃）；
- 仍只用 target 作预测目标；
- run 终态 succeeded，默认榜单指标为 mase。
"""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.tsfile_store import TsFileSlicer

FLOW_TEMPLATE = Path(__file__).parents[3] / "test" / "flow_template.csv"


def test_real_csv_full_column_ingest_tsfile_and_mase_ranking():
    client = TestClient(create_app())

    with FLOW_TEMPLATE.open("rb") as file:
        upload = client.post("/dataset-manifests/upload", files={"file": ("flow_template.csv", file, "text/csv")})
    assert upload.status_code == 200
    source_uri = upload.json()["source_uri"]

    # value_columns 省略 → 自动摄入除 time 外的所有数值列（target + extra）。
    manifest = client.post(
        "/dataset-manifests",
        json={"name": "flow template", "domain": "demo", "source_uri": source_uri, "file_format": "csv", "time_column": "time"},
    )
    assert manifest.status_code == 200
    manifest_id = manifest.json()["dataset_manifest_id"]

    job = client.post(
        "/dataset-load-jobs",
        json={
            "dataset_manifest_id": manifest_id,
            "split_config": {"context_length": 12, "horizon": 6, "stride": 6, "target_columns": ["target"]},
        },
    )
    assert job.status_code == 200, job.text
    assert job.json()["status"] == "succeeded"
    shard_id = job.json()["output_shard_id"]

    shard = client.get(f"/shards/{shard_id}").json()
    # 全列摄入：extra 进了 TsFile；目标仍是 target。
    assert set(shard["value_columns"]) == {"target", "extra"}
    assert shard["target_columns"] == ["target"]
    assert shard["sample_count"] == 3

    # 直接读 TsFile 证明 extra 列真的落盘且值与 CSV 一致（前 3 行 20.0/20.4/20.8）。
    slicer = TsFileSlicer(shard["tsfile_uri"])
    try:
        extra_head = slicer.slice(shard["dataset_id"], ["extra"], 0, 3)
    finally:
        slicer.close()
    assert extra_head == [[20.0], [20.4], [20.8]]

    track = client.post("/wizard/real-dataset-track", json={"name": "real track", "shard_ids": [shard_id]})
    assert track.status_code == 200
    track_id = track.json()["track_id"]

    model_ids = [item["model_id"] for item in client.get("/models").json()["items"][:2]]
    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": model_ids})
    assert run.status_code == 200
    run_id = run.json()["benchmarking_run_id"]

    progress = {}
    for _ in range(30):
        progress = client.get(f"/benchmarking-runs/{run_id}/progress").json()
        if progress["status"] == "succeeded":
            break
    assert progress["status"] == "succeeded"

    # 默认榜单指标随 track.primary_metric_id = mase。
    ranking = client.get(f"/tracks/{track_id}/ranking")
    assert ranking.status_code == 200
    assert ranking.json()["metric"] == "mase"
    assert len(ranking.json()["items"]) == 2

    report_id = progress["report_id"]
    sample_id = client.get(f"/reports/{report_id}").json()["sample_forecast_links"][0]["sample_id"]
    sample_forecast = client.get(f"/samples/{sample_id}/forecast", params={"run_id": run_id})
    assert sample_forecast.status_code == 200
    body = sample_forecast.json()
    assert body["target_history"] and body["target_future"]
    assert "mase" in body["models"][0]["metrics"]
