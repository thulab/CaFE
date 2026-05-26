"""Track A 基线 walker（plan Task A0）：对一个已起的后端走完整 API 链，落一份 baseline-record.md。

由 scripts/baseline-run.sh 调起（用隔离 runtime + 进程内确定性桩适配器）。
用法：baseline_run.py <base_url> <csv_path> <record_out.md>
"""
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    base_url, csv_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    client = httpx.Client(base_url=base_url, timeout=30.0, trust_env=False)

    # 鉴权重构后所有写操作都要 admin token。baseline-run.sh 已注入 admin 密码到 env。
    admin_password = os.environ.get("TSBENCHMARK_ADMIN_PASSWORD", "baseline-admin-pw")
    login = client.post("/auth/login", json={"username": "admin", "password": admin_password})
    if login.status_code != 200:
        raise SystemExit(f"[login] HTTP {login.status_code}: {login.text}")
    client.headers["Authorization"] = f"Bearer {login.json()['access_token']}"

    def expect(response: httpx.Response, label: str) -> dict:
        if response.status_code != 200:
            raise SystemExit(f"[{label}] HTTP {response.status_code}: {response.text}")
        return response.json()

    with csv_path.open("rb") as handle:
        upload = expect(
            client.post("/dataset-manifests/upload", files={"file": (csv_path.name, handle, "text/csv")}),
            "upload",
        )
    columns = [column["name"] for column in upload["columns"]]

    manifest = expect(
        client.post(
            "/dataset-manifests",
            json={"name": "baseline", "domain": "demo", "source_uri": upload["source_uri"], "file_format": "csv", "time_column": "time"},
        ),
        "manifest",
    )
    split_config = {"context_length": 12, "horizon": 6, "stride": 6, "target_columns": ["target"]}
    job = expect(
        client.post("/dataset-load-jobs", json={"dataset_manifest_id": manifest["dataset_manifest_id"], "split_config": split_config}),
        "load-job",
    )
    if job["status"] != "succeeded":
        raise SystemExit(f"load job not succeeded: {job}")
    shard = expect(client.get(f"/shards/{job['output_shard_id']}"), "shard")

    track = expect(
        client.post("/wizard/real-dataset-track", json={"name": "baseline track", "shard_ids": [shard["shard_id"]]}),
        "wizard",
    )
    track_id = track["track_id"]

    models = expect(client.get("/models"), "models")["items"]
    name_by_id = {model["model_id"]: model["name"] for model in models}
    model_ids = [model["model_id"] for model in models[:2]]

    run = expect(client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": model_ids}), "run")
    run_id = run["benchmarking_run_id"]

    progress: dict = {}
    for _ in range(120):
        progress = expect(client.get(f"/benchmarking-runs/{run_id}/progress"), "progress")
        if progress["status"] in {"succeeded", "partial_succeeded", "failed"}:
            break
        time.sleep(0.5)
    if progress.get("status") != "succeeded":
        raise SystemExit(f"run did not succeed: {progress.get('status')}")

    rankings = {
        metric: expect(client.get(f"/tracks/{track_id}/ranking", params={"metric": metric}), f"ranking-{metric}")
        for metric in ["mase", "mse", "mae"]
    }
    report = expect(client.get(f"/reports/{progress['report_id']}"), "report")
    sample_id = report["sample_forecast_links"][0]["sample_id"]
    sample_forecast = expect(client.get(f"/samples/{sample_id}/forecast", params={"run_id": run_id}), "sample-forecast")

    _write_record(out_path, base_url, csv_path, columns, shard, split_config, model_ids, name_by_id, rankings, sample_forecast)
    print(f"run status: {progress['status']}  models: {len(model_ids)}  samples/shard: {shard['sample_count']}")
    print(f"default ranking metric: {rankings['mase']['metric']}")
    return 0


def _write_record(out_path, base_url, csv_path, columns, shard, split_config, model_ids, name_by_id, rankings, sample_forecast) -> None:
    lines: list[str] = []
    lines.append("# TSBenchmark 基线记录（Track A · plan A0）")
    lines.append("")
    lines.append(f"- **生成时间**：{datetime.now(UTC).isoformat()}")
    lines.append(f"- **数据**：`{csv_path}`，列 = {columns}")
    lines.append(f"- **后端**：`{base_url}`（隔离 runtime + 进程内确定性桩适配器 `TSBENCHMARK_MODEL_ADAPTER=stub`）")
    lines.append(f"- **切分**：{split_config}")
    lines.append(f"- **Shard**：value_columns={shard['value_columns']}，target_columns={shard['target_columns']}，sample_count={shard['sample_count']}（序列真值存 SQLite SeriesPoint）")
    lines.append("")
    lines.append("> 通路：CSV/TsFile 输入 → 全列摄入 → SQLite SeriesPoint(逐点行) → 指针切片 → ModelInput(无 target_future) → 桩推理 → MSE/MAE/MASE → 榜单。")
    lines.append("> **桩确定性**：同一次 load 内，相同 (model, sample, seed) 必得相同 forecast；但 `sample_id` 为随机 UUID，故**绝对预测值不跨重载可比**（既有特性）。")
    lines.append("")

    for metric in ["mase", "mse", "mae"]:
        ranking = rankings[metric]
        primary = " （默认主排名）" if metric == "mase" else ""
        lines.append(f"## 榜单 · {metric}{primary}")
        lines.append("")
        lines.append("| rank | model | value |")
        lines.append("| --- | --- | --- |")
        for item in ranking["items"]:
            name = name_by_id.get(item["model_id"], item["model_id"])
            lines.append(f"| {item['rank']} | {name} | {_fmt(item['metric_value'])} |")
        lines.append("")

    lines.append("## 样本预测视图（首个样本）")
    lines.append("")
    lines.append(f"- sample_id：`{sample_forecast['sample_id']}`")
    truth = [row[0] for row in sample_forecast["target_future"]]
    lines.append(f"- 真值 target_future：{[_fmt(value, 4) for value in truth]}")
    lines.append("")
    lines.append("| model | forecast (h=1..n) | mase | mse | mae |")
    lines.append("| --- | --- | --- | --- | --- |")
    for model in sample_forecast["models"]:
        forecast = [_fmt(row[0], 4) for row in (model.get("forecast") or [])]
        metrics = model.get("metrics") or {}
        lines.append(
            f"| {model.get('model_name', model['model_id'])} | {forecast} | "
            f"{_fmt(metrics['mase']) if 'mase' in metrics else '—'} | "
            f"{_fmt(metrics['mse']) if 'mse' in metrics else '—'} | "
            f"{_fmt(metrics['mae']) if 'mae' in metrics else '—'} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> 由 `scripts/baseline-run.sh` 自动生成；重跑即覆盖。重构前后对照见 plan 的「实现状态」节。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
