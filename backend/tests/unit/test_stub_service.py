from fastapi.testclient import TestClient

from stub_service.main import create_app

client = TestClient(create_app())


def _forecast_body(model_id="Timer-3.5", horizon=3):
    return {
        "model_id": model_id,
        "targets": [
            {
                "columns": ["time", "value"],
                "data": [
                    ["2024-01-01T00:00:00", 1.0],
                    ["2024-01-01T01:00:00", 2.0],
                    ["2024-01-01T02:00:00", 3.0],
                ],
            }
        ],
        "output_length": [horizon],
        "time_col": ["time"],
    }


def test_health_endpoints_return_bare_status():
    for path, status in (("startup", "started"), ("liveness", "alive"), ("readiness", "ready")):
        body = client.get(f"/health/{path}").json()
        assert body["status"] == status
        assert "version" in body


def test_models_list_returns_builtin_models_in_envelope():
    body = client.get("/ai/api/v1/models/list").json()
    assert body["code"] == 200
    ids = [m["model_id"] for m in body["data"]["models"]]
    assert ids[:3] == ["Timer-3.5", "Timer-3.0", "Chronos-2"]


def test_forecast_returns_horizon_rows_in_contract_envelope():
    resp = client.post("/ai/api/v1/forecast", json=_forecast_body(horizon=3))
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 200
    result = body["data"]["results"][0]
    assert result["columns"] == ["time", "value"]
    assert len(result["data"]) == 3
    assert all(len(row) == 2 for row in result["data"])


def test_forecast_is_deterministic_for_same_request():
    first = client.post("/ai/api/v1/forecast", json=_forecast_body()).json()
    second = client.post("/ai/api/v1/forecast", json=_forecast_body()).json()
    assert first["data"] == second["data"]


def test_forecast_different_models_differ():
    a = client.post("/ai/api/v1/forecast", json=_forecast_body(model_id="Timer-3.5")).json()
    b = client.post("/ai/api/v1/forecast", json=_forecast_body(model_id="Chronos-2")).json()
    assert a["data"]["results"][0]["data"] != b["data"]["results"][0]["data"]


def test_forecast_rejects_empty_targets_with_422():
    resp = client.post("/ai/api/v1/forecast", json={"model_id": "Timer-3.5", "targets": []})
    assert resp.status_code == 422
    assert resp.json()["code"] == 422


def test_forecast_unknown_model_is_accepted():
    # 桩是通用替身：未注册的 model_id（如后端的 toto-mvp）也应可跑通。
    resp = client.post("/ai/api/v1/forecast", json=_forecast_body(model_id="toto-mvp"))
    assert resp.status_code == 200


# -- 模型管理 ------------------------------------------------------------------ #
def test_model_register_load_unload_delete_lifecycle():
    c = TestClient(create_app())
    reg = c.post("/ai/api/v1/models/register", json={"model_id": "my-model", "uri": "file:///tmp/m", "base_model_id": "Timer-3.0"})
    assert reg.status_code == 200

    assert c.post("/ai/api/v1/models/load", json={"model_id": "my-model"}).status_code == 200
    loaded_ids = [m["model_id"] for m in c.get("/ai/api/v1/models/list_loaded").json()["data"]["models"]]
    assert "my-model" in loaded_ids

    assert c.post("/ai/api/v1/models/unload", json={"model_id": "my-model"}).status_code == 200
    assert c.post("/ai/api/v1/models/delete", json={"model_id": "my-model"}).status_code == 200
    ids = [m["model_id"] for m in c.get("/ai/api/v1/models/list").json()["data"]["models"]]
    assert "my-model" not in ids


def test_model_register_conflict_and_missing_base():
    c = TestClient(create_app())
    assert c.post("/ai/api/v1/models/register", json={"model_id": "Timer-3.5", "uri": "file:///x"}).status_code == 409
    assert c.post("/ai/api/v1/models/register", json={"model_id": "x", "uri": "file:///x", "base_model_id": "nope"}).status_code == 404
    assert c.post("/ai/api/v1/models/register", json={"model_id": "x"}).status_code == 422


def test_model_delete_builtin_forbidden_and_unknown_not_found():
    c = TestClient(create_app())
    assert c.post("/ai/api/v1/models/delete", json={"model_id": "Timer-3.5"}).status_code == 403
    assert c.post("/ai/api/v1/models/delete", json={"model_id": "ghost"}).status_code == 404


def test_unload_then_forecast_known_model_returns_503():
    c = TestClient(create_app())
    assert c.post("/ai/api/v1/models/unload", json={"model_id": "Timer-3.5"}).status_code == 200
    resp = c.post("/ai/api/v1/forecast", json=_forecast_body(model_id="Timer-3.5"))
    assert resp.status_code == 503


# -- 数据集评估 / 治理 --------------------------------------------------------- #
def _eval_body(dimensions=None):
    return {
        "input": {
            "inline": {
                "columns": ["time", "temperature", "humidity"],
                "data": [
                    ["2024-01-01T00:00:00", 20.1, 65.0],
                    ["2024-01-01T01:00:00", 20.4, 63.5],
                    ["2024-01-01T02:00:00", None, 64.2],
                    ["2024-01-01T03:00:00", 21.2, 66.1],
                ],
                "time_col": "time",
            }
        },
        "dimensions": dimensions or ["integrity", "forecastability", "pearson"],
    }


def test_evaluate_requires_input():
    resp = client.post("/ai/api/v1/dataset/evaluate/execute", json={"dimensions": ["integrity"]})
    assert resp.status_code == 422


def test_evaluate_returns_scores_and_series_reports():
    resp = client.post("/ai/api/v1/dataset/evaluate/execute", json=_eval_body())
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert set(data["dimension_scores"]) == {"integrity", "forecastability", "pearson"}
    temp = next(r for r in data["series_reports"] if r["series_id"] == "temperature")
    assert temp["integrity"]["completeness"] == 75.0  # 3/4 非空


def test_evaluate_list_dimensions():
    names = [d["name"] for d in client.get("/ai/api/v1/dataset/evaluate/list_dimensions").json()["data"]["dimensions"]]
    assert names == ["integrity", "forecastability", "pearson"]


def test_govern_causal_mean_imputation_fills_nulls():
    resp = client.post(
        "/ai/api/v1/dataset/govern/execute",
        json={"input": _eval_body()["input"], "dimension": "causal_mean_imputation"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["summary"]["causal_mean_imputation"]["changes_count"] == 1
    assert all(cell is not None for row in data["inline"]["data"] for cell in row)


def test_govern_requires_valid_dimension():
    resp = client.post("/ai/api/v1/dataset/govern/execute", json={"input": _eval_body()["input"], "dimension": "nope"})
    assert resp.status_code == 422


def test_govern_list_dimensions():
    names = [d["name"] for d in client.get("/ai/api/v1/dataset/govern/list_dimensions").json()["data"]["dimensions"]]
    assert names == ["timestamp_repair", "causal_mean_imputation", "flat_series_removal", "zscore_normalization", "extreme_value_clipping"]


# -- 运维 --------------------------------------------------------------------- #
def test_metrics_exposes_prometheus_text():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "timer_service:" in resp.text


def test_reboot_returns_202_without_killing():
    resp = client.post("/reboot")
    assert resp.status_code == 202
    assert resp.json()["status"] == "rebooting"
