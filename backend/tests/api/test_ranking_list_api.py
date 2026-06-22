from fastapi.testclient import TestClient

from app.main import create_app
from tests.api.test_benchmarking_run_create import create_track


def test_ranking_list_api_returns_policy_view(client):
    track_id, model_id = create_track(client)
    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_id]}).json()
    for _ in range(20):
        progress = client.get(f"/benchmarking-runs/{run['benchmarking_run_id']}/progress").json()
        if progress["status"] == "succeeded":
            break

    response = client.get(f"/tracks/{track_id}/ranking", params={"metric": "mse", "policy": "latest_valid_result"})

    assert response.status_code == 200
    assert response.json()["items"][0]["model_id"] == model_id
    assert response.json()["items"][0]["rank"] == 1


def test_ranking_lists_top_carries_model_name_for_anonymous(client, anon_client):
    """匿名用户访问 /ranking-lists 拿不到 /models（authed），所以 top 项必须自带名字。"""
    track_id, model_id = create_track(client)
    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_id]}).json()
    for _ in range(20):
        progress = client.get(f"/benchmarking-runs/{run['benchmarking_run_id']}/progress").json()
        if progress["status"] == "succeeded":
            break

    model_name = client.get(f"/models").json()["items"][0]["name"]
    r = anon_client.get("/ranking-lists")
    assert r.status_code == 200
    boards = [b for b in r.json()["items"] if b["track_id"] == track_id]
    assert boards, "leaderboard for new track missing"
    top = boards[0]["top"]
    assert top, "expected at least one ranked entry"
    assert top[0]["model_id"] == model_id
    assert top[0]["model_name"] == model_name


def test_hidden_ranking_is_invisible_to_anonymous_but_visible_to_admin(client, anon_client):
    track_id, _model_id = create_track(client)
    board = next(item for item in client.get("/ranking-lists").json()["items"] if item["track_id"] == track_id)

    update = client.patch(f"/ranking-lists/{board['ranking_list_id']}/visibility", json={"public_visible": False})

    assert update.status_code == 200
    assert update.json()["public_visible"] is False
    assert [item for item in anon_client.get("/ranking-lists").json()["items"] if item["track_id"] == track_id] == []

    direct = anon_client.get(f"/tracks/{track_id}/ranking")
    assert direct.status_code == 404
    assert direct.json()["error_code"] == "ranking_not_found"

    admin_direct = client.get(f"/tracks/{track_id}/ranking")
    assert admin_direct.status_code == 200
    assert admin_direct.json()["public_visible"] is False
    admin_boards = [item for item in client.get("/ranking-lists").json()["items"] if item["track_id"] == track_id]
    assert admin_boards and admin_boards[0]["public_visible"] is False


def test_viewer_cannot_update_ranking_visibility(client, viewer_client):
    track_id, _model_id = create_track(client)
    board = next(item for item in client.get("/ranking-lists").json()["items"] if item["track_id"] == track_id)

    response = viewer_client.patch(f"/ranking-lists/{board['ranking_list_id']}/visibility", json={"public_visible": False})

    assert response.status_code == 403
    assert response.json()["details"]["required_permission"] == "track.manage"
