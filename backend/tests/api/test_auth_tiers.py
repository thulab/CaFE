"""三层访问模型的端到端边界：匿名 / viewer / admin 在典型路由上的行为。

公开层（Tier 0）：anonymous OK
登录层（Tier 1）：anonymous 401，登录后 200
权限层（Tier 2）：anonymous 401，viewer 403，admin 200
"""


def test_tier0_leaderboards_open_to_anonymous(anon_client):
    r = anon_client.get("/ranking-lists")
    assert r.status_code == 200
    assert "items" in r.json()


def test_tier0_tracks_list_open_to_anonymous(anon_client):
    r = anon_client.get("/tracks")
    assert r.status_code == 200


def test_tier1_models_list_rejects_anonymous(anon_client):
    r = anon_client.get("/models")
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_required"


def test_tier1_models_list_accepts_admin(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert "items" in r.json()


def test_tier2_create_track_rejects_anonymous(anon_client):
    r = anon_client.post("/tracks", json={"name": "x", "capability_block_ids": []})
    assert r.status_code == 401


def test_tier2_create_track_rejects_viewer(viewer_client):
    r = viewer_client.post("/tracks", json={"name": "x", "capability_block_ids": []})
    assert r.status_code == 403
    body = r.json()
    assert body["error_code"] == "forbidden"
    assert body["details"]["required_permission"] == "track.manage"


def test_tier2_create_track_accepts_admin(client):
    # Empty capability_block_ids triggers track_requires_block 400, not auth.
    r = client.post("/tracks", json={"name": "x", "capability_block_ids": []})
    assert r.status_code == 400
    assert r.json()["error_code"] == "track_requires_block"


def test_user_management_rejects_viewer(viewer_client):
    r = viewer_client.get("/users")
    assert r.status_code == 403


def test_role_management_rejects_viewer(viewer_client):
    r = viewer_client.get("/roles")
    assert r.status_code == 403


def test_auth_me_returns_role_and_permissions_for_viewer(viewer_client):
    r = viewer_client.get("/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["roles"] == ["viewer"]
    # viewer 应包含 *.read 但不含 *.write/*.manage/*.execute
    assert all(c.endswith(".read") for c in body["permissions"])


def test_system_role_perm_change_returns_409(client):
    roles = client.get("/roles").json()["items"]
    admin_role_id = next(r["role_id"] for r in roles if r["name"] == "admin")
    r = client.put(f"/roles/{admin_role_id}/permissions", json={"permission_codes": []})
    assert r.status_code == 409
    assert r.json()["error_code"] == "role_is_system"
