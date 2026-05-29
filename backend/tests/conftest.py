import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings

# 进程级默认：测试始终走进程内确定性桩。设在模块级是因为 create_run 会派生
# 守护线程异步执行，线程可能晚于单测的 env 清理才读取配置；进程级设置可避免
# 该竞态导致后台线程回退到 REST 适配器。需验证 REST 的用例在用例内用
# monkeypatch.setenv("TSBENCHMARK_MODEL_ADAPTER", "rest") 临时覆盖。
os.environ.setdefault("TSBENCHMARK_MODEL_ADAPTER", "stub")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    get_settings.cache_clear()
    runtime_dir = tmp_path / "runtime"
    db_path = runtime_dir / "tsbenchmark.db"
    monkeypatch.setenv("TSBENCHMARK_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("TSBENCHMARK_DATABASE_URL", f"sqlite:///{db_path}")
    # 测试默认走进程内确定性桩，避免依赖网络上的 timer-rest-service。
    # 需要验证 REST 路径的用例可在用例内 setenv("TSBENCHMARK_MODEL_ADAPTER", "rest") 覆盖。
    monkeypatch.setenv("TSBENCHMARK_MODEL_ADAPTER", "stub")
    monkeypatch.setenv("TSBENCHMARK_MODEL_LIFECYCLE_MODE", "sequential_unload")
    # 鉴权层需要 JWT 密钥与可预测的 admin 密码；缺这两条 create_app() 会拒绝启动。
    monkeypatch.setenv("TSBENCHMARK_AUTH_SECRET", "test-secret-32-bytes-long-padding-xx")
    monkeypatch.setenv("TSBENCHMARK_ADMIN_PASSWORD", "test-admin-pw")
    yield
    get_settings.cache_clear()
    os.environ.pop("TSBENCHMARK_RUNTIME_DIR", None)
    os.environ.pop("TSBENCHMARK_DATABASE_URL", None)
    os.environ.pop("TSBENCHMARK_MODEL_ADAPTER", None)
    os.environ.pop("TSBENCHMARK_MODEL_LIFECYCLE_MODE", None)
    os.environ.pop("TSBENCHMARK_AUTH_SECRET", None)
    os.environ.pop("TSBENCHMARK_ADMIN_PASSWORD", None)


# ------------------------- Auth fixtures -------------------------
# 默认 client = admin-authenticated TestClient。Tier 0 / 边界用例用 anon_client / viewer_client。

def _login(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture
def app():
    from app.main import create_app
    return create_app()


@pytest.fixture
def anon_client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def admin_token(app) -> str:
    c = TestClient(app)
    return _login(c, "admin", "test-admin-pw")


@pytest.fixture
def client(app, admin_token) -> TestClient:
    """Admin-authenticated TestClient.

    现有大部分测试改成依赖此 fixture 即可（去掉 `client = TestClient(create_app())`）。
    """
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {admin_token}"
    return c


@pytest.fixture
def viewer_token(app, admin_token) -> str:
    """创建 viewer 用户并返回其 token；用于权限边界用例（应 403 的写操作）。"""
    admin = TestClient(app)
    admin.headers["Authorization"] = f"Bearer {admin_token}"
    # 找 viewer 角色 id
    roles = admin.get("/roles").json()["items"]
    viewer_role_id = next(r["role_id"] for r in roles if r["name"] == "viewer")
    admin.post(
        "/users",
        json={
            "username": "viewer-fix",
            "password": "viewer-pw",
            "role_ids": [viewer_role_id],
        },
    )
    c = TestClient(app)
    return _login(c, "viewer-fix", "viewer-pw")


@pytest.fixture
def viewer_client(app, viewer_token) -> TestClient:
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {viewer_token}"
    return c
