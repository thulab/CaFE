import os
from collections.abc import Iterator

import pytest

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
    yield
    get_settings.cache_clear()
    os.environ.pop("TSBENCHMARK_RUNTIME_DIR", None)
    os.environ.pop("TSBENCHMARK_DATABASE_URL", None)
    os.environ.pop("TSBENCHMARK_MODEL_ADAPTER", None)
