"""timer-rest-service 桩服务包。

实现 docs/reference/rest-api.md 的精简子集（推理 + 模型列表 + 健康探针），
供本地无真实推理服务时让 TSBenchmark 后端通过 REST 正常跑通。
"""
from stub_service.main import app, create_app

__all__ = ["app", "create_app"]
