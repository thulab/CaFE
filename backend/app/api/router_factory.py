"""make_router() —— 替代 FastAPI 原生 APIRouter，承载 tier=/perm= 装饰参数。

用法：
    from app.api.router_factory import make_router
    router = make_router(prefix="/runs", tags=["runs"])

    @router.get("", tier="authed")
    def list_runs(...): ...

    @router.post("", tier="perm", perm="run.execute")
    def create_run(...): ...

    @router.get("/public", tier="public")
    def public_thing(...): ...

漏标的路由默认 tier="authed"（要登录），不是 public —— 安全语义反转。
"""

from typing import Any

from fastapi import APIRouter

from app.core.route_class import TieredRoute


class TieredAPIRouter(APIRouter):
    def _decorate(self, super_decorator, tier: str, perm: str | None):
        def apply(func):
            func._tier = tier
            func._perm = perm
            return super_decorator(func)

        return apply

    def get(self, path: str, *, tier: str = "authed", perm: str | None = None, **kwargs: Any):  # type: ignore[override]
        return self._decorate(super().get(path, **kwargs), tier, perm)

    def post(self, path: str, *, tier: str = "authed", perm: str | None = None, **kwargs: Any):  # type: ignore[override]
        return self._decorate(super().post(path, **kwargs), tier, perm)

    def put(self, path: str, *, tier: str = "authed", perm: str | None = None, **kwargs: Any):  # type: ignore[override]
        return self._decorate(super().put(path, **kwargs), tier, perm)

    def patch(self, path: str, *, tier: str = "authed", perm: str | None = None, **kwargs: Any):  # type: ignore[override]
        return self._decorate(super().patch(path, **kwargs), tier, perm)

    def delete(self, path: str, *, tier: str = "authed", perm: str | None = None, **kwargs: Any):  # type: ignore[override]
        return self._decorate(super().delete(path, **kwargs), tier, perm)


def make_router(**kwargs: Any) -> TieredAPIRouter:
    kwargs.setdefault("route_class", TieredRoute)
    return TieredAPIRouter(**kwargs)
