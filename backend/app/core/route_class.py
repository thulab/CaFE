"""所有 HTTP 请求统一过 TieredRoute 这层切面（方案 C · AOP 风格）。

切面顺序：
  1. 解析 Authorization 头 -> User | None
  2. 写入 request.state.user 供 handler 复用
  3. 按 endpoint 上的 _tier / _perm 标签决定放行 / 401 / 403
  4. 通过则继续走 FastAPI 原生 handler

endpoint 的 _tier 与 _perm 由 TieredAPIRouter（见 router_factory.py）注入：
    @router.get("/foo", tier="public")
    @router.post("/bar", tier="perm", perm="run.execute")
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, get_args

from fastapi import Request, Response
from fastapi.routing import APIRoute
from sqlmodel import Session

from app.core.errors import ApiError
from app.core.security import decode_token
from app.models.auth import User
from app.services.auth_service import user_permission_codes

Tier = Literal["public", "authed", "perm"]
VALID_TIERS: tuple[str, ...] = get_args(Tier)


class TieredRoute(APIRoute):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # APIRoute.__init__ 内部就会调用 self.get_route_handler()，
        # 所以 tier/perm 必须在 super().__init__() 之前已就位。
        endpoint = kwargs["endpoint"]
        # 默认 tier="authed" —— 漏标的新路由不会变成 public，是反向安全语义。
        self.tier: Tier = getattr(endpoint, "_tier", "authed")
        self.perm: str | None = getattr(endpoint, "_perm", None)
        super().__init__(*args, **kwargs)

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            user = _resolve_user(request)
            request.state.user = user

            if self.tier == "public":
                return await original(request)
            if user is None:
                raise ApiError("auth_required", "login required", status_code=401)
            if not user.is_active:
                raise ApiError("auth_required", "user disabled", status_code=401)
            if self.tier == "perm" and not user.is_superuser:
                with Session(request.app.state.engine) as session:
                    if self.perm not in user_permission_codes(session, user):
                        raise ApiError(
                            "forbidden",
                            "missing permission",
                            details={"required_permission": self.perm},
                            status_code=403,
                        )
            return await original(request)

        return handler


def _resolve_user(request: Request) -> User | None:
    """Authorization: Bearer <token> -> User | None.

    无 token => None（Tier 0 才放行）。
    有 token 但格式 / 签名 / 过期非法 => 401。
    """
    auth = request.headers.get("authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ApiError("auth_required", "invalid Authorization header", status_code=401)
    user_id = decode_token(parts[1])
    with Session(request.app.state.engine) as session:
        user = session.get(User, user_id)
    if user is None:
        raise ApiError("auth_required", "user not found", status_code=401)
    return user
