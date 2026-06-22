from collections.abc import Iterator

from fastapi import Request
from sqlmodel import Session

from app.core.errors import ApiError


def get_db_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session


def current_user(request: Request):
    """取 TieredRoute 注入到 request.state.user 的 User。

    handler 标 tier="authed"/"perm" 时切面已保证 user 非 None；
    tier="public" 路由若想拿用户对象，自己 getattr 即可。
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise ApiError("auth_required", "login required", status_code=401)
    return user
