from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import current_user, get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.core.security import encode_token, hash_password, verify_password
from app.models.auth import User
from app.services.auth_service import user_permission_codes, user_role_names

router = make_router(prefix="/auth", tags=["auth"])


class LoginPayload(BaseModel):
    username: str
    password: str


class PasswordChangePayload(BaseModel):
    old_password: str
    new_password: str


@router.post("/login", tier="public")
def login(payload: LoginPayload, session: Session = Depends(get_db_session)) -> dict:
    user = session.exec(select(User).where(User.username == payload.username)).first()
    # 统一返回相同错误，避免 username 枚举：用户不存在 / 密码错 / 被禁用 都是 401。
    if user is None or not verify_password(payload.password, user.password_hash) or not user.is_active:
        raise ApiError("invalid_credentials", "invalid username or password", status_code=401)
    token, ttl = encode_token(user.user_id)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl}


@router.get("/me", tier="authed")
def me(user: User = Depends(current_user), session: Session = Depends(get_db_session)) -> dict:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "roles": user_role_names(session, user),
        "permissions": sorted(user_permission_codes(session, user)),
    }


@router.post("/password", tier="authed")
def change_password(
    payload: PasswordChangePayload,
    user: User = Depends(current_user),
    session: Session = Depends(get_db_session),
) -> dict:
    if not verify_password(payload.old_password, user.password_hash):
        raise ApiError("invalid_credentials", "old password does not match", status_code=400)
    if not payload.new_password or len(payload.new_password) < 6:
        raise ApiError("weak_password", "new password must be at least 6 chars", status_code=400)
    user.password_hash = hash_password(payload.new_password)
    session.add(user)
    session.commit()
    return {"ok": True}
