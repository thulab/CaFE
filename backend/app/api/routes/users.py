from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.core.security import hash_password
from app.core.time import utc_now
from app.models.auth import Role, User, UserRole

router = make_router(prefix="/users", tags=["users"])


class UserCreate(BaseModel):
    username: str
    email: str | None = None
    password: str
    role_ids: list[str] = []
    is_superuser: bool = False
    is_active: bool = True


class UserUpdate(BaseModel):
    email: str | None = None
    is_active: bool | None = None


class RoleAssignment(BaseModel):
    role_ids: list[str]


class PasswordReset(BaseModel):
    new_password: str


def _user_dto(session: Session, user: User) -> dict:
    user_roles = session.exec(select(UserRole).where(UserRole.user_id == user.user_id)).all()
    role_ids = [ur.role_id for ur in user_roles]
    role_names: list[str] = []
    if role_ids:
        roles = session.exec(select(Role).where(Role.role_id.in_(role_ids))).all()
        role_names = sorted(r.name for r in roles)
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "role_ids": role_ids,
        "role_names": role_names,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def _validate_roles(session: Session, role_ids: list[str]) -> None:
    if not role_ids:
        return
    found = session.exec(select(Role).where(Role.role_id.in_(role_ids))).all()
    missing = set(role_ids) - {r.role_id for r in found}
    if missing:
        raise ApiError("role_not_found", "role not found", {"role_ids": sorted(missing)}, 404)


@router.get("", tier="perm", perm="user.manage")
def list_users(session: Session = Depends(get_db_session)) -> dict:
    users = session.exec(select(User).order_by(User.created_at)).all()
    return {"items": [_user_dto(session, u) for u in users]}


@router.post("", tier="perm", perm="user.manage")
def create_user(payload: UserCreate, session: Session = Depends(get_db_session)) -> dict:
    if not payload.password or len(payload.password) < 6:
        raise ApiError("weak_password", "password must be at least 6 chars", status_code=400)
    if session.exec(select(User).where(User.username == payload.username)).first():
        raise ApiError("username_taken", "username already exists", {"username": payload.username})
    _validate_roles(session, payload.role_ids)
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        is_active=payload.is_active,
        is_superuser=payload.is_superuser,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    for rid in payload.role_ids:
        session.add(UserRole(user_id=user.user_id, role_id=rid))
    if payload.role_ids:
        session.commit()
    return _user_dto(session, user)


@router.get("/{user_id}", tier="perm", perm="user.manage")
def get_user(user_id: str, session: Session = Depends(get_db_session)) -> dict:
    user = session.get(User, user_id)
    if user is None:
        raise ApiError("user_not_found", "user not found", {"user_id": user_id}, 404)
    return _user_dto(session, user)


@router.patch("/{user_id}", tier="perm", perm="user.manage")
def update_user(user_id: str, payload: UserUpdate, session: Session = Depends(get_db_session)) -> dict:
    user = session.get(User, user_id)
    if user is None:
        raise ApiError("user_not_found", "user not found", {"user_id": user_id}, 404)
    if payload.email is not None:
        user.email = payload.email or None
    if payload.is_active is not None:
        user.is_active = payload.is_active
    user.updated_at = utc_now()
    session.add(user)
    session.commit()
    session.refresh(user)
    return _user_dto(session, user)


@router.delete("/{user_id}", tier="perm", perm="user.manage")
def delete_user(user_id: str, session: Session = Depends(get_db_session)) -> dict:
    user = session.get(User, user_id)
    if user is None:
        raise ApiError("user_not_found", "user not found", {"user_id": user_id}, 404)
    # 清掉关联，避免脏数据。
    for ur in session.exec(select(UserRole).where(UserRole.user_id == user_id)).all():
        session.delete(ur)
    session.delete(user)
    session.commit()
    return {"ok": True}


@router.put("/{user_id}/roles", tier="perm", perm="user.manage")
def set_user_roles(user_id: str, payload: RoleAssignment, session: Session = Depends(get_db_session)) -> dict:
    user = session.get(User, user_id)
    if user is None:
        raise ApiError("user_not_found", "user not found", {"user_id": user_id}, 404)
    _validate_roles(session, payload.role_ids)
    existing = session.exec(select(UserRole).where(UserRole.user_id == user_id)).all()
    for ur in existing:
        session.delete(ur)
    for rid in payload.role_ids:
        session.add(UserRole(user_id=user_id, role_id=rid))
    user.updated_at = utc_now()
    session.add(user)
    session.commit()
    session.refresh(user)
    return _user_dto(session, user)


@router.post("/{user_id}/reset-password", tier="perm", perm="user.manage")
def reset_password(user_id: str, payload: PasswordReset, session: Session = Depends(get_db_session)) -> dict:
    user = session.get(User, user_id)
    if user is None:
        raise ApiError("user_not_found", "user not found", {"user_id": user_id}, 404)
    if not payload.new_password or len(payload.new_password) < 6:
        raise ApiError("weak_password", "new password must be at least 6 chars", status_code=400)
    user.password_hash = hash_password(payload.new_password)
    user.updated_at = utc_now()
    session.add(user)
    session.commit()
    return {"ok": True}
