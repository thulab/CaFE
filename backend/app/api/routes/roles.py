from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.core.permissions import ALL_PERMISSION_CODES
from app.models.auth import Permission, Role, RolePermission

router = make_router(tags=["roles"])


class RoleCreate(BaseModel):
    name: str
    description: str | None = None
    permission_codes: list[str] = []


class RoleUpdate(BaseModel):
    description: str | None = None


class RolePermissionsAssignment(BaseModel):
    permission_codes: list[str]


def _role_dto(session: Session, role: Role) -> dict:
    rps = session.exec(
        select(RolePermission).where(RolePermission.role_id == role.role_id)
    ).all()
    perm_ids = [rp.permission_id for rp in rps]
    codes: list[str] = []
    if perm_ids:
        perms = session.exec(
            select(Permission).where(Permission.permission_id.in_(perm_ids))
        ).all()
        codes = sorted(p.code for p in perms)
    return {
        "role_id": role.role_id,
        "name": role.name,
        "description": role.description,
        "is_system": role.is_system,
        "permission_codes": codes,
        "created_at": role.created_at,
    }


def _validate_codes(session: Session, codes: list[str]) -> list[Permission]:
    if not codes:
        return []
    found = session.exec(select(Permission).where(Permission.code.in_(codes))).all()
    missing = set(codes) - {p.code for p in found}
    if missing:
        raise ApiError(
            "permission_not_found", "permission code not found", {"codes": sorted(missing)}, 404
        )
    return list(found)


@router.get("/roles", tier="perm", perm="role.manage")
def list_roles(session: Session = Depends(get_db_session)) -> dict:
    roles = session.exec(select(Role).order_by(Role.created_at)).all()
    return {"items": [_role_dto(session, r) for r in roles]}


@router.post("/roles", tier="perm", perm="role.manage")
def create_role(payload: RoleCreate, session: Session = Depends(get_db_session)) -> dict:
    if session.exec(select(Role).where(Role.name == payload.name)).first():
        raise ApiError("role_name_taken", "role name already exists", {"name": payload.name})
    perms = _validate_codes(session, payload.permission_codes)
    role = Role(name=payload.name, description=payload.description, is_system=False)
    session.add(role)
    session.commit()
    session.refresh(role)
    for p in perms:
        session.add(RolePermission(role_id=role.role_id, permission_id=p.permission_id))
    if perms:
        session.commit()
    return _role_dto(session, role)


@router.patch("/roles/{role_id}", tier="perm", perm="role.manage")
def update_role(role_id: str, payload: RoleUpdate, session: Session = Depends(get_db_session)) -> dict:
    role = session.get(Role, role_id)
    if role is None:
        raise ApiError("role_not_found", "role not found", {"role_id": role_id}, 404)
    if role.is_system:
        raise ApiError("role_is_system", "system role cannot be modified", {"role_id": role_id}, 409)
    if payload.description is not None:
        role.description = payload.description
    session.add(role)
    session.commit()
    session.refresh(role)
    return _role_dto(session, role)


@router.delete("/roles/{role_id}", tier="perm", perm="role.manage")
def delete_role(role_id: str, session: Session = Depends(get_db_session)) -> dict:
    role = session.get(Role, role_id)
    if role is None:
        raise ApiError("role_not_found", "role not found", {"role_id": role_id}, 404)
    if role.is_system:
        raise ApiError("role_is_system", "system role cannot be deleted", {"role_id": role_id}, 409)
    for rp in session.exec(
        select(RolePermission).where(RolePermission.role_id == role_id)
    ).all():
        session.delete(rp)
    session.delete(role)
    session.commit()
    return {"ok": True}


@router.put("/roles/{role_id}/permissions", tier="perm", perm="role.manage")
def set_role_permissions(
    role_id: str,
    payload: RolePermissionsAssignment,
    session: Session = Depends(get_db_session),
) -> dict:
    role = session.get(Role, role_id)
    if role is None:
        raise ApiError("role_not_found", "role not found", {"role_id": role_id}, 404)
    if role.is_system:
        raise ApiError(
            "role_is_system",
            "system role permissions cannot be modified",
            {"role_id": role_id},
            409,
        )
    perms = _validate_codes(session, payload.permission_codes)
    for rp in session.exec(
        select(RolePermission).where(RolePermission.role_id == role_id)
    ).all():
        session.delete(rp)
    for p in perms:
        session.add(RolePermission(role_id=role_id, permission_id=p.permission_id))
    session.commit()
    session.refresh(role)
    return _role_dto(session, role)


@router.get("/permissions", tier="perm", perm="role.manage")
def list_permissions(session: Session = Depends(get_db_session)) -> dict:
    # 直接返回代码侧枚举顺序（带 description）；不依赖 DB 顺序。
    by_code = {p.code: p for p in session.exec(select(Permission)).all()}
    items = []
    for code, default_desc in ALL_PERMISSION_CODES:
        perm = by_code.get(code)
        items.append(
            {
                "code": code,
                "description": (perm.description if perm else default_desc),
                "group": code.split(".")[0],
            }
        )
    return {"items": items}
