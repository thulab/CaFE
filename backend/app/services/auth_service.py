"""认证 / 鉴权数据落地：seed + 用户/角色服务函数。

启动时调用：seed_permissions → seed_system_roles → seed_admin_if_empty
"""

import logging

from sqlmodel import Session, select

from app.core.permissions import ADMIN_ROLE, ALL_CODES, ALL_PERMISSION_CODES, VIEWER_CODES, VIEWER_ROLE
from app.core.security import generate_random_password, hash_password
from app.models.auth import Permission, Role, RolePermission, User, UserRole

log = logging.getLogger(__name__)


def seed_permissions(session: Session) -> None:
    existing = {p.code for p in session.exec(select(Permission)).all()}
    added = False
    for code, desc in ALL_PERMISSION_CODES:
        if code in existing:
            continue
        session.add(Permission(code=code, description=desc))
        added = True
    if added:
        session.commit()


def seed_system_roles(session: Session) -> None:
    by_code = {p.code: p for p in session.exec(select(Permission)).all()}

    admin = _ensure_role(session, ADMIN_ROLE, "Full administrative access")
    _sync_role_permissions(session, admin, ALL_CODES, by_code)

    viewer = _ensure_role(session, VIEWER_ROLE, "Read-only access")
    _sync_role_permissions(session, viewer, VIEWER_CODES, by_code)


def _ensure_role(session: Session, name: str, description: str) -> Role:
    row = session.exec(select(Role).where(Role.name == name)).first()
    if row is None:
        row = Role(name=name, description=description, is_system=True)
        session.add(row)
        session.commit()
        session.refresh(row)
    elif not row.is_system:
        row.is_system = True
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def _sync_role_permissions(
    session: Session, role: Role, codes: set[str], by_code: dict[str, Permission]
) -> None:
    desired_pids = {by_code[c].permission_id for c in codes if c in by_code}
    current = session.exec(
        select(RolePermission).where(RolePermission.role_id == role.role_id)
    ).all()
    current_pids = {rp.permission_id for rp in current}

    changed = False
    for rp in current:
        if rp.permission_id not in desired_pids:
            session.delete(rp)
            changed = True
    for pid in desired_pids - current_pids:
        session.add(RolePermission(role_id=role.role_id, permission_id=pid))
        changed = True
    if changed:
        session.commit()


def seed_admin_if_empty(session: Session, admin_password: str | None) -> None:
    has_any = session.exec(select(User)).first() is not None
    if has_any:
        return
    password = admin_password or generate_random_password()
    user = User(
        username="admin",
        password_hash=hash_password(password),
        is_superuser=True,
        is_active=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    admin_role = session.exec(select(Role).where(Role.name == ADMIN_ROLE)).first()
    if admin_role is not None:
        session.add(UserRole(user_id=user.user_id, role_id=admin_role.role_id))
        session.commit()

    if admin_password:
        log.info("seeded initial admin user 'admin' (password from TSBENCHMARK_ADMIN_PASSWORD)")
    else:
        # 关键：随机密码必须打印给用户，否则后续无法登录。
        print(
            "\n[TSBenchmark] Seeded initial admin user.\n"
            "  username: admin\n"
            f"  password: {password}\n"
            "  (no TSBENCHMARK_ADMIN_PASSWORD provided; password is shown only once)\n",
            flush=True,
        )


def user_permission_codes(session: Session, user: User) -> set[str]:
    """计算用户实有的权限码集合（基于 UserRole + RolePermission + Permission）。"""
    if user.is_superuser:
        return set(ALL_CODES)
    user_roles = session.exec(select(UserRole).where(UserRole.user_id == user.user_id)).all()
    role_ids = [ur.role_id for ur in user_roles]
    if not role_ids:
        return set()
    role_perms = session.exec(
        select(RolePermission).where(RolePermission.role_id.in_(role_ids))
    ).all()
    perm_ids = [rp.permission_id for rp in role_perms]
    if not perm_ids:
        return set()
    perms = session.exec(select(Permission).where(Permission.permission_id.in_(perm_ids))).all()
    return {p.code for p in perms}


def user_role_names(session: Session, user: User) -> list[str]:
    user_roles = session.exec(select(UserRole).where(UserRole.user_id == user.user_id)).all()
    role_ids = [ur.role_id for ur in user_roles]
    if not role_ids:
        return []
    roles = session.exec(select(Role).where(Role.role_id.in_(role_ids))).all()
    return sorted(r.name for r in roles)
