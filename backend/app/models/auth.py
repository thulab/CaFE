from datetime import datetime

from sqlmodel import Field, SQLModel

from app.core.ids import new_id
from app.core.time import utc_now


class User(SQLModel, table=True):
    user_id: str = Field(default_factory=new_id, primary_key=True)
    username: str = Field(unique=True, index=True)
    email: str | None = Field(default=None, unique=True, index=True)
    password_hash: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Role(SQLModel, table=True):
    role_id: str = Field(default_factory=new_id, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str | None = None
    is_system: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class Permission(SQLModel, table=True):
    permission_id: str = Field(default_factory=new_id, primary_key=True)
    code: str = Field(unique=True, index=True)
    description: str | None = None


class UserRole(SQLModel, table=True):
    user_id: str = Field(primary_key=True, index=True)
    role_id: str = Field(primary_key=True, index=True)


class RolePermission(SQLModel, table=True):
    role_id: str = Field(primary_key=True, index=True)
    permission_id: str = Field(primary_key=True, index=True)
