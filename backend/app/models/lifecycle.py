from datetime import datetime

from sqlmodel import Field, SQLModel

from app.core.time import utc_now


class ArchivedResource(SQLModel, table=True):
    resource_type: str = Field(primary_key=True)
    resource_id: str = Field(primary_key=True)
    archived_reason: str | None = None
    archived_at: datetime = Field(default_factory=utc_now)
