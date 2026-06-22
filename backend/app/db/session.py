from collections.abc import Iterator

from sqlmodel import Session, create_engine

from app.core.config import get_settings


def create_db_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def get_session() -> Iterator[Session]:
    engine = create_db_engine()
    with Session(engine) as session:
        yield session
