from sqlmodel import Session

from app.services.run_executor import recover_interrupted_runs


def recover_runs_on_startup(session: Session) -> None:
    recover_interrupted_runs(session)
