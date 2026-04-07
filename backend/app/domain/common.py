from __future__ import annotations

from datetime import datetime, timezone

from ..config import get_settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def user_submission_defaults():
    return get_settings().ui.user_model_submission


def admin_batch_defaults():
    return get_settings().ui.admin_batch_generation
