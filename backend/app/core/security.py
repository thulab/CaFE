import datetime as _dt
import secrets

import bcrypt
import jwt

from app.core.config import get_settings
from app.core.errors import ApiError


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _secret() -> str:
    s = get_settings().auth_secret
    if not s:
        raise RuntimeError(
            "TSBENCHMARK_AUTH_SECRET is not configured; set it in env before issuing or decoding tokens"
        )
    return s


def encode_token(user_id: str, ttl_seconds: int | None = None) -> tuple[str, int]:
    settings = get_settings()
    ttl = ttl_seconds if ttl_seconds is not None else settings.auth_ttl_seconds
    now = _dt.datetime.now(_dt.UTC)
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + _dt.timedelta(seconds=ttl)).timestamp()),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256"), ttl


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, _secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise ApiError("auth_required", "token expired", status_code=401) from exc
    except jwt.InvalidTokenError as exc:
        raise ApiError("auth_required", "invalid token", status_code=401) from exc
    user_id = payload.get("sub")
    if not user_id or not isinstance(user_id, str):
        raise ApiError("auth_required", "invalid token payload", status_code=401)
    return user_id


def generate_random_password(length: int = 16) -> str:
    return secrets.token_urlsafe(length)[:length]
