import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Response
from pwdlib import PasswordHash

from app.config import get_settings

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"
REFRESH_COOKIE_PATH = "/api/v1/auth"

_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _hasher.verify(password, password_hash)


def create_token(user_id: uuid.UUID, token_type: str, lifetime: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + lifetime,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, get_settings().jwt_secret, algorithm="HS256")


def create_token_pair(user_id: uuid.UUID) -> tuple[str, str]:
    settings = get_settings()
    access = create_token(user_id, "access", timedelta(minutes=settings.access_token_minutes))
    refresh = create_token(user_id, "refresh", timedelta(days=settings.refresh_token_days))
    return access, refresh


def decode_token(token: str, expected_type: str) -> uuid.UUID | None:
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None
    if payload.get("type") != expected_type:
        return None
    try:
        return uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        return None


def set_auth_cookies(response: Response, access: str, refresh: str) -> None:
    settings = get_settings()
    response.set_cookie(
        ACCESS_COOKIE,
        access,
        max_age=settings.access_token_minutes * 60,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE,
        refresh,
        max_age=settings.refresh_token_days * 86400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path=REFRESH_COOKIE_PATH,
    )


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_COOKIE_PATH)
