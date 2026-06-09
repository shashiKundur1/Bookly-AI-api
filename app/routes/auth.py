from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select

from app.dependencies import SessionDep
from app.models import User
from app.schemas.auth import AuthResponse, LoginRequest, RefreshRequest, RegisterRequest
from app.schemas.user import UserOut
from app.security import (
    REFRESH_COOKIE,
    clear_auth_cookies,
    create_token_pair,
    decode_token,
    hash_password,
    set_auth_cookies,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _authenticated(response: Response, user: User) -> AuthResponse:
    access, refresh = create_token_pair(user.id)
    set_auth_cookies(response, access, refresh)
    return AuthResponse(user=UserOut.from_model(user), access_token=access, refresh_token=refresh)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, response: Response, session: SessionDep) -> AuthResponse:
    email = payload.email.lower()
    existing = await session.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(email=email, password_hash=hash_password(payload.password), name=payload.name)
    session.add(user)
    await session.commit()
    return _authenticated(response, user)


@router.post("/login")
async def login(payload: LoginRequest, response: Response, session: SessionDep) -> AuthResponse:
    user = await session.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return _authenticated(response, user)


@router.post("/refresh")
async def refresh(
    request: Request,
    response: Response,
    session: SessionDep,
    payload: RefreshRequest | None = None,
) -> AuthResponse:
    token = request.cookies.get(REFRESH_COOKIE)
    if not token and payload is not None:
        token = payload.refresh_token
    user_id = decode_token(token, "refresh") if token else None
    user = await session.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")
    return _authenticated(response, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    clear_auth_cookies(response)
