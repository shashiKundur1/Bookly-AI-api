import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Book, User
from app.security import ACCESS_COOKIE, decode_token

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _extract_token(request: Request) -> str | None:
    token = request.cookies.get(ACCESS_COOKIE)
    if token:
        return token
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:]
    return None


async def get_current_user(request: Request, session: SessionDep) -> User:
    token = _extract_token(request)
    user_id = decode_token(token, "access") if token else None
    user = await session.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_owned_book(book_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> Book:
    book = await session.get(Book, book_id)
    if book is None or book.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Book not found")
    return book


OwnedBook = Annotated[Book, Depends(get_owned_book)]
