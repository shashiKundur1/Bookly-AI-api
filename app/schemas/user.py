import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from app.models import User

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
Password = Annotated[str, StringConstraints(min_length=8, max_length=128)]


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    has_avatar: bool
    created_at: datetime

    @classmethod
    def from_model(cls, user: User) -> "UserOut":
        return cls(
            id=user.id,
            email=user.email,
            name=user.name,
            has_avatar=user.avatar_path is not None,
            created_at=user.created_at,
        )


class UserUpdate(BaseModel):
    name: Name


class PasswordChange(BaseModel):
    current_password: str
    new_password: Password


class SessionSummary(BaseModel):
    id: uuid.UUID
    book_id: uuid.UUID
    book_title: str
    started_at: datetime
    ended_at: datetime | None
    start_page: int
    end_page: int | None


class UserStats(BaseModel):
    total_books: int
    by_status: dict[str, int]
    favorites: int
    pages_read: int
    reading_seconds: int
    recent_sessions: list[SessionSummary]
