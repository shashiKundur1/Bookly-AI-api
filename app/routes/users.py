from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from app.config import get_settings
from app.dependencies import CurrentUser, SessionDep
from app.models import Book, ReadingProgress, ReadingSession
from app.schemas.user import PasswordChange, SessionSummary, UserOut, UserStats, UserUpdate
from app.security import hash_password, verify_password
from app.services.storage import save_image

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def get_me(user: CurrentUser) -> UserOut:
    return UserOut.from_model(user)


@router.patch("/me")
async def update_me(payload: UserUpdate, user: CurrentUser, session: SessionDep) -> UserOut:
    user.name = payload.name
    await session.commit()
    return UserOut.from_model(user)


@router.put("/me/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(payload: PasswordChange, user: CurrentUser, session: SessionDep) -> None:
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password is incorrect")
    user.password_hash = hash_password(payload.new_password)
    await session.commit()


@router.post("/me/avatar")
async def upload_avatar(file: UploadFile, user: CurrentUser, session: SessionDep) -> UserOut:
    path = await save_image(file, get_settings().avatars_dir, str(user.id))
    user.avatar_path = str(path)
    await session.commit()
    return UserOut.from_model(user)


@router.get("/me/avatar")
async def get_avatar(user: CurrentUser) -> FileResponse:
    if user.avatar_path is None or not Path(user.avatar_path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No avatar uploaded")
    return FileResponse(user.avatar_path, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/me/stats")
async def get_stats(user: CurrentUser, session: SessionDep) -> UserStats:
    status_rows = (
        await session.execute(
            select(Book.status, func.count()).where(Book.user_id == user.id).group_by(Book.status)
        )
    ).all()
    by_status = {row[0]: row[1] for row in status_rows}

    favorites = (
        await session.scalar(
            select(func.count())
            .select_from(Book)
            .where(Book.user_id == user.id, Book.is_favorite.is_(True))
        )
    ) or 0

    pages_read = (
        await session.scalar(
            select(func.coalesce(func.sum(func.jsonb_array_length(ReadingProgress.pages_read)), 0))
            .join(Book, Book.id == ReadingProgress.book_id)
            .where(Book.user_id == user.id)
        )
    ) or 0

    reading_seconds = (
        await session.scalar(
            select(
                func.coalesce(
                    func.sum(
                        func.extract("epoch", ReadingSession.ended_at - ReadingSession.started_at)
                    ),
                    0,
                )
            )
            .join(Book, Book.id == ReadingSession.book_id)
            .where(Book.user_id == user.id, ReadingSession.ended_at.is_not(None))
        )
    ) or 0

    recent_rows = (
        await session.execute(
            select(ReadingSession, Book.title)
            .join(Book, Book.id == ReadingSession.book_id)
            .where(Book.user_id == user.id)
            .order_by(ReadingSession.started_at.desc())
            .limit(10)
        )
    ).all()
    recent_sessions = [
        SessionSummary(
            id=row.ReadingSession.id,
            book_id=row.ReadingSession.book_id,
            book_title=row.title,
            started_at=row.ReadingSession.started_at,
            ended_at=row.ReadingSession.ended_at,
            start_page=row.ReadingSession.start_page,
            end_page=row.ReadingSession.end_page,
        )
        for row in recent_rows
    ]

    return UserStats(
        total_books=sum(by_status.values()),
        by_status=by_status,
        favorites=favorites,
        pages_read=int(pages_read),
        reading_seconds=int(reading_seconds),
        recent_sessions=recent_sessions,
    )
