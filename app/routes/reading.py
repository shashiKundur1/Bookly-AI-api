import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.dependencies import CurrentUser, OwnedBook, SessionDep
from app.models import Book, ReadingProgress, ReadingSession
from app.schemas.reading import ProgressOut, ProgressUpdate, SessionOut, SessionStart, SessionUpdate

router = APIRouter(tags=["reading"])


def _clamp_pages(pages: list[int] | None, page_count: int) -> set[int]:
    if not pages:
        return set()
    if page_count:
        return {page for page in pages if 1 <= page <= page_count}
    return {page for page in pages if page >= 1}


@router.get("/books/{book_id}/progress")
async def get_progress(book: OwnedBook) -> ProgressOut:
    progress = book.progress or ReadingProgress(book_id=book.id, current_page=1, pages_read=[])
    return ProgressOut.from_model(progress, book.page_count)


@router.put("/books/{book_id}/progress")
async def update_progress(payload: ProgressUpdate, book: OwnedBook, session: SessionDep) -> ProgressOut:
    progress = book.progress
    if progress is None:
        progress = ReadingProgress(book_id=book.id, current_page=1, pages_read=[])
        session.add(progress)
    read = set(progress.pages_read or [])
    read |= _clamp_pages(payload.mark_read, book.page_count)
    read -= set(payload.unmark_read or [])
    progress.pages_read = sorted(read)
    if payload.current_page is not None:
        progress.current_page = (
            min(payload.current_page, book.page_count) if book.page_count else payload.current_page
        )
    book.last_read_at = datetime.now(timezone.utc)
    if book.status == "to_read":
        book.status = "reading"
    if book.page_count and len(read) >= book.page_count:
        book.status = "finished"
    await session.commit()
    await session.refresh(progress)
    return ProgressOut.from_model(progress, book.page_count)


@router.post("/books/{book_id}/sessions", status_code=status.HTTP_201_CREATED)
async def start_session(payload: SessionStart, book: OwnedBook, session: SessionDep) -> SessionOut:
    reading_session = ReadingSession(book_id=book.id, start_page=payload.start_page)
    session.add(reading_session)
    book.last_read_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(reading_session)
    return SessionOut.model_validate(reading_session)


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: uuid.UUID,
    payload: SessionUpdate,
    user: CurrentUser,
    session: SessionDep,
) -> SessionOut:
    reading_session = (
        await session.execute(
            select(ReadingSession)
            .join(Book, Book.id == ReadingSession.book_id)
            .where(ReadingSession.id == session_id, Book.user_id == user.id)
        )
    ).scalar_one_or_none()
    if reading_session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    reading_session.ended_at = datetime.now(timezone.utc)
    if payload.end_page is not None:
        reading_session.end_page = payload.end_page
    await session.commit()
    return SessionOut.model_validate(reading_session)
