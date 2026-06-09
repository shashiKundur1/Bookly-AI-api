import uuid
from pathlib import Path
from typing import Annotated, Literal

import anyio
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import case, func, or_, select

from app.config import get_settings
from app.dependencies import CurrentUser, OwnedBook, SessionDep
from app.models import Book
from app.schemas.book import BookOut, BookUpdate, ReorderRequest
from app.services.covers import generate_cover
from app.services.processing import process_book
from app.services.storage import delete_book_files, save_image, save_pdf

router = APIRouter(prefix="/books", tags=["books"])

PRIORITY_ORDER = case((Book.priority == "high", 3), (Book.priority == "medium", 2), else_=1)
SORT_COLUMNS = {
    "position": Book.position,
    "title": func.lower(Book.title),
    "created_at": Book.created_at,
    "updated_at": Book.updated_at,
    "last_read_at": Book.last_read_at,
    "priority": PRIORITY_ORDER,
}

BookSort = Literal["position", "title", "created_at", "updated_at", "last_read_at", "priority"]


@router.get("")
async def list_books(
    user: CurrentUser,
    session: SessionDep,
    status_filter: Annotated[Literal["to_read", "reading", "finished"] | None, Query(alias="status")] = None,
    priority: Annotated[Literal["low", "medium", "high"] | None, Query()] = None,
    color: Annotated[str | None, Query(max_length=9)] = None,
    favorite: Annotated[bool | None, Query()] = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    sort: Annotated[BookSort, Query()] = "position",
    order: Annotated[Literal["asc", "desc"], Query()] = "asc",
) -> list[BookOut]:
    query = select(Book).where(Book.user_id == user.id)
    if status_filter is not None:
        query = query.where(Book.status == status_filter)
    if priority is not None:
        query = query.where(Book.priority == priority)
    if color is not None:
        query = query.where(Book.color == color)
    if favorite is not None:
        query = query.where(Book.is_favorite.is_(favorite))
    if q:
        pattern = f"%{q}%"
        query = query.where(or_(Book.title.ilike(pattern), Book.author.ilike(pattern)))
    column = SORT_COLUMNS[sort]
    direction = column.desc().nulls_last() if order == "desc" else column.asc().nulls_last()
    books = (await session.scalars(query.order_by(direction, Book.position))).all()
    return [BookOut.from_model(book, include_pages=False) for book in books]


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_book(
    user: CurrentUser,
    session: SessionDep,
    background: BackgroundTasks,
    file: UploadFile,
    title: Annotated[str | None, Form(max_length=300)] = None,
    author: Annotated[str | None, Form(max_length=200)] = None,
) -> BookOut:
    book_id = uuid.uuid4()
    path, size = await save_pdf(file, book_id)
    next_position = (
        await session.scalar(
            select(func.coalesce(func.max(Book.position), 0)).where(Book.user_id == user.id)
        )
    ) + 1
    fallback_title = Path(file.filename).stem.strip() if file.filename else ""
    book = Book(
        id=book_id,
        user_id=user.id,
        title=(title or fallback_title or "Untitled").strip()[:300],
        author=author,
        file_path=str(path),
        file_size=size,
        position=next_position,
    )
    session.add(book)
    await session.commit()
    await session.refresh(book, ["progress"])
    background.add_task(process_book, book.id, title is None, author is None)
    return BookOut.from_model(book)


@router.put("/reorder", status_code=status.HTTP_204_NO_CONTENT)
async def reorder_books(payload: ReorderRequest, user: CurrentUser, session: SessionDep) -> None:
    books = (
        await session.scalars(
            select(Book).where(Book.user_id == user.id, Book.id.in_(payload.book_ids))
        )
    ).all()
    positions = {book_id: index for index, book_id in enumerate(payload.book_ids, start=1)}
    for book in books:
        book.position = positions[book.id]
    await session.commit()


@router.get("/{book_id}")
async def get_book(book: OwnedBook) -> BookOut:
    return BookOut.from_model(book)


@router.post("/{book_id}/reprocess")
async def reprocess_book(book: OwnedBook, session: SessionDep, background: BackgroundTasks) -> BookOut:
    book.extraction_status = "pending"
    book.extraction_error = None
    await session.commit()
    await session.refresh(book)
    background.add_task(process_book, book.id, False, False)
    return BookOut.from_model(book)


@router.patch("/{book_id}")
async def update_book(payload: BookUpdate, book: OwnedBook, session: SessionDep) -> BookOut:
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(book, field, value)
    await session.commit()
    await session.refresh(book)
    return BookOut.from_model(book)


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(book: OwnedBook, session: SessionDep) -> None:
    book_id = book.id
    await session.delete(book)
    await session.commit()
    delete_book_files(book_id)


@router.post("/{book_id}/cover")
async def upload_cover(file: UploadFile, book: OwnedBook, session: SessionDep) -> BookOut:
    path = await save_image(file, get_settings().covers_dir, str(book.id))
    book.cover_path = str(path)
    book.has_custom_cover = True
    await session.commit()
    await session.refresh(book)
    return BookOut.from_model(book)


@router.delete("/{book_id}/cover")
async def reset_cover(book: OwnedBook, session: SessionDep) -> BookOut:
    settings = get_settings()
    for existing in settings.covers_dir.glob(f"{book.id}.*"):
        existing.unlink(missing_ok=True)
    cover_path = settings.covers_dir / f"{book.id}.jpg"
    try:
        await anyio.to_thread.run_sync(generate_cover, Path(book.file_path), cover_path)
        book.cover_path = str(cover_path)
    except Exception:
        book.cover_path = None
    book.has_custom_cover = False
    await session.commit()
    await session.refresh(book)
    return BookOut.from_model(book)


@router.get("/{book_id}/cover")
async def get_cover(book: OwnedBook) -> FileResponse:
    if book.cover_path is None or not Path(book.cover_path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No cover available")
    return FileResponse(book.cover_path, headers={"Cache-Control": "private, no-cache"})


@router.get("/{book_id}/file")
async def get_file(book: OwnedBook) -> FileResponse:
    if not Path(book.file_path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Book file is missing")
    return FileResponse(
        book.file_path,
        media_type="application/pdf",
        headers={"Cache-Control": "private, max-age=86400"},
    )
