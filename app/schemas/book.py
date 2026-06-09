import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from app.models import Book
from app.schemas.reading import ProgressOut

BookStatus = Literal["to_read", "reading", "finished"]
BookPriority = Literal["low", "medium", "high"]
HexColor = Annotated[str, StringConstraints(pattern=r"^#[0-9a-fA-F]{6}$")]


class BookOut(BaseModel):
    id: uuid.UUID
    title: str
    author: str | None
    description: str | None
    status: BookStatus
    priority: BookPriority
    color: str | None
    is_favorite: bool
    position: int
    page_count: int
    file_size: int
    extraction_status: str
    extraction_error: str | None
    has_cover: bool
    created_at: datetime
    updated_at: datetime
    last_read_at: datetime | None
    progress: ProgressOut | None

    @classmethod
    def from_model(cls, book: Book) -> "BookOut":
        progress = None
        if book.progress is not None:
            progress = ProgressOut.from_model(book.progress, book.page_count)
        return cls(
            id=book.id,
            title=book.title,
            author=book.author,
            description=book.description,
            status=book.status,
            priority=book.priority,
            color=book.color,
            is_favorite=book.is_favorite,
            position=book.position,
            page_count=book.page_count,
            file_size=book.file_size,
            extraction_status=book.extraction_status,
            extraction_error=book.extraction_error,
            has_cover=book.cover_path is not None,
            created_at=book.created_at,
            updated_at=book.updated_at,
            last_read_at=book.last_read_at,
            progress=progress,
        )


class BookUpdate(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)] | None = None
    author: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] | None = None
    description: Annotated[str, StringConstraints(max_length=5000)] | None = None
    status: BookStatus | None = None
    priority: BookPriority | None = None
    color: HexColor | None = None
    is_favorite: bool | None = None


class ReorderRequest(BaseModel):
    book_ids: list[uuid.UUID] = Field(min_length=1)
