import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models import ReadingProgress


class ProgressOut(BaseModel):
    current_page: int
    pages_read: list[int]
    percent: float
    updated_at: datetime | None

    @classmethod
    def from_model(
        cls, progress: ReadingProgress, page_count: int, include_pages: bool = True
    ) -> "ProgressOut":
        read = sorted(set(progress.pages_read or []))
        percent = round(len(read) / page_count * 100, 1) if page_count else 0.0
        return cls(
            current_page=progress.current_page,
            pages_read=read if include_pages else [],
            percent=min(percent, 100.0),
            updated_at=progress.updated_at,
        )


class ProgressUpdate(BaseModel):
    current_page: int | None = Field(default=None, ge=1)
    mark_read: list[int] | None = None
    unmark_read: list[int] | None = None


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    book_id: uuid.UUID
    started_at: datetime
    ended_at: datetime | None
    start_page: int
    end_page: int | None


class SessionStart(BaseModel):
    start_page: int = Field(default=1, ge=1)


class SessionUpdate(BaseModel):
    end_page: int | None = Field(default=None, ge=1)
