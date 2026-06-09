import json
import logging
import uuid
from pathlib import Path
from typing import Any

import anyio

from app.config import get_settings
from app.database import SessionFactory
from app.models import Book
from app.services.content import invalidate
from app.services.covers import generate_cover
from app.services.extraction import extract_book

logger = logging.getLogger(__name__)


def _write_content(path: Path, result: dict[str, Any]) -> None:
    payload = {
        "page_count": result["page_count"],
        "toc": result["toc"],
        "pages": result["pages"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


async def process_book(book_id: uuid.UUID, use_meta_title: bool, use_meta_author: bool) -> None:
    settings = get_settings()
    async with SessionFactory() as session:
        book = await session.get(Book, book_id)
        if book is None:
            return
        book.extraction_status = "processing"
        await session.commit()
        try:
            pdf_path = Path(book.file_path)
            result = await anyio.to_thread.run_sync(extract_book, pdf_path)
            await anyio.to_thread.run_sync(
                _write_content, settings.content_dir / f"{book_id}.json", result
            )
            if not book.has_custom_cover:
                cover_path = settings.covers_dir / f"{book_id}.jpg"
                await anyio.to_thread.run_sync(generate_cover, pdf_path, cover_path)
                book.cover_path = str(cover_path)
            book.page_count = result["page_count"]
            if use_meta_title and result["title"]:
                book.title = result["title"][:300]
            if use_meta_author and result["author"]:
                book.author = result["author"][:200]
            book.extraction_status = "ready"
            book.extraction_error = None
        except Exception as exc:
            logger.exception("Failed to process book %s", book_id)
            book.extraction_status = "failed"
            book.extraction_error = str(exc)[:2000]
        invalidate(book_id)
        await session.commit()
