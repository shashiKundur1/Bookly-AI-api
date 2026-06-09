import json
import uuid
from pathlib import Path
from typing import Any

import anyio

from app.config import get_settings

MAX_CACHED_BOOKS = 6

_cache: dict[uuid.UUID, dict[str, Any]] = {}
_order: list[uuid.UUID] = []


def content_path(book_id: uuid.UUID) -> Path:
    return get_settings().content_dir / f"{book_id}.json"


def invalidate(book_id: uuid.UUID) -> None:
    _cache.pop(book_id, None)
    if book_id in _order:
        _order.remove(book_id)


async def load_content(book_id: uuid.UUID) -> dict[str, Any] | None:
    if book_id in _cache:
        return _cache[book_id]
    path = content_path(book_id)
    if not path.exists():
        return None
    data = await anyio.to_thread.run_sync(lambda: json.loads(path.read_text()))
    _cache[book_id] = data
    _order.append(book_id)
    while len(_order) > MAX_CACHED_BOOKS:
        _cache.pop(_order.pop(0), None)
    return data


def _page_chunks(content: dict[str, Any], page_number: int) -> list[dict[str, Any]]:
    if 1 <= page_number <= len(content["pages"]):
        return content["pages"][page_number - 1]["chunks"]
    return []


def find_chunk(content: dict[str, Any], chunk_id: str) -> dict[str, Any] | None:
    try:
        page_number = int(chunk_id.split("-", 1)[0])
    except ValueError:
        return None
    for chunk in _page_chunks(content, page_number):
        if chunk["id"] == chunk_id:
            return chunk
    return None


def chunks_after(content: dict[str, Any], chunk_id: str, count: int) -> list[dict[str, Any]]:
    try:
        page_number = int(chunk_id.split("-", 1)[0])
    except ValueError:
        return []
    collected: list[dict[str, Any]] = []
    passed_current = False
    page = page_number
    while len(collected) < count and page <= len(content["pages"]):
        for chunk in _page_chunks(content, page):
            if passed_current:
                collected.append(chunk)
                if len(collected) >= count:
                    break
            elif chunk["id"] == chunk_id:
                passed_current = True
        if not passed_current:
            return []
        page += 1
    return collected
