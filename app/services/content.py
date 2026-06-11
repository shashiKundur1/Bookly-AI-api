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


async def save_content(book_id: uuid.UUID, data: dict[str, Any]) -> None:
    path = content_path(book_id)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    await anyio.to_thread.run_sync(path.write_text, text)
    _cache[book_id] = data


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


def adjacent_chunk(
    content: dict[str, Any], chunk_id: str, direction: int
) -> dict[str, Any] | None:
    try:
        page_number = int(chunk_id.split("-", 1)[0])
    except ValueError:
        return None
    page_chunks = _page_chunks(content, page_number)
    index = next((i for i, chunk in enumerate(page_chunks) if chunk["id"] == chunk_id), None)
    if index is None:
        return None
    target = index + direction
    if 0 <= target < len(page_chunks):
        return page_chunks[target]
    step = 1 if direction > 0 else -1
    page = page_number + step
    while 1 <= page <= len(content["pages"]):
        candidates = _page_chunks(content, page)
        if candidates:
            return candidates[0] if step > 0 else candidates[-1]
        page += step
    return None


def first_chunk_at_or_after(content: dict[str, Any], page_number: int) -> dict[str, Any] | None:
    page = max(1, page_number)
    while page <= len(content["pages"]):
        candidates = _page_chunks(content, page)
        if candidates:
            return candidates[0]
        page += 1
    return None
