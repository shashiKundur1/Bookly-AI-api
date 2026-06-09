import json
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.config import get_settings
from app.dependencies import OwnedBook
from app.services import ai
from app.services.content import chunks_after, find_chunk, load_content
from app.services.tts import VOICE_IDS, VOICES, audio_path, ensure_audio, prefetch, timing_path

router = APIRouter(tags=["narration"])

PREFETCH_COUNT = 2


@router.get("/voices")
async def list_voices() -> list[dict[str, str]]:
    return VOICES


@router.get("/books/{book_id}/content")
async def get_content_overview(book: OwnedBook) -> dict[str, Any]:
    content = await load_content(book.id)
    if content is None:
        return {
            "extraction_status": book.extraction_status,
            "page_count": book.page_count,
            "toc": [],
        }
    return {
        "extraction_status": book.extraction_status,
        "page_count": content["page_count"],
        "toc": content["toc"],
    }


@router.get("/books/{book_id}/pages/{page_number}")
async def get_page_content(page_number: int, book: OwnedBook) -> dict[str, Any]:
    content = await load_content(book.id)
    if content is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Book content is still processing")
    if page_number < 1 or page_number > len(content["pages"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")
    return content["pages"][page_number - 1]


async def _resolve_audio(book: OwnedBook, chunk_id: str, voice: str | None):
    resolved_voice = voice or get_settings().default_voice
    if resolved_voice not in VOICE_IDS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown voice")
    content = await load_content(book.id)
    if content is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Book content is still processing")
    chunk = find_chunk(content, chunk_id)
    if chunk is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chunk not found")
    if ai.is_enabled() and not audio_path(book.id, resolved_voice, chunk_id).exists():
        await ai.ensure_page_polished(book.id, chunk["page"])
        content = await load_content(book.id)
        chunk = find_chunk(content, chunk_id) or chunk
    path = await ensure_audio(book.id, resolved_voice, chunk)
    prefetch(book.id, resolved_voice, chunks_after(content, chunk_id, PREFETCH_COUNT))
    return path


@router.get("/books/{book_id}/audio/{chunk_id}")
async def get_chunk_audio(
    chunk_id: str,
    book: OwnedBook,
    voice: Annotated[str | None, Query()] = None,
) -> FileResponse:
    path = await _resolve_audio(book, chunk_id, voice)
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=604800"},
    )


@router.get("/books/{book_id}/audio/{chunk_id}/timing")
async def get_chunk_timing(
    chunk_id: str,
    book: OwnedBook,
    voice: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    path = await _resolve_audio(book, chunk_id, voice)
    timing = timing_path(path)
    if not timing.exists():
        return {"words": []}
    return await anyio.to_thread.run_sync(lambda: json.loads(timing.read_text()))
