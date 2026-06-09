import asyncio
import json
import uuid
from typing import Annotated, Any

import anyio
import numpy as np
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.database import SessionFactory
from app.dependencies import OwnedBook
from app.models import Book
from app.security import ACCESS_COOKIE, decode_token
from app.services import ai
from app.services.cache import is_fresh, weak_etag
from app.services.content import (
    adjacent_chunk,
    chunks_after,
    find_chunk,
    first_chunk_at_or_after,
    load_content,
)
from app.services.tts import (
    SAMPLE_RATE,
    VOICE_IDS,
    VOICES,
    audio_path,
    engine,
    ensure_audio,
    pcm16_bytes,
    prefetch,
    resolve_voice,
    split_for_streaming,
    timing_path,
    write_chunk_cache,
)

router = APIRouter(tags=["narration"])

PREFETCH_COUNT = 2
PCM_FRAME_BYTES = 64 * 1024
CONTENT_CACHE_CONTROL = "private, max-age=300"
AUDIO_CACHE_CONTROL = "private, max-age=604800"


@router.get("/voices")
async def list_voices(response: Response) -> list[dict[str, str]]:
    response.headers["Cache-Control"] = "public, max-age=86400"
    return VOICES


@router.get("/books/{book_id}/content")
async def get_content_overview(book: OwnedBook, request: Request) -> Response:
    etag = weak_etag(book.id, book.updated_at, "overview")
    if is_fresh(request, etag):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})
    content = await load_content(book.id)
    if content is None:
        payload: dict[str, Any] = {
            "extraction_status": book.extraction_status,
            "page_count": book.page_count,
            "toc": [],
        }
    else:
        payload = {
            "extraction_status": book.extraction_status,
            "page_count": content["page_count"],
            "toc": content["toc"],
        }
    return JSONResponse(payload, headers={"ETag": etag, "Cache-Control": CONTENT_CACHE_CONTROL})


@router.get("/books/{book_id}/pages/{page_number}")
async def get_page_content(page_number: int, book: OwnedBook, request: Request) -> Response:
    etag = weak_etag(book.id, book.updated_at, "page", page_number)
    if is_fresh(request, etag):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})
    content = await load_content(book.id)
    if content is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Book content is still processing")
    if page_number < 1 or page_number > len(content["pages"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found")
    return JSONResponse(
        content["pages"][page_number - 1],
        headers={"ETag": etag, "Cache-Control": CONTENT_CACHE_CONTROL},
    )


async def _resolve_audio(book: OwnedBook, chunk_id: str, voice: str | None):
    resolved_voice = resolve_voice(voice)
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
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": AUDIO_CACHE_CONTROL})


class NarrationSession:
    def __init__(self, voice: str, speed: float) -> None:
        self.voice = voice
        self.speed = speed
        self.current: str | None = None
        self.pending: str | None = None
        self.flush = False
        self.acked = False
        self.stopped = False
        self.wake = asyncio.Event()

    def signal(self) -> None:
        self.wake.set()

    async def wait(self) -> None:
        await self.wake.wait()
        self.wake.clear()


def _clamp_speed(value: Any) -> float:
    try:
        return min(2.0, max(0.5, float(value)))
    except (TypeError, ValueError):
        return 1.0


def _resolve_start(content: dict[str, Any], message: dict[str, Any]) -> str | None:
    chunk_id = message.get("chunk")
    if isinstance(chunk_id, str):
        chunk = find_chunk(content, chunk_id)
        if chunk is not None:
            return chunk["id"]
    page = message.get("page")
    if isinstance(page, int):
        chunk = first_chunk_at_or_after(content, page)
        if chunk is not None:
            return chunk["id"]
    return None


async def _narration_receiver(
    websocket: WebSocket, session: NarrationSession, content: dict[str, Any]
) -> None:
    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "start":
                voice = message.get("voice")
                if voice in VOICE_IDS:
                    session.voice = voice
                session.speed = _clamp_speed(message.get("speed", session.speed))
                target = _resolve_start(content, message)
                if target is not None:
                    session.pending = target
                    session.flush = True
            elif kind == "ack":
                session.acked = True
            elif kind == "seek":
                target = None
                chunk_id = message.get("chunk")
                direction = message.get("direction")
                if isinstance(chunk_id, str):
                    chunk = find_chunk(content, chunk_id)
                    target = chunk["id"] if chunk else None
                elif direction in (1, -1) and session.current is not None:
                    chunk = adjacent_chunk(content, session.current, direction)
                    target = chunk["id"] if chunk else None
                if target is not None:
                    session.pending = target
                    session.flush = True
            elif kind == "voice":
                voice = message.get("voice")
                if voice in VOICE_IDS:
                    session.voice = voice
            elif kind == "speed":
                session.speed = _clamp_speed(message.get("speed"))
            elif kind == "stop":
                session.stopped = True
            session.signal()
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError, KeyError):
        session.stopped = True
        session.signal()


async def _send_pcm(websocket: WebSocket, pcm: bytes) -> None:
    for index in range(0, len(pcm), PCM_FRAME_BYTES):
        await websocket.send_bytes(pcm[index : index + PCM_FRAME_BYTES])


async def _stream_cached(websocket: WebSocket, chunk: dict[str, Any], path) -> None:
    data = await anyio.to_thread.run_sync(path.read_bytes)
    pcm = data[44:]
    words: list[dict[str, Any]] = []
    timing = timing_path(path)
    if timing.exists():
        try:
            words = json.loads(timing.read_text()).get("words", [])
        except (json.JSONDecodeError, OSError):
            words = []
    await websocket.send_json(
        {
            "type": "sentence",
            "chunk_id": chunk["id"],
            "text": chunk["speech"],
            "offset": 0.0,
            "words": words,
        }
    )
    await _send_pcm(websocket, pcm)


async def _stream_chunk(
    websocket: WebSocket,
    session: NarrationSession,
    book_id: uuid.UUID,
    chunk: dict[str, Any],
) -> None:
    voice = session.voice
    speed = session.speed
    cache = audio_path(book_id, voice, chunk["id"])
    cached = speed == 1.0 and cache.exists()
    await websocket.send_json(
        {
            "type": "chunk",
            "id": chunk["id"],
            "page": chunk["page"],
            "blocks": chunk["blocks"],
            "speech": chunk["speech"],
            "sample_rate": SAMPLE_RATE,
            "cached": cached,
        }
    )
    if cached:
        await _stream_cached(websocket, chunk, cache)
        duration = (cache.stat().st_size - 44) / 2 / SAMPLE_RATE
        await websocket.send_json(
            {"type": "chunk_end", "id": chunk["id"], "duration": round(duration, 3)}
        )
        return
    offset = 0.0
    parts: list[np.ndarray] = []
    collected: list[dict[str, Any]] = []
    rate = SAMPLE_RATE
    complete = True
    for piece in split_for_streaming(chunk["speech"]):
        if session.flush or session.stopped:
            complete = False
            break
        samples, rate, words = await engine.synthesize_sentence(piece, session.voice, session.speed)
        shifted = [
            {**word, "start": round(word["start"] + offset, 3), "end": round(word["end"] + offset, 3)}
            for word in words
        ]
        await websocket.send_json(
            {
                "type": "sentence",
                "chunk_id": chunk["id"],
                "text": piece,
                "offset": round(offset, 3),
                "words": shifted,
            }
        )
        await _send_pcm(websocket, pcm16_bytes(samples))
        offset += len(samples) / rate
        parts.append(samples)
        collected.extend(shifted)
    if complete and parts and session.speed == 1.0 and session.voice == voice:
        combined = np.concatenate(parts)
        await anyio.to_thread.run_sync(write_chunk_cache, cache, combined, rate, collected)
    if complete:
        await websocket.send_json(
            {"type": "chunk_end", "id": chunk["id"], "duration": round(offset, 3)}
        )


@router.websocket("/books/{book_id}/narrate")
async def narrate_socket(websocket: WebSocket, book_id: uuid.UUID) -> None:
    settings = get_settings()
    origin = websocket.headers.get("origin")
    if settings.environment == "production" and origin and origin not in settings.cors_origins:
        await websocket.close(code=4403)
        return
    token = websocket.cookies.get(ACCESS_COOKIE)
    user_id = decode_token(token, "access") if token else None
    authorized = False
    if user_id is not None:
        async with SessionFactory() as db:
            book = await db.get(Book, book_id)
            authorized = book is not None and book.user_id == user_id
    if not authorized:
        await websocket.close(code=4401)
        return
    content = await load_content(book_id)
    if content is None:
        await websocket.close(code=4409)
        return
    await websocket.accept()
    session = NarrationSession(resolve_voice(None), 1.0)
    receiver = asyncio.create_task(_narration_receiver(websocket, session, content))
    try:
        while not session.stopped:
            if session.pending is None:
                await session.wait()
                continue
            chunk_id = session.pending
            session.pending = None
            session.flush = False
            chunk = find_chunk(content, chunk_id)
            if chunk is None:
                await websocket.send_json({"type": "error", "message": "Chunk not found"})
                continue
            session.current = chunk["id"]
            await _stream_chunk(websocket, session, book_id, chunk)
            if session.stopped or session.pending is not None:
                continue
            session.acked = False
            while not (session.acked or session.pending is not None or session.stopped):
                await session.wait()
            if session.acked and session.pending is None:
                next_chunk = adjacent_chunk(content, session.current, 1)
                if next_chunk is not None:
                    session.pending = next_chunk["id"]
                else:
                    await websocket.send_json({"type": "end"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()


@router.get("/books/{book_id}/audio/{chunk_id}/timing")
async def get_chunk_timing(
    chunk_id: str,
    book: OwnedBook,
    response: Response,
    voice: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    path = await _resolve_audio(book, chunk_id, voice)
    response.headers["Cache-Control"] = AUDIO_CACHE_CONTROL
    timing = timing_path(path)
    if not timing.exists():
        return {"words": []}
    return await anyio.to_thread.run_sync(lambda: json.loads(timing.read_text()))
