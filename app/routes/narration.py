import asyncio
import json
import logging
import uuid
from typing import Any

import anyio
import numpy as np
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import SessionFactory
from app.dependencies import OwnedBook
from app.models import Book
from app.security import ACCESS_COOKIE, decode_token
from app.services import ai
from app.services.cache import is_fresh, weak_etag
from app.services.content import (
    adjacent_chunk,
    find_chunk,
    first_chunk_at_or_after,
    load_content,
)
from app.services.emotion import DEFAULT_EMOTION, EMOTIONS, plan, resolve_emotion
from app.services.tts import (
    SAMPLE_RATE,
    VOICE_IDS,
    VOICES,
    audio_path,
    engine,
    pcm16_bytes,
    resolve_voice,
    split_for_streaming,
    timing_path,
    write_chunk_cache,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["narration"])

PCM_FRAME_BYTES = 64 * 1024
CONTENT_CACHE_CONTROL = "private, max-age=300"
ESTIMATED_CHARS_PER_SECOND = 13.5

_polish_tasks: set[asyncio.Task] = set()


@router.get("/voices")
async def list_voices(response: Response) -> list[dict[str, str]]:
    response.headers["Cache-Control"] = "public, max-age=86400"
    return VOICES


@router.get("/emotions")
async def list_emotions(response: Response) -> list[dict[str, str]]:
    response.headers["Cache-Control"] = "public, max-age=86400"
    return EMOTIONS


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


class WarmChunk:
    """Partially synthesized chunk held in session memory for instant starts."""

    def __init__(
        self, chunk_id: str, speech: str, voice: str, emotion: str, speed: float
    ) -> None:
        self.chunk_id = chunk_id
        self.speech = speech
        self.voice = voice
        self.emotion = emotion
        self.speed = speed
        self.index = 0
        self.offset = 0.0
        self.parts: list[np.ndarray] = []
        self.frames: list[dict[str, Any]] = []
        self.collected: list[dict[str, Any]] = []
        self.complete = False

    def matches(self, session: "NarrationSession", chunk_id: str, speech: str) -> bool:
        return (
            self.chunk_id == chunk_id
            and self.speech == speech
            and self.voice == session.voice
            and self.emotion == session.emotion
            and self.speed == session.speed
        )


class NarrationSession:
    def __init__(self, voice: str, speed: float, emotion: str) -> None:
        self.voice = voice
        self.speed = speed
        self.emotion = emotion
        self.current: str | None = None
        self.pending: str | None = None
        self.warm_target: str | None = None
        self.warm: WarmChunk | None = None
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
                if isinstance(message.get("emotion"), str):
                    session.emotion = resolve_emotion(message["emotion"])
                target = _resolve_start(content, message)
                if target is not None:
                    session.pending = target
                    session.flush = True
            elif kind == "ack":
                session.acked = True
            elif kind == "seek":
                chunk_id = message.get("chunk")
                direction = message.get("direction")
                page = message.get("page")
                if isinstance(page, int) and not isinstance(chunk_id, str):
                    chunk = first_chunk_at_or_after(content, page)
                else:
                    base = (
                        chunk_id if isinstance(chunk_id, str) else session.pending or session.current
                    )
                    chunk = find_chunk(content, base) if base else None
                    if chunk is not None and direction in (1, -1):
                        chunk = adjacent_chunk(content, chunk["id"], direction) or chunk
                if chunk is not None:
                    session.pending = chunk["id"]
                    session.flush = True
            elif kind == "voice":
                voice = message.get("voice")
                if voice in VOICE_IDS:
                    session.voice = voice
            elif kind == "emotion":
                if isinstance(message.get("emotion"), str):
                    session.emotion = resolve_emotion(message["emotion"])
            elif kind == "speed":
                session.speed = _clamp_speed(message.get("speed"))
            elif kind == "stop":
                session.stopped = True
            session.signal()
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError, KeyError):
        session.stopped = True
        session.signal()


def _kick_polish(book_id: uuid.UUID, page: int) -> None:
    if not ai.is_enabled():
        return
    task = asyncio.create_task(ai.ensure_page_polished(book_id, page))
    _polish_tasks.add(task)
    task.add_done_callback(_polish_done)


def _polish_done(task: asyncio.Task) -> None:
    _polish_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Background polish failed: %s", task.exception())


async def _synthesize_piece(
    piece: str, index: int, pieces: list[str], session: NarrationSession
) -> tuple[np.ndarray, int, list[dict[str, Any]], Any]:
    prosody = await asyncio.to_thread(
        lambda: plan(
            piece,
            session.emotion,
            first=index == 0,
            last=index == len(pieces) - 1,
            next_chars=len(pieces[index + 1]) if index + 1 < len(pieces) else None,
        )
    )
    samples, rate, words = await engine.synthesize_sentence(
        piece, session.voice, session.speed, prosody
    )
    pre = prosody.pre_pause / session.speed
    post = prosody.post_pause / session.speed
    samples = _pad_silence(samples, rate, pre, post)
    return samples, rate, words, prosody


def _piece_frame(
    piece: str, offset: float, pre: float, words: list[dict[str, Any]], prosody
) -> dict[str, Any]:
    shifted = [
        {
            **word,
            "start": round(word["start"] + offset + pre, 3),
            "end": round(word["end"] + offset + pre, 3),
        }
        for word in words
    ]
    return {
        "text": piece,
        "offset": round(offset, 3),
        "words": shifted,
        "cues": {
            "lead": prosody.lead_tag if engine.performs(prosody.lead_tag) else "",
            "trail": prosody.trail_tag if engine.performs(prosody.trail_tag) else "",
        },
    }


async def _warm_step(
    session: NarrationSession, book_id: uuid.UUID, content: dict[str, Any]
) -> bool:
    """Synthesize one piece of the warm-target chunk into session memory.

    Runs whenever the main loop is otherwise idle, so a Play (or the next-chunk
    advance) lands on audio that already exists. Returns False when there is
    nothing (more) to warm.
    """
    target = session.warm_target
    if target is None or session.speed != 1.0:
        session.warm_target = None
        return False
    chunk = find_chunk(content, target)
    if chunk is None:
        session.warm_target = None
        return False
    cache = audio_path(book_id, session.voice, session.emotion, chunk["id"], chunk["speech"])
    if cache.exists() and not _cache_is_stale(cache):
        session.warm_target = None
        return False
    warm = session.warm
    if warm is None or not warm.matches(session, chunk["id"], chunk["speech"]):
        warm = WarmChunk(
            chunk["id"], chunk["speech"], session.voice, session.emotion, session.speed
        )
        session.warm = warm
    if warm.complete:
        session.warm_target = None
        return False
    pieces = split_for_streaming(warm.speech)
    piece = pieces[warm.index]
    samples, rate, words, prosody = await _synthesize_piece(piece, warm.index, pieces, session)
    # Settings or polish may have changed underneath the synthesis; discard stale warms.
    if not warm.matches(session, chunk["id"], chunk["speech"]):
        session.warm = None
        return True
    pre = prosody.pre_pause / session.speed
    warm.frames.append(_piece_frame(piece, warm.offset, pre, words, prosody))
    warm.parts.append(samples)
    warm.collected.extend(warm.frames[-1]["words"])
    warm.offset += len(samples) / rate
    warm.index += 1
    if warm.index >= len(pieces):
        warm.complete = True
        session.warm_target = None
        combined = np.concatenate(warm.parts)
        await anyio.to_thread.run_sync(
            write_chunk_cache, cache, combined, rate, warm.collected, warm.frames
        )
    return True


def _cache_is_stale(path) -> bool:
    if not engine.has_word_timings:
        return False
    timing = timing_path(path)
    if not timing.exists():
        return True
    try:
        return len(json.loads(timing.read_text()).get("words", [])) == 0
    except (json.JSONDecodeError, OSError):
        return True


def _pad_silence(samples: np.ndarray, rate: int, pre: float, post: float) -> np.ndarray:
    if pre <= 0 and post <= 0:
        return samples
    return np.concatenate(
        [
            np.zeros(int(pre * rate), dtype=np.float32),
            samples,
            np.zeros(int(post * rate), dtype=np.float32),
        ]
    )


async def _send_pcm(websocket: WebSocket, pcm: bytes) -> None:
    for index in range(0, len(pcm), PCM_FRAME_BYTES):
        await websocket.send_bytes(pcm[index : index + PCM_FRAME_BYTES])


async def _stream_cached(websocket: WebSocket, chunk: dict[str, Any], speech: str, path) -> None:
    data = await anyio.to_thread.run_sync(path.read_bytes)
    pcm = data[44:]
    timing_data: dict[str, Any] = {}
    timing = timing_path(path)
    if timing.exists():
        try:
            timing_data = json.loads(timing.read_text())
        except (json.JSONDecodeError, OSError):
            timing_data = {}
    sentences = timing_data.get("sentences")
    if not sentences:
        sentences = [
            {
                "text": speech,
                "offset": 0.0,
                "words": timing_data.get("words", []),
                "cues": {"lead": "", "trail": ""},
            }
        ]
    for frame in sentences:
        await websocket.send_json(
            {
                "type": "sentence",
                "chunk_id": chunk["id"],
                "text": frame.get("text", ""),
                "offset": frame.get("offset", 0.0),
                "words": frame.get("words", []),
                "cues": frame.get("cues", {"lead": "", "trail": ""}),
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
    emotion = session.emotion
    speed = session.speed
    speech = chunk["speech"]
    cache = audio_path(book_id, voice, emotion, chunk["id"], speech)
    cached = speed == 1.0 and cache.exists() and not _cache_is_stale(cache)
    await websocket.send_json(
        {
            "type": "chunk",
            "id": chunk["id"],
            "page": chunk["page"],
            "blocks": chunk["blocks"],
            "speech": speech,
            "sample_rate": SAMPLE_RATE,
            "cached": cached,
        }
    )
    if cached:
        await _stream_cached(websocket, chunk, speech, cache)
        duration = (cache.stat().st_size - 44) / 2 / SAMPLE_RATE
        await websocket.send_json(
            {"type": "chunk_end", "id": chunk["id"], "duration": round(duration, 3)}
        )
        return
    offset = 0.0
    parts: list[np.ndarray] = []
    collected: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    rate = SAMPLE_RATE
    complete = True
    pieces = split_for_streaming(speech)
    start_index = 0
    warm = session.warm
    if warm is not None and warm.matches(session, chunk["id"], speech) and warm.parts:
        # Warm start: flush everything already synthesized in the background,
        # then continue live from where the warm-up left off.
        for frame, samples in zip(warm.frames, warm.parts):
            await websocket.send_json({"type": "sentence", "chunk_id": chunk["id"], **frame})
            await _send_pcm(websocket, pcm16_bytes(samples))
        offset = warm.offset
        parts = warm.parts
        collected = list(warm.collected)
        frames = list(warm.frames)
        start_index = warm.index
        session.warm = None
    for index, piece in enumerate(pieces[start_index:], start=start_index):
        if session.flush or session.stopped:
            complete = False
            break
        prosody = await asyncio.to_thread(
            lambda p=piece, i=index: plan(
                p,
                emotion,
                first=i == 0,
                last=i == len(pieces) - 1,
                next_chars=len(pieces[i + 1]) if i + 1 < len(pieces) else None,
            )
        )
        pre = prosody.pre_pause / session.speed
        post = prosody.post_pause / session.speed
        cues = {
            "lead": prosody.lead_tag if engine.performs(prosody.lead_tag) else "",
            "trail": prosody.trail_tag if engine.performs(prosody.trail_tag) else "",
        }

        def shift(words: list[dict[str, Any]], base: float) -> list[dict[str, Any]]:
            return [
                {
                    **word,
                    "start": round(word["start"] + base, 3),
                    "end": round(word["end"] + base, 3),
                }
                for word in words
            ]

        if engine.streams_pcm and session.speed == 1.0:
            # Streaming engines push PCM the moment frames decode, so playback
            # starts in well under a second. The sentence frame goes out first
            # with pace-estimated word timings; the exact timings follow in a
            # sentence_update once the piece finishes.
            estimated = engine._estimate_words(
                piece, len(piece) / ESTIMATED_CHARS_PER_SECOND, prosody
            )
            frame = {
                "text": piece,
                "offset": round(offset, 3),
                "words": shift(estimated, offset + pre),
                "cues": cues,
            }
            await websocket.send_json({"type": "sentence", "chunk_id": chunk["id"], **frame})
            if pre > 0:
                await _send_pcm(websocket, pcm16_bytes(np.zeros(int(pre * rate), np.float32)))

            async def forward(block: np.ndarray) -> None:
                await _send_pcm(websocket, pcm16_bytes(block))

            samples, rate, words = await engine.stream_sentence(
                piece, session.voice, session.speed, prosody, forward
            )
            if post > 0:
                await _send_pcm(websocket, pcm16_bytes(np.zeros(int(post * rate), np.float32)))
            samples = _pad_silence(samples, rate, pre, post)
            shifted = shift(words, offset + pre)
            frame["words"] = shifted
            await websocket.send_json(
                {
                    "type": "sentence_update",
                    "chunk_id": chunk["id"],
                    "offset": frame["offset"],
                    "words": shifted,
                }
            )
        else:
            samples, rate, words = await engine.synthesize_sentence(
                piece, session.voice, session.speed, prosody
            )
            samples = _pad_silence(samples, rate, pre, post)
            shifted = shift(words, offset + pre)
            frame = {
                "text": piece,
                "offset": round(offset, 3),
                "words": shifted,
                "cues": cues,
            }
            await websocket.send_json({"type": "sentence", "chunk_id": chunk["id"], **frame})
            await _send_pcm(websocket, pcm16_bytes(samples))
        offset += len(samples) / rate
        parts.append(samples)
        collected.extend(shifted)
        frames.append(frame)
    if (
        complete
        and parts
        and session.speed == 1.0
        and session.voice == voice
        and session.emotion == emotion
    ):
        combined = np.concatenate(parts)
        await anyio.to_thread.run_sync(write_chunk_cache, cache, combined, rate, collected, frames)
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
    session = NarrationSession(resolve_voice(None), 1.0, DEFAULT_EMOTION)
    receiver = asyncio.create_task(_narration_receiver(websocket, session, content))
    polished_page = -1
    try:
        while not session.stopped:
            if session.pending is None:
                if session.warm_target is not None and await _warm_step(
                    session, book_id, content
                ):
                    continue
                await session.wait()
                continue
            chunk_id = session.pending
            session.pending = None
            session.flush = False
            chunk = find_chunk(content, chunk_id)
            if chunk is None:
                await websocket.send_json({"type": "error", "message": "Chunk not found"})
                continue
            if chunk["page"] != polished_page:
                polished_page = chunk["page"]
                _kick_polish(book_id, polished_page)
                _kick_polish(book_id, polished_page + 1)
            session.current = chunk["id"]
            await _stream_chunk(websocket, session, book_id, chunk)
            if session.stopped or session.pending is not None:
                continue
            upcoming = adjacent_chunk(content, chunk["id"], 1)
            if upcoming is not None:
                session.warm_target = upcoming["id"]
            session.acked = False
            while not (session.acked or session.pending is not None or session.stopped):
                if session.warm_target is not None and await _warm_step(
                    session, book_id, content
                ):
                    continue
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
