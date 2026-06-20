import logging
import time
import uuid
from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, StringConstraints
from sqlalchemy import func, select

from app.config import get_settings
from app.dependencies import CurrentUser, SessionDep
from app.models import Book
from app.schemas.book import BookOut
from app.security import create_ws_ticket, decode_token
from app.services import content as content_store
from app.services.emotion import DEFAULT_EMOTION, resolve_emotion
from app.services.narration import build_chunks
from app.services.synthesis import stream_pieces
from app.services.tts import SAMPLE_RATE, VOICE_IDS, resolve_voice

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extension", tags=["extension"])

MAX_SELECTION_CHARS = 20_000
MAX_ARTICLE_CHARS = 200_000
ARTICLE_EXTRACTOR_VERSION = 1

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]

TICKET_RATE_WINDOW = 60.0
TICKET_RATE_LIMIT = 20
_ticket_hits: dict[uuid.UUID, list[float]] = defaultdict(list)


def _rate_limited(user_id: uuid.UUID) -> bool:
    now = time.monotonic()
    hits = [stamp for stamp in _ticket_hits[user_id] if now - stamp < TICKET_RATE_WINDOW]
    hits.append(now)
    _ticket_hits[user_id] = hits
    return len(hits) > TICKET_RATE_LIMIT


class WsTicket(BaseModel):
    ticket: str
    expires_in: int


class ArticleCapture(BaseModel):
    title: Title
    author: str | None = None
    url: str | None = None
    text: str


@router.post("/ws-ticket")
async def issue_ws_ticket(user: CurrentUser) -> WsTicket:
    logger.info("ws-ticket requested by user %s", user.id)
    if _rate_limited(user.id):
        logger.warning("ws-ticket rate limited for user %s", user.id)
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many narration requests")
    return WsTicket(ticket=create_ws_ticket(user.id), expires_in=60)


def _clamp_speed(value: object) -> float:
    try:
        return min(2.0, max(0.5, float(value)))
    except (TypeError, ValueError):
        return 1.0


@router.websocket("/narrate")
async def narrate_selection(websocket: WebSocket) -> None:
    settings = get_settings()
    origin = websocket.headers.get("origin")
    if (
        settings.environment == "production"
        and origin
        and not origin.startswith("chrome-extension://")
        and origin not in settings.cors_origins
    ):
        logger.warning("Selection narrate rejected origin=%s", origin)
        await websocket.close(code=4403)
        return
    ticket = websocket.query_params.get("ticket")
    user_id = decode_token(ticket, "ws") if ticket else None
    if user_id is None:
        logger.warning("Selection narrate rejected: invalid ticket")
        await websocket.close(code=4401)
        return
    await websocket.accept()
    logger.info("Selection narrate connected user=%s", user_id)

    async def send_json(payload: dict) -> None:
        await websocket.send_json(payload)

    async def send_pcm(pcm: bytes) -> None:
        await websocket.send_bytes(pcm)

    stopped = False
    try:
        while not stopped:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "stop":
                logger.info("Selection narrate stop user=%s", user_id)
                stopped = True
                continue
            if kind != "speak":
                continue
            text = (message.get("text") or "").strip()[:MAX_SELECTION_CHARS]
            if not text:
                await send_json({"type": "error", "message": "Nothing to read"})
                continue
            voice = message.get("voice")
            voice = voice if voice in VOICE_IDS else resolve_voice(None)
            speed = _clamp_speed(message.get("speed", 1.0))
            emotion = resolve_emotion(message.get("emotion") or DEFAULT_EMOTION)
            chunk_id = uuid.uuid4().hex[:8]
            logger.info(
                "Selection speak user=%s chars=%d voice=%s emotion=%s speed=%.2f",
                user_id, len(text), voice, emotion, speed,
            )
            await send_json(
                {"type": "begin", "chunk_id": chunk_id, "sample_rate": SAMPLE_RATE}
            )
            duration = await stream_pieces(
                text=text,
                voice=voice,
                speed=speed,
                emotion=emotion,
                chunk_id=chunk_id,
                send_json=send_json,
                send_pcm=send_pcm,
            )
            await send_json(
                {"type": "end", "chunk_id": chunk_id, "duration": round(duration, 3)}
            )
            logger.info("Selection speak done user=%s duration=%.3f", user_id, duration)
    except (WebSocketDisconnect, RuntimeError) as exc:
        logger.info("Selection narrate disconnected user=%s (%s)", user_id, exc)
    except Exception:
        logger.exception("Selection narrate failed user=%s", user_id)


def _content_from_text(text: str) -> dict:
    paragraphs = [" ".join(part.split()) for part in text.split("\n") if part.strip()]
    blocks = [
        {"type": "paragraph", "text": para, "bbox": [0, 0, 1, 0], "i": index}
        for index, para in enumerate(paragraphs)
    ]
    return {
        "page_count": 1,
        "toc": [],
        "source": "extension",
        "extractor_version": ARTICLE_EXTRACTOR_VERSION,
        "pages": [{"page": 1, "blocks": blocks, "chunks": build_chunks(1, blocks)}],
    }


@router.post("/articles", status_code=status.HTTP_201_CREATED)
async def capture_article(
    payload: ArticleCapture, user: CurrentUser, session: SessionDep
) -> BookOut:
    text = payload.text.strip()[:MAX_ARTICLE_CHARS]
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Article has no readable text")
    book_id = uuid.uuid4()
    logger.info("Capture article user=%s book=%s chars=%d", user.id, book_id, len(text))
    next_position = (
        await session.scalar(
            select(func.coalesce(func.max(Book.position), 0)).where(Book.user_id == user.id)
        )
    ) + 1
    book = Book(
        id=book_id,
        user_id=user.id,
        title=payload.title.strip()[:300],
        author=(payload.author or "Bookify Extension").strip()[:200] or None,
        description=(payload.url or None),
        file_path="",
        file_size=len(text.encode("utf-8")),
        page_count=1,
        position=next_position,
        is_favorite=True,
        extraction_status="ready",
    )
    session.add(book)
    await session.commit()
    await session.refresh(book, ["progress"])
    get_settings().content_dir.mkdir(parents=True, exist_ok=True)
    document = _content_from_text(text)
    await content_store.save_content(book_id, document)
    logger.info(
        "Captured article saved book=%s chunks=%d", book_id, len(document["pages"][0]["chunks"])
    )
    return BookOut.from_model(book)
