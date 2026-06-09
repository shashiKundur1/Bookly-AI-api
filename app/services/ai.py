import asyncio
import json
import logging
import urllib.request
import uuid

from app.config import get_settings
from app.services import content as content_store

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_SECONDS = 45
MAX_CONSECUTIVE_FAILURES = 2
MAX_SPEECH_CHARS = 600

PROMPT = (
    "You prepare book pages for audiobook narration. Rewrite each chunk's speech text so it "
    "reads naturally when spoken aloud: turn fragmented table cells and broken lines into "
    "flowing sentences, smooth awkward abbreviations, and keep a calm narrator tone. Preserve "
    "every piece of information, never invent facts, never merge or drop chunks, and keep each "
    "rewritten chunk under 600 characters. Return only a JSON array of objects with keys id "
    "and speech, one per input chunk, in the same order.\n\nChunks:\n"
)

_locks: dict[str, asyncio.Lock] = {}
_consecutive_failures = 0


def is_enabled() -> bool:
    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        return False
    return bool(get_settings().gemini_api_key)


def _call_gemini(chunks: list[dict[str, str]]) -> dict[str, str] | None:
    settings = get_settings()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={settings.gemini_api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": PROMPT + json.dumps(chunks, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read())
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    rewritten = json.loads(text)
    if not isinstance(rewritten, list):
        return None
    result: dict[str, str] = {}
    for entry in rewritten:
        chunk_id = entry.get("id")
        speech = entry.get("speech")
        if isinstance(chunk_id, str) and isinstance(speech, str) and speech.strip():
            result[chunk_id] = speech.strip()[:MAX_SPEECH_CHARS]
    return result or None


async def ensure_page_polished(book_id: uuid.UUID, page_number: int) -> None:
    global _consecutive_failures
    if not is_enabled():
        return
    key = f"{book_id}:{page_number}"
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        data = await content_store.load_content(book_id)
        if data is None or not (1 <= page_number <= len(data["pages"])):
            return
        page = data["pages"][page_number - 1]
        if page.get("polished") or not page["chunks"]:
            return
        payload = [{"id": chunk["id"], "speech": chunk["speech"]} for chunk in page["chunks"]]
        try:
            rewritten = await asyncio.to_thread(_call_gemini, payload)
        except Exception as exc:
            _consecutive_failures += 1
            logger.warning("Gemini polish failed for book %s page %s: %s", book_id, page_number, exc)
            return
        if rewritten is None:
            _consecutive_failures += 1
            return
        _consecutive_failures = 0
        for chunk in page["chunks"]:
            if chunk["id"] in rewritten:
                chunk["speech"] = rewritten[chunk["id"]]
        page["polished"] = True
        await content_store.save_content(book_id, data)
    _locks.pop(key, None)
