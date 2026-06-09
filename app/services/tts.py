import asyncio
import io
import json
import logging
import re
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")
FIRST_PIECE_MAX_CHARS = 90
PIECE_MAX_CHARS = 220
ONNX_MODEL_FILE = "kokoro-v1.0.int8.onnx"
ONNX_VOICES_FILE = "voices-v1.0.bin"

KOKORO_VOICES = [
    {"id": "af_heart", "name": "Heart", "gender": "female", "accent": "american"},
    {"id": "af_bella", "name": "Bella", "gender": "female", "accent": "american"},
    {"id": "bf_emma", "name": "Emma", "gender": "female", "accent": "british"},
    {"id": "am_michael", "name": "Michael", "gender": "male", "accent": "american"},
    {"id": "am_fenrir", "name": "Fenrir", "gender": "male", "accent": "american"},
    {"id": "bm_george", "name": "George", "gender": "male", "accent": "british"},
]

EDGE_VOICES = [
    {"id": "en-US-AriaNeural", "name": "Aria", "gender": "female", "accent": "american"},
    {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "female", "accent": "american"},
    {"id": "en-US-GuyNeural", "name": "Guy", "gender": "male", "accent": "american"},
    {"id": "en-US-ChristopherNeural", "name": "Christopher", "gender": "male", "accent": "american"},
    {"id": "en-GB-SoniaNeural", "name": "Sonia", "gender": "female", "accent": "british"},
    {"id": "en-GB-RyanNeural", "name": "Ryan", "gender": "male", "accent": "british"},
]


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _words_from_tokens(tokens: list[Any], offset: float) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    current = ""
    start: float | None = None
    end = offset

    def push() -> None:
        nonlocal current, start
        if current.strip():
            begin = start if start is not None else end
            words.append({"word": current.strip(), "start": round(begin, 3), "end": round(end, 3)})
        current = ""
        start = None

    for token in tokens:
        if token.start_ts is not None and start is None:
            start = offset + token.start_ts
        if token.end_ts is not None:
            end = offset + token.end_ts
        current += token.text
        if token.whitespace:
            push()
    push()
    return words


def split_for_streaming(text: str) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_BOUNDARY.split(text.strip()):
        if not sentence:
            continue
        limit = FIRST_PIECE_MAX_CHARS if not pieces and not current else PIECE_MAX_CHARS
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
        while len(current) > PIECE_MAX_CHARS * 2:
            cut = current.rfind(" ", 0, PIECE_MAX_CHARS)
            if cut <= 0:
                break
            pieces.append(current[:cut])
            current = current[cut + 1 :]
    if current:
        pieces.append(current)
    return pieces


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class SpeechEngine:
    voices: list[dict[str, str]] = KOKORO_VOICES

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.voice_ids = {voice["id"] for voice in self.voices}

    def _synthesize_samples(
        self, text: str, voice: str, speed: float
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        raise NotImplementedError

    def _synthesize(self, text: str, voice: str) -> tuple[bytes, list[dict[str, Any]]]:
        samples, sample_rate, words = self._synthesize_samples(text, voice, 1.0)
        return _wav_bytes(samples, sample_rate), words

    async def synthesize_sentence(
        self, text: str, voice: str, speed: float
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_samples, text, voice, speed)

    async def synthesize_to_file(self, text: str, voice: str, path: Path) -> Path:
        if path.exists():
            return path
        async with self._lock:
            if path.exists():
                return path
            audio, words = await asyncio.to_thread(self._synthesize, text, voice)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f"{path.name}.tmp")
            temp.write_bytes(audio)
            temp.rename(path)
            timing_path(path).write_text(json.dumps({"words": words}, ensure_ascii=False))
            return path


class TorchSpeechEngine(SpeechEngine):
    def __init__(self) -> None:
        super().__init__()
        self._pipelines: dict[str, Any] = {}

    def _pipeline(self, lang_code: str):
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            from kokoro import KPipeline

            existing = next(iter(self._pipelines.values()), None)
            model = existing.model if existing is not None else True
            pipeline = KPipeline(lang_code=lang_code, model=model)
            self._pipelines[lang_code] = pipeline
        return pipeline

    def _synthesize_samples(
        self, text: str, voice: str, speed: float
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        pipeline = self._pipeline("b" if voice.startswith("b") else "a")
        segments: list[np.ndarray] = []
        words: list[dict[str, Any]] = []
        offset = 0.0
        for result in pipeline(text, voice=voice, speed=speed):
            audio = result.audio.numpy()
            words.extend(_words_from_tokens(result.tokens or [], offset))
            offset += len(audio) / SAMPLE_RATE
            segments.append(audio)
        combined = np.concatenate(segments) if segments else np.zeros(1, dtype=np.float32)
        return combined, SAMPLE_RATE, words


class OnnxSpeechEngine(SpeechEngine):
    def __init__(self) -> None:
        super().__init__()
        self._kokoro: Any = None

    def _load(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            settings = get_settings()
            self._kokoro = Kokoro(
                str(settings.models_dir / ONNX_MODEL_FILE),
                str(settings.models_dir / ONNX_VOICES_FILE),
            )
        return self._kokoro

    def _synthesize_samples(
        self, text: str, voice: str, speed: float
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        kokoro = self._load()
        lang = "en-gb" if voice.startswith("b") else "en-us"
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        return np.asarray(samples, dtype=np.float32), sample_rate, []


class EdgeSpeechEngine(SpeechEngine):
    voices = EDGE_VOICES

    def _synthesize_samples(
        self, text: str, voice: str, speed: float
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        return asyncio.run(self._collect(text, voice, speed))

    async def _collect(
        self, text: str, voice: str, speed: float
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        import edge_tts

        rate = f"{round((speed - 1) * 100):+d}%"
        communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        mp3 = bytearray()
        words: list[dict[str, Any]] = []
        async for event in communicate.stream():
            if event["type"] == "audio":
                mp3.extend(event["data"])
            elif event["type"] == "WordBoundary":
                start = event["offset"] / 10_000_000
                words.append(
                    {
                        "word": str(event["text"]),
                        "start": round(start, 3),
                        "end": round(start + event["duration"] / 10_000_000, 3),
                    }
                )
        samples = _decode_mp3(bytes(mp3))
        return samples, SAMPLE_RATE, words


def _decode_mp3(data: bytes) -> np.ndarray:
    import subprocess

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32767.0


def write_chunk_cache(path: Path, samples: np.ndarray, rate: int, words: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_bytes(_wav_bytes(samples, rate))
    temp.rename(path)
    timing_path(path).write_text(json.dumps({"words": words}, ensure_ascii=False))


def _create_engine() -> SpeechEngine:
    selected = get_settings().tts_engine
    if selected == "kokoro-onnx":
        return OnnxSpeechEngine()
    if selected == "edge":
        return EdgeSpeechEngine()
    return TorchSpeechEngine()


engine = _create_engine()
VOICES = engine.voices
VOICE_IDS = engine.voice_ids


def resolve_voice(requested: str | None) -> str:
    if requested and requested in VOICE_IDS:
        return requested
    preferred = get_settings().default_voice
    if preferred in VOICE_IDS:
        return preferred
    return VOICES[0]["id"]

_prefetch_tasks: set[asyncio.Task] = set()


def audio_path(book_id: uuid.UUID, voice: str, chunk_id: str) -> Path:
    return get_settings().audio_dir / str(book_id) / voice / f"{chunk_id}.wav"


def timing_path(audio_file: Path) -> Path:
    return audio_file.with_suffix(".json")


async def ensure_audio(book_id: uuid.UUID, voice: str, chunk: dict) -> Path:
    return await engine.synthesize_to_file(
        chunk["speech"], voice, audio_path(book_id, voice, chunk["id"])
    )


def _prefetch_done(task: asyncio.Task) -> None:
    _prefetch_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Audio prefetch failed: %s", task.exception())


def prefetch(book_id: uuid.UUID, voice: str, chunks: list[dict]) -> None:
    for chunk in chunks:
        if audio_path(book_id, voice, chunk["id"]).exists():
            continue
        task = asyncio.create_task(ensure_audio(book_id, voice, chunk))
        _prefetch_tasks.add(task)
        task.add_done_callback(_prefetch_done)


async def warmup() -> None:
    try:
        await asyncio.to_thread(engine._synthesize, "Bookly is ready.", get_settings().default_voice)
        logger.info("TTS engine warmed up")
    except Exception:
        logger.exception("TTS warmup failed")
