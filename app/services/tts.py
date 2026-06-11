import asyncio
import base64
import hashlib
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
from app.services.emotion import Prosody

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")
FIRST_PIECE_MAX_CHARS = 60
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

ORPHEUS_VOICES = [
    {"id": "tara", "name": "Tara", "gender": "female", "accent": "american"},
    {"id": "leah", "name": "Leah", "gender": "female", "accent": "american"},
    {"id": "jess", "name": "Jess", "gender": "female", "accent": "american"},
    {"id": "leo", "name": "Leo", "gender": "male", "accent": "american"},
    {"id": "dan", "name": "Dan", "gender": "male", "accent": "american"},
    {"id": "mia", "name": "Mia", "gender": "female", "accent": "american"},
    {"id": "zac", "name": "Zac", "gender": "male", "accent": "american"},
    {"id": "zoe", "name": "Zoe", "gender": "female", "accent": "american"},
]

# Semantic acting cue -> Orpheus emotive tag. Orpheus performs these audibly
# (actual laughter, gasps, breath); cues it cannot voice fall back to the
# prosody/pause channel only. Sighs/exhales are deliberately absent: the only
# breath-like sound a narrator makes between lines is an inhale.
ORPHEUS_TAGS = {
    "laugh": "<laugh>",
    "chuckle": "<chuckle>",
    "gasp": "<gasp>",
    "groan": "<groan>",
    "yawn": "<yawn>",
    "sniffle": "<sniffle>",
    "cry": "<sniffle>",
    "cough": "<cough>",
}


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


def _effective_speed(speed: float, prosody: Prosody | None) -> float:
    if prosody is not None:
        speed *= prosody.rate
    return max(0.5, min(2.0, speed))


class SpeechEngine:
    voices: list[dict[str, str]] = KOKORO_VOICES
    has_word_timings = False
    streams_pcm = False

    LEAD_TAG_SECONDS = 0.8
    TRAIL_TAG_SECONDS = 0.7

    def performs(self, tag: str) -> bool:
        """Whether the engine audibly acts out the given semantic cue."""
        return False

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.voice_ids = {voice["id"] for voice in self.voices}

    def _tag_reserves(self, duration: float, prosody: Prosody | None) -> tuple[float, float]:
        lead = self.LEAD_TAG_SECONDS if self.performs(prosody.lead_tag if prosody else "") else 0.0
        trail = self.TRAIL_TAG_SECONDS if self.performs(prosody.trail_tag if prosody else "") else 0.0
        if lead + trail > duration * 0.5 and lead + trail > 0:
            scale = duration * 0.5 / (lead + trail)
            lead *= scale
            trail *= scale
        return lead, trail

    def _estimate_words(
        self, text: str, duration: float, prosody: Prosody | None
    ) -> list[dict[str, Any]]:
        # Emotive cues (a chuckle, a gasp) are performed as audio before/after
        # the words; reserve their time so the karaoke highlight stays in sync.
        lead, trail = self._tag_reserves(duration, prosody)
        spoken = duration - lead - trail
        tokens = [token for token in text.split() if token]
        weights = [len(token) + 1 for token in tokens]
        total = sum(weights) or 1
        words: list[dict[str, Any]] = []
        cursor = lead
        for token, weight in zip(tokens, weights):
            span = spoken * weight / total
            words.append({"word": token, "start": round(cursor, 3), "end": round(cursor + span, 3)})
            cursor += span
        return words

    def _align_words(
        self, text: str, samples: np.ndarray, rate: int, prosody: Prosody | None
    ) -> list[dict[str, Any]]:
        """Word timings aligned to the actual speech energy in the audio.

        An RMS envelope marks voiced regions; words are distributed over voiced
        time only, so the highlight waits through real pauses and through the
        performed cues (chuckles, gasps) instead of drifting ahead.
        """
        duration = len(samples) / rate
        hop = int(rate * 0.02)
        if hop == 0 or len(samples) < hop * 8:
            return self._estimate_words(text, duration, prosody)
        frame_count = len(samples) // hop
        frames = samples[: frame_count * hop].reshape(frame_count, hop)
        energy = np.sqrt(np.mean(frames * frames, axis=1))
        threshold = max(float(np.percentile(energy, 15)) * 3.0, float(energy.max()) * 0.07)
        voiced = energy > threshold
        # Close sub-100ms dips so single words are not split apart.
        gap = 0
        for index in range(len(voiced)):
            if voiced[index]:
                if 0 < gap <= 5:
                    voiced[index - gap : index] = True
                gap = 0
            else:
                gap += 1
        spans: list[list[float]] = []
        start: int | None = None
        for index, flag in enumerate(voiced):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                spans.append([start * 0.02, index * 0.02])
                start = None
        if start is not None:
            spans.append([start * 0.02, frame_count * 0.02])
        lead, trail = self._tag_reserves(duration, prosody)
        if lead > 0:
            spans = [span for span in spans if span[1] > lead * 0.75]
            if spans and spans[0][0] < lead * 0.5:
                spans[0][0] = min(spans[0][1], lead * 0.75)
        if trail > 0:
            cutoff = duration - trail * 0.75
            spans = [span for span in spans if span[0] < cutoff]
        if not spans:
            return self._estimate_words(text, duration, prosody)
        voiced_total = sum(end - start for start, end in spans)
        tokens = [token for token in text.split() if token]
        weights = [len(token) + 1 for token in tokens]
        total = sum(weights) or 1

        def absolute(position: float) -> float:
            remaining = position
            for span_start, span_end in spans:
                length = span_end - span_start
                if remaining <= length:
                    return span_start + remaining
                remaining -= length
            return spans[-1][1]

        words: list[dict[str, Any]] = []
        cumulative = 0.0
        for token, weight in zip(tokens, weights):
            begin = absolute(cumulative / total * voiced_total)
            cumulative += weight
            end = absolute(cumulative / total * voiced_total)
            words.append({"word": token, "start": round(begin, 3), "end": round(end, 3)})
        return words

    def _synthesize_samples(
        self, text: str, voice: str, speed: float, prosody: Prosody | None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        raise NotImplementedError

    async def synthesize_sentence(
        self, text: str, voice: str, speed: float, prosody: Prosody | None = None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_samples, text, voice, speed, prosody)

    async def stream_sentence(
        self,
        text: str,
        voice: str,
        speed: float,
        prosody: Prosody | None,
        on_pcm,
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        """Synthesize while pushing PCM to ``on_pcm`` as audio becomes available.

        The base implementation synthesizes fully and emits once; streaming
        engines override this to emit audio with sub-second latency.
        """
        samples, rate, words = await self.synthesize_sentence(text, voice, speed, prosody)
        await on_pcm(samples)
        return samples, rate, words


class TorchSpeechEngine(SpeechEngine):
    has_word_timings = True

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
        self, text: str, voice: str, speed: float, prosody: Prosody | None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        pipeline = self._pipeline("b" if voice.startswith("b") else "a")
        segments: list[np.ndarray] = []
        words: list[dict[str, Any]] = []
        offset = 0.0
        for result in pipeline(text, voice=voice, speed=_effective_speed(speed, prosody)):
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
        self, text: str, voice: str, speed: float, prosody: Prosody | None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        kokoro = self._load()
        lang = "en-gb" if voice.startswith("b") else "en-us"
        samples, sample_rate = kokoro.create(
            text, voice=voice, speed=_effective_speed(speed, prosody), lang=lang
        )
        return np.asarray(samples, dtype=np.float32), sample_rate, []


class EdgeSpeechEngine(SpeechEngine):
    voices = EDGE_VOICES
    has_word_timings = True

    def _synthesize_samples(
        self, text: str, voice: str, speed: float, prosody: Prosody | None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        return asyncio.run(self._collect(text, voice, speed, prosody))

    async def _collect(
        self, text: str, voice: str, speed: float, prosody: Prosody | None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        import edge_tts

        rate_pct = round((_effective_speed(speed, prosody) - 1) * 100)
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=f"{max(-50, min(100, rate_pct)):+d}%",
            pitch=f"{prosody.pitch_hz if prosody else 0:+d}Hz",
            volume=f"{prosody.volume_pct if prosody else 0:+d}%",
            boundary="WordBoundary",
        )
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


class OrpheusSpeechEngine(SpeechEngine):
    """Orpheus 3B (Apache-2.0) served by llama.cpp, decoded with SNAC ONNX.

    Performs truly emotive narration: laughter, sighs, gasps and natural breath are
    generated as audio, directed by the semantic acting cues from the emotion
    planner. Falls back to edge-tts per piece if the inference host is down so
    narration never stalls.
    """

    voices = ORPHEUS_VOICES
    has_word_timings = False
    streams_pcm = True

    SNAC_FILE = "snac_24khz.decoder.onnx"
    TOKEN_PATTERN = re.compile(r"<custom_token_(\d+)>")
    TOKEN_OFFSET = 10
    CODEBOOK = 4096
    FRAME = 7
    WINDOW_FRAMES = 4  # sliding SNAC decode window
    SAMPLES_PER_FRAME = 2048

    def __init__(self) -> None:
        super().__init__()
        self._snac: Any = None
        self._snac_inputs: list[str] = []
        self._fallback = EdgeSpeechEngine()

    def _load_snac(self):
        if self._snac is None:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            self._snac = onnxruntime.InferenceSession(
                str(get_settings().models_dir / self.SNAC_FILE),
                options,
                providers=["CPUExecutionProvider"],
            )
            self._snac_inputs = [item.name for item in self._snac.get_inputs()]
        return self._snac

    def _fallback_voice(self, voice: str) -> str:
        gender = next((v["gender"] for v in self.voices if v["id"] == voice), "female")
        return "en-US-AriaNeural" if gender == "female" else "en-US-GuyNeural"

    def _tagged_text(self, text: str, prosody: Prosody | None) -> str:
        if prosody is None:
            return text
        lead = ORPHEUS_TAGS.get(prosody.lead_tag, "")
        trail = ORPHEUS_TAGS.get(prosody.trail_tag, "")
        return f"{lead} {text} {trail}".strip()

    def _completion(self, text: str, voice: str, stability: float) -> list[int]:
        import urllib.request

        body = {
            "prompt": f"<custom_token_3>{voice}: {text}<|eot_id|><custom_token_4>",
            "n_predict": 6144,
            "temperature": 0.6 + (0.5 - stability) * 0.4,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "cache_prompt": False,
        }
        request = urllib.request.Request(
            f"{get_settings().orpheus_url}/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            content = json.loads(response.read())["content"]
        # The model prefixes marker tokens and opens the audio stream with
        # <custom_token_1>; only tokens after that marker are SNAC codes.
        if "<custom_token_1>" in content:
            content = content.rsplit("<custom_token_1>", 1)[1]
        return [int(match) for match in self.TOKEN_PATTERN.findall(content) if int(match) >= 10]

    def _decode_tokens(self, tokens: list[int]) -> np.ndarray:
        codes = [token - self.TOKEN_OFFSET - ((index % self.FRAME) * self.CODEBOOK)
                 for index, token in enumerate(tokens)]
        frames = len(codes) // self.FRAME
        if frames == 0:
            raise ValueError("Orpheus returned no audio frames")
        layer0: list[int] = []
        layer1: list[int] = []
        layer2: list[int] = []
        for index in range(frames):
            chunk = codes[index * self.FRAME : (index + 1) * self.FRAME]
            if any(code < 0 or code >= self.CODEBOOK for code in chunk):
                continue
            layer0.append(chunk[0])
            layer1.extend((chunk[1], chunk[4]))
            layer2.extend((chunk[2], chunk[3], chunk[5], chunk[6]))
        if not layer0:
            raise ValueError("Orpheus returned malformed audio frames")
        snac = self._load_snac()
        feeds = dict(
            zip(
                self._snac_inputs,
                (
                    np.array([layer0], dtype=np.int64),
                    np.array([layer1], dtype=np.int64),
                    np.array([layer2], dtype=np.int64),
                ),
            )
        )
        waveform = snac.run(None, feeds)[0]
        return np.asarray(waveform, dtype=np.float32).reshape(-1)

    def performs(self, tag: str) -> bool:
        return tag in ORPHEUS_TAGS

    def _synthesize_samples(
        self, text: str, voice: str, speed: float, prosody: Prosody | None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        try:
            stability = prosody.stability if prosody else 0.5
            tokens = self._completion(self._tagged_text(text, prosody), voice, stability)
            samples = self._decode_tokens(tokens)
        except Exception as exc:
            logger.warning("Orpheus synthesis failed (%s); falling back to edge-tts", exc)
            return self._fallback._synthesize_samples(
                text, self._fallback_voice(voice), speed, prosody
            )
        if abs(speed - 1.0) > 0.01:
            samples = _atempo(samples, speed)
        words = self._align_words(text, samples, SAMPLE_RATE, prosody)
        return samples, SAMPLE_RATE, words

    def _stream_tokens(self, text: str, voice: str, stability: float):
        """Yield SNAC token ids as llama-server streams the completion."""
        import urllib.request

        body = {
            "prompt": f"<custom_token_3>{voice}: {text}<|eot_id|><custom_token_4>",
            "n_predict": 6144,
            "temperature": 0.6 + (0.5 - stability) * 0.4,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "cache_prompt": False,
            "stream": True,
        }
        request = urllib.request.Request(
            f"{get_settings().orpheus_url}/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        buffer = ""
        audio_started = False
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                buffer += payload.get("content", "")
                if not audio_started:
                    marker = buffer.rfind("<custom_token_1>")
                    if marker < 0:
                        continue
                    buffer = buffer[marker + len("<custom_token_1>") :]
                    audio_started = True
                while True:
                    match = self.TOKEN_PATTERN.search(buffer)
                    if match is None or match.end() == len(buffer):
                        break
                    buffer = buffer[match.end() :]
                    value = int(match.group(1))
                    if value >= self.TOKEN_OFFSET:
                        yield value
                if payload.get("stop"):
                    match = self.TOKEN_PATTERN.search(buffer)
                    if match is not None and int(match.group(1)) >= self.TOKEN_OFFSET:
                        yield int(match.group(1))
                    return

    def _decode_window(self, codes: list[int], start_frame: int, end_frame: int) -> np.ndarray:
        layer0: list[int] = []
        layer1: list[int] = []
        layer2: list[int] = []
        for index in range(start_frame, end_frame):
            chunk = codes[index * self.FRAME : (index + 1) * self.FRAME]
            layer0.append(chunk[0])
            layer1.extend((chunk[1], chunk[4]))
            layer2.extend((chunk[2], chunk[3], chunk[5], chunk[6]))
        snac = self._load_snac()
        feeds = dict(
            zip(
                self._snac_inputs,
                (
                    np.array([layer0], dtype=np.int64),
                    np.array([layer1], dtype=np.int64),
                    np.array([layer2], dtype=np.int64),
                ),
            )
        )
        waveform = snac.run(None, feeds)[0]
        return np.asarray(waveform, dtype=np.float32).reshape(-1)

    def _stream_blocks(self, text: str, voice: str, stability: float):
        """Yield PCM blocks with a sliding 4-frame SNAC window for clean joins."""
        spf = self.SAMPLES_PER_FRAME
        window = self.WINDOW_FRAMES
        codes: list[int] = []
        valid_frames = 0
        pending: list[int] = []
        for token in self._stream_tokens(text, voice, stability):
            position = len(codes) + len(pending)
            pending.append(token - self.TOKEN_OFFSET - (position % self.FRAME) * self.CODEBOOK)
            if len(pending) < self.FRAME:
                continue
            if all(0 <= code < self.CODEBOOK for code in pending):
                codes.extend(pending)
                valid_frames += 1
            pending = []
            if valid_frames == window:
                yield self._decode_window(codes, 0, window)[: spf * (window - 1)]
            elif valid_frames > window:
                yield self._decode_window(codes, valid_frames - window, valid_frames)[
                    spf * (window - 2) : spf * (window - 1)
                ]
        if valid_frames == 0:
            raise ValueError("Orpheus returned no audio frames")
        if valid_frames < window:
            yield self._decode_window(codes, 0, valid_frames)
        else:
            yield self._decode_window(codes, valid_frames - window, valid_frames)[spf * (window - 1) :]

    async def stream_sentence(
        self,
        text: str,
        voice: str,
        speed: float,
        prosody: Prosody | None,
        on_pcm,
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        if abs(speed - 1.0) > 0.01:
            return await super().stream_sentence(text, voice, speed, prosody, on_pcm)
        async with self._lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            stability = prosody.stability if prosody else 0.5
            tagged = self._tagged_text(text, prosody)

            def produce() -> None:
                try:
                    for block in self._stream_blocks(tagged, voice, stability):
                        loop.call_soon_threadsafe(queue.put_nowait, ("pcm", block))
                    loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

            producer = loop.run_in_executor(None, produce)
            parts: list[np.ndarray] = []
            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "pcm":
                        parts.append(payload)
                        await on_pcm(payload)
                    elif kind == "end":
                        break
                    else:
                        raise payload
            except Exception as exc:
                if parts:
                    raise
                logger.warning("Orpheus stream failed (%s); falling back to edge-tts", exc)
                samples, rate, words = await asyncio.to_thread(
                    self._fallback._synthesize_samples,
                    text,
                    self._fallback_voice(voice),
                    speed,
                    prosody,
                )
                await on_pcm(samples)
                return samples, rate, words
            finally:
                await producer
            samples = np.concatenate(parts)
            words = self._align_words(text, samples, SAMPLE_RATE, prosody)
            return samples, SAMPLE_RATE, words


def _atempo(samples: np.ndarray, speed: float) -> np.ndarray:
    import subprocess

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-filter:a",
            f"atempo={max(0.5, min(2.0, speed))}",
            "-f",
            "s16le",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32767.0


def write_chunk_cache(
    path: Path,
    samples: np.ndarray,
    rate: int,
    words: list[dict[str, Any]],
    sentences: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_bytes(_wav_bytes(samples, rate))
    temp.rename(path)
    payload: dict[str, Any] = {"words": words}
    if sentences:
        payload["sentences"] = sentences
    timing_path(path).write_text(json.dumps(payload, ensure_ascii=False))


GEMINI_VOICES = [
    {"id": "Kore", "name": "Kore", "gender": "female", "accent": "american"},
    {"id": "Aoede", "name": "Aoede", "gender": "female", "accent": "american"},
    {"id": "Leda", "name": "Leda", "gender": "female", "accent": "american"},
    {"id": "Puck", "name": "Puck", "gender": "male", "accent": "american"},
    {"id": "Charon", "name": "Charon", "gender": "male", "accent": "american"},
    {"id": "Fenrir", "name": "Fenrir", "gender": "male", "accent": "american"},
]

GEMINI_STYLE = {
    "narrator": "Narrate like a seasoned audiobook narrator, even and engaging.",
    "storyteller": "Narrate like a warm storyteller by a fireside.",
    "dramatic": "Narrate with weighty, theatrical drama.",
    "cinematic": "Narrate like an epic movie-trailer voice, low and grand.",
    "excited": "Narrate with fast, breathless excitement.",
    "cheerful": "Narrate brightly, upbeat and smiling.",
    "calm": "Narrate slowly and soothingly, soft as a lullaby.",
    "whisper": "Whisper the whole line, hushed and intimate.",
    "mysterious": "Narrate low and mysterious, full of intrigue.",
    "suspense": "Narrate tense and uneasy, like something is about to happen.",
    "melancholy": "Narrate softly, heavy with sorrow.",
    "announcer": "Narrate like a punchy stage announcer, projected and confident.",
}

GEMINI_CUES = {
    "laugh": "Laugh briefly before the line.",
    "chuckle": "Give a soft chuckle before the line.",
    "gasp": "Gasp before the line.",
    "cry": "Let a quiet sob break into the voice.",
    "nervous": "Sound nervous and on edge.",
    "sad": "Let real sadness show in the voice.",
    "curious": "Sound genuinely curious.",
    "soft": "Keep the voice gentle and tender.",
    "happy": "Sound delighted.",
    "excited": "Sound thrilled.",
    "angry": "Sound angry.",
    "shout": "Raise the voice, almost shouting.",
    "dramatic": "Lean into the drama.",
    "whisper": "Whisper it.",
    "cheerful": "Sound cheerful.",
}

GEMINI_PERFORMED_CUES = {"laugh", "chuckle", "gasp", "cry"}


class GeminiSpeechEngine(SpeechEngine):
    """Gemini native TTS: cloud-side emotional acting on the free tier.

    The model takes natural-language direction (built from the emotion preset
    and the per-sentence acting cues) and returns 24 kHz PCM. Costs no droplet
    memory and no money; on quota errors a circuit breaker routes pieces to
    edge-tts so narration never stalls.
    """

    voices = GEMINI_VOICES
    has_word_timings = False

    COOLDOWN_SECONDS = 240.0

    def __init__(self) -> None:
        super().__init__()
        self._fallback = EdgeSpeechEngine()
        self._cooldown_until = 0.0

    def performs(self, tag: str) -> bool:
        return tag in GEMINI_PERFORMED_CUES

    def _fallback_voice(self, voice: str) -> str:
        gender = next((v["gender"] for v in self.voices if v["id"] == voice), "female")
        return "en-US-AriaNeural" if gender == "female" else "en-US-GuyNeural"

    def _direction(self, text: str, emotion_style: str, prosody: Prosody | None) -> str:
        parts = [emotion_style]
        if prosody is not None:
            for tag in (prosody.lead_tag, prosody.trail_tag):
                cue = GEMINI_CUES.get(tag)
                if cue and cue not in parts:
                    parts.append(cue)
        # Narrators breathe in quietly between lines; an audible exhale reads
        # as unintended emotion, so forbid it unless a cue asked for one.
        parts.append("Breathe naturally with quick quiet inhales; never sigh or exhale audibly.")
        direction = " ".join(parts)
        return f"{direction} Read only this text: {text}"

    def _request(self, prompt: str, voice: str) -> np.ndarray:
        import time
        import urllib.error
        import urllib.request

        settings = get_settings()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_tts_model}:generateContent?key={settings.gemini_api_key}"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        }
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        part = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    self._cooldown_until = time.monotonic() + self.COOLDOWN_SECONDS
                raise
            try:
                part = data["candidates"][0]["content"]["parts"][0]["inlineData"]
                break
            except (KeyError, IndexError):
                # The preview model occasionally returns a candidate without
                # audio parts; one retry almost always recovers it.
                if attempt == 1:
                    raise ValueError("Gemini TTS returned no audio")
        mime = part["mimeType"]
        rate = int(mime.split("rate=")[1].split(";")[0]) if "rate=" in mime else SAMPLE_RATE
        samples = (
            np.frombuffer(base64.b64decode(part["data"]), dtype="<i2").astype(np.float32)
            / 32767.0
        )
        if rate != SAMPLE_RATE:
            samples = _resample(samples, rate, SAMPLE_RATE)
        return samples

    def _synthesize_samples(
        self, text: str, voice: str, speed: float, prosody: Prosody | None
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        import time

        emotion_style = GEMINI_STYLE.get(
            prosody.emotion if prosody else "", GEMINI_STYLE["narrator"]
        )
        if time.monotonic() < self._cooldown_until:
            return self._fallback._synthesize_samples(
                text, self._fallback_voice(voice), speed, prosody
            )
        try:
            samples = self._request(self._direction(text, emotion_style, prosody), voice)
        except Exception as exc:
            logger.warning("Gemini TTS failed (%s); falling back to edge-tts", exc)
            return self._fallback._synthesize_samples(
                text, self._fallback_voice(voice), speed, prosody
            )
        if abs(speed - 1.0) > 0.01:
            samples = _atempo(samples, speed)
        words = self._align_words(text, samples, SAMPLE_RATE, prosody)
        return samples, SAMPLE_RATE, words


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    positions = np.arange(0, len(samples), source_rate / target_rate)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def _create_engine() -> SpeechEngine:
    selected = get_settings().tts_engine
    if selected == "kokoro-onnx":
        return OnnxSpeechEngine()
    if selected == "edge":
        return EdgeSpeechEngine()
    if selected == "orpheus":
        return OrpheusSpeechEngine()
    if selected == "gemini":
        return GeminiSpeechEngine()
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

def audio_path(
    book_id: uuid.UUID, voice: str, emotion: str, chunk_id: str, speech: str
) -> Path:
    digest = hashlib.sha1(speech.encode("utf-8")).hexdigest()[:8]
    return get_settings().audio_dir / str(book_id) / voice / emotion / f"{chunk_id}-{digest}.wav"


def timing_path(audio_file: Path) -> Path:
    return audio_file.with_suffix(".json")


async def warmup() -> None:
    try:
        await engine.synthesize_sentence("Bookly is ready.", resolve_voice(None), 1.0)
        logger.info("TTS engine warmed up")
    except Exception:
        logger.exception("TTS warmup failed")
