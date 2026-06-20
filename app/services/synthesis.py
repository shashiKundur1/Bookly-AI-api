import logging
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from app.services.emotion import plan
from app.services.tts import (
    SAMPLE_RATE,
    engine,
    pcm16_bytes,
    split_for_streaming,
)

logger = logging.getLogger(__name__)

ESTIMATED_CHARS_PER_SECOND = 13.5

SendJson = Callable[[dict[str, Any]], Awaitable[None]]
SendPcm = Callable[[bytes], Awaitable[None]]


def pad_silence(samples: np.ndarray, rate: int, pre: float, post: float) -> np.ndarray:
    if pre <= 0 and post <= 0:
        return samples
    return np.concatenate(
        [
            np.zeros(int(pre * rate), dtype=np.float32),
            samples,
            np.zeros(int(post * rate), dtype=np.float32),
        ]
    )


def shift_words(words: list[dict[str, Any]], base: float) -> list[dict[str, Any]]:
    return [
        {
            **word,
            "start": round(word["start"] + base, 3),
            "end": round(word["end"] + base, 3),
        }
        for word in words
    ]


async def stream_pieces(
    *,
    text: str,
    voice: str,
    speed: float,
    emotion: str,
    chunk_id: str,
    send_json: SendJson,
    send_pcm: SendPcm,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> float:
    """Synthesize ``text`` piece by piece, streaming PCM and karaoke timings.

    This is the engine-agnostic streaming core shared by book narration and the
    extension's ad-hoc selection narration: it runs the same sentence split,
    emotion planning and per-piece synthesis, emitting ``sentence`` frames with
    word timings followed by raw PCM. Returns the total audio duration in
    seconds, or the duration produced before cancellation.
    """
    pieces = split_for_streaming(text)
    logger.info(
        "stream_pieces start chunk=%s pieces=%d voice=%s emotion=%s speed=%.2f chars=%d",
        chunk_id, len(pieces), voice, emotion, speed, len(text),
    )
    offset = 0.0
    rate = SAMPLE_RATE
    for index, piece in enumerate(pieces):
        if is_cancelled():
            logger.info("stream_pieces cancelled chunk=%s at piece %d/%d", chunk_id, index, len(pieces))
            break
        prosody = plan(
            piece,
            emotion,
            first=index == 0,
            last=index == len(pieces) - 1,
            next_chars=len(pieces[index + 1]) if index + 1 < len(pieces) else None,
        )
        pre = prosody.pre_pause / speed
        post = prosody.post_pause / speed
        cues = {
            "lead": prosody.lead_tag if engine.performs(prosody.lead_tag) else "",
            "trail": prosody.trail_tag if engine.performs(prosody.trail_tag) else "",
        }
        logger.debug("stream_pieces synth chunk=%s piece=%d offset=%.3f", chunk_id, index, offset)

        if engine.streams_pcm and speed == 1.0:
            estimated = engine._estimate_words(
                piece, len(piece) / ESTIMATED_CHARS_PER_SECOND, prosody
            )
            frame = {
                "text": piece,
                "offset": round(offset, 3),
                "words": shift_words(estimated, offset + pre),
                "cues": cues,
            }
            await send_json({"type": "sentence", "chunk_id": chunk_id, **frame})
            if pre > 0:
                await send_pcm(pcm16_bytes(np.zeros(int(pre * rate), np.float32)))

            async def forward(block: np.ndarray) -> None:
                await send_pcm(pcm16_bytes(block))

            samples, rate, words = await engine.stream_sentence(
                piece, voice, speed, prosody, forward
            )
            if post > 0:
                await send_pcm(pcm16_bytes(np.zeros(int(post * rate), np.float32)))
            samples = pad_silence(samples, rate, pre, post)
            await send_json(
                {
                    "type": "sentence_update",
                    "chunk_id": chunk_id,
                    "offset": frame["offset"],
                    "words": shift_words(words, offset + pre),
                }
            )
        else:
            samples, rate, words = await engine.synthesize_sentence(piece, voice, speed, prosody)
            samples = pad_silence(samples, rate, pre, post)
            frame = {
                "text": piece,
                "offset": round(offset, 3),
                "words": shift_words(words, offset + pre),
                "cues": cues,
            }
            await send_json({"type": "sentence", "chunk_id": chunk_id, **frame})
            await send_pcm(pcm16_bytes(samples))
        offset += len(samples) / rate

    logger.info("stream_pieces done chunk=%s duration=%.3f", chunk_id, offset)
    return offset
