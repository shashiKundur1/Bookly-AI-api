"""Emotion-driven prosody planning for narration.

Each emotion preset defines a baseline delivery (pace, pitch, loudness, pausing)
plus a `dynamics` factor that scales how strongly the text itself — punctuation,
energy words, dialogue — modulates that baseline sentence by sentence. A small
deterministic jitter derived from the text keeps long narration from sounding
metronomic while staying byte-identical across runs, so synthesized audio stays
cacheable.
"""

import hashlib
import re
from dataclasses import dataclass

from app.services import affect


@dataclass(frozen=True)
class Prosody:
    rate: float  # multiplier combined with the listener's speed setting
    pitch_hz: int  # shift for engines that support pitch (edge-tts)
    volume_pct: int  # loudness delta for engines that support it (edge-tts)
    pre_pause: float  # seconds of silence injected before the piece
    post_pause: float  # seconds of silence injected after the piece
    lead_tag: str = ""  # semantic acting cue opening the piece (engine translates)
    trail_tag: str = ""  # semantic acting cue closing the piece
    stability: float = 0.5  # expressiveness: 0.0 = most theatrical, 1.0 = most even


@dataclass(frozen=True)
class _Preset:
    name: str
    tagline: str
    rate: float
    pitch_hz: int
    volume_pct: int
    pause: float  # multiplier on injected pauses
    dynamics: float  # how strongly text cues swing the baseline
    stability: float  # expressiveness (0.0 = most theatrical)
    leads: tuple[str, ...]  # acting-cue palette for piece openings ("" = none)
    trails: tuple[str, ...]  # acting-cue palette for piece closings
    excl_tag: str  # acting cue when the text shouts
    soft_tag: str  # acting cue when the text trails off


# Breathing model (Wang et al. 2010; Fuchs et al. 2017; Hirai et al. 2022;
# Goldman-Eisler 1968): narrators speak on the exhale and take a short, quiet
# inhale *inside* the boundary pause before the next sentence — never an
# audible exhale after one. Kept breaths in audiobooks sit 20–35 dB below
# speech, so a correctly sized silence IS the catch breath; the generative
# voice already breathes naturally within sentences. Sighs are reserved for
# scripted emotional beats. Pause baselines: sentence 600–900 ms, paragraph
# 1000–1500 ms, scaled by the preset's pace multiplier (slow 1.2–1.5x,
# fast 0.7–0.85x) and never below the 250 ms perception threshold.
PRESETS: dict[str, _Preset] = {
    "narrator": _Preset(
        "Narrator", "Classic audiobook voice", 1.0, 0, 0, 1.0, 0.7,
        0.5, ("", "", "", ""), ("",), "", "",
    ),
    "storyteller": _Preset(
        "Storyteller", "Warm fireside telling", 0.97, 2, 0, 1.15, 1.0,
        0.0, ("", "", "curious", "soft"), ("", "", "", "chuckle"), "gasp", "",
    ),
    "dramatic": _Preset(
        "Dramatic", "Weighty, theatrical delivery", 0.92, -8, 4, 1.3, 1.4,
        0.0, ("dramatic", "", ""), ("",), "gasp", "",
    ),
    "cinematic": _Preset(
        "Cinematic", "Epic movie-trailer gravitas", 0.88, -14, 10, 1.45, 1.2,
        0.0, ("dramatic", "", ""), ("",), "shout", "",
    ),
    "excited": _Preset(
        "Excited", "Fast, breathless energy", 1.12, 18, 6, 0.75, 1.5,
        0.0, ("excited", "excited", ""), ("", "", "laugh"), "excited", "",
    ),
    "cheerful": _Preset(
        "Cheerful", "Bright and upbeat", 1.05, 12, 4, 0.85, 1.1,
        0.0, ("cheerful", "", "happy"), ("", "", "chuckle"), "laugh", "",
    ),
    "calm": _Preset(
        "Calm", "Slow, soothing wind-down", 0.93, -4, -4, 1.35, 0.4,
        0.5, ("soft", "", ""), ("",), "", "",
    ),
    "whisper": _Preset(
        "Whisper", "Hushed and intimate", 0.95, -6, -25, 1.25, 0.3,
        0.0, ("whisper",), ("",), "whisper", "",
    ),
    "mysterious": _Preset(
        "Mysterious", "Low, probing intrigue", 0.90, -10, -8, 1.4, 0.9,
        0.0, ("whisper", "curious", ""), ("",), "gasp", "",
    ),
    "suspense": _Preset(
        "Suspense", "Tense, clipped unease", 0.97, -6, 0, 1.3, 1.3,
        0.0, ("nervous", "", "whisper"), ("",), "gasp", "",
    ),
    "melancholy": _Preset(
        "Melancholy", "Soft, sorrowful fall", 0.90, -10, -6, 1.45, 0.7,
        0.0, ("sad", "", "soft"), ("",), "cry", "",
    ),
    "announcer": _Preset(
        "Announcer", "Punchy stage projection", 1.08, 6, 12, 0.8, 1.2,
        0.0, ("excited", "", ""), ("",), "shout", "",
    ),
}

DEFAULT_EMOTION = "narrator"
EMOTION_IDS = set(PRESETS)

EMOTIONS = [
    {"id": key, "name": preset.name, "tagline": preset.tagline}
    for key, preset in PRESETS.items()
]

# GoEmotions label -> semantic acting cue. Only labels with a performable
# delivery get one; the rest act through prosody alone. Breath-like cues are
# inhale-only (gasp): narrators never audibly exhale between lines, so no
# label maps to a sigh or exhale — low moods act through tone and pacing.
AFFECT_TAGS = {
    "amusement": "chuckle",
    "anger": "angry",
    "caring": "soft",
    "confusion": "curious",
    "curiosity": "curious",
    "embarrassment": "nervous",
    "excitement": "excited",
    "fear": "nervous",
    "grief": "cry",
    "joy": "happy",
    "love": "soft",
    "nervousness": "nervous",
    "realization": "gasp",
    "remorse": "sad",
    "sadness": "sad",
    "surprise": "gasp",
}

# GoEmotions label -> (arousal, valence), each in [-1, 1].
AFFECT_DYNAMICS = {
    "admiration": (0.3, 0.6),
    "amusement": (0.5, 0.7),
    "anger": (0.8, -0.7),
    "annoyance": (0.4, -0.4),
    "approval": (0.1, 0.3),
    "caring": (-0.2, 0.5),
    "confusion": (0.1, -0.1),
    "curiosity": (0.3, 0.2),
    "desire": (0.4, 0.4),
    "disappointment": (-0.3, -0.5),
    "disapproval": (0.2, -0.4),
    "disgust": (0.3, -0.6),
    "embarrassment": (0.2, -0.3),
    "excitement": (0.9, 0.7),
    "fear": (0.7, -0.6),
    "gratitude": (0.1, 0.5),
    "grief": (-0.4, -0.9),
    "joy": (0.6, 0.8),
    "love": (0.1, 0.7),
    "nervousness": (0.5, -0.4),
    "optimism": (0.3, 0.5),
    "pride": (0.4, 0.5),
    "realization": (0.4, 0.1),
    "relief": (-0.3, 0.4),
    "remorse": (-0.3, -0.6),
    "sadness": (-0.5, -0.7),
    "surprise": (0.7, 0.1),
}

AFFECT_TAG_THRESHOLD = 0.35
AFFECT_SCORE_FLOOR = 0.20

_HIGH_ENERGY = re.compile(
    r"(?i)\b(sudden(?:ly)?|burst|scream(?:ed|ing)?|shout(?:ed|ing)?|ran|rush(?:ed|ing)?|"
    r"crash(?:ed|ing)?|explod\w+|attack\w*|fight\w*|danger(?:ous)?|amazing|incredible|"
    r"victory|win|leap(?:ed|t)?|fierce|thunder\w*|storm\w*|blaz\w+|roar\w*|slam\w+|"
    r"chase[ds]?|strike[sd]?|struck|grab\w+|fl(?:ee|ed)|panic\w*)\b"
)
_LOW_ENERGY = re.compile(
    r"(?i)\b(whisper\w*|quiet(?:ly)?|slow(?:ly)?|gentl[ey]|soft(?:ly)?|sleep\w*|dream\w*|"
    r"dark(?:ness)?|silen(?:t|ce)|alone|lonel\w+|sorrow\w*|grie(?:f|ve|ving)|tears?|wept|"
    r"weep(?:ing)?|dying|death|dead|cold|still(?:ness)?|fading|drift(?:ed|ing)?|sigh\w*)\b"
)
_CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")


def _jitter(text: str) -> float:
    """Deterministic pseudo-random in [-1, 1) derived from the text."""
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") / 32768.0 - 1.0


def plan(
    text: str, emotion: str, *, first: bool, last: bool, next_chars: int | None = None
) -> Prosody:
    preset = PRESETS.get(emotion, PRESETS[DEFAULT_EMOTION])
    dyn = preset.dynamics

    rate = preset.rate
    pitch = float(preset.pitch_hz)
    volume = float(preset.volume_pct)

    exclaims = text.count("!")
    if exclaims:
        rate += 0.04 * dyn * min(exclaims, 2)
        pitch += 9 * dyn * min(exclaims, 2)
        volume += 4 * dyn
    if text.rstrip().endswith("?"):
        pitch += 8 * dyn
    if '"' in text or "“" in text:
        pitch += 4 * dyn
    if _CAPS_WORD.search(text):
        volume += 5 * dyn

    # Emotion detection: the GoEmotions classifier reads the sentence and its
    # arousal/valence steer pacing while the strongest label picks the acting
    # tag. Heuristic word lists only apply when the model is unavailable.
    affect_tag = ""
    detected = affect.detect(text)
    if detected is not None:
        arousal = 0.0
        valence = 0.0
        for label, score in detected:
            if score < AFFECT_SCORE_FLOOR or label == "neutral":
                continue
            label_arousal, label_valence = AFFECT_DYNAMICS.get(label, (0.0, 0.0))
            arousal += label_arousal * score
            valence += label_valence * score
            if not affect_tag and score >= AFFECT_TAG_THRESHOLD:
                affect_tag = AFFECT_TAGS.get(label, "")
        arousal = max(-1.0, min(1.0, arousal))
        valence = max(-1.0, min(1.0, valence))
        rate += 0.08 * dyn * arousal
        pitch += 14 * dyn * arousal + 6 * dyn * valence
        volume += 7 * dyn * max(0.0, arousal)
        if arousal < -0.15 and valence < 0:
            post_extra = 0.15 * preset.pause * -arousal
        else:
            post_extra = 0.0
    else:
        energy = len(_HIGH_ENERGY.findall(text)) - len(_LOW_ENERGY.findall(text))
        energy = max(-3, min(3, energy))
        rate += 0.015 * dyn * energy
        pitch += 3 * dyn * energy
        post_extra = 0.0

    # Boundary silence is the narrator's quiet inhale (speech rides the
    # exhale). Sentence boundaries get 600–900 ms and chunk ends a paragraph
    # beat of 1000–1500 ms at neutral pace; real readers also breathe deeper
    # before longer upcoming sentences, so the pause stretches slightly when
    # a long sentence follows. Trailing-off punctuation earns an extra beat.
    pace = max(0.7, min(1.5, preset.pause))
    post = (1.1 if last else 0.75) * pace + post_extra
    if next_chars is not None:
        post *= 0.9 + min(next_chars, 220) / 220 * 0.25
    if re.search(r"(\.\.\.|…|—|--)\s*$", text):
        post += 0.25 * pace
    post = max(0.3, min(post, 2.0 if last else 1.5))
    pre = 0.22 * pace if first else 0.0

    wobble = _jitter(text)
    rate += 0.02 * dyn * wobble
    pitch += 4 * dyn * wobble

    # ElevenLabs v3 acting direction. The classifier's verdict on the sentence
    # outranks the preset's generic palette; shouting and trailing-off text
    # override both. The whisper preset always keeps its identity.
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    lead = preset.leads[digest[2] % len(preset.leads)]
    trail = preset.trails[digest[3] % len(preset.trails)]
    if affect_tag and dyn >= 0.4:
        lead = affect_tag
    if exclaims and preset.excl_tag:
        lead = preset.excl_tag
    if re.search(r"(\.\.\.|…)\s*$", text) and preset.soft_tag:
        trail = preset.soft_tag
    if emotion == "whisper":
        lead = "whisper"

    return Prosody(
        rate=max(0.7, min(1.45, rate)),
        pitch_hz=int(max(-40, min(40, round(pitch)))),
        volume_pct=int(max(-40, min(35, round(volume)))),
        pre_pause=round(pre, 3),
        post_pause=round(post, 3),
        lead_tag=lead,
        trail_tag=trail,
        stability=preset.stability,
    )


def resolve_emotion(requested: str | None) -> str:
    if requested in EMOTION_IDS:
        return requested
    return DEFAULT_EMOTION
