"""Sentence-level emotion detection on the GoEmotions taxonomy.

Runs SamLowe/roberta-base-go_emotions-onnx (int8-quantized RoBERTa, trained on
Google Research's GoEmotions corpus of 58k human-labeled examples across 28
emotion labels) via onnxruntime on CPU. The model files are seeded into
``data/models`` the same way the kokoro voice models are; when they are
missing the classifier reports unavailable and narration falls back to
punctuation-based heuristics.
"""

import logging
from functools import lru_cache

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

MODEL_FILE = "go_emotions.quant.onnx"
TOKENIZER_FILE = "go_emotions.tokenizer.json"
MAX_TOKENS = 128

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


class _Classifier:
    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []
        self._failed = False

    def _load(self) -> bool:
        if self._failed:
            return False
        if self._session is not None:
            return True
        settings = get_settings()
        model_path = settings.models_dir / MODEL_FILE
        tokenizer_path = settings.models_dir / TOKENIZER_FILE
        if not model_path.exists() or not tokenizer_path.exists():
            self._failed = True
            logger.warning("GoEmotions model files missing in %s", settings.models_dir)
            return False
        try:
            import onnxruntime
            from tokenizers import Tokenizer

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            self._session = onnxruntime.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
            self._input_names = [item.name for item in self._session.get_inputs()]
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
            logger.info("GoEmotions classifier loaded")
            return True
        except Exception:
            self._failed = True
            self._session = None
            logger.exception("GoEmotions classifier failed to load")
            return False

    def scores(self, text: str) -> dict[str, float] | None:
        if not self._load():
            return None
        encoding = self._tokenizer.encode(text)
        feeds = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)
        logits = self._session.run(None, feeds)[0][0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return dict(zip(LABELS, (float(p) for p in probabilities)))


_classifier = _Classifier()


def available() -> bool:
    return _classifier._load()


@lru_cache(maxsize=1024)
def detect(text: str) -> tuple[tuple[str, float], ...] | None:
    """Top-3 detected emotions for the text, or None when the model is unavailable."""
    scores = _classifier.scores(text)
    if scores is None:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return tuple(ranked[:3])
