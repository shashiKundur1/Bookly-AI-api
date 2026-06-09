import re

MAX_CHUNK_CHARS = 420

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
DOT_LEADER_PAGE = re.compile(r"\s*\.{4,}\s*([0-9]+|[ivxlcdmIVXLCDM]+)\b")
DOT_LEADER = re.compile(r"\s*\.{4,}\s*")
URL = re.compile(r"https?://(?:www\.)?([^\s/]+)\S*")
WHITESPACE = re.compile(r"\s+")
LIST_MARKER = re.compile(r"^\s*(?:(\d{1,3})[.)]|([a-zA-Z])[.)]|[•▪◦‣·∙○●♦►▶*–—-])\s+")
NUMBERED_TITLE = re.compile(r"^(\d{1,3})\s+(.{3,})$")
TITLE_KEYWORD = re.compile(r"(?i)^(chapter|part|section|appendix|unit|lesson)\b")


def clean_for_speech(text: str) -> str:
    text = DOT_LEADER_PAGE.sub(r", page \1,", text)
    text = DOT_LEADER.sub(", ", text)
    text = URL.sub(r"\1", text)
    return WHITESPACE.sub(" ", text).strip()


def heading_speech(block: dict) -> str:
    text = clean_for_speech(block["text"]).rstrip(".:;,")
    if not text:
        return ""
    match = NUMBERED_TITLE.match(text)
    if block.get("level") == 1 and match and not TITLE_KEYWORD.match(match.group(2)):
        return f"Chapter {match.group(1)}. {match.group(2)}."
    return f"{text}."


def item_speech(text: str) -> str:
    cleaned = clean_for_speech(text)
    match = LIST_MARKER.match(cleaned)
    if match:
        number = match.group(1) or match.group(2)
        body = cleaned[match.end() :].strip()
        cleaned = f"{number}. {body}" if number else body
    if cleaned and cleaned[-1] not in ".!?:;":
        cleaned += "."
    return cleaned


def split_text(text: str) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(text):
        for part in _split_long_sentence(sentence):
            if current and len(current) + len(part) + 1 > MAX_CHUNK_CHARS:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
    if current:
        pieces.append(current)
    return pieces


def _split_long_sentence(sentence: str) -> list[str]:
    if len(sentence) <= MAX_CHUNK_CHARS:
        return [sentence] if sentence else []
    parts: list[str] = []
    current = ""
    for word in sentence.split(" "):
        if current and len(current) + len(word) + 1 > MAX_CHUNK_CHARS:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def build_chunks(page: int, blocks: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    pending_items: list[tuple[str, str, int]] = []

    def add(text: str, speech: str, block_ids: list[int]) -> None:
        speech = speech.strip()
        if not speech:
            return
        chunks.append(
            {
                "id": f"{page}-{len(chunks)}",
                "page": page,
                "blocks": block_ids,
                "text": text.strip(),
                "speech": speech,
            }
        )

    def flush_items() -> None:
        nonlocal pending_items
        group: list[tuple[str, str, int]] = []
        length = 0
        for display, speech, block_id in pending_items:
            if group and length + len(speech) > MAX_CHUNK_CHARS:
                _emit(group)
                group, length = [], 0
            group.append((display, speech, block_id))
            length += len(speech)
        if group:
            _emit(group)
        pending_items = []

    def _emit(group: list[tuple[str, str, int]]) -> None:
        add(
            "\n".join(item[0] for item in group),
            " ".join(item[1] for item in group),
            [item[2] for item in group],
        )

    for block in blocks:
        index = block["i"]
        if block["type"] == "heading":
            flush_items()
            add(block["text"], heading_speech(block), [index])
        elif block["type"] == "list_item":
            pending_items.append((block["text"], item_speech(block["text"]), index))
        else:
            flush_items()
            for piece in split_text(clean_for_speech(block["text"])):
                add(piece, piece, [index])
    flush_items()
    return chunks
