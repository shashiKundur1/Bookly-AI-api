import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from app.services.narration import build_chunks

MARGIN_BAND = 0.08
REPEAT_MIN_PAGES = 3
REPEAT_RATIO = 0.18
BOLD_FLAG = 16
MAX_HEADING_CHARS = 120

WHITESPACE = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")
PAGE_LABEL = re.compile(r"^[\s.\-–—·]*(?:\d{1,4}|[ivxlcdm]{1,8}|[IVXLCDM]{1,8})[\s.\-–—·]*$")
LIST_MARKER = re.compile(r"^\s*(?:[•▪◦‣·∙○●♦►▶]|\d{1,3}[.)]|[a-zA-Z][.)])\s+")
TERMINAL = (".", "!", "?", ":", ";")


def extract_book(pdf_path: Path) -> dict[str, Any]:
    with pymupdf.open(pdf_path) as doc:
        toc = _table_of_contents(doc)
        body_size, repeated_margins = _document_profile(doc)
        toc_by_page = _toc_by_page(toc)
        pages = []
        for page in doc:
            number = page.number + 1
            blocks = _page_blocks(page, body_size, repeated_margins, toc_by_page.get(number, []))
            pages.append({"page": number, "blocks": blocks, "chunks": build_chunks(number, blocks)})
        metadata = doc.metadata or {}
        return {
            "page_count": doc.page_count,
            "title": (metadata.get("title") or "").strip(),
            "author": (metadata.get("author") or "").strip(),
            "toc": toc,
            "pages": pages,
        }


def _table_of_contents(doc: pymupdf.Document) -> list[dict[str, Any]]:
    entries = []
    for level, title, page in doc.get_toc(simple=True):
        title = WHITESPACE.sub(" ", title or "").strip()
        if title and page >= 1:
            entries.append({"level": level, "title": title, "page": page})
    return entries


def _simplify(text: str) -> str:
    return WHITESPACE.sub(" ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def _toc_by_page(toc: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    mapping: dict[int, list[dict[str, Any]]] = {}
    for entry in toc:
        simplified = _simplify(entry["title"])
        if simplified:
            mapping.setdefault(entry["page"], []).append(
                {"level": entry["level"], "simplified": simplified}
            )
    return mapping


def _normalize_margin(text: str) -> str:
    return WHITESPACE.sub(" ", DIGITS.sub("#", text)).strip().lower()


def _line_text(line: dict[str, Any]) -> str:
    return WHITESPACE.sub(" ", "".join(span["text"] for span in line["spans"])).strip()


def _document_profile(doc: pymupdf.Document) -> tuple[float, set[str]]:
    sizes: Counter[float] = Counter()
    margin_lines: Counter[str] = Counter()
    for page in doc:
        height = page.rect.height or 1
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                if abs(line["dir"][1]) > 0.3:
                    continue
                text = _line_text(line)
                if not text:
                    continue
                for span in line["spans"]:
                    stripped = span["text"].strip()
                    if stripped:
                        sizes[round(span["size"] * 2) / 2] += len(stripped)
                center = (line["bbox"][1] + line["bbox"][3]) / 2
                if center < height * MARGIN_BAND or center > height * (1 - MARGIN_BAND):
                    margin_lines[_normalize_margin(text)] += 1
    body_size = sizes.most_common(1)[0][0] if sizes else 11.0
    threshold = max(REPEAT_MIN_PAGES, int(doc.page_count * REPEAT_RATIO))
    repeated = {text for text, count in margin_lines.items() if count >= threshold}
    return body_size, repeated


def _heading_level(line: dict[str, Any], body_size: float, toc_titles: list[dict[str, Any]]) -> int | None:
    text = line["text"]
    if len(text) > MAX_HEADING_CHARS:
        return None
    if toc_titles:
        simplified = _simplify(text)
        for entry in toc_titles:
            if simplified and simplified == entry["simplified"]:
                return min(entry["level"], 3)
    ratio = line["size"] / body_size if body_size else 1.0
    if ratio >= 1.6:
        return 1
    if ratio >= 1.32:
        return 2
    if ratio >= 1.15:
        return 3
    if (
        line["bold"]
        and ratio >= 1.02
        and len(text) <= 80
        and not text.rstrip().endswith((".", ",", ";", ":"))
        and not LIST_MARKER.match(text)
    ):
        return 3
    return None


def _join_lines(parts: list[str]) -> str:
    text = ""
    for part in parts:
        if not text:
            text = part
        elif text.endswith("-") and part[:1].islower():
            text = text[:-1] + part
        else:
            text = f"{text} {part}"
    return WHITESPACE.sub(" ", text).strip()


def _norm_bbox(bbox: tuple[float, float, float, float], width: float, height: float) -> list[float]:
    return [
        round(bbox[0] / width, 4),
        round(bbox[1] / height, 4),
        round(bbox[2] / width, 4),
        round(bbox[3] / height, 4),
    ]


def _merge_bbox(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _classify(
    lines: list[dict[str, Any]],
    body_size: float,
    toc_titles: list[dict[str, Any]],
    width: float,
    height: float,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    paragraph: list[dict[str, Any]] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        bbox = _norm_bbox(paragraph[0]["bbox"], width, height)
        for line in paragraph[1:]:
            bbox = _merge_bbox(bbox, _norm_bbox(line["bbox"], width, height))
        blocks.append(
            {
                "type": "paragraph",
                "text": _join_lines([line["text"] for line in paragraph]),
                "bbox": bbox,
            }
        )
        paragraph.clear()

    for line in lines:
        line_bbox = _norm_bbox(line["bbox"], width, height)
        level = _heading_level(line, body_size, toc_titles)
        if level is not None:
            flush_paragraph()
            previous = blocks[-1] if blocks else None
            if previous is not None and previous["type"] == "heading" and previous["level"] == level:
                previous["text"] = _join_lines([previous["text"], line["text"]])
                previous["bbox"] = _merge_bbox(previous["bbox"], line_bbox)
            else:
                blocks.append(
                    {"type": "heading", "level": level, "text": line["text"], "bbox": line_bbox}
                )
        elif LIST_MARKER.match(line["text"]):
            flush_paragraph()
            blocks.append({"type": "list_item", "text": line["text"], "bbox": line_bbox})
        else:
            last = blocks[-1] if blocks and not paragraph else None
            if (
                last is not None
                and last["type"] == "list_item"
                and not last["text"].rstrip().endswith(TERMINAL)
            ):
                last["text"] = _join_lines([last["text"], line["text"]])
                last["bbox"] = _merge_bbox(last["bbox"], line_bbox)
            else:
                paragraph.append(line)
    flush_paragraph()
    return blocks


def _page_blocks(
    page: pymupdf.Page,
    body_size: float,
    repeated_margins: set[str],
    toc_titles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    width = page.rect.width or 1
    height = page.rect.height or 1
    blocks: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        lines = []
        for line in block["lines"]:
            if abs(line["dir"][1]) > 0.3:
                continue
            text = _line_text(line)
            if not text:
                continue
            center = (line["bbox"][1] + line["bbox"][3]) / 2
            in_margin = center < height * MARGIN_BAND or center > height * (1 - MARGIN_BAND)
            if in_margin and (
                _normalize_margin(text) in repeated_margins or PAGE_LABEL.match(text)
            ):
                continue
            visible_spans = [span for span in line["spans"] if span["text"].strip()]
            size = max((span["size"] for span in visible_spans), default=body_size)
            bold = bool(visible_spans) and all(
                span["flags"] & BOLD_FLAG for span in visible_spans
            )
            lines.append({"text": text, "size": size, "bold": bold, "bbox": line["bbox"]})
        if lines:
            blocks.extend(_classify(lines, body_size, toc_titles, width, height))
    for index, block in enumerate(blocks):
        block["i"] = index
    return blocks
