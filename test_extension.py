import uuid

from app.routes.extension import _content_from_text
from app.security import create_ws_ticket, decode_token
from app.services.tts import FIRST_PIECE_MAX_CHARS, split_for_streaming


def test_fast_first_piece() -> None:
    long_open = (
        "Although the morning had started with a great deal of confusion and noise, "
        "everyone eventually found their seats and the meeting began on time."
    )
    pieces = split_for_streaming(long_open)
    assert pieces, "expected pieces"
    assert len(pieces[0]) <= 90, f"first piece too long for fast start: {pieces[0]!r}"
    assert "".join(pieces).replace(" ", "") == long_open.replace(" ", "")


def test_short_text_single_piece() -> None:
    pieces = split_for_streaming("Read this.")
    assert pieces == ["Read this."]


def test_content_from_text() -> None:
    document = _content_from_text("Hello world.\n\nA second paragraph, longer than the first one.")
    assert document["page_count"] == 1
    page = document["pages"][0]
    assert page["page"] == 1
    assert len(page["blocks"]) == 2
    chunks = page["chunks"]
    assert chunks, "expected at least one narratable chunk"
    assert all(chunk["id"].startswith("1-") for chunk in chunks)
    assert all(chunk["speech"].strip() for chunk in chunks)
    joined = " ".join(chunk["speech"] for chunk in chunks)
    assert "Hello world" in joined and "second paragraph" in joined


def test_content_from_text_empty_lines() -> None:
    document = _content_from_text("\n\n   \n")
    assert document["pages"][0]["blocks"] == []
    assert document["pages"][0]["chunks"] == []


def test_ws_ticket_roundtrip() -> None:
    user_id = uuid.uuid4()
    ticket = create_ws_ticket(user_id)
    assert decode_token(ticket, "ws") == user_id
    assert decode_token(ticket, "access") is None


if __name__ == "__main__":
    test_content_from_text()
    test_content_from_text_empty_lines()
    test_ws_ticket_roundtrip()
    test_fast_first_piece()
    test_short_text_single_piece()
    print("ok")
