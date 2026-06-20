import uuid

from app.routes.extension import _content_from_text
from app.security import create_ws_ticket, decode_token


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
    print("ok")
