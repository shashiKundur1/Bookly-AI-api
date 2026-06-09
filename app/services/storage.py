import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings

PDF_MAGIC = b"%PDF-"
CHUNK_SIZE = 1024 * 1024


def ensure_data_dirs() -> None:
    settings = get_settings()
    for directory in (
        settings.books_dir,
        settings.covers_dir,
        settings.avatars_dir,
        settings.content_dir,
        settings.audio_dir,
        settings.models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _image_extension(chunk: bytes) -> str | None:
    if chunk.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if chunk.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if chunk.startswith(b"RIFF") and chunk[8:12] == b"WEBP":
        return "webp"
    return None


async def save_pdf(file: UploadFile, book_id: uuid.UUID) -> tuple[Path, int]:
    settings = get_settings()
    destination = settings.books_dir / f"{book_id}.pdf"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with destination.open("wb") as handle:
            first = True
            while chunk := await file.read(CHUNK_SIZE):
                if first:
                    if not chunk.startswith(PDF_MAGIC):
                        raise HTTPException(
                            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File is not a PDF"
                        )
                    first = False
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds the {settings.max_upload_mb} MB limit",
                    )
                handle.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")
    return destination, size


async def save_image(file: UploadFile, directory: Path, name: str) -> Path:
    settings = get_settings()
    max_bytes = settings.max_image_mb * 1024 * 1024
    temp = directory / f"{name}.tmp"
    size = 0
    extension: str | None = None
    try:
        with temp.open("wb") as handle:
            while chunk := await file.read(CHUNK_SIZE):
                if extension is None:
                    extension = _image_extension(chunk)
                    if extension is None:
                        raise HTTPException(
                            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            "Only JPEG, PNG and WebP images are supported",
                        )
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"Image exceeds the {settings.max_image_mb} MB limit",
                    )
                handle.write(chunk)
    except HTTPException:
        temp.unlink(missing_ok=True)
        raise
    if extension is None:
        temp.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")
    for existing in directory.glob(f"{name}.*"):
        if existing != temp:
            existing.unlink(missing_ok=True)
    final = directory / f"{name}.{extension}"
    temp.rename(final)
    return final


def delete_book_files(book_id: uuid.UUID) -> None:
    settings = get_settings()
    (settings.books_dir / f"{book_id}.pdf").unlink(missing_ok=True)
    (settings.content_dir / f"{book_id}.json").unlink(missing_ok=True)
    for cover in settings.covers_dir.glob(f"{book_id}.*"):
        cover.unlink(missing_ok=True)
    shutil.rmtree(settings.audio_dir / str(book_id), ignore_errors=True)
