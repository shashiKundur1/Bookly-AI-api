import hashlib

from fastapi import Request


def weak_etag(*parts: object) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode()).hexdigest()[:20]
    return f'W/"{digest}"'


def is_fresh(request: Request, etag: str) -> bool:
    return request.headers.get("if-none-match") == etag
