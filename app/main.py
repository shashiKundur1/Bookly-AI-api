import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import get_settings
from app.database import engine
from app.routes import api_router
from app.services.storage import ensure_data_dirs
from app.services.tts import warmup

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("app").setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

_startup_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ensure_data_dirs()
    if settings.environment != "development":
        task = asyncio.create_task(warmup())
        _startup_tasks.add(task)
        task.add_done_callback(_startup_tasks.discard)
    yield
    await engine.dispose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api/v1")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}
