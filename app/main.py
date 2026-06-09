import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import engine
from app.routes import api_router
from app.services.storage import ensure_data_dirs
from app.services.tts import warmup

settings = get_settings()
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api/v1")


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}
