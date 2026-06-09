from fastapi import APIRouter

from app.routes.auth import router as auth_router
from app.routes.books import router as books_router
from app.routes.narration import router as narration_router
from app.routes.reading import router as reading_router
from app.routes.users import router as users_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(books_router)
api_router.include_router(reading_router)
api_router.include_router(narration_router)
