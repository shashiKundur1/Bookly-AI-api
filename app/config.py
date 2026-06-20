from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Bookly API"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://bookly:bookly@localhost:5432/bookly"
    jwt_secret: str = "change-me"
    access_token_minutes: int = 30
    refresh_token_days: int = 30
    cookie_secure: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]
    data_dir: Path = Path("data")
    max_upload_mb: int = 200
    max_image_mb: int = 10
    default_voice: str = "af_heart"
    tts_engine: str = "kokoro"
    gemini_api_key: str = ""
    gemini_tts_model: str = "gemini-3.1-flash-tts-preview"
    orpheus_url: str = "http://localhost:8080"

    @property
    def books_dir(self) -> Path:
        return self.data_dir / "books"

    @property
    def covers_dir(self) -> Path:
        return self.data_dir / "covers"

    @property
    def avatars_dir(self) -> Path:
        return self.data_dir / "avatars"

    @property
    def content_dir(self) -> Path:
        return self.data_dir / "content"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
