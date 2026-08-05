from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = "fat-ai-server"
    environment: str = "development"
    cors_origins: list[str] = Field(default_factory=list)
    request_timeout_seconds: float = Field(default=20, gt=0, le=120)
    docling_server_url: AnyHttpUrl = "http://127.0.0.1:5001"
    max_document_size_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    openai_api_key: str | None = None
    openai_base_url: AnyHttpUrl | None = None
    default_chat_model: str = "gpt-4o-mini"
    database_url: str = "sqlite+aiosqlite:///./fat_ai.db"
    jwt_secret: str = "change-this-development-secret-before-production"
    jwt_expiration_minutes: int = Field(default=60 * 24 * 7, gt=0)
    upload_directory: str = "data/uploads"
    allow_local_document_paths: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
