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
    # 文档转换是慢路径(本地 CPU 上大图 OCR/多页 PDF 常超过 20s 的通用超时):
    # 独立的转换超时, 与 docling-serve 自身的 document_timeout 默认值一致。
    docling_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    max_document_size_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    openai_api_key: str | None = None
    openai_base_url: AnyHttpUrl | None = None
    default_chat_model: str = "gpt-4o-mini"
    database_url: str = "sqlite+aiosqlite:///./fat_ai.db"
    jwt_secret: str = "change-this-development-secret-before-production"
    jwt_expiration_minutes: int = Field(default=60 * 24 * 7, gt=0)
    upload_directory: str = "data/uploads"
    allow_local_document_paths: bool = True

    # RAG: OpenAI-compatible embedding endpoint (local Ollama by default, hosted BYOK for production)
    embedding_base_url: str = "http://127.0.0.1:11434/v1"
    embedding_api_key: str = "ollama"
    embedding_model: str = "bge-m3"
    embedding_dimensions: int = Field(default=1024, ge=64, le=8192)
    rag_top_k_memory: int = Field(default=8, ge=1, le=50)
    rag_top_k_document: int = Field(default=5, ge=1, le=50)
    rag_min_score: float = Field(default=0.45, ge=0, le=1)
    rag_chunk_chars: int = Field(default=800, ge=100, le=10_000)
    rag_sweep_seconds: float = Field(default=5.0, gt=0, le=3600)


@lru_cache
def get_settings() -> Settings:
    return Settings()
