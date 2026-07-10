from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Knowledge Assistant"
    storage_dir: Path = Path("storage")
    uploads_dir: Path = Path("storage/uploads")
    qdrant_path: Path | None = Path("storage/qdrant")
    qdrant_collection: str = "documents"
    embedding_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    chunk_size: int = 700
    chunk_overlap: int = 100
    retrieval_k: int = 5
    max_context_chunks: int = 5
    rerank_enabled: bool = False
    rerank_top_n: int = 3
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    temp_upload_dir: Path = Path("storage/tmp")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
