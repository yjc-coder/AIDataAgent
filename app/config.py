"""Runtime configuration.

Replaces Java's `application.yml` + `AiModelRegistry` dynamic proxy.
Single source of truth: environment variables (and optional `.env`).

Use `get_settings()` anywhere — it returns a cached singleton.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime knobs in one place."""

    # ---------- LLM (OpenAI-compatible) ----------------------------------
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = "your-api-key-here"
    llm_model: str = "qwen-plus"

    # ---------- Embedding (Phase 5) --------------------------------------
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str = "your-api-key-here"
    embedding_model: str = "text-embedding-v3"

    # ---------- Server ---------------------------------------------------
    app_host: str = "0.0.0.0"
    app_port: int = 8065
    app_debug: bool = False

    # ---------- MySQL (Phase 2) -------------------------------------------
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "readonly"
    db_password: str = "readonly"
    db_name: str = "demo"
    db_charset: str = "utf8mb4"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # ---------- Phase 5: RAG (Chroma) --------------------------------------
    # Where the persistent Chroma collection lives on disk.
    chroma_dir: str = "./data/chroma"
    # Name of the collection inside Chroma.
    chroma_collection: str = "ai_data_agent_knowledge"
    # How many top chunks to recall per question.
    rag_top_k: int = 4
    # Force a fresh embed & rebuild on next launch. Useful after editing
    # data/knowledge/*.md or when upgrading the embedding model.
    rag_force_rebuild: bool = False
    # Max chars per chunk when splitting markdown — keeps retrieval focused.
    rag_chunk_size: int = 400
    # Overlap between consecutive chunks (helps when a rule straddles the cut).
    rag_chunk_overlap: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — Settings construction reads env once."""
    return Settings()
