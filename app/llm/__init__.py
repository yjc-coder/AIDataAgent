"""LLM sub-package.

Exports the chat and embeddings factories used app-wide.
"""
from app.llm.client import get_chat_model, get_embeddings

__all__ = ["get_chat_model", "get_embeddings"]
