"""Chat API — non-streaming Phase 1 endpoint.

POST /api/chat
    Single-turn LLM call that returns {answer, conversation_id, model}.

The streaming sibling at GET /api/stream/search lands in Phase 9,
once LangGraph is in place (Phase 6) and we have something to stream.

We do NOT use Depends() for the chat model so tests can simply
monkeypatch `app.api.chat.get_chat_model` without touching FastAPI's
dependency_overrides graph.
"""
from fastapi import APIRouter

from app.config import get_settings
from app.llm.client import get_chat_model
from app.models import ChatRequest, ChatResponse


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Echo the user's question to the LLM and return raw text answer.

    Phase 1 has no RAG, no schema, no agent workflow — the goal is purely:
        FastAPI up  →  config loaded  →  LLM reachable  →  answer returned.

    Phases 3-7 will turn this into a proper Text-to-SQL pipeline:
        Intent → Schema RAG → SQL Generate → SQL Execute → Report.
    """
    settings = get_settings()
    chat_model = get_chat_model()
    message = await chat_model.ainvoke(req.query)
    return ChatResponse(
        answer=message.content,
        conversation_id=req.conversation_id,
        model=settings.llm_model,
    )
