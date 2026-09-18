"""Shared pytest fixtures.

Three roles:
    1. Provide a hermetic Stub LLM for the FastAPI endpoint tests.
    2. Provide a DB pool fixture (auto-skipped if MySQL is unreachable).
    3. Provide a Stub retriever so RAG-related tests don't trigger real
       embedding API calls (or any Chroma disk I/O outside of an isolated
       temp dir).

We monkeypatch the chat-model factory rather than going through FastAPI's
`dependency_overrides` — keeps the test surface minimal and free of
Depends() plumbing at the endpoint.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.api.chat as chat_module
import app.api.sql as sql_module


# ---------------------------------------------------------------------------
# Stub LLM (Phase 1)
# ---------------------------------------------------------------------------


class _StubChat:
    """Minimal stand-in for `ChatOpenAI` exposing `ainvoke`."""

    def __init__(self, content: str = "stub-answer") -> None:
        self._content = content

    async def ainvoke(self, *_args, **_kwargs):
        class _Msg:
            def __init__(self, c):
                self.content = c

        return _Msg(self._content)


@pytest.fixture(autouse=True)
def restore_graph_helpers():
    """Snapshot & restore the LangGraph node helpers around every test.

    Several test files install stubs via `set_helpers(...)` (a plain module
    global write, no undo). Without this fixture the stubs leak into every
    later test file — e.g. test_multi_turn's stubs broke test_sql's hostile
    SQL tests only when run as part of the full suite.
    """
    from app.agent import nodes as nodes_module

    _keys = (
        "_intent_classifier",
        "_schema_recaller",
        "_sql_generator",
        "_sql_runner",
        "_report_generator",
        "_tool_decider",
    )
    saved = {k: getattr(nodes_module, k) for k in _keys}
    yield
    for k, v in saved.items():
        setattr(nodes_module, k, v)


@pytest.fixture
def stub_chat() -> _StubChat:
    return _StubChat("hello from stub")


@pytest.fixture
def client(stub_chat, monkeypatch):
    """FastAPI TestClient with the chat model patched to a stub."""
    monkeypatch.setattr(chat_module, "get_chat_model", lambda: stub_chat)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Stub Text2SqlGenerator (Phase 3)
# ---------------------------------------------------------------------------


class _StubT2S:
    """Returns a fixed SQL and reasoning — no real LLM call."""

    def __init__(
        self,
        sql: str = "SELECT 1 AS x",
        reasoning: str = "(stub)",
        chunks: list | None = None,
    ) -> None:
        self._sql = sql
        self._reasoning = reasoning
        self._chunks = chunks or []

    async def agenerate(self, **_kwargs):
        from app.agent.text2sql import Text2SqlResult

        return Text2SqlResult(
            sql=self._sql,
            reasoning=self._reasoning,
            retrieved_chunks=self._chunks,
        )


@pytest.fixture
def stub_generator():
    return _StubT2S(
        sql="SELECT 1 AS hello",
        reasoning="phase3 stub",
    )


@pytest.fixture
def sql_client(stub_generator, monkeypatch):
    """TestClient for /api/sql with the Text2SqlGenerator stubbed.

    NOTE: the `execute` request body will still try to run the SQL when
    `execute=true`. To avoid a DB round-trip we also monkeypatch `run_select`
    on the *caller's* namespace (`app.api.sql.run_select`), since that's where
    `from app.agent.sql_runner import run_select` captured the reference at
    import time.
    """
    async def _fake_run(sql: str, **_kw):
        return 0, []

    import app.agent.text2sql as t2s_module
    import app.agent.sql_runner as sql_runner_module

    # Phase 6: graph reads the factory via the module attribute
    # `text2sql_module.get_text2sql_generator` (see `app.agent.graph`).
    monkeypatch.setattr(t2s_module.text2sql_module, "get_text2sql_generator", lambda: stub_generator)
    monkeypatch.setattr(sql_runner_module, "run_select", _fake_run)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Stub Retriever (Phase 5)
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Returns canned schema/evidence without touching Chroma or the network."""

    def __init__(self, schema_text: str = "(stub schema)", evidence_text: str = "(stub evidence)", chunks: list | None = None) -> None:
        self._schema = schema_text
        self._evidence = evidence_text
        self._chunks = chunks or []

    def recall(self, question: str):
        from app.rag.retriever import RecallResult

        return RecallResult(
            schema_text=self._schema,
            evidence_text=self._evidence,
            chunks=self._chunks,
        )


@pytest.fixture
def stub_retriever():
    """Default stub retriever — also injected into Text2SqlGenerator."""
    return _StubRetriever()


@pytest.fixture
def stub_generator_with_rag(stub_retriever):
    """Stub T2S generator that has already 'called' RAG and returns chunks."""
    from app.rag.documents import KnowledgeChunk

    chunk = KnowledgeChunk(
        id="schema/ecommerce.md::000::orders",
        text="orders table info...",
        type="schema",
        source="schema/ecommerce.md",
        title="orders",
    )
    return _StubT2S(
        sql="SELECT 1 AS hello",
        reasoning="phase5 stub",
        chunks=[chunk],
    )


@pytest.fixture
def sql_client_with_rag(stub_generator_with_rag, stub_retriever, monkeypatch):
    """TestClient that exposes `retrieved_chunks` in the response.

    Phase 6 routes /api/sql through a LangGraph. The graph calls
    ``get_text2sql_generator`` from ``app.agent.text2sql`` (not the
    re-export in ``app.api.sql``), so we patch the agent module.
    """
    async def _fake_run(sql: str, **_kw):
        return 0, []

    import app.agent.text2sql as t2s_module
    import app.agent.sql_runner as sql_runner_module

    # Patch both the generator factory AND the module-level retriever getter
    # so that even if some code path falls back to `get_retriever()` we don't
    # actually build the real vector store.
    monkeypatch.setattr(t2s_module.text2sql_module, "get_text2sql_generator", lambda: stub_generator_with_rag)
    monkeypatch.setattr(sql_runner_module, "run_select", _fake_run)
    from app.rag import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "get_retriever", lambda: stub_retriever)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# DB fixture (Phase 2) — skips gracefully when MySQL is unreachable
# ---------------------------------------------------------------------------


def _mysql_reachable() -> bool:
    """Quick TCP probe; avoids a full aiomysql round-trip per collected test."""
    try:
        from app.config import get_settings
        import socket

        s = get_settings()
        with socket.create_connection((s.db_host, s.db_port), timeout=1.5):
            return True
    except Exception:
        return False


@pytest.fixture
async def db_ready():
    """Auto-skip if MySQL isn't reachable or auth fails; yield otherwise.

    Both "port closed" and "auth error" are treated as skip — we don't
    want a half-broken local DB (or absent credentials) to fail CI.
    """
    if not _mysql_reachable():
        pytest.skip("MySQL not reachable; run `docker compose up -d mysql`")
    from app.database.engine import init_pool, close_pool

    try:
        await init_pool()
    except Exception as e:  # pragma: no cover - depends on local DB
        pytest.skip(f"MySQL pool init failed ({type(e).__name__}): {e}")
    try:
        yield
    finally:
        await close_pool()