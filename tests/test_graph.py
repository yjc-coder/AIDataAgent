"""Phase 6 tests — LangGraph 5-node workflow.

All tests use stub helpers so the suite runs offline without an LLM, a real
DB, or a real Chroma collection.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Auto-save / restore helpers so a stub set in one test doesn't leak into
# another (test_graph sets globals via `nodes.set_helpers(...)`).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_node_helpers():
    from app.agent import nodes as _nodes

    saved = {
        k: getattr(_nodes, k)
        for k in (
            "_intent_classifier",
            "_schema_recaller",
            "_sql_generator",
            "_sql_runner",
            "_report_generator",
            "_tool_decider",
        )
    }
    yield
    for k, v in saved.items():
        setattr(_nodes, k, v)


# ---------------------------------------------------------------------------
# Per-node unit tests — no LangGraph involved; just call the function directly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_node_data_query_with_classifier(monkeypatch):
    from app.agent import nodes
    from app.agent.state import INTENT_DATA_QUERY, KEY_QUESTION, AgentState

    async def _classifier(_q):
        return INTENT_DATA_QUERY

    nodes.set_helpers(
        intent_classifier=_classifier,
        schema_recaller=lambda *a, **k: None,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=lambda *a, **k: None,
        report_generator=lambda *a, **k: None,
    )

    state: AgentState = {KEY_QUESTION: "查询销售额"}
    out = await nodes.intent_node(state)
    assert out["intent"] == INTENT_DATA_QUERY
    assert out["should_continue"] is True
    assert out["node_path"][-1] == "intent"


@pytest.mark.asyncio
async def test_intent_node_chat_skips_pipeline(monkeypatch):
    from app.agent import nodes
    from app.agent.state import INTENT_CHAT, KEY_QUESTION, AgentState

    async def _classifier(_q):
        return INTENT_CHAT

    nodes.set_helpers(
        intent_classifier=_classifier,
        schema_recaller=lambda *a, **k: None,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=lambda *a, **k: None,
        report_generator=lambda *a, **k: None,
    )

    state: AgentState = {KEY_QUESTION: "你好"}
    out = await nodes.intent_node(state)
    assert out["intent"] == INTENT_CHAT
    assert out["should_continue"] is False


@pytest.mark.asyncio
async def test_intent_node_classifier_failure_falls_back_to_data_query():
    from app.agent import nodes
    from app.agent.state import INTENT_DATA_QUERY, KEY_QUESTION, AgentState

    async def _broken(_q):
        raise RuntimeError("LLM down")

    nodes.set_helpers(
        intent_classifier=_broken,
        schema_recaller=lambda *a, **k: None,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=lambda *a, **k: None,
        report_generator=lambda *a, **k: None,
    )

    state: AgentState = {KEY_QUESTION: "查询"}
    out = await nodes.intent_node(state)
    assert out["intent"] == INTENT_DATA_QUERY  # fail-safe default
    assert "意图分类器异常" in (out["error"] or "")


@pytest.mark.asyncio
async def test_schema_recall_node_returns_chunks():
    from app.agent import nodes
    from app.agent.state import (
        KEY_QUESTION,
        KEY_SHOULD_CONTINUE,
        AgentState,
    )
    from app.rag.documents import KnowledgeChunk

    captured: dict = {}

    async def _recall(question):
        captured["q"] = question
        return [
            KnowledgeChunk(
                id="schema/u",
                text="users",
                type="schema",
                source="schema/u.md",
                title="users",
            )
        ], "schema text", "evidence text"

    nodes.set_helpers(
        intent_classifier=lambda *a, **k: None,
        schema_recaller=_recall,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=lambda *a, **k: None,
        report_generator=lambda *a, **k: None,
    )

    state: AgentState = {
        KEY_QUESTION: "q",
        KEY_SHOULD_CONTINUE: True,
    }
    out = await nodes.schema_recall_node(state)
    assert captured["q"] == "q"
    assert len(out["retrieved_chunks"]) == 1
    assert out["retrieved_chunks"][0].id == "schema/u"
    assert out["node_path"][-1] == "schema_recall"


@pytest.mark.asyncio
async def test_schema_recall_skipped_for_chat():
    from app.agent import nodes
    from app.agent.state import KEY_SHOULD_CONTINUE, AgentState

    called = {"flag": False}

    async def _recall(_q):
        called["flag"] = True
        return [], "", ""

    nodes.set_helpers(
        intent_classifier=lambda *a, **k: None,
        schema_recaller=_recall,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=lambda *a, **k: None,
        report_generator=lambda *a, **k: None,
    )

    state: AgentState = {KEY_SHOULD_CONTINUE: False}
    out = await nodes.schema_recall_node(state)
    assert called["flag"] is False
    assert "retrieved_chunks" not in out


@pytest.mark.asyncio
async def test_sql_generate_node_returns_sql_and_reasoning():
    from app.agent import nodes
    from app.agent.state import KEY_QUESTION, KEY_SHOULD_CONTINUE, AgentState

    captured: dict = {}

    async def _gen(question, chunks, prev_steps="(none)"):
        captured["q"] = question
        captured["chunks"] = chunks
        return "SELECT 1", "trivial", []

    nodes.set_helpers(
        intent_classifier=lambda *a, **k: None,
        schema_recaller=lambda *a, **k: None,
        sql_generator=_gen,
        sql_runner=lambda *a, **k: None,
        report_generator=lambda *a, **k: None,
    )

    state: AgentState = {
        KEY_QUESTION: "q",
        KEY_SHOULD_CONTINUE: True,
    }
    out = await nodes.sql_generate_node(state)
    assert out["generated_sql"] == "SELECT 1"
    assert out["generated_reasoning"] == "trivial"
    assert captured["q"] == "q"


@pytest.mark.asyncio
async def test_sql_execute_node_runs_sql():
    from app.agent import nodes
    from app.agent.state import KEY_GENERATED_SQL, KEY_SHOULD_CONTINUE, AgentState

    captured: dict = {}

    async def _run(sql):
        captured["sql"] = sql
        return 1, [{"x": 1}]

    nodes.set_helpers(
        intent_classifier=lambda *a, **k: None,
        schema_recaller=lambda *a, **k: None,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=_run,
        report_generator=lambda *a, **k: None,
    )

    state: AgentState = {
        KEY_GENERATED_SQL: "SELECT 1",
        KEY_SHOULD_CONTINUE: True,
    }
    out = await nodes.sql_execute_node(state)
    assert out["row_count"] == 1
    assert out["query_result"] == [{"x": 1}]
    assert captured["sql"] == "SELECT 1"


@pytest.mark.asyncio
async def test_sql_execute_node_no_sql():
    from app.agent import nodes
    from app.agent.state import AgentState

    nodes.set_helpers(
        intent_classifier=lambda *a, **k: None,
        schema_recaller=lambda *a, **k: None,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=lambda *a, **k: None,
        report_generator=lambda *a, **k: None,
    )

    out = await nodes.sql_execute_node({})  # no SQL → no execution
    assert out["query_result"] is None
    assert out["row_count"] is None
    assert "待执行SQL为空" in (out["error"] or "")


@pytest.mark.asyncio
async def test_report_node_calls_generator():
    from app.agent import nodes
    from app.agent.state import AgentState

    captured: dict = {}

    async def _report(question, rows, sql):
        captured["q"] = question
        captured["rows"] = rows
        captured["sql"] = sql
        return "answer"

    nodes.set_helpers(
        intent_classifier=lambda *a, **k: None,
        schema_recaller=lambda *a, **k: None,
        sql_generator=lambda *a, **k: ("", "", []),
        sql_runner=lambda *a, **k: None,
        report_generator=_report,
    )

    state: AgentState = {"question": "q", "query_result": [{"a": 1}], "generated_sql": "SELECT 1"}
    out = await nodes.report_node(state)
    assert out["final_answer"] == "answer"
    assert captured["sql"] == "SELECT 1"


# ---------------------------------------------------------------------------
# End-to-end graph tests with stubbed helpers.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_graph_data_query_path():
    """All 5 nodes execute in order for a DATA_QUERY question."""
    from app.agent.graph import get_compiled_graph, reset_compiled_graph
    from app.agent.nodes import set_helpers
    from app.rag.documents import KnowledgeChunk

    calls: list[str] = []

    async def _intent(q):
        calls.append("intent")
        return "DATA_QUERY"

    async def _recall(q):
        calls.append("recall")
        return [KnowledgeChunk(id="x", text="t", type="schema", source="s", title="t")], "s", "e"

    async def _gen(q, chunks, prev_steps="(none)"):
        calls.append("gen")
        return "SELECT 1", "trivial", []

    async def _run(sql):
        calls.append("run")
        return 1, [{"x": 1}]

    async def _report(q, rows, sql):
        calls.append("report")
        return "the answer"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()

    app = get_compiled_graph()
    cfg = {"configurable": {"thread_id": "t-data-query"}}
    final = await app.ainvoke({"question": "查询销售额"}, config=cfg)

    # All 5 nodes visited (intent + 4 below).
    assert calls == ["intent", "recall", "gen", "run", "report"]
    assert final["intent"] == "DATA_QUERY"
    assert final["generated_sql"] == "SELECT 1"
    assert final["row_count"] == 1
    assert final["final_answer"] == "the answer"
    assert final["node_path"][-1] == "report"


@pytest.mark.asyncio
async def test_full_graph_chat_path_skips_sql_pipeline():
    """CHAT intent must route directly to report, skipping SQL pipeline."""
    from app.agent.graph import get_compiled_graph, reset_compiled_graph
    from app.agent.nodes import set_helpers

    calls: list[str] = []

    async def _intent(q):
        calls.append("intent")
        return "CHAT"

    async def _recall(q):
        calls.append("recall")
        return [], "", ""

    async def _gen(q, chunks, prev_steps="(none)"):
        calls.append("gen")
        return "", "", []

    async def _run(sql):
        calls.append("run")
        return 0, []

    async def _report(q, rows, sql):
        calls.append("report")
        return "hi there"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()

    app = get_compiled_graph()
    cfg = {"configurable": {"thread_id": "t-chat"}}
    final = await app.ainvoke({"question": "你好"}, config=cfg)

    assert calls == ["intent", "report"]
    assert final["intent"] == "CHAT"
    assert final["final_answer"] == "hi there"
    # sql_generate node never ran → its key isn't present in final state.
    assert final.get("generated_sql", "") == ""


@pytest.mark.asyncio
async def test_full_graph_records_node_path_for_trace():
    from app.agent.graph import get_compiled_graph, reset_compiled_graph
    from app.agent.nodes import set_helpers

    async def _intent(q):
        return "DATA_QUERY"

    async def _recall(q):
        from app.rag.documents import KnowledgeChunk

        return [KnowledgeChunk(id="x", text="t", type="schema", source="s", title="t")], "s", "e"

    async def _gen(q, chunks, prev_steps="(none)"):
        return "SELECT 1", "", []

    async def _run(sql):
        return 0, []

    async def _report(q, rows, sql):
        return "ok"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()

    app = get_compiled_graph()
    cfg = {"configurable": {"thread_id": "t-trace"}}
    final = await app.ainvoke({"question": "查询"}, config=cfg)
    assert final["node_path"] == ["intent", "schema_recall", "sql_generate", "sql_execute", "report"]


# ---------------------------------------------------------------------------
# API integration — /api/graph/structure & /api/graph/trace
# ---------------------------------------------------------------------------


def test_graph_structure_endpoint_returns_nodes_and_edges():
    from fastapi.testclient import TestClient
    from app.main import app
    from app.agent.graph import reset_compiled_graph

    reset_compiled_graph()
    with TestClient(app) as c:
        r = c.get("/api/graph/structure")
    assert r.status_code == 200
    data = r.json()
    names = {n["id"] for n in data["nodes"]}
    # Should include our 6 nodes + the special __start__ / __end__.
    for n in ("intent", "schema_recall", "tool_call", "sql_generate", "sql_execute", "report"):
        assert n in names
    assert data["node_count"] >= 6


def test_graph_trace_endpoint_returns_path():
    from fastapi.testclient import TestClient
    from app.main import app
    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers

    async def _intent(q):
        return "DATA_QUERY"

    async def _recall(q):
        from app.rag.documents import KnowledgeChunk

        return [KnowledgeChunk(id="x", text="t", type="schema", source="s", title="t")], "s", "e"

    async def _gen(q, chunks, prev_steps="(none)"):
        return "SELECT 1", "", []

    async def _run(sql):
        return 0, []

    async def _report(q, rows, sql):
        return "ok"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()

    with TestClient(app) as c:
        # First run the graph so MemorySaver has state to read.
        r = c.post(
            "/api/sql",
            json={"question": "test", "execute": False, "conversation_id": "t-api-trace"},
        )
        assert r.status_code == 200
        # Then ask for the trace.
        r2 = c.get("/api/graph/trace", params={"conversation_id": "t-api-trace"})
    assert r2.status_code == 200
    body = r2.json()
    assert body["conversation_id"] == "t-api-trace"
    assert body["node_path"][:1] == ["intent"]
    assert "report" in body["node_path"]
    assert body["intent"] == "DATA_QUERY"


def test_sql_endpoint_still_returns_after_graph_rewrite():
    """Regression: Phase 6 must not break /api/sql's response shape."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers

    async def _intent(q):
        return "DATA_QUERY"

    async def _recall(q):
        from app.rag.documents import KnowledgeChunk

        return [KnowledgeChunk(id="x", text="t", type="schema", source="s", title="t")], "s", "e"

    async def _gen(q, chunks, prev_steps="(none)"):
        return "SELECT 1", "trivial", []

    async def _run(sql):
        return 1, [{"x": 1}]

    async def _report(q, rows, sql):
        return "the answer"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()

    with TestClient(app) as c:
        r = c.post(
            "/api/sql",
            json={"question": "查询", "execute": True, "conversation_id": "t-regress"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["question"] == "查询"
    assert data["sql"] == "SELECT 1"
    assert data["reasoning"] == "trivial"
    assert data["executed"] is True
    assert data["rowcount"] == 1
    assert data["rows"] == [{"x": 1}]
    assert data["intent"] == "DATA_QUERY"
    assert "report" in data["node_path"]
    assert data["final_answer"] == "the answer"
    assert data["error"] is None