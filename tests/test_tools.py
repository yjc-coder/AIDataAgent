"""Phase 7 tests — Tool Calling.

Covers:
    1. The two @tool wrappers (search_schema / execute_sql) — offline via
       monkeypatched dependencies, plus REAL sqlglot rejection of malicious
       SQL through execute_sql (no DB needed: validation fails pre-connection).
    2. tool_call_node — one LLM decision round, tool execution, chunk merging,
       failure degradation, per-turn call cap.
    3. Graph-level conditional edge — low-confidence recall detours through
       tool_call; good recall skips it.

No test here talks to a real LLM or MySQL.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from app.rag.documents import KnowledgeChunk
from app.rag.retriever import RecallResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id_: str, type_: str = "schema", title: str = "") -> KnowledgeChunk:
    return KnowledgeChunk(id=id_, text=f"text of {id_}", type=type_, source=f"{id_}.md", title=title or id_)


class _StubRetriever:
    """Retriever stand-in returning canned chunks — no Chroma, no network."""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = chunks
        self.called_with: list[str] = []

    def recall(self, question: str) -> RecallResult:
        self.called_with.append(question)
        schema_text, evidence_text = None, None  # not used by the tool
        from app.rag.retriever import chunks_to_prompt_parts

        s, e = chunks_to_prompt_parts(self._chunks)
        return RecallResult(schema_text=s, evidence_text=e, chunks=list(self._chunks))


def _ai_with_tools(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def _search_call(keyword: str, id_: str = "c1") -> dict:
    return {"name": "search_schema", "args": {"keyword": keyword}, "id": id_, "type": "tool_call"}


def _sql_call(sql: str, id_: str = "c2") -> dict:
    return {"name": "execute_sql", "args": {"sql": sql}, "id": id_, "type": "tool_call"}


# ---------------------------------------------------------------------------
# 1. The @tool wrappers
# ---------------------------------------------------------------------------


def test_search_schema_tool_returns_parseable_json(monkeypatch):
    from app.tools import search_schema as mod

    stub = _StubRetriever([_chunk("schema/ecommerce.md::000::orders"), _chunk("business/sales.md::001", type_="business")])
    monkeypatch.setattr(mod, "get_retriever", lambda: stub)

    out = mod.search_schema.invoke({"keyword": "orders"})

    data = json.loads(out)
    assert data["keyword"] == "orders"
    assert data["count"] == 2
    ids = [c["id"] for c in data["chunks"]]
    assert "schema/ecommerce.md::000::orders" in ids
    assert stub.called_with == ["orders"]  # retriever got the keyword, not the full question


def test_chunks_from_tool_output_roundtrip_and_garbage():
    from app.tools.search_schema import chunks_from_tool_output

    payload = json.dumps(
        {
            "keyword": "orders",
            "count": 1,
            "chunks": [{"id": "x", "type": "schema", "title": "orders", "source": "s", "text": "t"}],
        },
        ensure_ascii=False,
    )
    chunks = chunks_from_tool_output(payload)
    assert len(chunks) == 1
    assert chunks[0].id == "x"
    assert chunks[0].type == "schema"

    # Garbage / error strings must never raise.
    assert chunks_from_tool_output("not json at all") == []
    assert chunks_from_tool_output(json.dumps({"error": "boom"})) == []


@pytest.mark.asyncio
async def test_execute_sql_tool_returns_rows(monkeypatch):
    from app.tools import execute_sql as mod

    async def _fake_run(sql, **_kw):
        assert sql == "SELECT id FROM users"
        return 2, [{"id": 1}, {"id": 2}]

    monkeypatch.setattr(mod, "run_select", _fake_run)

    out = await mod.execute_sql.ainvoke({"sql": "SELECT id FROM users"})
    data = json.loads(out)
    assert data["error"] is None
    assert data["row_count"] == 2
    assert data["rows"] == [{"id": 1}, {"id": 2}]
    assert data["truncated"] is False


@pytest.mark.asyncio
async def test_execute_sql_tool_truncates_huge_results(monkeypatch):
    from app.tools import execute_sql as mod

    rows = [{"i": i} for i in range(50)]

    async def _fake_run(_sql, **_kw):
        return 50, rows

    monkeypatch.setattr(mod, "run_select", _fake_run)

    data = json.loads(await mod.execute_sql.ainvoke({"sql": "SELECT * FROM orders"}))
    assert data["row_count"] == 50
    assert len(data["rows"]) == 20  # MAX_PREVIEW_ROWS
    assert data["truncated"] is True


@pytest.mark.asyncio
async def test_execute_sql_tool_blocks_malicious_sql():
    """恶意 SQL 经工具路径仍被 sqlglot 拦截（真实 validator，无需 DB）。

    validate_and_rewrite raises BEFORE any connection is opened, so this
    runs fully offline with the real defence chain in place.
    """
    from app.tools import execute_sql as mod

    hostile = [
        "DROP TABLE users",
        "DELETE FROM orders",
        "UPDATE users SET id = 1",
        "INSERT INTO users VALUES (1)",
        "SELECT SLEEP(10)",
        "SELECT 1; DROP TABLE users",
        "ALTER TABLE users ADD COLUMN x INT",
    ]
    for sql in hostile:
        out = await mod.execute_sql.ainvoke({"sql": sql})
        data = json.loads(out)
        assert data["error"], f"{sql} should be rejected"
        assert data["row_count"] is None
        assert data["rows"] == []


@pytest.mark.asyncio
async def test_execute_sql_tool_rejects_empty_sql():
    from app.tools import execute_sql as mod

    data = json.loads(await mod.execute_sql.ainvoke({"sql": "   "}))
    assert data["error"]


# ---------------------------------------------------------------------------
# 2. tool_call_node
# ---------------------------------------------------------------------------


def _set_min_helpers(**overrides):
    """set_helpers with throwaway stubs for nodes the test doesn't exercise."""
    from app.agent import nodes

    kwargs = dict(
        intent_classifier=None,
        schema_recaller=None,
        sql_generator=None,
        sql_runner=None,
        report_generator=None,
    )
    kwargs.update(overrides)
    nodes.set_helpers(**kwargs)


@pytest.mark.asyncio
async def test_tool_call_node_runs_search_schema_and_merges_chunks(monkeypatch):
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION, KEY_RETRIEVED_CHUNKS
    from app.tools import search_schema as search_mod

    stub = _StubRetriever([_chunk("schema/orders")])
    monkeypatch.setattr(search_mod, "get_retriever", lambda: stub)

    async def _decider(_messages):
        return _ai_with_tools(_search_call("orders"))

    _set_min_helpers(tool_decider=_decider)

    state: AgentState = {KEY_QUESTION: "查询订单", KEY_RETRIEVED_CHUNKS: []}
    out = await nodes.tool_call_node(state)

    assert out["tool_called"] is True
    assert out["schema_needs_tool"] is False
    assert len(out["tool_results"]) == 1
    entry = out["tool_results"][0]
    assert entry["ok"] is True and entry["tool"] == "search_schema"
    # tool-found knowledge landed in retrieved_chunks for sql_generate
    assert [c.id for c in out[KEY_RETRIEVED_CHUNKS]] == ["schema/orders"]
    assert out["node_path"][-1] == "tool_call"


@pytest.mark.asyncio
async def test_tool_call_node_malicious_execute_sql_is_blocked():
    """LLM 若被诱导执行 DELETE，工具路径必须拦截且不炸图（真实 validator）。"""
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION, KEY_TOOL_ERROR

    async def _decider(_messages):
        return _ai_with_tools(_sql_call("DELETE FROM orders"))

    _set_min_helpers(tool_decider=_decider)

    out = await nodes.tool_call_node({KEY_QUESTION: "删掉所有订单"})
    entry = out["tool_results"][0]
    assert entry["ok"] is False
    assert "Forbidden" in (entry["error"] or "") or "Only SELECT" in (entry["error"] or "")
    assert out[KEY_TOOL_ERROR]  # all calls failed → tool_error set
    assert out["tool_called"] is True
    assert out["schema_needs_tool"] is False  # one shot per turn


@pytest.mark.asyncio
async def test_tool_call_node_mixed_calls_reported_individually(monkeypatch):
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION
    from app.tools import execute_sql as sql_mod
    from app.tools import search_schema as search_mod

    monkeypatch.setattr(search_mod, "get_retriever", lambda: _StubRetriever([_chunk("schema/orders")]))

    async def _fake_run(_sql, **_kw):
        return 1, [{"n": 1}]

    monkeypatch.setattr(sql_mod, "run_select", _fake_run)

    async def _decider(_messages):
        return _ai_with_tools(_search_call("orders"), _sql_call("SELECT COUNT(*) AS n FROM orders"))

    _set_min_helpers(tool_decider=_decider)

    out = await nodes.tool_call_node({KEY_QUESTION: "多少订单"})
    assert [r["ok"] for r in out["tool_results"]] == [True, True]
    assert "tool_error" not in out  # at least one succeeded → no error


@pytest.mark.asyncio
async def test_tool_call_node_unknown_tool_reported():
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION

    async def _decider(_messages):
        return _ai_with_tools({"name": "send_email", "args": {}, "id": "c9", "type": "tool_call"})

    _set_min_helpers(tool_decider=_decider)

    out = await nodes.tool_call_node({KEY_QUESTION: "发邮件"})
    entry = out["tool_results"][0]
    assert entry["ok"] is False
    assert "未知工具" in entry["error"]


@pytest.mark.asyncio
async def test_tool_call_node_without_tool_calls_degrades():
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION

    async def _decider(_messages):
        return AIMessage(content="I don't need tools.")

    _set_min_helpers(tool_decider=_decider)

    out = await nodes.tool_call_node({KEY_QUESTION: "q"})
    assert out["tool_called"] is False
    assert "LLM未选择任何工具" in out["tool_error"]
    assert out["schema_needs_tool"] is False  # graph must not revisit tool_call


@pytest.mark.asyncio
async def test_tool_call_node_decider_failure_degrades():
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION

    async def _broken(_messages):
        raise RuntimeError("LLM down")

    _set_min_helpers(tool_decider=_broken)

    out = await nodes.tool_call_node({KEY_QUESTION: "q"})
    assert out["tool_called"] is False
    assert "工具决策失败" in out["tool_error"]


@pytest.mark.asyncio
async def test_tool_call_node_caps_tool_calls_per_turn(monkeypatch):
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION
    from app.tools import search_schema as search_mod

    monkeypatch.setattr(search_mod, "get_retriever", lambda: _StubRetriever([_chunk("schema/x")]))

    async def _decider(_messages):
        return _ai_with_tools(*[_search_call("x", id_=f"c{i}") for i in range(5)])

    _set_min_helpers(tool_decider=_decider)

    out = await nodes.tool_call_node({KEY_QUESTION: "q"})
    assert len(out["tool_results"]) == nodes.MAX_TOOL_CALLS_PER_TURN == 3


@pytest.mark.asyncio
async def test_tool_call_node_dedupes_chunks_against_existing(monkeypatch):
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION, KEY_RETRIEVED_CHUNKS
    from app.tools import search_schema as search_mod

    monkeypatch.setattr(search_mod, "get_retriever", lambda: _StubRetriever([_chunk("schema/orders"), _chunk("schema/users")]))

    async def _decider(_messages):
        return _ai_with_tools(_search_call("orders"))

    _set_min_helpers(tool_decider=_decider)

    state: AgentState = {KEY_QUESTION: "q", KEY_RETRIEVED_CHUNKS: [_chunk("schema/orders")]}
    out = await nodes.tool_call_node(state)
    ids = [c.id for c in out[KEY_RETRIEVED_CHUNKS]]
    assert ids.count("schema/orders") == 1  # deduped
    assert "schema/users" in ids


# ---------------------------------------------------------------------------
# 3. schema_recall_node low-confidence flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_recall_flags_low_confidence_without_schema_chunks():
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_QUESTION, KEY_SCHEMA_NEEDS_TOOL, KEY_SHOULD_CONTINUE

    async def _recall(_q):
        return [_chunk("business/sales", type_="business")], "s", "e"

    _set_min_helpers(schema_recaller=_recall)

    out = await nodes.schema_recall_node({KEY_QUESTION: "q", KEY_SHOULD_CONTINUE: True})
    assert out[KEY_SCHEMA_NEEDS_TOOL] is True  # business-only → needs the tool


@pytest.mark.asyncio
async def test_schema_recall_confident_with_schema_chunk():
    from app.agent import nodes
    from app.agent.state import KEY_QUESTION, KEY_SCHEMA_NEEDS_TOOL, KEY_SHOULD_CONTINUE

    async def _recall(_q):
        return [_chunk("schema/orders")], "s", "e"

    _set_min_helpers(schema_recaller=_recall)

    out = await nodes.schema_recall_node({KEY_QUESTION: "q", KEY_SHOULD_CONTINUE: True})
    assert out[KEY_SCHEMA_NEEDS_TOOL] is False


@pytest.mark.asyncio
async def test_schema_recall_failure_sets_needs_tool():
    from app.agent import nodes
    from app.agent.state import KEY_QUESTION, KEY_SCHEMA_NEEDS_TOOL, KEY_SHOULD_CONTINUE

    async def _broken(_q):
        raise RuntimeError("chroma down")

    _set_min_helpers(schema_recaller=_broken)

    out = await nodes.schema_recall_node({KEY_QUESTION: "q", KEY_SHOULD_CONTINUE: True})
    assert out[KEY_SCHEMA_NEEDS_TOOL] is True
    assert "表结构检索失败" in (out["error"] or "")


# ---------------------------------------------------------------------------
# 4. Graph-level conditional edge
# ---------------------------------------------------------------------------


def _wire_graph(stubs: dict) -> None:
    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers

    set_helpers(**stubs)
    reset_compiled_graph()


@pytest.mark.asyncio
async def test_full_graph_detours_through_tool_call_on_low_confidence(monkeypatch):
    """空召回 → intent, schema_recall, tool_call, sql_generate, sql_execute, report."""
    from app.agent.graph import get_compiled_graph
    from app.tools import search_schema as search_mod

    calls: list[str] = []
    gen_saw_chunks: list[str] = []

    async def _intent(q):
        return "DATA_QUERY"

    async def _recall(q):
        calls.append("recall")
        return [], "", ""  # nothing found → needs tool

    async def _gen(q, chunks, prev_steps="(none)"):
        calls.append("gen")
        gen_saw_chunks.extend(c.id for c in (chunks or []))
        return "SELECT 1", "", list(chunks or [])

    async def _run(sql):
        calls.append("run")
        return 1, [{"x": 1}]

    async def _report(q, rows, sql):
        calls.append("report")
        return "done"

    async def _decider(_messages):
        return _ai_with_tools(_search_call("orders"))

    monkeypatch.setattr(search_mod, "get_retriever", lambda: _StubRetriever([_chunk("schema/orders")]))

    _wire_graph(
        dict(
            intent_classifier=_intent,
            schema_recaller=_recall,
            sql_generator=_gen,
            sql_runner=_run,
            report_generator=_report,
            tool_decider=_decider,
        )
    )

    app = get_compiled_graph()
    final = await app.ainvoke({"question": "查询"}, config={"configurable": {"thread_id": "t-tool-detour"}})

    assert calls == ["recall", "gen", "run", "report"]  # recall ran once; no second recall
    assert final["node_path"] == ["intent", "schema_recall", "tool_call", "sql_generate", "sql_execute", "report"]
    assert final["tool_called"] is True
    assert final["row_count"] == 1
    # tool-found schema flowed into sql_generate without a second RAG query
    assert gen_saw_chunks == ["schema/orders"]


@pytest.mark.asyncio
async def test_full_graph_skips_tool_call_on_good_recall():
    """高置信度召回 → 不进 tool_call，与 Phase 6 路径一致。"""
    from app.agent.graph import get_compiled_graph

    async def _intent(q):
        return "DATA_QUERY"

    async def _recall(q):
        return [_chunk("schema/orders")], "s", "e"

    async def _gen(q, chunks, prev_steps="(none)"):
        return "SELECT 1", "", list(chunks or [])

    async def _run(sql):
        return 0, []

    async def _report(q, rows, sql):
        return "ok"

    _wire_graph(
        dict(
            intent_classifier=_intent,
            schema_recaller=_recall,
            sql_generator=_gen,
            sql_runner=_run,
            report_generator=_report,
            # no tool_decider on purpose — the node must never be visited
        )
    )

    app = get_compiled_graph()
    final = await app.ainvoke({"question": "查询"}, config={"configurable": {"thread_id": "t-no-tool"}})

    assert "tool_call" not in final["node_path"]
    assert final["node_path"] == ["intent", "schema_recall", "sql_generate", "sql_execute", "report"]


@pytest.mark.asyncio
async def test_full_graph_chat_intent_never_enters_tool_call():
    from app.agent.graph import get_compiled_graph

    async def _intent(q):
        return "CHAT"

    async def _report(q, rows, sql):
        return "hi"

    _wire_graph(
        dict(
            intent_classifier=_intent,
            schema_recaller=None,  # must not be called
            sql_generator=None,
            sql_runner=None,
            report_generator=_report,
        )
    )

    app = get_compiled_graph()
    final = await app.ainvoke({"question": "你好"}, config={"configurable": {"thread_id": "t-chat2"}})
    assert "tool_call" not in final["node_path"]
    assert final["final_answer"] == "hi"
