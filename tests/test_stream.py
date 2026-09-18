"""Phase 9 tests — SSE streaming endpoint (POST /api/chat/stream).

The endpoint streams one SSE event per graph node. Tests stub the node
helpers (offline) and parse the event stream with httpx via TestClient.
"""

from __future__ import annotations

import json

import pytest


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


def _wire_graph() -> None:
    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers
    from app.rag.documents import KnowledgeChunk

    async def _intent(_q):
        return "DATA_QUERY"

    async def _recall(_q):
        return [KnowledgeChunk(id="x", text="t", type="schema", source="s", title="t")], "s", "e"

    async def _gen(_q, _chunks, _prev):
        return "SELECT 1", "", []

    async def _run(_sql):
        return 2, [{"x": 1}, {"x": 2}]

    async def _report(_q, _rows, _sql):
        return "最终答案"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()


def _post_stream(client, payload: dict):
    """POST /api/chat/stream and return the raw SSE text."""
    with client.stream(
        "POST",
        "/api/chat/stream",
        json=payload,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return "".join(chunk for chunk in response.iter_text())


def _parse_events(raw: str) -> list[tuple[str, object]]:
    """Parse 'event: X\\ndata: Y\\n\\n' blocks into (event, data) tuples."""
    events: list[tuple[str, object]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name, data_line = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_line = json.loads(line[len("data: "):])
        if event_name is not None:
            events.append((event_name, data_line))
    return events


def test_stream_emits_node_progress_events_in_order():
    from fastapi.testclient import TestClient

    from app.main import app

    _wire_graph()
    with TestClient(app) as client:
        raw = _post_stream(client, {"question": "查询2026年销售额", "conversation_id": "sse-1"})

    events = _parse_events(raw)
    names = [e for e, _ in events]

    assert names == ["thinking", "retrieval", "sql", "result", "answer", "done"]
    # answer event carries the final natural-language answer
    assert events[4][1] == "最终答案"
    # sql event contains the generated SQL
    assert "SELECT 1" in str(events[2][1])
    # result event reports the row count
    assert "2" in str(events[3][1])
    # terminal sentinel
    assert events[5][1] == "[DONE]"


def test_stream_chat_intent_skips_sql_events():
    from fastapi.testclient import TestClient

    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers
    from app.main import app

    async def _intent(_q):
        return "CHAT"

    async def _report(_q, _rows, _sql):
        return "你好呀"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=None,
        sql_generator=None,
        sql_runner=None,
        report_generator=_report,
    )
    reset_compiled_graph()

    with TestClient(app) as client:
        raw = _post_stream(client, {"question": "你好", "conversation_id": "sse-chat"})

    names = [e for e, _ in _parse_events(raw)]
    assert names == ["thinking", "answer", "done"]


def test_stream_low_confidence_recall_includes_tool_event(monkeypatch):
    from fastapi.testclient import TestClient

    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers
    from app.main import app
    from app.rag.documents import KnowledgeChunk
    from app.tools import search_schema as search_mod

    class _StubRetriever:
        def recall(self, _q):
            chunk = KnowledgeChunk(id="schema/orders", text="t", type="schema", source="s", title="orders")
            from app.rag.retriever import RecallResult

            return RecallResult(schema_text="s", evidence_text="e", chunks=[chunk])

    async def _intent(_q):
        return "DATA_QUERY"

    async def _recall(_q):
        return [], "", ""  # empty → tool detour

    async def _gen(_q, _chunks, _prev):
        return "SELECT 1", "", []

    async def _run(_sql):
        return 0, []

    async def _report(_q, _rows, _sql):
        return "ok"

    async def _decider(_messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="",
            tool_calls=[{"name": "search_schema", "args": {"keyword": "orders"}, "id": "c1", "type": "tool_call"}],
        )

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
        tool_decider=_decider,
    )
    reset_compiled_graph()
    monkeypatch.setattr(search_mod, "get_retriever", lambda: _StubRetriever())

    with TestClient(app) as client:
        raw = _post_stream(client, {"question": "查询", "conversation_id": "sse-tool"})

    names = [e for e, _ in _parse_events(raw)]
    assert names == ["thinking", "retrieval", "tool", "sql", "result", "answer", "done"]
    assert "search_schema" in str(dict(_parse_events(raw))["tool"])


def test_stream_survives_node_errors_and_always_ends_with_done():
    """节点内部错误被 state 吸收（图设计为永不抛出），流必须照常走完。"""
    from fastapi.testclient import TestClient

    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers
    from app.main import app

    async def _intent(_q):
        raise RuntimeError("LLM exploded")

    async def _report(_q, _rows, _sql):
        return "降级回答"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=None,
        sql_generator=None,
        sql_runner=None,
        report_generator=_report,
    )
    reset_compiled_graph()

    with TestClient(app) as client:
        raw = _post_stream(client, {"question": "查询", "conversation_id": "sse-err"})

    names = [e for e, _ in _parse_events(raw)]
    assert names[-1] == "done"  # stream always terminates cleanly
    assert "answer" in names  # degraded answer still reaches the user


def test_stream_multi_turn_second_call_receives_previous_question():
    """Phase 8+9 串联：同一 conversation_id 的第二个 SSE 流必须带上第一轮上下文。

    技巧：生成器把收到的 prev_steps 写进"SQL"，通过 sql 事件断言上下文，
    无需额外 mock 设施。
    """
    from fastapi.testclient import TestClient

    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers
    from app.main import app
    from app.rag.documents import KnowledgeChunk

    async def _intent(_q):
        return "DATA_QUERY"

    async def _recall(_q):
        return [KnowledgeChunk(id="x", text="t", type="schema", source="s", title="t")], "s", "e"

    async def _gen(_q, _chunks, prev_steps):
        return f"SELECT '{prev_steps}' AS ctx", "", []

    async def _run(_sql):
        return 0, []

    async def _report(_q, _rows, _sql):
        return "ok"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=_gen,
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()

    with TestClient(app) as client:
        _post_stream(client, {"question": "查询2026年销售额", "conversation_id": "sse-mt"})
        raw2 = _post_stream(client, {"question": "那2025年呢？", "conversation_id": "sse-mt"})

    events = dict(_parse_events(raw2))
    assert "上一轮用户问题: 查询2026年销售额" in str(events["sql"])


def test_stream_emits_error_event_when_setup_fails(monkeypatch):
    from fastapi.testclient import TestClient

    import app.api.stream as stream_mod
    from app.main import app

    def _broken():
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(stream_mod, "get_compiled_graph", _broken)

    with TestClient(app) as client:
        raw = _post_stream(client, {"question": "查询", "conversation_id": "sse-err2"})

    names = [e for e, _ in _parse_events(raw)]
    assert "error" in names
    assert names[-1] == "done"
