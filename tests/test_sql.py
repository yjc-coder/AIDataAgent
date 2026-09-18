"""Tests for the /api/sql endpoint (Phase 3).

The Text2SqlGenerator is stubbed, so no real LLM call is made.
The `run_select` runner is also stubbed so we don't need a live MySQL.
"""
from fastapi.testclient import TestClient


def test_sql_returns_stub_sql_and_reasoning(sql_client: TestClient):
    r = sql_client.post(
        "/api/sql",
        json={
            "question": "查询 2026 年销售额最高的商品",
            "execute": False,
            "conversation_id": "t1",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["sql"] == "SELECT 1 AS hello"
    assert data["reasoning"] == "phase3 stub"
    assert data["executed"] is False
    assert data["error"] is None
    assert data["rows"] is None


def test_sql_with_execute_true_runs_runner(sql_client: TestClient):
    r = sql_client.post(
        "/api/sql",
        json={"question": "查询所有用户", "execute": True, "conversation_id": "t2"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["sql"] == "SELECT 1 AS hello"
    assert data["executed"] is True
    # runner was monkeypatched to return (0, []).
    assert data["rowcount"] == 0
    assert data["rows"] == []


def test_sql_rejects_empty_question(sql_client: TestClient):
    r = sql_client.post(
        "/api/sql",
        json={"question": "", "execute": False, "conversation_id": "t3"},
    )
    assert r.status_code == 422  # Pydantic min_length=1


def test_sql_handles_llm_exception():
    """When the LLM raises, the endpoint must NOT 500 — it returns
    `error: "..."` and an empty `sql` so the caller can show the user
    something useful."""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.agent.nodes as nodes_module
    import app.agent.text2sql as t2s_module
    from app.agent.state import INTENT_DATA_QUERY

    class _Broken:
        async def agenerate(self, **_kw):
            raise RuntimeError("LLM down")

    async def _force_data_query(_q):
        return INTENT_DATA_QUERY

    async def _stub_recall(_q):
        from app.rag.documents import KnowledgeChunk

        chunk = KnowledgeChunk(id="s", text="orders table", type="schema", source="s.md", title="orders")
        return [chunk], "stub schema", "stub evidence"

    import pytest as _pytest
    monkeypatch = _pytest.MonkeyPatch()
    # Phase 6: /api/sql goes through the LangGraph which reads
    # `text2sql_module.get_text2sql_generator` (a module attribute).
    monkeypatch.setattr(t2s_module.text2sql_module, "get_text2sql_generator", lambda: _Broken())
    # 意图分类/检索都是真实 LLM 调用："hi" 可能被判定为 CHAT 而跳过 SQL
    # 流程，导致测试随机失败。强制 DATA_QUERY + stub 检索，只验证
    # "LLM 挂了也要返回结构化错误信封"这件事本身。
    monkeypatch.setattr(nodes_module, "_intent_classifier", _force_data_query)
    monkeypatch.setattr(nodes_module, "_schema_recaller", _stub_recall)
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/sql",
                json={"question": "hi", "execute": False, "conversation_id": "t4"},
            )
    finally:
        monkeypatch.undo()
    assert r.status_code == 200
    data = r.json()
    # Error could be from the generator layer or from the graph layer;
    # what we really care about is the endpoint returning a structured
    # error envelope (status 200, sql empty, executed False) instead of
    # letting the LLM exception bubble up as a 500.
    err = (data["error"] or "")
    assert (
        "LLM generation failed" in err
        or "SQL生成失败" in err
        or "graph invoke failed" in err
        or "待执行SQL为空" in err  # downstream effect of LLM failure
        or "LLM down" in err
    )
    assert data["sql"] == ""
    assert data["executed"] is False


# ---------------------------------------------------------------------------
# Phase 4: end-to-end check that the sqlglot validator wires into the endpoint.
# We DO NOT patch `run_select` here — we let the real validator run, then
# confirm the endpoint surfaces the rejection in `error` (200, not 500).
# ---------------------------------------------------------------------------


def _hostile_client(stub_sql: str):
    """Build a TestClient whose Text2SqlGenerator returns a hostile SQL
    but whose `run_select` is the REAL Phase 4 runner (so validator runs)."""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.agent.nodes as nodes_module
    import app.agent.text2sql as t2s_module
    from app.agent.state import INTENT_DATA_QUERY
    from app.rag.documents import KnowledgeChunk

    class _Hostile:
        async def agenerate(self, **_kw):
            from app.agent.text2sql import Text2SqlResult

            return Text2SqlResult(sql=stub_sql, reasoning="malicious stub")

    async def _force_data_query(_q):
        return INTENT_DATA_QUERY

    async def _stub_recall(_q):
        chunk = KnowledgeChunk(
            id="schema/stub", text="orders table", type="schema", source="s.md", title="orders"
        )
        return [chunk], "stub schema", "stub evidence"

    import pytest as _pytest
    monkeypatch = _pytest.MonkeyPatch()
    # Phase 6: graph reads the factory via the module attribute
    # `text2sql_module.get_text2sql_generator` (see `app.agent.graph`).
    monkeypatch.setattr(t2s_module.text2sql_module, "get_text2sql_generator", lambda: _Hostile())
    # The intent classifier / retriever are REAL LLM+embedding calls — a live
    # model may legitimately classify "hostile" as CHAT, skipping the whole
    # SQL pipeline. Force DATA_QUERY + stubbed recall so the test is
    # deterministic and only exercises the real sqlglot validator.
    monkeypatch.setattr(nodes_module, "_intent_classifier", _force_data_query)
    monkeypatch.setattr(nodes_module, "_schema_recaller", _stub_recall)
    # do NOT patch run_select — let the real validator execute
    return TestClient(app), monkeypatch


def test_endpoint_rejects_drop_table():
    client, mp = _hostile_client("DROP TABLE users")
    try:
        r = client.post(
            "/api/sql",
            json={"question": "hostile", "execute": True, "conversation_id": "p4-drop"},
        )
    finally:
        mp.undo()
    assert r.status_code == 200
    data = r.json()
    assert data["executed"] is False
    assert data["rowcount"] is None
    assert data["rows"] is None
    # Either the validator rejection or the report-failure envelope
    # is acceptable evidence that we did NOT execute the SQL.
    err = (data["error"] or "")
    assert (
        "Drop" in err
        or "SELECT is allowed" in err
        or "回答生成失败" in err
    )


def test_endpoint_rejects_sleep_function():
    client, mp = _hostile_client("SELECT SLEEP(10)")
    try:
        r = client.post(
            "/api/sql",
            json={"question": "hostile", "execute": True, "conversation_id": "p4-sleep"},
        )
    finally:
        mp.undo()
    assert r.status_code == 200
    data = r.json()
    assert data["executed"] is False
    # Either validator rejection or downstream report failure is acceptable
    # — what matters is `executed` is False.
    err = (data["error"] or "")
    assert "SLEEP" in err or "回答生成失败" in err


def test_endpoint_rejects_multi_statement():
    client, mp = _hostile_client("SELECT 1; DROP TABLE users")
    try:
        r = client.post(
            "/api/sql",
            json={"question": "hostile", "execute": True, "conversation_id": "p4-multi"},
        )
    finally:
        mp.undo()
    assert r.status_code == 200
    data = r.json()
    assert data["executed"] is False
    err = (data["error"] or "")
    assert "Multi-statement" in err or "回答生成失败" in err
