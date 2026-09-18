"""Phase 8 tests — multi-turn conversation context.

The checkpointer (MemorySaver, keyed by conversation_id) already persists
state across turns. Phase 8 makes run_question *read* that state and inject
prev_question/prev_sql/prev_answer, so sql_generate's Prompt can resolve
references like "查询2026年销售额" → "那2025年呢?".

All tests are offline: LLM / retriever / runner are stubbed.
"""

from __future__ import annotations

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


class _CapturingT2S:
    """Text2SqlGenerator stand-in that records every agenerate() call."""

    def __init__(self, sql: str = "SELECT 1 AS v") -> None:
        self._sql = sql
        self.calls: list[dict] = []

    async def agenerate(self, **kwargs):
        from app.agent.text2sql import Text2SqlResult

        self.calls.append(kwargs)
        return Text2SqlResult(sql=self._sql, reasoning="", retrieved_chunks=[])


def _wire_graph(monkeypatch, captor: _CapturingT2S) -> None:
    """Stub intent/recall/run/report; keep the production _generate_sql so
    the run_question → nodes → graph._generate_sql → captor path is real."""
    from app.agent import graph as graph_mod
    from app.agent.graph import reset_compiled_graph
    from app.agent.nodes import set_helpers
    from app.rag.documents import KnowledgeChunk

    async def _intent(_q):
        return "DATA_QUERY"

    async def _recall(_q):
        return [KnowledgeChunk(id="x", text="t", type="schema", source="s", title="t")], "s", "e"

    async def _run(_sql):
        return 1, [{"x": 1}]

    async def _report(_q, _rows, _sql):
        return "answer"

    set_helpers(
        intent_classifier=_intent,
        schema_recaller=_recall,
        sql_generator=graph_mod._generate_sql,  # real — delegates to the captor
        sql_runner=_run,
        report_generator=_report,
    )
    reset_compiled_graph()

    import app.agent.text2sql as t2s_module

    monkeypatch.setattr(t2s_module, "get_text2sql_generator", lambda: captor)


# ---------------------------------------------------------------------------
# run_question level — the real multi-turn behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_turn_has_no_previous_context(monkeypatch):
    from app.agent.run_graph import run_question

    captor = _CapturingT2S()
    _wire_graph(monkeypatch, captor)

    res = await run_question("查询2026年销售额", "conv-mt-1")
    assert res.error is None
    assert len(captor.calls) == 1
    assert captor.calls[0]["previous_steps"] == "(none)"


@pytest.mark.asyncio
async def test_second_turn_receives_previous_question_and_sql(monkeypatch):
    """查询2026年销售额 → 那2025年呢? — 第二轮必须带上第一轮上下文。"""
    from app.agent.run_graph import run_question

    captor = _CapturingT2S()
    _wire_graph(monkeypatch, captor)

    await run_question("查询2026年销售额", "conv-mt-2")
    res = await run_question("那2025年呢？", "conv-mt-2")

    assert res.error is None
    assert len(captor.calls) == 2
    prev = captor.calls[1]["previous_steps"]
    assert "上一轮用户问题: 查询2026年销售额" in prev
    assert f"上一轮生成SQL: {captor._sql}" in prev
    assert "上一轮返回答案: answer" in prev


@pytest.mark.asyncio
async def test_different_conversations_do_not_leak_context(monkeypatch):
    from app.agent.run_graph import run_question

    captor = _CapturingT2S()
    _wire_graph(monkeypatch, captor)

    await run_question("查询2026年销售额", "conv-a")
    await run_question("那2025年呢？", "conv-b")  # different thread

    assert captor.calls[1]["previous_steps"] == "(none)"


@pytest.mark.asyncio
async def test_third_turn_carries_both_earlier_turns(monkeypatch):
    """【跨多轮指代】第三轮同时看到第二轮（上一轮）和第一轮（更早一轮）。"""
    from app.agent.run_graph import run_question

    captor = _CapturingT2S(sql="SELECT 2 AS v")
    _wire_graph(monkeypatch, captor)

    await run_question("q1", "conv-mt-3")
    await run_question("q2", "conv-mt-3")
    await run_question("q3", "conv-mt-3")

    prev = captor.calls[2]["previous_steps"]
    assert "上一轮用户问题: q2" in prev
    assert "更早一轮用户问题: q1" in prev  # 跨三轮引用：第一轮上下文仍在场


# ---------------------------------------------------------------------------
# node level — _format_prev_steps rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sql_generate_node_formats_previous_steps():
    from app.agent import nodes
    from app.agent.state import AgentState, KEY_PREV_ANSWER, KEY_PREV_QUESTION, KEY_PREV_SQL

    seen: dict = {}

    async def _gen(question, chunks, prev_steps):
        seen["prev"] = prev_steps
        return "SELECT 1", "", []

    nodes.set_helpers(
        intent_classifier=None,
        schema_recaller=None,
        sql_generator=_gen,
        sql_runner=None,
        report_generator=None,
    )

    state: AgentState = {
        "question": "那2025年呢？",
        KEY_PREV_QUESTION: "查询2026年销售额",
        KEY_PREV_SQL: "SELECT SUM(amount) FROM orders",
        KEY_PREV_ANSWER: "2026年销售额为100万",
    }
    await nodes.sql_generate_node(state)
    assert "上一轮用户问题: 查询2026年销售额" in seen["prev"]
    assert "上一轮生成SQL: SELECT SUM(amount) FROM orders" in seen["prev"]
    assert "上一轮返回答案: 2026年销售额为100万" in seen["prev"]


@pytest.mark.asyncio
async def test_sql_generate_node_first_turn_prev_is_none():
    from app.agent import nodes
    from app.agent.state import AgentState

    seen: dict = {}

    async def _gen(question, chunks, prev_steps):
        seen["prev"] = prev_steps
        return "SELECT 1", "", []

    nodes.set_helpers(
        intent_classifier=None,
        schema_recaller=None,
        sql_generator=_gen,
        sql_runner=None,
        report_generator=None,
    )

    state: AgentState = {"question": "查询销售额"}
    await nodes.sql_generate_node(state)
    assert seen["prev"] == "(none)"


# ---------------------------------------------------------------------------
# Text2SqlGenerator level — history lands in the rendered prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text2sql_prompt_receives_history():
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from app.agent.text2sql import Text2SqlGenerator
    from app.rag.documents import KnowledgeChunk

    captured: dict = {}

    async def _capture(prompt_value):
        captured["value"] = prompt_value
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "SqlGeneration", "args": {"sql": "SELECT 1", "reasoning": ""}, "id": "t1", "type": "tool_call"}
            ],
        )

    class _ChatStub:
        def bind_tools(self, _tools, **_kwargs):
            return RunnableLambda(_capture)

    gen = Text2SqlGenerator(chat_model=_ChatStub())
    chunk = KnowledgeChunk(id="s", text="orders table", type="schema", source="s.md", title="orders")

    await gen.agenerate(
        question="那2025年呢？",
        previous_steps="上一轮用户问题: 查询2026年销售额",
        chunks=[chunk],  # skip internal RAG recall
    )

    text = captured["value"].to_string()
    assert "上一轮用户问题: 查询2026年销售额" in text
    assert "那2025年呢？" in text
    assert "orders table" in text  # tool-provided chunk reached the schema slot


# ---------------------------------------------------------------------------
# 跨多轮指代（第11阶段）—— 查询2026年销售额 → 那2025年呢 → 那前年呢
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_third_turn_receives_three_turns_of_context(monkeypatch):
    """第三轮"那前年呢？"必须同时看到前两轮的问题/SQL/答案。"""
    from app.agent.run_graph import run_question

    captor = _CapturingT2S()
    _wire_graph(monkeypatch, captor)

    conv = "conv-mt-cross3"
    await run_question("查询2026年销售额", conv)
    await run_question("那2025年呢？", conv)
    res = await run_question("那前年呢？", conv)

    assert res.error is None
    assert len(captor.calls) == 3
    prev = captor.calls[2]["previous_steps"]

    # 最近一轮
    assert "上一轮用户问题: 那2025年呢？" in prev
    # 更早的两轮也在场（跨三轮引用的核心）
    assert "更早一轮用户问题: 查询2026年销售额" in prev
    # 提示词包含跨轮指代引导
    assert "前年" in prev


@pytest.mark.asyncio
async def test_history_is_capped_at_three_turns(monkeypatch):
    """上限=最近3轮：第4轮能看到第3/2/1轮；第5轮起第1轮被截断。"""
    from app.agent.run_graph import run_question

    captor = _CapturingT2S()
    _wire_graph(monkeypatch, captor)

    conv = "conv-mt-cap"
    await run_question("查询第1个问题", conv)
    await run_question("查询第2个问题", conv)
    await run_question("查询第3个问题", conv)
    await run_question("查询第4个问题", conv)
    res = await run_question("查询第5个问题", conv)

    assert res.error is None
    prev = captor.calls[4]["previous_steps"]
    assert "查询第4个问题" in prev   # 上一轮
    assert "查询第3个问题" in prev   # 更早一轮
    assert "查询第2个问题" in prev   # 更早两轮
    assert "查询第1个问题" not in prev  # 超出3轮上限，最早一轮被截断
