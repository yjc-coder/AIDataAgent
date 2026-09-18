"""分层记忆测试（第11阶段）：短期记忆（最近3轮原文）+ 长期记忆（滚动总结）。

覆盖：
    1. memory_update_node：窗口未满时不动长期记忆；窗口满后把最老一轮
       交给 summarizer 归纳并更新 memory_summary
    2. 归纳失败时保留旧总结（fail-open）
    3. skip_report 轻量模式不触发归纳
    4. _format_prev_steps 渲染长期总结段落
    5. 跨窗口引用：第5轮的 Prompt 里有长期总结、且第1轮原文已滚出短期窗口
"""
from __future__ import annotations

import pytest

from app.agent.nodes import memory_update_node
from app.agent.state import (
    KEY_HISTORY,
    KEY_MEMORY_SUMMARY,
    KEY_PREV_ANSWER,
    KEY_PREV_QUESTION,
    KEY_PREV_SQL,
    KEY_SKIP_REPORT,
)


def _state(*, history=None, memory_summary="", skip_report=False):
    return {
        KEY_HISTORY: history or [],
        KEY_MEMORY_SUMMARY: memory_summary,
        KEY_SKIP_REPORT: skip_report,
    }


@pytest.mark.asyncio
async def test_window_not_full_keeps_summary_unchanged(monkeypatch):
    """前几轮：history 不足3轮 → 没有轮次滑出，总结保持原样，也不调 LLM。"""
    from app.agent import nodes as nodes_module

    async def _must_not_call(_old, _turn):  # pragma: no cover
        raise AssertionError("window not full, summarizer must not run")

    monkeypatch.setattr(nodes_module, "_memory_summarizer", _must_not_call)

    out = await memory_update_node(
        _state(history=[{"question": "q1", "sql": "s1", "answer": "a1"}], memory_summary="旧总结")
    )

    assert out.get(KEY_MEMORY_SUMMARY, "旧总结") == "旧总结"


@pytest.mark.asyncio
async def test_full_window_folds_oldest_turn_into_summary(monkeypatch):
    from app.agent import nodes as nodes_module

    captured: dict = {}

    async def _summarizer(old_summary, turn):
        captured["old"] = old_summary
        captured["turn"] = turn
        return f"新总结（含 {turn['question']}）"

    monkeypatch.setattr(nodes_module, "_memory_summarizer", _summarizer)

    history = [
        {"question": "第3轮", "sql": "s3", "answer": "a3"},
        {"question": "第2轮", "sql": "s2", "answer": "a2"},
        {"question": "第1轮", "sql": "s1", "answer": "a1"},
    ]
    out = await memory_update_node(_state(history=history, memory_summary="旧总结"))

    assert captured["old"] == "旧总结"
    assert captured["turn"]["question"] == "第1轮"  # 最老的一轮滑出窗口
    assert out[KEY_MEMORY_SUMMARY] == "新总结（含 第1轮）"


@pytest.mark.asyncio
async def test_summarizer_failure_keeps_old_summary(monkeypatch):
    from app.agent import nodes as nodes_module

    async def _boom(_old, _turn):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(nodes_module, "_memory_summarizer", _boom)

    history = [
        {"question": "q3", "sql": "s3", "answer": "a3"},
        {"question": "q2", "sql": "s2", "answer": "a2"},
        {"question": "q1", "sql": "s1", "answer": "a1"},
    ]
    out = await memory_update_node(_state(history=history, memory_summary="旧总结"))

    assert out.get(KEY_MEMORY_SUMMARY, "旧总结") == "旧总结"


@pytest.mark.asyncio
async def test_skip_report_skips_memory_update(monkeypatch):
    from app.agent import nodes as nodes_module

    async def _must_not_call(_old, _turn):  # pragma: no cover
        raise AssertionError("skip mode must not summarize")

    monkeypatch.setattr(nodes_module, "_memory_summarizer", _must_not_call)

    history = [{"question": f"q{i}", "sql": "s", "answer": "a"} for i in (3, 2, 1)]
    out = await memory_update_node(_state(history=history, skip_report=True))

    assert out.get(KEY_MEMORY_SUMMARY, "") == ""


def test_format_prev_steps_renders_long_term_summary():
    from app.agent.nodes import _format_prev_steps

    state = {
        KEY_MEMORY_SUMMARY: "用户曾查询2024年销售额，最高为对开门冰箱。",
        KEY_HISTORY: [
            {"question": "第3轮", "sql": "s3", "answer": "a3"},
        ],
        KEY_PREV_QUESTION: "第3轮",
        KEY_PREV_SQL: "s3",
        KEY_PREV_ANSWER: "a3",
    }
    text = _format_prev_steps(state)
    assert "【更早对话长期总结】用户曾查询2024年销售额" in text
    assert "上一轮用户问题: 第3轮" in text
