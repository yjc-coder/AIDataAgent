import pytest


@pytest.mark.asyncio
async def test_debug_mem(monkeypatch):
    from app.agent import nodes as nodes_module

    captured = {}

    async def _summarizer(old, turn):
        captured["old"] = old
        return "新总结"

    monkeypatch.setattr(nodes_module, "_memory_summarizer", _summarizer)
    print("AFTER SETATTR:", nodes_module._memory_summarizer)

    state = {
        "history": [
            {"question": "q3", "sql": "s3", "answer": "a3"},
            {"question": "q2", "sql": "s2", "answer": "a2"},
            {"question": "q1", "sql": "s1", "answer": "a1"},
        ],
        "memory_summary": "旧总结",
    }
    out = await nodes_module.memory_update_node(state)
    print("CAPTURED:", captured)
    print("OUT:", out)
    assert captured.get("old") == "旧总结"
