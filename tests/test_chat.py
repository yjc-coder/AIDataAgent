"""Chat endpoint tests using a stub LLM (no real API call)."""
from fastapi.testclient import TestClient


def test_chat_returns_stub_answer(client: TestClient):
    r = client.post(
        "/api/chat",
        json={"query": "你好", "conversation_id": "c1"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "hello from stub"
    assert data["conversation_id"] == "c1"
    assert "model" in data


def test_chat_rejects_empty_query(client: TestClient):
    r = client.post("/api/chat", json={"query": "", "conversation_id": "c1"})
    assert r.status_code == 422  # Pydantic min_length=1


def test_chat_rejects_missing_conversation_id(client: TestClient):
    r = client.post("/api/chat", json={"query": "hi"})
    assert r.status_code == 422  # conversation_id required


def test_chat_accepts_optional_agent_id(client: TestClient):
    r = client.post(
        "/api/chat",
        json={
            "query": "列出所有用户",
            "conversation_id": "c2",
            "agent_id": "agent-007",
        },
    )
    assert r.status_code == 200
    assert r.json()["conversation_id"] == "c2"
