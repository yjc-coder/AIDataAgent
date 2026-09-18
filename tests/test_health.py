"""Health endpoint must respond 200 with a predictable payload — no LLM call."""
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok():
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "ai-data-agent"
    assert data["phase"] == "1"


def test_health_does_not_require_llm():
    """Even without an API key, /health must work — it's purely local."""
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
