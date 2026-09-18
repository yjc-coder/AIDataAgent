"""Phase 10 tests — the Vue3 frontend is served by FastAPI."""

from __future__ import annotations


def test_root_serves_chat_page():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert 'id="app"' in body          # Vue mount point
    assert "chat/stream" in body       # frontend consumes the SSE endpoint
    assert "vue.global" in body        # Vue3 global build via CDN


def test_api_routes_take_precedence_over_static_mount():
    """StaticFiles is mounted at '/' but must not shadow the API."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/graph/structure").status_code == 200
