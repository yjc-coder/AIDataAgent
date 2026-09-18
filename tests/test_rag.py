"""Tests for the Phase 5 RAG pipeline.

Covers three layers independently:
    1. documents.py  — load + chunk from data/knowledge/
    2. vector_store.py — Chroma round-trip with a deterministic EF
    3. retriever.py   — partition into schema vs business

Plus a thin endpoint-level check that `retrieved_chunks` shows up in
the /api/sql response when the generator is stubbed to carry chunks.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1) documents.py
# ---------------------------------------------------------------------------


def test_load_knowledge_chunks_returns_one_per_section():
    from app.rag.documents import KNOWLEDGE_DIR, load_knowledge_chunks

    if not KNOWLEDGE_DIR.exists():
        pytest.skip(f"knowledge dir missing: {KNOWLEDGE_DIR}")

    chunks = load_knowledge_chunks()
    # 5 tables in ecommerce.md + sales + refund = 7 sections minimum.
    assert len(chunks) >= 7, f"expected >=7 chunks, got {len(chunks)}"

    types = {c.type for c in chunks}
    assert "schema" in types
    assert "business" in types

    # Every chunk must have a stable id, a type, and a non-empty text.
    for c in chunks:
        assert c.id
        assert c.text
        assert c.type in {"schema", "business"}


def test_load_knowledge_chunks_idempotent_when_dir_empty(tmp_path: Path, monkeypatch):
    """If the user deletes data/knowledge/, the loader must not crash."""
    from app.rag import documents as docs_mod

    monkeypatch.setattr(docs_mod, "KNOWLEDGE_DIR", tmp_path)
    assert docs_mod.load_knowledge_chunks() == []


def test_split_markdown_into_sections_keeps_pre_h2_content():
    from app.rag.documents import _split_markdown_into_sections

    text = "# Title\npreamble line\n\n## A\nbody A\n\n## B\nbody B\n"
    sections = _split_markdown_into_sections(text)
    # Three sections:
    #   [0] preamble (before first `##`)
    #   [1] `## A` and its body
    #   [2] `## B` and its body
    # Note: the splitter saves the *current* buffer when it hits a `##` line,
    # so `## A` is the *first* line of section [1], not section [0].
    assert len(sections) == 3
    assert "preamble line" in sections[0]
    assert "# Title" in sections[0]
    assert "## A" in sections[1]
    assert "body A" in sections[1]
    assert sections[1].startswith("## A")
    assert sections[2].startswith("## B")
    assert "body B" in sections[2]


# ---------------------------------------------------------------------------
# 2) vector_store.py — round-trip with a deterministic EF
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_chroma_dir(tmp_path: Path):
    """Yield a temp dir; ensure it's wiped after the test."""
    d = tmp_path / "chroma_test"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class _DeterministicEF:
    """Tiny EF that returns a fixed-dim hash-based vector. Compatible with
    chromadb 1.5.x (has `name`, `embed_query`, `embed_documents`, `is_legacy`)."""

    _DIM = 8
    is_legacy = False

    def name(self) -> str:
        return "deterministic-test-v1"

    def _vec(self, text: str) -> list[float]:
        import hashlib, random

        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        return [rng.random() for _ in range(self._DIM)]

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return [self._vec(t) for t in input]

    def embed_documents(self, input):
        if isinstance(input, str):
            input = [input]
        return [self._vec(t) for t in input]

    def __call__(self, input):
        return self.embed_documents(input)


def test_vector_store_round_trip_with_deterministic_ef(isolated_chroma_dir):
    from app.rag.documents import KnowledgeChunk
    from app.rag.vector_store import VectorStore

    chunks = [
        KnowledgeChunk(
            id="schema/orders::001::orders",
            text="orders table contains order_id, user_id, amount",
            type="schema",
            source="schema/orders.md",
            title="orders",
            metadata={"type": "schema", "source": "schema/orders.md", "title": "orders"},
        ),
        KnowledgeChunk(
            id="business/sales::001::sales",
            text="sales = sum(amount) where status='paid'",
            type="business",
            source="business/sales.md",
            title="sales",
            metadata={"type": "business", "source": "business/sales.md", "title": "sales"},
        ),
    ]

    store = VectorStore(
        persist_dir=str(isolated_chroma_dir),
        name="test_1",
        embedding_fn=_DeterministicEF(),
    )
    assert store.is_empty()

    store.add(chunks)
    assert store.count() == 2

    # Re-adding the same chunks must be a no-op (idempotent).
    store.add(chunks)
    assert store.count() == 2

    # Query for the orders chunk.
    results = store.query("orders", top_k=1)
    assert len(results) == 1
    assert results[0].type == "schema"
    assert "orders" in results[0].text


def test_hash_embedding_fallback_works_offline(isolated_chroma_dir):
    """`get_vector_store()` falls back to hash EF when no real key is set."""
    from app.rag.documents import KnowledgeChunk
    from app.rag.vector_store import VectorStore, _HashEmbedding

    store = VectorStore(
        persist_dir=str(isolated_chroma_dir),
        name="test_2",
        embedding_fn=_HashEmbedding(),
    )
    store.add(
        [
            KnowledgeChunk(
                id="x",
                text="杭州销售额",
                type="schema",
                source="t.md",
                title="t",
                metadata={"type": "schema", "source": "t.md", "title": "t"},
            )
        ]
    )
    out = store.query("杭州销售额", top_k=1)
    assert out and out[0].id == "x"


# ---------------------------------------------------------------------------
# 3) retriever.py — partition & formatting
# ---------------------------------------------------------------------------


def test_knowledge_retriever_partitions_schema_vs_business(monkeypatch):
    from app.rag.documents import KnowledgeChunk
    from app.rag.retriever import KnowledgeRetriever
    from app.rag import retriever as retriever_module

    class _FakeVS:
        def __init__(self, chunks):
            self._chunks = chunks

        def query(self, _text, top_k):
            return self._chunks[:top_k]

    chunks = [
        KnowledgeChunk(
            id="schema/u",
            text="users table has columns user_id, city",
            type="schema",
            source="schema/users.md",
            title="users",
        ),
        KnowledgeChunk(
            id="business/s",
            text="sales rule",
            type="business",
            source="business/sales.md",
            title="sales",
        ),
    ]
    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: _FakeVS(chunks))
    r = KnowledgeRetriever(top_k=2)
    out = r.recall("anything")

    assert "users table" in out.schema_text
    assert "sales rule" in out.evidence_text
    assert out.evidence_text  # not empty


def test_retriever_returns_empty_when_collection_empty(monkeypatch):
    from app.rag.retriever import KnowledgeRetriever
    from app.rag import retriever as retriever_module

    class _EmptyVS:
        def query(self, _text, top_k):
            return []

        def count(self):
            return 0

    monkeypatch.setattr(retriever_module, "get_vector_store", lambda: _EmptyVS())
    r = KnowledgeRetriever(top_k=3)
    out = r.recall("x")
    assert "no schema knowledge retrieved" in out.schema_text
    assert "no business knowledge retrieved" in out.evidence_text
    assert out.chunks == []


# ---------------------------------------------------------------------------
# 4) endpoint-level — retrieved_chunks surfaces in /api/sql response
# ---------------------------------------------------------------------------


def test_endpoint_surfaces_retrieved_chunks(sql_client_with_rag):
    r = sql_client_with_rag.post(
        "/api/sql",
        json={
            "question": "查询 2026 年销售额最高的商品",
            "execute": False,
            "conversation_id": "p5-t1",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "retrieved_chunks" in data
    assert isinstance(data["retrieved_chunks"], list)
    assert len(data["retrieved_chunks"]) >= 1
    chunk = data["retrieved_chunks"][0]
    assert {"id", "title", "source", "type"} <= set(chunk.keys())
    assert chunk["type"] == "schema"