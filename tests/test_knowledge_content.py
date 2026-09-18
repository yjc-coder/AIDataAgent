"""Phase 11 tests — knowledge-base content guards.

The RAG layer has its own unit tests (chunking, retrieval, endpoints) — but
if somebody deletes or renames the sales-rule / table docs, retrieval keeps
working (silently returning garbage). These tests read the REAL
data/knowledge/*.md files and turn such accidents into loud failures.
"""

from __future__ import annotations

from app.rag.documents import load_knowledge_chunks


def test_knowledge_base_is_chunked_and_typed():
    chunks = load_knowledge_chunks()
    assert chunks, "data/knowledge/ produced zero chunks — did the folder move?"
    types = {c.type for c in chunks}
    assert "schema" in types, "no schema knowledge found"
    assert "business" in types, "no business-rule knowledge found"


def test_all_five_business_tables_are_documented():
    text = "\n".join(c.text for c in load_knowledge_chunks() if c.type == "schema")
    for table in ("users", "categories", "products", "orders", "order_items"):
        assert table in text, f"table {table} missing from schema knowledge"


def test_sales_and_refund_business_rules_are_documented():
    text = "\n".join(c.text for c in load_knowledge_chunks() if c.type == "business")
    # 销售额定义 + 退款规则 are the two rules the demo questions rely on.
    assert "销售额" in text, "sales definition missing from business knowledge"
    assert "退款" in text, "refund rule missing from business knowledge"
