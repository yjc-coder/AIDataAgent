"""LangChain tools exposed to the LLM (Phase 7).

Only two tools, on purpose:

    search_schema(keyword)  — RAG lookup of schema / business knowledge
    execute_sql(sql)        — validated, read-only SELECT against MySQL

Design rule: the LLM *decides* which tool to call (via bind_tools), but
LangGraph nodes *execute* the tools and stay in control of the workflow.
There is no autonomous agent loop anywhere in this app.

NOTE: the tools are re-exported under ``*_tool`` aliases so the package
attribute ``app.tools.search_schema`` keeps pointing at the *module* (same
for execute_sql) — avoids the classic shadowing trap for tests.
"""

from app.tools.execute_sql import execute_sql as execute_sql_tool
from app.tools.search_schema import (
    chunks_from_tool_output,
    search_schema as search_schema_tool,
)

__all__ = ["execute_sql_tool", "search_schema_tool", "chunks_from_tool_output"]
