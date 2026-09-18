"""Agent sub-package.

Phase 3 surface area:
    - `prompts`           — load templates from app/prompts/*.txt
    - `business_knowledge`— load rule markdowns from data/knowledge/**/*.md
    - `text2sql`          — Text2SqlGenerator (LLM -> SqlGeneration)
    - `sql_runner`        — thin SQL execution wrapper (Phase 4 injects sqlglot)

The same `data/knowledge/*.md` files are later vectorised by Phase 5 RAG,
so the source-of-truth is written ONCE — prompt injection today,
semantic retrieval tomorrow.
"""
from app.agent.sql_runner import run_select
from app.agent.text2sql import SqlGeneration, Text2SqlGenerator

__all__ = ["SqlGeneration", "Text2SqlGenerator", "run_select"]
