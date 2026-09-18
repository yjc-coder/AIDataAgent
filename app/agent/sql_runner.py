"""SQL execution runner.

Phase 4: validates SQL through `security.sql_validator` BEFORE execution.
Phase 3: pass-through. Now sqlglot-validated.

Pipeline:
    raw_sql
        -> validate_and_rewrite()  (raises SqlValidationError on violation)
        -> execute()                (aiomysql.DictCursor)
        -> (rowcount, rows)

The validator is the code-level check; we explicitly do NOT remove any
of the other defences (prompt soft-constraint, LLM structured output,
read-only DB account) — the README's four layers all stay in place.
"""
from __future__ import annotations

from app.database.execute import execute
from app.security.sql_validator import validate_and_rewrite


async def run_select(sql: str, *, max_rows: int = 1000) -> tuple[int, list[dict]]:
    """Validate, then execute a SELECT.

    Args:
        sql:      Raw SQL produced by the LLM (or any caller).
        max_rows: Default LIMIT applied when the SQL doesn't include one.

    Returns:
        (rowcount, rows) — same shape as `app.database.execute.execute`.

    Raises:
        SqlValidationError: if the SQL is not a safe SELECT.
        (Other): DB-layer errors surface as their native exceptions.
    """
    validated = validate_and_rewrite(sql, default_max_rows=max_rows)
    return await execute(validated.sql)


# Module self-reference so consumers (e.g. app.agent.graph) can access
# `run_select` via `sql_runner_module.run_select` for monkeypatchability.
import sys as _sys
sql_runner_module = _sys.modules[__name__]
