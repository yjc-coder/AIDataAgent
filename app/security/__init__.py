"""Security sub-package.

Phase 4: `sql_validator` (sqlglot-based SQL safety check).

The README's "Four layers of defence" enumerates this as the **code-level**
check (the other three are: prompt soft-constraint, LLM structured output,
external read-only DB account).
"""
from app.security.sql_validator import (
    SqlValidationError,
    ValidatedSql,
    validate_and_rewrite,
)

__all__ = ["SqlValidationError", "ValidatedSql", "validate_and_rewrite"]
