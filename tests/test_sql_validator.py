"""Tests for `app.security.sql_validator` (Phase 4).

Pure logic, no LLM, no MySQL. Every assertion is a sqlglot-level fact.
"""
import pytest

from app.security.sql_validator import (
    SqlValidationError,
    validate_and_rewrite,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_simple_select_passes_and_injects_limit():
    v = validate_and_rewrite("SELECT 1")
    assert v.limit_added is True
    assert "SELECT 1" in v.sql
    assert "LIMIT 1000" in v.sql


def test_cte_select_passes():
    sql = "WITH cte AS (SELECT 1 AS x) SELECT * FROM cte"
    v = validate_and_rewrite(sql)
    assert v.limit_added is True
    assert "LIMIT 1000" in v.sql


def test_existing_limit_preserved():
    v = validate_and_rewrite("SELECT 1 LIMIT 5")
    assert v.limit_added is False
    assert "LIMIT 5" in v.sql
    assert "1000" not in v.sql


def test_trailing_semicolon_stripped():
    v = validate_and_rewrite("SELECT 1;")
    assert v.sql.endswith("LIMIT 1000")
    assert ";" not in v.sql


def test_custom_default_max_rows():
    v = validate_and_rewrite("SELECT 1", default_max_rows=42)
    assert v.limit_added is True
    assert "LIMIT 42" in v.sql


def test_safe_aggregates_allowed():
    v = validate_and_rewrite("SELECT SUM(amount), COUNT(*), MAX(id) FROM t")
    assert v.limit_added is True


def test_join_with_where_allowed():
    sql = (
        "SELECT p.name, SUM(oi.amount) AS sales "
        "FROM order_items oi "
        "JOIN orders o ON o.id = oi.order_id "
        "JOIN products p ON p.id = oi.product_id "
        "WHERE o.status = 'paid' AND YEAR(o.created_at) = 2026 "
        "GROUP BY p.name ORDER BY sales DESC"
    )
    v = validate_and_rewrite(sql)
    assert v.limit_added is True
    assert "GROUP BY" in v.sql.upper()


# ---------------------------------------------------------------------------
# Rejection: empty
# ---------------------------------------------------------------------------


def test_empty_sql_rejected():
    with pytest.raises(SqlValidationError, match="empty"):
        validate_and_rewrite("")


def test_whitespace_only_rejected():
    with pytest.raises(SqlValidationError, match="empty"):
        validate_and_rewrite("   \n\t  ")


# ---------------------------------------------------------------------------
# Rejection: write / management statements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE users",
        "INSERT INTO users (name) VALUES ('a')",
        "UPDATE users SET name='x'",
        "DELETE FROM users",
        "MERGE INTO x USING y ON x.id = y.id",
        "CREATE TABLE x (id INT)",
        "ALTER TABLE x ADD COLUMN y INT",
        "TRUNCATE TABLE users",
        "GRANT ALL ON x TO y",
        "REVOKE ALL ON x FROM y",
    ],
)
def test_write_statements_rejected(sql):
    with pytest.raises(SqlValidationError):
        validate_and_rewrite(sql)


# ---------------------------------------------------------------------------
# Rejection: dangerous functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SLEEP(5)",
        "SELECT BENCHMARK(1000000, MD5(1))",
        "SELECT LOAD_FILE('/etc/passwd')",
    ],
)
def test_dangerous_functions_rejected(sql):
    with pytest.raises(SqlValidationError, match="function"):
        validate_and_rewrite(sql)


# ---------------------------------------------------------------------------
# Rejection: multi-statement
# ---------------------------------------------------------------------------


def test_multi_statement_rejected():
    with pytest.raises(SqlValidationError, match="Multi-statement"):
        validate_and_rewrite("SELECT 1; SELECT 2")


def test_multi_statement_with_drop_rejected():
    """The classic injection pattern: 'SELECT 1; DROP TABLE users'."""
    with pytest.raises(SqlValidationError, match="Multi-statement"):
        validate_and_rewrite("SELECT 1; DROP TABLE users")


# ---------------------------------------------------------------------------
# Rejection: parse error
# ---------------------------------------------------------------------------


def test_unparseable_sql_rejected():
    with pytest.raises(SqlValidationError, match="parse failed"):
        validate_and_rewrite("SELECT FROM WHERE")
