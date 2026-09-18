"""DB-layer smoke tests.

Skipped automatically when MySQL isn't reachable. Run with:
    docker compose up -d mysql
    .venv\\Scripts\\python data/seed_data.py
    pytest tests/test_db.py -v
"""
from app.database.execute import execute, execute_scalar
from app.database.repository import Stats


async def test_select_one(db_ready):
    """The cheapest possible round-trip."""
    n = await execute_scalar("SELECT 1")
    assert n == 1


async def test_tables_present(db_ready):
    _, rows = await execute("SHOW TABLES")
    names = {list(r.values())[0] for r in rows}
    assert {
        "users",
        "categories",
        "products",
        "orders",
        "order_items",
    } <= names


async def test_repository_counts_within_expected_range(db_ready):
    """Counts must fall in seeded ranges.

    Seed is approximate by design (deterministic only when `random.seed`
    is fixed), so we check ranges rather than exact numbers.
    """
    users = await Stats.users_count()
    products = await Stats.products_count()
    orders = await Stats.orders_count()

    assert 100 <= users <= 1_000
    assert 30 <= products <= 200
    assert 1_000 <= orders <= 5_000


async def test_top_products_returns_rows(db_ready):
    rows = await Stats.top_products_by_sales(2026, limit=5)
    assert len(rows) == 5
    for r in rows:
        assert {"id", "name", "qty", "sales"} <= set(r.keys())
        assert r["sales"] is not None


async def test_sales_total_is_nonnegative(db_ready):
    val = await Stats.sales_total(year=2025)
    assert val >= 0
