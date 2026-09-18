"""Seed the demo MySQL database with realistic mock data.

Run from project root:
    .venv\\Scripts\\python data/seed_data.py

It will:
    1. Connect using app.config (DB_HOST/USER/PASSWORD/...),
    2. TRUNCATE the demo tables,
    3. Insert ~200 users, 8 categories, ~60 products,
       ~2000 orders, ~5000 order_items spread across 2024-2026.

The target size is large enough that 销售额 / 销量 / 月度 等 aggregations
feel real, while still seeding in a few seconds.

Re-running this script resets and re-seeds (safe for demos).
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal

import aiomysql

from app.config import get_settings


# ---------------------------------------------------------------------------
# Data dictionaries (Chinese-friendly e-commerce demo)
# ---------------------------------------------------------------------------

CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉",
    "南京", "西安", "重庆", "苏州", "天津", "长沙", "青岛",
    "厦门", "宁波", "郑州", "济南", "合肥", "福州",
]

CATEGORIES = [
    ("电子产品", "手机、电脑、智能设备等"),
    ("家用电器", "冰箱、洗衣机、空调等"),
    ("服饰鞋帽", "衣服、鞋、帽子等"),
    ("美妆护肤", "化妆品、护肤品等"),
    ("食品饮料", "零食、酒水、饮料"),
    ("图书音像", "图书、音像制品"),
    ("运动户外", "运动器材、户外装备"),
    ("母婴用品", "婴儿用品、玩具等"),
]

PRODUCTS = {
    "电子产品": [
        ("智能手机 Pro", 5999.00),
        ("轻薄笔记本", 7299.00),
        ("无线耳机", 899.00),
        ("智能手表", 1999.00),
        ("平板电脑", 3299.00),
        ("蓝牙音箱", 499.00),
        ("电竞键盘", 699.00),
        ("4K 显示器", 2299.00),
    ],
    "家用电器": [
        ("变频空调", 3299.00),
        ("对开门冰箱", 4999.00),
        ("滚筒洗衣机", 2899.00),
        ("电热水器", 1299.00),
        ("智能扫地机", 1799.00),
        ("空气净化器", 1599.00),
    ],
    "服饰鞋帽": [
        ("经典 T 恤", 129.00),
        ("商务衬衫", 299.00),
        ("牛仔裤", 399.00),
        ("冲锋衣", 899.00),
        ("运动鞋", 599.00),
        ("鸭舌帽", 89.00),
    ],
    "美妆护肤": [
        ("保湿面霜", 269.00),
        ("防晒霜", 159.00),
        ("口红", 199.00),
        ("精华液", 599.00),
        ("卸妆水", 99.00),
    ],
    "食品饮料": [
        ("坚果礼盒", 168.00),
        ("进口巧克力", 89.00),
        ("矿泉水 24 瓶", 39.00),
        ("红酒 750ml", 299.00),
        ("速溶咖啡", 79.00),
    ],
    "图书音像": [
        ("Python 编程从入门到实践", 69.00),
        ("深入理解计算机系统", 139.00),
        ("三体套装", 99.00),
        ("经济学原理", 88.00),
    ],
    "运动户外": [
        ("瑜伽垫", 119.00),
        ("哑铃 20kg", 269.00),
        ("跑步机", 3999.00),
        ("登山杖", 199.00),
        ("帐篷", 599.00),
    ],
    "母婴用品": [
        ("婴儿推车", 1299.00),
        ("奶粉 1 段", 299.00),
        ("婴儿湿巾", 49.00),
        ("儿童积木", 159.00),
    ],
}

ORDER_STATUS_WEIGHTS = [("paid", 0.80), ("refunded", 0.10), ("cancelled", 0.10)]


def _weighted_status() -> str:
    r = random.random()
    cum = 0.0
    for s, w in ORDER_STATUS_WEIGHTS:
        cum += w
        if r < cum:
            return s
    return "paid"


def _random_created_at(year_low: int = 2024, year_high: int = 2026) -> datetime:
    """Random timestamp between 2024-01-01 and end of `year_high`."""
    start = datetime(year_low, 1, 1, 0, 0, 0)
    end = datetime(year_high, 12, 31, 23, 59, 59)
    delta_seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, delta_seconds))


# ---------------------------------------------------------------------------
# Insertion helpers (executemany for speed)
# ---------------------------------------------------------------------------


async def _truncate(cur: aiomysql.Cursor) -> None:
    """Wipe in FK-safe order. Disable FK checks for the duration."""
    await cur.execute("SET FOREIGN_KEY_CHECKS=0")
    for table in ("order_items", "orders", "products", "categories", "users"):
        await cur.execute(f"TRUNCATE TABLE {table}")
    await cur.execute("SET FOREIGN_KEY_CHECKS=1")


async def _insert_users(cur: aiomysql.Cursor, n: int = 200) -> list[int]:
    rows = [
        (
            f"user_{i:04d}",
            f"user_{i:04d}@demo.local",
            random.choice(CITIES),
            _random_created_at(2023, 2025),
        )
        for i in range(1, n + 1)
    ]
    await cur.executemany(
        "INSERT INTO users (username, email, city, created_at) VALUES (%s, %s, %s, %s)",
        rows,
    )
    return list(range(1, n + 1))


async def _insert_categories(cur: aiomysql.Cursor) -> list[int]:
    rows = [(name, desc) for name, desc in CATEGORIES]
    await cur.executemany(
        "INSERT INTO categories (name, description) VALUES (%s, %s)", rows
    )
    # IDs are auto-increment, starting at 1 in fresh truncate.
    return list(range(1, len(rows) + 1))


async def _insert_products(
    cur: aiomysql.Cursor, cat_ids: list[int]
) -> list[tuple[int, str, Decimal]]:
    rows: list[tuple] = []
    product_meta: list[tuple[int, str, Decimal]] = []
    pid = 1
    for cat_id, cat_name in zip(cat_ids, [c[0] for c in CATEGORIES]):
        for name, price in PRODUCTS[cat_name]:
            rows.append(
                (
                    name,
                    cat_id,
                    f"{price:.2f}",
                    random.randint(0, 500),
                    "on_sale",
                    _random_created_at(2023, 2024),
                )
            )
            product_meta.append((pid, name, Decimal(str(price))))
            pid += 1
    await cur.executemany(
        "INSERT INTO products (name, category_id, price, stock, status, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        rows,
    )
    return product_meta


async def _insert_orders_and_items(
    cur: aiomysql.Cursor,
    user_ids: list[int],
    products: list[tuple[int, str, Decimal]],
    n_orders: int = 2000,
) -> None:
    order_rows: list[tuple] = []
    item_rows: list[tuple] = []
    for oid in range(1, n_orders + 1):
        user_id = random.choice(user_ids)
        n_items = random.choices([1, 2, 3, 4, 5], weights=[35, 30, 20, 10, 5])[0]
        total = Decimal("0.00")
        for _ in range(n_items):
            pid, _, price = random.choice(products)
            qty = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5])[0]
            unit_price = price
            amount = unit_price * qty
            item_rows.append((oid, pid, qty, f"{unit_price:.2f}", f"{amount:.2f}"))
            total += amount
        status = _weighted_status()
        order_rows.append(
            (oid, user_id, f"{total:.2f}", status, _random_created_at(2024, 2026))
        )
    # executemany is fastest for large batches.
    # Split into chunks just to keep the wire easy to read on errors.
    CHUNK = 500
    for i in range(0, len(order_rows), CHUNK):
        await cur.executemany(
            "INSERT INTO orders (id, user_id, total_amount, status, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            order_rows[i : i + CHUNK],
        )
    for i in range(0, len(item_rows), CHUNK):
        await cur.executemany(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price, amount) "
            "VALUES (%s, %s, %s, %s, %s)",
            item_rows[i : i + CHUNK],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    s = get_settings()
    print(
        f"Connecting to MySQL {s.db_host}:{s.db_port} as {s.db_user} db={s.db_name} ..."
    )
    conn = await aiomysql.connect(
        host=s.db_host,
        port=s.db_port,
        user=s.db_user,
        password=s.db_password,
        db=s.db_name,
        charset=s.db_charset,
        autocommit=True,
    )
    try:
        async with conn.cursor() as cur:
            print("Truncating tables ...")
            await _truncate(cur)

            print("Seeding users ...")
            user_ids = await _insert_users(cur, n=200)

            print("Seeding categories ...")
            cat_ids = await _insert_categories(cur)

            print("Seeding products ...")
            product_meta = await _insert_products(cur, cat_ids)
            print(f"  {len(product_meta)} products")

            print("Seeding orders + items ...")
            await _insert_orders_and_items(cur, user_ids, product_meta, n_orders=2000)

        # Final counts via a fresh cursor.
        async with conn.cursor() as cur:
            for table in ("users", "categories", "products", "orders", "order_items"):
                await cur.execute(f"SELECT COUNT(*) FROM {table}")
                (n,) = await cur.fetchone()
                print(f"  {table:>12} = {n}")
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
