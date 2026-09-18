"""轻量 SQL 执行助手。

第2阶段刻意只暴露两个函数，且都是只读的：
    - `execute(sql, params) -> (rowcount, list[dict])`
        执行 SELECT，返回全部行（字典列表）和行数。
    - `execute_scalar(sql, params) -> Any | None`
        执行 SELECT，只返回首行首列的标量值。

命名设计考量：
    - `execute` 是所有 SELECT 的统一入口。第4阶段会在它之上包一层 sqlglot
      校验（只允许只读语句），然后才暴露给大模型生成的 SQL 使用。
    - `execute_scalar` 返回第一行的第一列（无结果时返回 None），
      特别适合 `SELECT COUNT(*)`、`SELECT MAX(price)` 这类单值聚合查询。

安全要求：参数一律使用 aiomysql 的位置占位符绑定（`%s`），
绝对禁止把用户输入通过字符串拼接内联进 SQL，以防 SQL 注入。
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import aiomysql

from app.database.engine import get_pool


async def execute(
    sql: str,
    params: Optional[Sequence[Any]] = None,
) -> tuple[int, list[dict[str, Any]]]:
    """执行一条 SELECT，以字典列表形式返回全部结果行。

    参数：
        sql:    带 %s 占位符的 SQL 语句。
        params: 与占位符一一对应的参数序列，交给驱动做参数绑定（防注入）；
                为 None 时按空元组处理。

    返回
    -------
    (rowcount, rows)
        `rowcount`：驱动报告的行数（cursor.rowcount）。
        `rows`：结果行列表，每行为一个 dict（键为列名），可能为空列表。
    """
    pool = get_pool()
    # 从连接池获取一条连接；async with 保证用完自动归还连接池
    async with pool.acquire() as conn:
        # 使用字典游标：结果行自动以 {列名: 值} 的形式返回，便于上层直接序列化
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, params or ())  # 参数绑定执行，杜绝 SQL 注入
            rows = await cur.fetchall()            # 拉取全部结果行
            return cur.rowcount, list(rows)


async def execute_scalar(
    sql: str,
    params: Optional[Sequence[Any]] = None,
) -> Any:
    """执行一条 SELECT，只返回第一行的第一列（标量值）。

    没有任何结果行时返回 None，适合 COUNT / MAX / MIN / SUM 等单值查询。
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        # 这里用默认元组游标即可，因为只取一个值，不需要列名映射
        async with conn.cursor() as cur:
            await cur.execute(sql, params or ())
            row = await cur.fetchone()             # 只取第一行
            return row[0] if row else None         # 取首列；无行则 None
