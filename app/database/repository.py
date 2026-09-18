"""面向业务领域的只读聚合查询封装。

这里是"SQL 能力展示"——即一个 Text-to-SQL Agent 理应能够生成的那类聚合查询。
第3阶段会让大模型根据自然语言生成这些 SQL；而在第2阶段先手写出来，
目的是先验证数据库连通性、数据形状以及"写入→查询"往返的正确性。

每个方法都遵守以下约定：
    - 使用 `%s` 参数绑定传参（绝不做字符串拼接内联），防止 SQL 注入。
    - 返回 Python 原生类型（int、float、list[dict]），方便直接序列化给前端。
    - 第4阶段安全校验落地后，这些 SQL 同样会经过 security.sql_validator 检查。
"""
from __future__ import annotations

from typing import Optional

from app.database.execute import execute, execute_scalar


class Stats:
    """统计 / 查询助手集合——对应用户最常问的几类数据分析问题。

    全部为 @staticmethod，无实例状态，直接通过 Stats.xxx() 调用。
    """

    @staticmethod
    async def orders_count() -> int:
        """订单表 orders 的总行数；无数据时 execute_scalar 返回 None，兜底为 0。"""
        return await execute_scalar("SELECT COUNT(*) FROM orders") or 0

    @staticmethod
    async def users_count() -> int:
        """用户表 users 的总行数。"""
        return await execute_scalar("SELECT COUNT(*) FROM users") or 0

    @staticmethod
    async def products_count() -> int:
        """商品表 products 的总行数。"""
        return await execute_scalar("SELECT COUNT(*) FROM products") or 0

    @staticmethod
    async def sales_total(*, year: Optional[int] = None) -> float:
        """已支付（paid）订单的销售总额（退款订单不计入）。

        这与第5阶段要教给大模型的业务规则一致：
            "销售额 = 有效订单 amount 之和；退款订单不计入销售额；单位 元"

        参数：
            year: 仅统计该年份的销售额；为 None 时统计全部年份。
                  通过 YEAR(o.created_at) 过滤，并用 %s 绑定年份参数。

        实现说明：order_items 明细表 JOIN orders 主表，
        只累加主表状态为 'paid' 的明细金额，COALESCE 保证无匹配时返回 0 而非 NULL。
        """
        if year is None:
            # 注意：下面第一次赋值的 SQL 立刻被覆盖，属于遗留的无效赋值（dead code），
            # 实际生效的是紧随其后的全量已支付订单求和语句。这里原样保留，不改动逻辑。
            sql = "SELECT COALESCE(SUM(amount), 0) FROM order_items WHERE 1=0"
            # 真正使用的 SQL：不带年份条件，统计全部已支付订单金额
            sql = (
                "SELECT COALESCE(SUM(oi.amount), 0) "
                "FROM order_items oi "
                "JOIN orders o ON o.id = oi.order_id "
                "WHERE o.status = 'paid'"
            )
            params: tuple = ()  # 无年份，不需要绑定参数
        else:
            # 指定年份：在已支付条件上追加 YEAR(created_at) = ? 过滤
            sql = (
                "SELECT COALESCE(SUM(oi.amount), 0) "
                "FROM order_items oi "
                "JOIN orders o ON o.id = oi.order_id "
                "WHERE o.status = 'paid' AND YEAR(o.created_at) = %s"
            )
            params = (year,)
        val = await execute_scalar(sql, params)
        return float(val or 0)  # 统一转 float，None / 0 都兜底为 0.0

    @staticmethod
    async def top_products_by_sales(
        year: int, limit: int = 10
    ) -> list[dict]:
        """查询指定年份销售额最高的若干商品，按销售额降序排名。

        参数：
            year:  统计年份（必填）。
            limit: 返回前 N 名，默认 10。

        返回每行结构形如：
            {id, name, qty, sales}
            其中 qty 为累计销量（quantity 之和），sales 为累计销售额（amount 之和）。

        实现说明：order_items 同时 JOIN orders（取支付状态和时间）
        与 products（取商品名），按商品分组聚合后排序、限制条数。
        """
        sql = (
            "SELECT p.id, p.name, "
            "       SUM(oi.quantity) AS qty, "   # 累计销量
            "       SUM(oi.amount)   AS sales "  # 累计销售额
            "FROM order_items oi "
            "JOIN orders o   ON o.id  = oi.order_id "
            "JOIN products p ON p.id  = oi.product_id "
            "WHERE o.status = 'paid' AND YEAR(o.created_at) = %s "
            "GROUP BY p.id, p.name "
            "ORDER BY sales DESC "
            "LIMIT %s"
        )
        # execute 返回 (rowcount, rows)，这里只需要 rows，用 _ 占位忽略行数
        _, rows = await execute(sql, (year, limit))
        return rows

    @staticmethod
    async def users_by_city(city: str) -> int:
        """统计指定城市的用户数量。

        参数：
            city: 城市名称，通过 %s 参数绑定传入，避免 SQL 注入。
        """
        return (
            await execute_scalar(
                "SELECT COUNT(*) FROM users WHERE city = %s",
                (city,),
            )
            or 0  # 无该城市用户时 COUNT 返回 0；异常空值兜底为 0
        )
