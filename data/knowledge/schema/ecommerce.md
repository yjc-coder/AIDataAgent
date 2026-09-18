# 业务库 schema 摘要

> 与 `data/product_schema.sql` 保持同步。Phase 3 直接注入到 LLM Prompt；Phase 5 RAG 会把同一份 Markdown 切片入 Chroma。

## users — 用户
| 字段 | 类型 | 说明 |
|------|------|------|
| id          | BIGINT       | 主键 |
| username    | VARCHAR(64)  | 用户名，唯一 |
| email       | VARCHAR(128) | 邮箱 |
| city        | VARCHAR(64)  | 城市，例如：北京 / 上海 / 杭州 / 深圳 |
| created_at  | DATETIME     | 注册时间 |

样例：
```sql
SELECT id, username, city FROM users WHERE city = '北京' LIMIT 10;
```

## categories — 商品分类
| 字段 | 类型 | 说明 |
|------|------|------|
| id          | BIGINT       | 主键 |
| name        | VARCHAR(64)  | 分类名，唯一，例如：电子产品 / 家用品 / 服饰鞋帽 / 美妆护肤 / 食品饮料 / 图书音像 / 运动户外 / 母婴用品 |
| description | VARCHAR(255) | 分类描述 |

## products — 商品
| 字段 | 类型 | 说明 |
|------|------|------|
| id          | BIGINT         | 主键 |
| name        | VARCHAR(128)   | 商品名 |
| category_id | BIGINT         | 分类 ID（外键 -> categories.id） |
| price       | DECIMAL(10, 2) | **单价，单位：人民币元** |
| stock       | INT            | 库存 |
| status      | VARCHAR(16)    | `on_sale` / `off_shelf` |
| created_at  | DATETIME       | 上架时间 |

## orders — 订单
| 字段 | 类型 | 说明 |
|------|------|------|
| id           | BIGINT         | 主键 |
| user_id      | BIGINT         | 用户 ID（外键 -> users.id） |
| total_amount | DECIMAL(12, 2) | 订单总额（= 该订单所有 order_items.amount 之和） |
| status       | VARCHAR(16)    | `paid`（已支付）/ `refunded`（已退款）/ `cancelled`（已取消） |
| created_at   | DATETIME       | 下单时间 |

## order_items — 订单明细
| 字段 | 类型 | 说明 |
|------|------|------|
| id         | BIGINT         | 主键 |
| order_id   | BIGINT         | 订单 ID（外键 -> orders.id） |
| product_id | BIGINT         | 商品 ID（外键 -> products.id） |
| quantity   | INT            | 购买数量 |
| unit_price | DECIMAL(10, 2) | 成交单价，单位人民币元 |
| amount     | DECIMAL(12, 2) | 小计金额 = unit_price × quantity |

## 常用 JOIN 模板

**订单-商品-用户**（常见视图）：
```sql
SELECT o.id            AS order_id,
       o.created_at,
       o.total_amount,
       o.status,
       u.username,
       u.city
FROM orders o
JOIN users u ON u.id = o.user_id
WHERE o.status = 'paid'
LIMIT 100;
```

**订单-商品-分类**（销售分析）：
```sql
SELECT oi.order_id,
       p.name           AS product,
       c.name           AS category,
       oi.quantity,
       oi.unit_price,
       oi.amount
FROM order_items oi
JOIN orders o     ON o.id = oi.order_id
JOIN products p   ON p.id = oi.product_id
JOIN categories c ON c.id = p.category_id
WHERE o.status = 'paid'
LIMIT 100;
```

**按商品聚合销售额**：
```sql
SELECT p.id,
       p.name,
       SUM(oi.quantity) AS qty,
       SUM(oi.amount)   AS sales
FROM order_items oi
JOIN orders o   ON o.id  = oi.order_id
JOIN products p ON p.id  = oi.product_id
WHERE o.status = 'paid'
  AND YEAR(o.created_at) = 2026
GROUP BY p.id, p.name
ORDER BY sales DESC
LIMIT 10;
```
