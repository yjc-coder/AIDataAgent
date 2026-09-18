# 销售业务规则

> 这是销售分析问题的"业务知识注入"。LLM 在看 schema 的同时，会从同一份文件读到这里。

## 销售额
- 销售额 = `SUM(order_items.amount)`，单位人民币元
- **仅统计 `orders.status = 'paid'` 的订单**；`refunded` / `cancelled` 不计入
- 推荐聚合自 `order_items.amount`（语义最清晰），与 `orders.total_amount` 数学上等价

## 时间维度
- 用户问 **"X 年销售额"** → `WHERE YEAR(o.created_at) = X`
- 用户问 **"上个月销售额"** → `WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH)`
- 用户问 **"最近 N 天"** → `WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL N DAY)`

## 商品维度
- 商品名模糊匹配：`WHERE p.name LIKE '%手机%'`
- 分类筛选：
  `JOIN categories c ON c.id = p.category_id WHERE c.name = ?`
- 多分类 OR：`WHERE c.name IN ('电子产品', '家用电器')`

## 用户维度
- 城市筛选：`JOIN users u ON u.id = o.user_id WHERE u.city = ?`
- 用户排行榜：按 `SUM(oi.amount) DESC` + `JOIN users`

## 排序与 TOP
- 销售额最高：`ORDER BY sales DESC LIMIT N`
- 销量最高：用 `SUM(quantity)`，**不是** `COUNT(*)`
- 平均订单金额：`AVG(o.total_amount)`
- 默认返回前 10 条；明确 LIMIT 时遵用户

## 关联模板（最高频三连）
1. 销售额按商品 → `order_items JOIN orders JOIN products`，按 `SUM(oi.amount)` 排序
2. 销售额按城市 → `order_items JOIN orders JOIN users`，按 `users.city` 分组
3. 销售额按月 → 加 `GROUP BY DATE_FORMAT(o.created_at, '%Y-%m')`
