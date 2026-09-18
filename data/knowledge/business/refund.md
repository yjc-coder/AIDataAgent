# 退款 / 取消业务规则

## 退款
- `orders.status = 'refunded'` 表示已退款订单
- 退款订单**不计入销售额**
- 退款订单数：`SELECT COUNT(*) FROM orders WHERE status = 'refunded'`
- 退款率：`refunded_count / paid_count`（注意分母只看 paid）

## 取消
- `orders.status = 'cancelled'` 表示已取消订单
- 同样**不计入销售额**
- 与 refunded 之和：`WHERE status IN ('refunded', 'cancelled')`

## "有效订单"
- 业务上"有效订单" = `paid`
- 任何聚合条件如：`WHERE o.status = 'paid'` 或显式列出合法状态
