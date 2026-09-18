"""数据库访问包。

第2阶段引入，包含三个模块，职责分层：
    - `engine`     ：aiomysql 异步连接池的生命周期管理（创建 / 获取 / 关闭）
    - `execute`    ：仅面向 SELECT 的轻量执行助手（返回 list[dict] 或标量）
    - `repository` ：面向业务领域的聚合查询封装，供第3阶段 Text-to-SQL 使用

安全校验（只允许 SELECT / 强制 LIMIT / 危险关键字黑名单）在第4阶段
通过 `security/sql_validator.py` 实现。第2阶段刻意保持接口面最小，
让演示数据和简单查询先跑通，后续再叠加安全层。
"""
