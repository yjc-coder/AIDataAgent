"""
SQL 安全校验器

四层防御体系，用来抵御提示词注入，或者大模型输出粗心生成的危险SQL：
    1. Prompt软约束    （提示词 new_sql_generate.txt 中要求禁止 INSERT 等写操作）
    2. LLM结构化输出    （SqlGeneration Pydantic模型，限制输出格式）
    3. **本模块**       （基于sqlglot做AST语法树解析 + 白名单 + 自动追加LIMIT）
    4. MySQL只读账号    （docker-compose.yml 配置只读数据库用户）

策略：**顶层节点使用白名单**，嵌套子节点、函数使用黑名单。
多层保险，因为sqlglot偶尔会把一些不常见的MySQL扩展语法映射成意料之外的AST节点类型。

本模块能力：
    - 拒绝非SELECT语句（包含 WITH...SELECT 公用表达式CTE）
    - 拒绝嵌套的写操作/管理操作（例如子查询里面写INSERT）
    - 拒绝危险函数（SLEEP, BENCHMARK, LOAD_FILE）
    - 拒绝多语句SQL（例如 `SELECT 1; DROP TABLE x`）
    - 如果SQL没有LIMIT，自动追加 LIMIT <default_max_rows>
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


class SqlValidationError(ValueError):
    """SQL违反安全策略时抛出异常

    message 是面向用户的提示文案，API接口会直接把这个消息放到返回的error字段。
    """


@dataclass(frozen=True)
class ValidatedSql:
    """已经通过校验、可以安全执行的SQL字符串

    limit_added 标记位：告诉上层接口，我们自动限制了返回行数
    方便告知用户：原始大模型生成的SQL没有行数限制，系统自动加上了。
    """

    sql: str
    limit_added: bool


# ---------------------------------------------------------------------------
# 白名单 / 黑名单定义
# ---------------------------------------------------------------------------

# 允许的根AST节点类型。WITH ... SELECT 解析后根节点依然是 exp.Select
# CTE公用表达式存在节点args["with"]内部，所以白名单只需要这一个类型
ALLOWED_ROOT_TYPES: tuple[type, ...] = (exp.Select,)

# 写操作/管理员操作黑名单：**在AST任何位置出现都禁止**
FORBIDDEN_NODE_TYPES: tuple[type, ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Command,    # 兜底：捕获无法解析的数据库命令
    exp.Block,      # 多语句块 ("SELECT 1; DROP TABLE x")
)

# 禁止调用的函数名称，对比时全部转小写
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "sleep",
        "benchmark",
        "load_file",
    }
)


# ---------------------------------------------------------------------------
# 对外暴露的主函数
# ---------------------------------------------------------------------------


def validate_and_rewrite(
    sql: str,
    *,
    default_max_rows: int = 1000,
    dialect: str = "mysql",
) -> ValidatedSql:
    """解析、校验SQL，必要时自动追加LIMIT

    抛出异常：
        SqlValidationError: 安全策略不通过。消息简短对用户友好，API直接透传给前端。

    返回：
        ValidatedSql：校验并可能修改后的SQL，附带标记是否自动增加了LIMIT
    """
    # --- 1. 基础格式检查 ----------------------------------------------------
    if not sql or not sql.strip():
        raise SqlValidationError("SQL is empty")

    raw = sql.strip()
    # 去掉末尾分号
    if raw.endswith(";"):
        raw = raw[:-1].rstrip()
    # 如果字符串中间还有分号，直接判定多语句攻击
    if ";" in raw:
        # sqlglot.parse_one 本身也会报错，但这里提前拦截，返回更友好提示
        raise SqlValidationError(
            "Multi-statement SQL is not allowed; only a single SELECT is permitted"
        )

    # --- 2. sqlglot 解析SQL生成AST语法树 ------------------------------------------
    try:
        tree = sqlglot.parse_one(raw, read=dialect)
    except sqlglot.errors.ParseError as e:
        raise SqlValidationError(f"SQL parse failed: {e}") from e

    # --- 3. 顶层节点白名单校验 -------------------------------------------
    # 根节点必须是 Select，不允许顶层是 CREATE / DROP 等
    if not isinstance(tree, ALLOWED_ROOT_TYPES):
        raise SqlValidationError(
            f"Only SELECT is allowed at the top level; got {type(tree).__name__}"
        )

    # --- 4. 遍历整棵AST，黑名单扫描（递归检查所有子节点） --------------------------------------------
    bad: list[str] = []
    # walk() 深度优先遍历AST全部节点，包括子查询、CTE内部节点
    for node in tree.walk():
        # 如果节点类型属于禁止操作，记录
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            bad.append(type(node).__name__)
        # 判断匿名函数调用，检查函数名黑名单
        elif isinstance(node, exp.Anonymous):
            name = (node.name or "").lower()
            if name in FORBIDDEN_FUNCTIONS:
                bad.append(f"function {name.upper()}")
    if bad:
        unique = sorted(set(bad))
        raise SqlValidationError(
            f"Forbidden statement/function detected: {', '.join(unique)}"
        )

    # --- 5. 自动追加 LIMIT 限制行数 ------------------------------------------
    limit_added = False
    # 判断AST有没有limit节点，没有就新增
    if tree.args.get("limit") is None:
        tree = tree.limit(
            exp.Limit(expression=exp.Literal.number(default_max_rows))
        )
        limit_added = True

    # 将AST重新序列化为SQL字符串返回
    return ValidatedSql(
        sql=tree.sql(dialect=dialect),
        limit_added=limit_added,
    )