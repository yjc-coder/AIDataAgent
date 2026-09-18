"""
execute_sql 工具（第7阶段）

作用：当大模型已经掌握足够信息，可以自行编写SQL时，允许LLM调用本工具执行只读查询。
⚠️ 本工具**不会直接执行原始SQL字符串**，而是把SQL交给 app.agent.sql_runner.run_select 处理；
run_select 内部会先用 sqlglot 做语法+安全校验：
    execute_sql(sql)
        -> validate_and_rewrite()   # 强制仅允许单条SELECT语句；禁止危险函数；缺少LIMIT时自动追加 LIMIT 1000
        -> execute()                # 执行SQL查询，校验失败不会连接数据库
        -> 返回JSON字符串

安全设计说明：
恶意SQL（DROP / DELETE / SLEEP / 多条语句）会在校验阶段直接拒绝，**在建立数据库连接之前就拦截**。
只读MySQL账号（docker-compose.yml中配置）属于纵深防御的最后一层安全兜底。
"""

from __future__ import annotations
# 允许在类定义前使用自身类型注解，Python3.7+语法

import json
import logging

from langchain_core.tools import tool
# LangChain 工具装饰器，把函数包装成Agent可识别调用的工具

from app.agent.sql_runner import run_select
# 模块级导入，方便单元测试时mock打桩，替换数据库查询逻辑
from app.security.sql_validator import SqlValidationError
# SQL校验自定义异常：SQL语法非法、存在高危操作时抛出该异常

logger = logging.getLogger(__name__)

# 限制返回给LLM上下文的最大预览行数，防止一次性返回1000行数据塞满大模型上下文窗口
MAX_PREVIEW_ROWS = 20


@tool
# LangChain工具装饰器：自动提取函数名、文档字符串、参数类型，生成工具描述schema，供大模型判断何时调用
async def execute_sql(sql: str) -> str:
    """
    在业务MySQL数据库上执行只读SELECT查询。

    SQL会先经过sqlglot安全校验：
    仅允许单条SELECT语句（或者WITH ... SELECT 公用表达式）；
    INSERT / UPDATE / DELETE / DROP / ALTER 以及 SLEEP 这类危险函数都会被拦截拒绝；
    如果SQL没有写LIMIT，会自动追加 LIMIT 1000。

    Args:
        sql: MySQL单条SELECT查询语句字符串
    Returns:
        JSON字符串，两种返回格式：
        成功：{"error": null, "row_count": N, "rows": [...前20条预览数据...],
               "total_rows": N, "truncated": bool}
        校验失败/拒绝：{"error": "<错误原因>", "row_count": null, "rows": []}
    """
    try:
        # 调用底层查询入口，内部包含sqlglot校验+数据库查询
        # row_count：受影响行数；rows：全部查询结果列表
        row_count, rows = await run_select(sql)
    except SqlValidationError as e:
        # SQL安全校验失败分支：预期内的拒绝场景，不向上抛出异常
        # 捕获异常，把错误信息封装成JSON返回给LLM，让Agent可以自行修正SQL，保证工作流继续运行
        logger.info("execute_sql工具拦截非法SQL：%s", e)
        return json.dumps(
            {"error": str(e), "row_count": None, "rows": []},
            ensure_ascii=False, # 支持中文，不转义unicode字符
        )

    # 截取前MAX_PREVIEW_ROWS行作为预览数据，传给大模型
    preview = list(rows or [])[:MAX_PREVIEW_ROWS]
    payload = {
        "error": None,
        "row_count": row_count,          # 本次查询返回行数
        "rows": preview,                 # 预览行（最多20行，避免上下文溢出）
        "total_rows": len(rows or []),   # 查询真实总行数
        "truncated": len(rows or []) > MAX_PREVIEW_ROWS, # 是否截断标记：true代表还有更多数据没返回给LLM
    }
    logger.info("execute_sql工具执行成功，查询总行数：%d", row_count)
    # default=str：自动把datetime、decimal等无法序列化的对象转为字符串
    return json.dumps(payload, ensure_ascii=False, default=str)