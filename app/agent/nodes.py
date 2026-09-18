"""
所有节点都是纯函数：``(AgentState) -> Partial[AgentState]``。
节点仅通过可替换的辅助函数（`_intent_classifier`、`_schema_recaller`、`_sql_generator`、
`_sql_runner`、`_report_generator`、`_tool_decider`）与外部系统交互，
这样单元测试时可以很方便地替换这些辅助函数做mock测试。

第7阶段新增 tool_call_node：当RAG检索结果为空、或没有检索到可用表结构时，
交给LLM做一轮工具决策，绑定两个工具（search_schema / execute_sql）。
该节点本地执行选中的工具，之后图继续流转；**不存在自主循环调用工具的智能体循环**。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import (
    INTENT_CHAT,
    INTENT_DATA_QUERY,
    KEY_CONVERSATION_ID,
    KEY_ERROR,
    KEY_FINAL_ANSWER,
    KEY_GENERATED_REASONING,
    KEY_GENERATED_SQL,
    KEY_INTENT,
    KEY_NODE_PATH,
    KEY_PREV_ANSWER,
    KEY_PREV_QUESTION,
    KEY_PREV_SQL,
    KEY_QUESTION,
    KEY_QUERY_RESULT,
    KEY_RETRIEVED_CHUNKS,
    KEY_ROW_COUNT,
    KEY_SCHEMA_NEEDS_TOOL,
    KEY_SHOULD_CONTINUE,
    KEY_SKIP_REPORT,
    KEY_SKIP_EXECUTE,
    KEY_TOOL_CALLED,
    KEY_TOOL_ERROR,
    KEY_TOOL_RESULTS,
    AgentState,
)
from app.rag.documents import KnowledgeChunk
from app.tools import chunks_from_tool_output, execute_sql_tool, search_schema_tool

logger = logging.getLogger(__name__)


# ===================== 可替换辅助函数（由 graph.py 注入，单元测试可覆盖） =====================
# 意图分类器：输入用户问题，异步返回识别出的意图
IntentClassifier = Callable[[str], Awaitable[str]]
# 表结构检索器：输入问题，异步返回（知识库片段列表，额外信息1，额外信息2）
SchemaRecaller = Callable[[str], Awaitable[tuple[list[KnowledgeChunk], str, str]]]
# SQL生成器：入参(用户问题，召回知识库片段，历史上下文)，异步返回(sql,推理过程,片段列表)
SQLGenerator = Callable[
    [str, list[KnowledgeChunk], str],
    Awaitable[tuple[str, str, list[KnowledgeChunk]]],
]
# SQL执行器：输入SQL，异步返回(结果行数，查询数据集)
SQLRunner = Callable[[str], Awaitable[tuple[int, list[dict] | None]]]
# 回答生成器：输入(问题，查询结果集，生成的SQL)，异步返回自然语言最终回答
ReportGenerator = Callable[
    [str, list[dict] | None, str | None],
    Awaitable[str],
]
# 工具决策器：传入消息列表，返回AI消息对象（可能携带tool_calls工具调用信息）
ToolDecider = Callable[[list], Awaitable[Any]]


# 模块级全局占位变量，方便单元测试做猴子补丁mock
# 真实图实例会在 `app.agent.graph.build_graph` 中替换成真实实现
_intent_classifier: IntentClassifier | None = None
_schema_recaller: SchemaRecaller | None = None
_sql_generator: SQLGenerator | None = None
_sql_runner: SQLRunner | None = None
_report_generator: ReportGenerator | None = None
_tool_decider: ToolDecider | None = None


def set_helpers(
    *,
    intent_classifier: IntentClassifier,
    schema_recaller: SchemaRecaller,
    sql_generator: SQLGenerator,
    sql_runner: SQLRunner,
    report_generator: ReportGenerator,
    tool_decider: ToolDecider | None = None,
) -> None:
    """
    注入各个辅助函数的真实实现（服务启动/单元测试时调用一次）

    tool_decider 为可选参数：仅工具调用节点依赖它，大部分单元测试不会走到这个节点。
    """
    global _intent_classifier, _schema_recaller, _sql_generator, _sql_runner
    global _report_generator, _tool_decider
    _intent_classifier = intent_classifier
    _schema_recaller = schema_recaller
    _sql_generator = sql_generator
    _sql_runner = sql_runner
    _report_generator = report_generator
    _tool_decider = tool_decider


def _mark(node: str, state: AgentState,** delta: Any) -> dict[str, Any]:
    """
    工具函数：把当前节点名称追加到执行链路记录，同时合并状态变更。
    返回需要合并到AgentState的字典。
    """
    path = list(state.get(KEY_NODE_PATH, []) or [])
    path.append(node)
    return {KEY_NODE_PATH: path, **delta}

def _err(msg: str) -> dict[str, Any]:
    """快捷函数：构造错误状态字典"""
    return {KEY_ERROR: msg}


# ===================== 5个核心工作流节点 =====================


async def intent_node(state: AgentState) -> dict[str, Any]:
    """
    意图识别节点：区分用户意图是【数据查询 DATA_QUERY】还是【闲聊 CHAT】

    使用轻量分类器（关键词+LLM兜底）；闲聊场景直接跳转到结果总结节点，
    避免浪费SQL生成、工具调用资源。
    """
    question = state.get(KEY_QUESTION, "")
    if _intent_classifier is None:
        # 未配置分类器，兜底默认走数据查询（更安全）
        intent = INTENT_DATA_QUERY
    else:
        try:
            intent = await _intent_classifier(question)
        except Exception as e:  # noqa: BLE001
            logger.exception("意图分类失败")
            return _mark(
                "intent",
                state,
                intent=INTENT_DATA_QUERY,
                should_continue=True,
                error=f"意图分类器异常: {type(e).__name__}: {e}",
            )

    # 闲聊则停止后续SQL流水线；数据查询继续往下走
    should_continue = intent != INTENT_CHAT
    return _mark(
        "intent",
        state,
        intent=intent,
        should_continue=should_continue,
    )


async def schema_recall_node(state: AgentState) -> dict[str, Any]:
    """
    表结构检索节点：调用RAG检索，获取数据表结构与业务知识库片段

    第7阶段新增：标记低置信度检索结果。
    "低置信度"：检索结果为空，或者没有任何表结构类型的片段。
    如果没有表定义，SQL生成器很容易编造不存在表和字段，此时交给LLM调用search_schema工具补救（tool_call_node）。
    """
    if not state.get(KEY_SHOULD_CONTINUE, True):
        return _mark("schema_recall", state)

    if _schema_recaller is None:
        return _mark(
            "schema_recall",
            state,
            retrieved_chunks=[],
            schema_needs_tool=True,
            error="schema_recaller 未配置",
        )

    try:
        chunks, _, _ = await _schema_recaller(state.get(KEY_QUESTION, ""))
    except Exception as e:  # noqa: BLE001
        logger.exception("表结构RAG检索失败")
        return _mark(
            "schema_recall",
            state,
            retrieved_chunks=[],
            schema_needs_tool=True,  # 检索异常，进入工具分支尝试补救
            error=f"表结构检索失败: {type(e).__name__}: {e}",
        )

    # 判断：无片段 OR 片段里没有表结构类型的数据 → 需要调用工具补全元数据
    needs_tool = (not chunks) or (not any(c.type == "schema" for c in chunks))
    return _mark(
        "schema_recall",
        state,
        retrieved_chunks=chunks,
        schema_needs_tool=needs_tool,
    )


# ===================== 第7阶段：工具调用节点 =====================
# 安全限制：单次轮次最多执行这么多工具调用。控制延迟，缩小故障影响面
MAX_TOOL_CALLS_PER_TURN = 3

_TOOL_SYSTEM_PROMPT = (
    "你是Text-to-SQL数据分析智能体的工具选择模块。\n"
    "自动知识库检索没有找到用户问题可用的表结构信息。你可以调用工具获取所需信息：\n"
    "- search_schema(keyword): 在知识库中查询表、字段、关联关系、业务规则。不确定表结构时优先选这个。\n"
    "- execute_sql(sql): 在业务MySQL数据库执行只读SELECT语句。SQL会预先校验，仅允许SELECT查询。\n"
    "选择能够帮助回答用户问题的工具。禁止编造不存在的表和字段。"
)


def _tool_messages(state: AgentState) -> list:
    """构造这一轮工具决策所需的对话消息列表"""
    question = state.get(KEY_QUESTION, "")
    known = [c.title or c.id for c in (state.get(KEY_RETRIEVED_CHUNKS, []) or [])]
    human = (
        f"用户问题：{question}\n\n"
        f"已检索到的知识库内容（可能不足）：{known or '无可用信息'}\n\n"
        "判断是否调用工具，以及调用哪个工具。"
    )
    return [SystemMessage(content=_TOOL_SYSTEM_PROMPT), HumanMessage(content=human)]


def _interpret_tool_output(output: str) -> tuple[bool, str | None]:
    """
    解析工具返回JSON结果，返回 (是否成功, 错误信息)

    工具预期内的拒绝（例如sqlglot拦截DELETE语句）会返回 `{"error": "..."}`，
    不会直接抛出异常。所以：没有抛出异常 ≠ 调用成功。非JSON输出视为成功。
    """
    try:
        data = json.loads(output)
    except (ValueError, TypeError):
        return True, None
    if isinstance(data, dict) and data.get("error"):
        return False, str(data["error"])
    return True, None


async def tool_call_node(state: AgentState) -> dict[str, Any]:
    """
    工具调用节点：LLM仅做一轮工具选择决策，由本节点执行选中工具

    【重点】**不是自主循环智能体**
      - LLM只负责选择工具（bind_tools绑定工具定义）
      - 当前节点执行工具，执行完成直接返回状态
      - LangGraph条件边判断下一跳节点（走向sql_generate）
    """
    if not state.get(KEY_SHOULD_CONTINUE, True):
        return _mark("tool_call", state)

    if _tool_decider is None:
        return _mark(
            "tool_call",
            state,
            tool_called=False,
            tool_error="tool_decider 未配置",
            schema_needs_tool=False,  # 关闭工具分支标记，防止循环回到本节点
        )

    # 1) 请求LLM选择工具。异常时优雅降级：带着已有上下文继续执行SQL生成
    try:
        ai_msg = await _tool_decider(_tool_messages(state))
    except Exception as e:  # noqa: BLE001
        logger.exception("工具决策LLM调用失败")
        return _mark(
            "tool_call",
            state,
            tool_called=False,
            tool_error=f"工具决策失败: {type(e).__name__}: {e}",
            schema_needs_tool=False,
        )

    # 取出工具调用列表，截断至最大调用次数限制
    calls = list(getattr(ai_msg, "tool_calls", None) or [])[:MAX_TOOL_CALLS_PER_TURN]
    if not calls:
        return _mark(
            "tool_call",
            state,
            tool_called=False,
            tool_error="LLM未选择任何工具，使用现有上下文继续",
            schema_needs_tool=False,
        )

    # 2) 在本地依次执行选中工具，**不会自动跳回LLM继续多轮工具决策**
    registry = {t.name: t for t in (search_schema_tool, execute_sql_tool)}
    results: list[dict] = []
    new_chunks: list[KnowledgeChunk] = []
    for tc in calls:
        name = str(tc.get("name", ""))
        args = dict(tc.get("args") or {})
        tool_obj = registry.get(name)
        if tool_obj is None:
            results.append(
                {"tool": name, "args": args, "ok": False, "result": "", "error": f"未知工具: {name}"}
            )
            continue
        try:
            out = await tool_obj.ainvoke(args)
            ok, tool_err = _interpret_tool_output(out)
            results.append({"tool": name, "args": args, "ok": ok, "result": out, "error": tool_err})
            # 如果是search_schema工具且执行成功，把返回结果转为知识库片段
            if ok and name == search_schema_tool.name:
                new_chunks.extend(chunks_from_tool_output(out))
        except Exception as e:  # noqa: BLE001 — 单个工具失败不能中断整个图流程
            logger.exception("工具 %s 执行异常", name)
            results.append(
                {"tool": name, "args": args, "ok": False, "result": "", "error": f"{type(e).__name__}: {e}"}
            )

    delta: dict[str, Any] = {
        KEY_TOOL_CALLED: True,
        KEY_TOOL_RESULTS: results,
        KEY_SCHEMA_NEEDS_TOOL: False,  # 本轮只尝试一次工具，不再重复进入工具节点
    }

    # 3) 将search_schema工具查到的新知识合并进retrieved_chunks
    # SQL生成节点可以像读取普通RAG召回结果一样读取工具获取的表结构信息
    if new_chunks:
        existing = list(state.get(KEY_RETRIEVED_CHUNKS, []) or [])
        seen = {c.id for c in existing}
        delta[KEY_RETRIEVED_CHUNKS] = existing + [c for c in new_chunks if c.id not in seen]

    # 如果所有工具调用全部失败，写入全局工具错误信息
    if all(not r["ok"] for r in results):
        delta[KEY_TOOL_ERROR] = "; ".join(filter(None, (r.get("error") for r in results))) or "所有工具调用均失败"

    return _mark("tool_call", state, **delta)


def _format_prev_steps(state: AgentState) -> str:
    """
    格式化上一轮对话上下文，填充Prompt里的PREVIOUS_STEPS占位

    第8阶段多轮对话场景：例如第一轮“查询2026年销售额”，第二轮“那2025年呢？”
    SQL生成器需要上一轮问题/SQL，才能理解指代（那）。
    首轮对话返回 "(none)"
    """
    prev_q = state.get(KEY_PREV_QUESTION, "") or ""
    if not prev_q:
        return "(none)"
    parts = [f"上一轮用户问题: {prev_q}"]
    prev_sql = state.get(KEY_PREV_SQL, "") or ""
    if prev_sql:
        parts.append(f"上一轮生成SQL: {prev_sql}")
    prev_answer = state.get(KEY_PREV_ANSWER, "") or ""
    if prev_answer:
        parts.append(f"上一轮返回答案: {prev_answer}")
    parts.append(
        "如果当前问题包含【那】这类指代，需要参考上一轮问题理解语义。"
    )
    return "\n".join(parts)


async def sql_generate_node(state: AgentState) -> dict[str, Any]:
    """
    SQL生成节点：基于召回的表结构信息 + 用户问题，让LLM编写SQL语句
    """
    if not state.get(KEY_SHOULD_CONTINUE, True):
        return _mark("sql_generate", state)

    if _sql_generator is None:
        return _mark(
            "sql_generate",
            state,
            generated_sql="",
            generated_reasoning="",
            error="sql_generator 未配置",
        )

    try:
        sql, reasoning, chunks = await _sql_generator(
            state.get(KEY_QUESTION, ""),
            state.get(KEY_RETRIEVED_CHUNKS, []) or [],
            _format_prev_steps(state),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("SQL生成失败")
        return _mark(
            "sql_generate",
            state,
            generated_sql="",
            generated_reasoning="",
            error=f"SQL生成失败: {type(e).__name__}: {e}",
        )

    return _mark(
        "sql_generate",
        state,
        generated_sql=sql,
        generated_reasoning=reasoning,
        retrieved_chunks=chunks,
    )


async def sql_execute_node(state: AgentState) -> dict[str, Any]:
    """
    SQL执行节点：将生成的SQL交给校验器，再执行数据库查询
    """
    if not state.get(KEY_SHOULD_CONTINUE, True):
        return _mark("sql_execute", state)

    if state.get(KEY_SKIP_EXECUTE):
        # 调用方只想要生成SQL文本，不需要访问数据库执行
        return _mark("sql_execute", state)

    sql = state.get(KEY_GENERATED_SQL, "")
    if not sql:
        return _mark(
            "sql_execute",
            state,
            query_result=None,
            row_count=None,
            error="待执行SQL为空",
        )

    if _sql_runner is None:
        return _mark(
            "sql_execute",
            state,
            error="sql_runner 未配置",
        )

    try:
        row_count, rows = await _sql_runner(sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("SQL执行失败")
        return _mark(
            "sql_execute",
            state,
            query_result=None,
            row_count=None,
            error=f"SQL执行失败: {type(e).__name__}: {e}",
        )

    return _mark(
        "sql_execute",
        state,
        query_result=rows,
        row_count=row_count,
    )


async def report_node(state: AgentState) -> dict[str, Any]:
    """
    结果总结节点：基于SQL查询结果，生成自然语言最终回答
    """
    if state.get(KEY_SKIP_REPORT):
        # API接口 /api/sql，execute=False场景：仅返回SQL文本，跳过LLM总结，保持返回结果干净
        return _mark(
            "report",
            state,
            final_answer="(已跳过结果总结)",
        )

    if _report_generator is None:
        return _mark(
            "report",
            state,
            final_answer="(report_generator 未配置)",
        )

    try:
        answer = await _report_generator(
            state.get(KEY_QUESTION, ""),
            state.get(KEY_QUERY_RESULT),
            state.get(KEY_GENERATED_SQL, "") or None,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("回答生成失败")
        return _mark(
            "report",
            state,
            final_answer="",
            error=f"回答生成失败: {type(e).__name__}: {e}",
        )

    return _mark("report", state, final_answer=answer)