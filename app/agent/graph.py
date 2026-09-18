"""
LangGraph StateGraph 图组装代码（第6阶段 + 第7阶段）

固定节点，两条条件分支边：
    - intent节点执行完成后：如果是闲聊CHAT，直接跳转到report节点
    - schema_recall节点执行完成后：RAG检索置信度不足，则绕行到tool_call节点（第7阶段新增）
      LLM只做一轮工具选择决策。

完整流水线：

    START（图入口）
      → intent        (判断用户意图：DATA_QUERY数据查询 / CHAT闲聊)
      → schema_recall (仅数据查询才会走到这里；闲聊会直接跳过)
      → tool_call     (仅RAG检索为空/没有表结构片段时才进入)
      → sql_generate
      → sql_execute
      → report        (所有分支最终都会走到report节点)
      → END（图结束）

LLM只在tool_call节点内部选择工具；工具调用节点本地执行工具，LangGraph负责路由下一跳。
**本设计不存在自主Agent循环，工具最多只执行一轮。**
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.agent.nodes import (
    intent_node,
    report_node,
    schema_recall_node,
    sql_execute_node,
    sql_generate_node,
    tool_call_node,
    set_helpers,
)
from app.agent.state import (
    INTENT_CHAT,
    KEY_INTENT,
    KEY_SCHEMA_NEEDS_TOOL,
    KEY_SHOULD_CONTINUE,
    KEY_TOOL_CALLED,
    AgentState,
)
from app.rag.retriever import retriever_module  # 模块引用，单元测试可以monkey-patch打补丁mock
from app.rag.documents import KnowledgeChunk
from app.agent.text2sql import text2sql_module  # 模块引用，单元测试可以monkey-patch打补丁mock
from app.agent.sql_runner import sql_runner_module  # 模块引用，单元测试可以monkey-patch打补丁mock
from app.llm.client import get_chat_model
from app.tools import execute_sql_tool, search_schema_tool

logger = logging.getLogger(__name__)


# ===================== 辅助适配器函数（各个节点底层真实能力实现） =====================

async def _classify_intent(question: str) -> str:
    """轻量LLM意图分类器

    LLM不可用时降级使用关键词启发式规则，保证图不会因为分类器异常卡死。
    """
    q = (question or "").strip().lower()
    # 先关键词兜底：明显的闲聊直接返回CHAT，省去一次LLM调用
    chat_markers = ("你好", "hello", "hi ", "hi!", "who are you", "你是谁", "介绍", "谢谢", "thanks")
    if any(m in q for m in chat_markers):
        return "CHAT"

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        chat = get_chat_model()
        from pydantic import BaseModel

        class IntentLabel(BaseModel):
            intent: str  # 可选值 "DATA_QUERY" | "CHAT"

        # 结构化输出，强制LLM返回固定格式
        structured = chat.with_structured_output(IntentLabel)
        resp = await structured.ainvoke(
            [
                SystemMessage(
                    content=(
                        "对用户问题做意图分类。\n"
                        "如果用户询问数据、指标、销售额、订单、商品、客户，或任何需要SQL查询的内容，返回`intent` = DATA_QUERY。\n"
                        "其余场景返回 CHAT。\n"
                    )
                ),
                HumanMessage(content=question),
            ]
        )
        label = (getattr(resp, "intent", "") or "").upper()
        # 非法输出兜底，默认走数据查询链路
        return label if label in {"DATA_QUERY", "CHAT"} else "DATA_QUERY"
    except Exception as e:  # noqa: BLE001
        logger.warning("意图分类LLM调用失败 (%s); 默认降级为DATA_QUERY", e)
        return "DATA_QUERY"


async def _retrieve(question: str) -> tuple[list[KnowledgeChunk], str, str]:
    """RAG检索适配器：调用检索器，返回知识库片段、表结构文本、证据文本"""
    r = retriever_module.get_retriever()
    out = r.recall(question)
    return out.chunks, out.schema_text, out.evidence_text


async def _generate_sql(
    question: str,
    chunks: list[KnowledgeChunk],
    previous_steps: str = "(none)",
) -> tuple[str, str, list[KnowledgeChunk]]:
    """SQL生成适配器

    使用模块属性获取生成器，方便单元测试对 text2sql_module 做猴子补丁mock。
    第7阶段：tool_call节点已经拿到表结构片段，直接传入，不再重复RAG检索。
    如果传入空列表，则由生成器内部自行做RAG召回。
    第8阶段：previous_steps 携带上一轮问答信息，用于处理多轮指代（“那去年呢”）。
    """
    gen_factory = text2sql_module.get_text2sql_generator
    gen = gen_factory()
    result = await gen.agenerate(
        question=question,
        chunks=list(chunks or []) or None,
        previous_steps=previous_steps,
    )
    return result.sql, result.reasoning, list(result.retrieved_chunks or [])


async def _execute(sql: str) -> tuple[int, list[dict] | None]:
    """SQL执行适配器：仅执行只读SELECT语句，返回行数和查询结果"""
    return await sql_runner_module.run_select(sql)


async def _decide_tools(messages: list):
    """工具决策适配器：绑定两个工具，LLM做出工具调用选择，返回AIMessage对象

    tool_call节点负责执行工具；这个函数只负责LLM决策。
    """
    chat = get_chat_model()
    # 将两个工具绑定到LLM
    bound = chat.bind_tools([search_schema_tool, execute_sql_tool])
    return await bound.ainvoke(messages)


async def _report(
    question: str,
    rows: list[dict] | None,
    sql: str | None,
) -> str:
    """最终回答生成适配器：把SQL结果转换成自然语言中文回答"""
    chat = get_chat_model()
    try:
        chat = chat.bind(temperature=0)
    except Exception:  # noqa: BLE001 — bind是可选配置，不支持也不报错
        pass
    from langchain_core.messages import SystemMessage, HumanMessage

    # 场景：闲聊分支，没有SQL和查询结果
    if not rows and not sql:
        try:
            msg = await chat.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "你是电商数据分析助手。用户这条消息不需要查询数据库：\n"
                            "- 如果是问候或闲聊，请自然友好地回应；\n"
                            "- 如果是想查数据但没能生成 SQL，请说明并建议用户换种问法"
                            "（例如指明年份、商品或指标）；\n"
                            "- 绝对不要编造任何数据。用中文简短回复。"
                        )
                    ),
                    HumanMessage(content=question),
                ]
            )
            return msg.content if hasattr(msg, "content") else str(msg)
        except Exception:  # noqa: BLE001 — LLM挂掉兜底文案
            logger.exception("闲聊兜底回答生成失败")
            return "（未生成可执行的 SQL，请补充更明确的问题。）"

    # 场景：有SQL查询结果，整理数据分析结论
    system = SystemMessage(
        content=(
            "你是数据分析师。根据用户问题、SQL语句和查询结果，\n"
            "写出简洁中文回答，重点标出关键数字。\n"
            "如果查询结果为空，如实说明。"
        )
    )
    user = HumanMessage(
        content=(
            f"用户问题:\n{question}\n\n"
            f"SQL:\n{sql or '(none)'}\n\n"
            f"结果行数 ({len(rows or [])}):\n{rows or []}\n"
        )
    )
    msg = await chat.ainvoke([system, user])
    return msg.content if hasattr(msg, "content") else str(msg)


# ===================== 构建LangGraph状态图 =====================

def _build_state() -> StateGraph:
    # 注意：调用此函数前必须先注入helper实现；生产环境在应用启动调用init_default_helpers()
    # 单元测试直接调用set_helpers，然后再编译图。这里不重复调用set_helpers，防止mock被覆盖。

    g = StateGraph(AgentState)
    # 注册所有节点
    g.add_node("intent", intent_node)
    g.add_node("schema_recall", schema_recall_node)
    g.add_node("tool_call", tool_call_node)
    g.add_node("sql_generate", sql_generate_node)
    g.add_node("sql_execute", sql_execute_node)
    g.add_node("report", report_node)

    # 图入口：START → intent节点
    g.add_edge(START, "intent")

    # ---------- 第一条条件边：intent执行完成后路由 ----------
    # 闲聊直接跳report；数据查询走schema_recall
    def _route_after_intent(state: AgentState) -> str:
        if state.get(KEY_INTENT) == INTENT_CHAT:
            return "report"
        return "schema_recall"

    g.add_conditional_edges(
        "intent",
        _route_after_intent,
        {"report": "report", "schema_recall": "schema_recall"},
    )

    # ---------- 第二条条件边：schema_recall执行完成后路由（第7阶段） ----------
    # 如果RAG召回不足，并且本轮还没有调用过工具 → 进入tool_call节点
    # `not tool_called` 保证本轮最多只走一次tool_call，防止循环
    def _route_after_recall(state: AgentState) -> str:
        if state.get(KEY_SCHEMA_NEEDS_TOOL) and not state.get(KEY_TOOL_CALLED):
            return "tool_call"
        return "sql_generate"

    g.add_conditional_edges(
        "schema_recall",
        _route_after_recall,
        {"tool_call": "tool_call", "sql_generate": "sql_generate"},
    )

    # 固定边：tool_call执行完一定走到sql_generate
    g.add_edge("tool_call", "sql_generate")
    # 固定边：sql生成 → sql执行
    g.add_edge("sql_generate", "sql_execute")
    # 固定边：sql执行 → report总结
    g.add_edge("sql_execute", "report")
    # report执行完成，流程结束
    g.add_edge("report", END)

    return g


@lru_cache(maxsize=1)
def get_compiled_graph():
    """获取编译好的LangGraph实例，全局单例：进程内只编译一次图"""
    g = _build_state()
    # MemorySaver：内存级检查点，用来保存多轮会话状态，支持多轮对话
    return g.compile(checkpointer=MemorySaver())


def init_default_helpers() -> None:
    """生产环境初始化：注入真实业务适配器，应用启动时调用一次。
    单元测试一般直接调用set_helpers，跳过这个函数。
    """
    set_helpers(
        intent_classifier=_classify_intent,
        schema_recaller=_retrieve,
        sql_generator=_generate_sql,
        sql_runner=_execute,
        report_generator=_report,
        tool_decider=_decide_tools,
    )


def reset_compiled_graph() -> None:
    """清空单例缓存，单元测试用：重新编译图，使用mock的helper"""
    get_compiled_graph.cache_clear()


def node_trace_from_state(state: AgentState) -> list[str]:
    """从状态中取出本次执行经过的节点路径，用于接口 /api/graph/trace 调试追踪"""
    return list(state.get("node_path", []) or [])