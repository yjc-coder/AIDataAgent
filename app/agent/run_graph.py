"""
LangGraph 执行入口模块，统一运行图并返回扁平化结果。
/api/sql 和 /api/graph/trace 两个接口都复用该模块。
第9阶段 SSE 流式接口会在这里接入 astream_events，而原有基于 invoke 的同步执行链路保持不变。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

# 获取编译完成的 LangGraph 图实例
from app.agent.graph import get_compiled_graph
# 导入Agent状态类、所有状态字段常量
from app.agent.state import (
    KEY_CONVERSATION_ID,
    KEY_ERROR,
    KEY_FINAL_ANSWER,
    KEY_GENERATED_REASONING,
    KEY_GENERATED_SQL,
    KEY_NODE_PATH,
    KEY_HISTORY,
    KEY_PREV_ANSWER,
    KEY_PREV_QUESTION,
    KEY_PREV_SQL,
    KEY_QUERY_RESULT,
    KEY_RETRIEVED_CHUNKS,
    KEY_ROW_COUNT,
    KEY_SCHEMA_NEEDS_TOOL,
    KEY_SKIP_REPORT,
    KEY_SKIP_EXECUTE,
    KEY_TOOL_CALLED,
    KEY_TOOL_ERROR,
    KEY_TOOL_RESULTS,
    AgentState,
)
# RAG知识库片段数据结构
from app.rag.documents import KnowledgeChunk

# 日志对象
logger = logging.getLogger(__name__)

# 跨多轮指代支持的历史轮数上限（如"那前年呢？"跨三轮引用）
MAX_HISTORY_TURNS = 3


@dataclass
class GraphRunResult:
    """
    图执行结果数据类：将AgentState复杂状态扁平化，提供给上层API处理器返回给前端。
    把LangGraph内部状态快照提取成简单字段，方便接口序列化。
    """

    question: str                          # 用户当前提问
    conversation_id: str                   # 对话唯一ID，LangGraph checkpointer thread_id
    intent: str | None                     # 用户意图识别结果
    node_path: list[str]                   # 本次执行经过的节点路径（用于链路追踪/调试）
    generated_sql: str                     # Agent生成的SQL语句
    generated_reasoning: str               # 大模型推理思考过程
    row_count: int | None                  # SQL查询返回行数
    query_result: list[dict] | None        # SQL查询结果集
    final_answer: str                      # 最终自然语言回答
    retrieved_chunks: list[KnowledgeChunk] # RAG检索到的知识库片段
    error: str | None                      # 错误信息，无错误则为None

    @classmethod
    def from_state(cls, state: AgentState, *, question: str, conversation_id: str) -> "GraphRunResult":
        """
        类方法：从LangGraph原始AgentState状态对象，构造扁平化GraphRunResult实例。
        :param state: LangGraph执行完成后的完整状态
        :param question: 用户原始问题
        :param conversation_id: 对话ID
        :return: 扁平化结果对象
        """
        return cls(
            question=question,
            conversation_id=conversation_id,
            intent=state.get("intent"),
            node_path=list(state.get(KEY_NODE_PATH, []) or []),
            generated_sql=state.get(KEY_GENERATED_SQL, "") or "",
            generated_reasoning=state.get(KEY_GENERATED_REASONING, "") or "",
            row_count=state.get(KEY_ROW_COUNT),
            query_result=state.get(KEY_QUERY_RESULT),
            final_answer=state.get(KEY_FINAL_ANSWER, "") or "",
            retrieved_chunks=list(state.get(KEY_RETRIEVED_CHUNKS, []) or []),
            error=state.get(KEY_ERROR),
        )


async def build_turn_input(
    app,
    question: str,
    conversation_id: str,
    *,
    skip_report: bool = False,
    skip_execute: bool = False,
) -> dict:
    """
    构建单次对话轮次的图输入状态字典。
    【第8阶段特性】：通过checkpointer读取上一轮对话的问题/SQL/回答，实现多轮上下文引用（例如用户追问“那2025年呢？”）。
    【第9阶段特性】：SSE流式接口与普通接口共用此函数，保证两种入口输入逻辑完全一致。
    容错设计：读取历史状态属于尽力而为逻辑；即使checkpointer读取异常，不能阻塞当前轮次正常执行。

    :param app: 已经编译好的LangGraph图实例
    :param question: 用户当前输入问题
    :param conversation_id: 对话ID，对应LangGraph thread_id
    :param skip_report: 是否跳过生成自然语言报告
    :param skip_execute: 是否跳过SQL真实执行（只生成SQL，不跑库）
    :return: dict，LangGraph图启动所需初始输入状态
    """
    # 初始化上一轮上下文为空字符串
    prev_question = prev_sql = prev_answer = ""
    # 最近多轮对话（新→旧），每轮 {"question","sql","answer"}；用于跨多轮指代
    turns: list[dict] = []
    try:
        cfg = {"configurable": {"thread_id": conversation_id or "default"}}
        # 从checkpointer读取该对话thread_id对应的最新状态快照（上一轮）
        snapshot = await app.aget_state(cfg)
        if snapshot and snapshot.values:
            # 提取上一轮的问题、SQL、最终答案，作为本轮多轮上下文
            prev_question = str(snapshot.values.get("question", "") or "")
            prev_sql = str(snapshot.values.get(KEY_GENERATED_SQL, "") or "")
            prev_answer = str(snapshot.values.get(KEY_FINAL_ANSWER, "") or "")

        # 【跨多轮指代】快照历史按节点步从新到旧产出——同一轮的每个节点
        # 都有一个快照且 question 相同，按 question 去重即得到"轮次"序列。
        # 最多取 MAX_HISTORY_TURNS 轮，支撑"那前年呢？"这类跨三轮引用。
        seen_questions: set[str] = set()
        async for snap in app.aget_state_history(cfg, limit=100):
            values = snap.values or {}
            q = str(values.get("question", "") or "")
            if not q or q in seen_questions:
                continue
            seen_questions.add(q)
            turns.append({
                "question": q,
                "sql": str(values.get(KEY_GENERATED_SQL, "") or ""),
                "answer": str(values.get(KEY_FINAL_ANSWER, "") or ""),
            })
            if len(turns) >= MAX_HISTORY_TURNS:
                break
    except Exception as e:  # noqa: BLE001 捕获全部异常，不向上抛出
        # 读取历史状态失败仅打警告日志，不阻断本轮执行
        logger.warning("读取上一轮对话状态失败: %s", e)

    return {
        "question": question,
        KEY_CONVERSATION_ID: conversation_id,
        KEY_SKIP_REPORT: skip_report,
        KEY_SKIP_EXECUTE: skip_execute,
        # 【第7阶段】清空上一轮遗留的工具调用标记，防止持久化状态干扰本轮工具分支判断
        KEY_SCHEMA_NEEDS_TOOL: False,
        KEY_TOOL_CALLED: False,
        KEY_TOOL_RESULTS: [],
        KEY_TOOL_ERROR: None,
        # 【第8阶段】多轮对话上下文，首轮对话为空字符串
        KEY_PREV_QUESTION: prev_question,
        KEY_PREV_SQL: prev_sql,
        KEY_PREV_ANSWER: prev_answer,
        # 【跨多轮指代】最近多轮对话（新→旧），Prompt渲染时逐轮展开
        KEY_HISTORY: turns,
    }


async def run_question(
    question: str,
    conversation_id: str = "default",
    *,
    skip_report: bool = False,
    skip_execute: bool = False,
) -> GraphRunResult:
    """
    对外主入口函数：执行LangGraph图，并返回扁平化快照结果。
    异常处理策略：所有异常都会捕获并写入state.error，不会向上抛出异常；上层API可以永远拿到GraphRunResult对象。

    :param question: 用户提问
    :param conversation_id: 对话ID，默认值 default
    :param skip_report: True=跳过生成自然语言总结报告
    :param skip_execute: True=仅生成SQL，跳过数据库执行
    :return: GraphRunResult 扁平化执行结果
    """
    # 获取编译完成的LangGraph Agent图
    app = get_compiled_graph()
    # LangGraph运行配置：thread_id绑定对话ID，checkpointer以此为key持久化状态
    cfg: RunnableConfig = {
        "configurable": {"thread_id": conversation_id or "default"},
    }
    # 组装本轮图的初始输入
    turn_input = await build_turn_input(
        app, question, conversation_id, skip_report=skip_report, skip_execute=skip_execute
    )

    try:
        # 异步调用LangGraph，完整跑完整个图所有节点，拿到最终状态
        final_state = await app.ainvoke(turn_input, config=cfg)
    except Exception as e:  # noqa: BLE001 捕获所有异常，保证API一定返回结构体
        # 图调用发生异常时，构造最小化结果对象，将异常信息存入error字段，供API包装错误返回
        return GraphRunResult(
            question=question,
            conversation_id=conversation_id,
            intent=None,
            node_path=[],
            generated_sql="",
            generated_reasoning="",
            row_count=None,
            query_result=None,
            retrieved_chunks=[],
            final_answer="",
            error=f"LangGraph图执行失败: {type(e).__name__}: {e}",
        )
    # 正常执行完成：从AgentState转为扁平化结果返回
    return GraphRunResult.from_state(
        final_state, question=question, conversation_id=conversation_id
    )