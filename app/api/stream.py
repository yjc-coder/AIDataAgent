"""
POST /api/chat/stream — SSE流式接口（第9阶段）

按照LangGraph的节点边界，每完成一个节点就推送一条SSE事件，前端可以实时看到Agent的执行过程：

    event: thinking    data: 正在理解问题...
    event: retrieval   data: 正在检索Schema...
    event: tool        data: 正在调用工具: search_schema     (只有进入工具分支才会推送)
    event: sql         data: 正在生成SQL...                 (附带生成的SQL文本)
    event: result      data: 查询完成, 共 N 行
    event: answer      data: <最终自然语言回答，JSON编码>
    event: done        data: [DONE]

实现原理：使用LangGraph的 `astream(stream_mode="updates")`，每执行完一个节点就返回该节点的状态增量。
代码将节点名称映射成对应的SSE事件类型。**原有LangGraph节点逻辑完全不用改动**；
输入字典由和 `/api/sql` 接口完全相同的 `build_turn_input` 函数构建，
所以工具调用（Phase7）、多轮对话（Phase8）在两个接口上行为保持一致。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import get_compiled_graph
from app.agent.run_graph import build_turn_input
from app.agent.state import (
    KEY_GENERATED_SQL,
    KEY_ROW_COUNT,
    KEY_TOOL_RESULTS,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# 请求体模型
class StreamRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = "default"


def _sse(event: str, data) -> str:
    """
    组装一条 Server-Sent Event 字符串。
    data 使用 json.dumps 编码，保证中文、换行符、引号可以正常在网络传输。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _node_to_event(node: str, delta: dict) -> tuple[str, object] | None:
    """
    把【已执行完毕的节点名称 + 节点状态增量delta】映射成 (事件名, 事件数据)
    如果节点不需要对外推送事件，则返回 None。
    """
    if node == "intent":
        intent = delta.get("intent")
        if intent == "CHAT":
            return ("thinking", "闲聊模式，跳过数据查询...")
        return ("thinking", "正在理解问题...")

    if node == "schema_recall":
        # 获取本次召回的知识库chunk数量
        n = len(delta.get("retrieved_chunks", []) or [])
        return ("retrieval", f"正在检索Schema... 命中 {n} 条知识")

    if node == "tool_call":
        # 提取本次调用的工具名称列表
        tools = [r.get("tool") for r in (delta.get(KEY_TOOL_RESULTS, []) or [])]
        return ("tool", f"正在调用工具: {', '.join(t for t in tools if t) or 'none'}")

    if node == "sql_generate":
        sql_text = delta.get(KEY_GENERATED_SQL, "") or ""
        return ("sql", f"正在生成SQL... {sql_text}".strip())

    if node == "sql_execute":
        rows = delta.get(KEY_ROW_COUNT)
        return ("result", f"查询完成, 共 {rows if rows is not None else 0} 行")

    if node == "report":
        final_ans = delta.get("final_answer", "") or ""
        return ("answer", final_ans)

    # 其他节点不推送事件
    return None


@router.post("/chat/stream")
async def chat_stream(req: StreamRequest) -> StreamingResponse:
    """以SSE流式方式，逐节点返回Agent执行进度"""
    conversation_id = req.conversation_id or "default"

    async def _gen():
        # 生成器函数：产出SSE文本流
        try:
            # 获取编译好的LangGraph图实例
            app = get_compiled_graph()
            # 对话thread_id，LangGraph持久化状态用
            cfg = {"configurable": {"thread_id": conversation_id}}
            # 构建本轮对话输入，和非流式接口共用同一个函数，保证逻辑一致
            turn_input = await build_turn_input(app, req.question, conversation_id)
        except Exception as e:
            logger.exception("stream setup failed")
            yield _sse("error", f"stream setup failed: {type(e).__name__}: {e}")
            yield _sse("done", "[DONE]")
            return

        try:
            # aream + stream_mode="updates"：每跑完一个节点，返回该节点的状态更新delta
            async for chunk in app.astream(turn_input, config=cfg, stream_mode="updates"):
                # langgraph的chunk有时是元组，取最后一项才是 {node:delta} 字典
                if isinstance(chunk, tuple):
                    chunk = chunk[-1]
                if not isinstance(chunk, dict):
                    continue
                # 遍历本次更新里所有节点
                for node, delta in chunk.items():
                    mapped = _node_to_event(node, delta if isinstance(delta, dict) else {})
                    if mapped:
                        yield _sse(mapped[0], mapped[1])
        except Exception as e:
            logger.exception("graph streaming failed")
            yield _sse("error", f"graph streaming failed: {type(e).__name__}: {e}")

        # 全部流程结束，推送done事件
        yield _sse("done", "[DONE]")

    # 返回FastAPI流式响应
    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Nginx关闭缓冲，保证SSE实时推送
        },
    )