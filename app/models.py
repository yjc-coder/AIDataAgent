"""
Pydantic 数据模型，相当于Java里的 DTO / BO / VO 类。

通用规范：所有跨API边界传输的数据结构都放在这里。
保持模型轻量化、强类型、支持序列化（可以转json）。
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """用户侧传入的聊天请求体。

    conversation_id 用于串联多轮对话历史（第8阶段）。
    """

    query: str = Field(..., min_length=1, description="用户自然语言问题")
    conversation_id: str = Field(..., min_length=1, description="对话线程ID")
    agent_id: Optional[str] = Field(
        None,
        description="多Agent场景下可选的Agent标识",
    )


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """对话历史中的单条消息。

    第8阶段会按照 conversation_id 把消息持久化到 SQLite。
    """

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""


class TextType:
    """SSE流式消息的文本类型标记（对齐Java端TextType的开始/结束标识）。

    第1阶段只输出 TEXT 类型，但协议预先预留在这里，方便第9阶段扩展，
    不会破坏前后端契约。

    SSE流式接口使用，告诉前端该如何渲染片段：代码块、表格、markdown、JSON、普通文本。
    """

    TEXT = "text"
    SQL = "sql"
    RESULT_SET = "result_set"
    MARK_DOWN = "markdown"
    JSON = "json"


class ChatResponse(BaseModel):
    """非流式聊天响应（第1阶段范围）。

    第9阶段会替换为SSE流式输出，流式会推送中间过程：意图识别、检索、SQL、查询结果、最终回答。
    """

    answer: str = Field(..., description="大模型生成的自然语言回答")
    conversation_id: str
    model: str = Field(..., description="实际使用的LLM模型名称")


# ---------------------------------------------------------------------------
# 元信息模型
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """服务存活健康检查返回体。"""

    status: str = "ok"
    service: str = "ai-data-agent"
    phase: str = "1"