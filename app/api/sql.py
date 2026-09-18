"""
POST /api/sql — Text-to-SQL 接口（第6阶段：基于LangGraph执行）

接口的请求/响应结构和第5阶段保持不变；唯一变化是：
核心业务逻辑交给 `app.agent.run_graph.run_question` 来执行。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.run_graph import GraphRunResult, run_question
from app.agent.text2sql import get_text2sql_generator  # 第4阶段单元测试用来做猴子补丁
from app.rag.documents import KnowledgeChunk

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------- 请求/响应模型 ------------------------------------
class SqlRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    execute: bool = True
    conversation_id: str | None = "default"


class RetrievedChunkDTO(BaseModel):
    """返回给前端的知识库chunk简易对象（DTO：数据传输对象）"""
    id: str
    title: str
    source: str
    type: str


class SqlResponse(BaseModel):
    question: str
    sql: str
    reasoning: str
    executed: bool
    rowcount: int | None = None
    rows: list[dict] | None = None
    retrieved_chunks: list[RetrievedChunkDTO]
    intent: str | None = None
    node_path: list[str] = []
    final_answer: str | None = None
    error: str | None = None


def _to_response(res: GraphRunResult, *, executed: bool) -> SqlResponse:
    """将LangGraph执行结果 GraphRunResult 转换为前端需要的响应模型"""
    return SqlResponse(
        question=res.question,
        sql=res.generated_sql,
        reasoning=res.generated_reasoning,
        executed=executed and res.row_count is not None,
        rowcount=res.row_count,
        rows=res.query_result,
        retrieved_chunks=[
            RetrievedChunkDTO(
                id=c.id,
                title=c.title,
                source=c.source,
                type=c.type,
            )
            for c in res.retrieved_chunks
        ],
        intent=res.intent,
        node_path=res.node_path,
        final_answer=res.final_answer or None,
        error=res.error,
    )


# ---------- 接口处理器 ------------------------------------------------------
@router.post("/sql", response_model=SqlResponse)
async def run(req: SqlRequest) -> SqlResponse:
    """
    通过LangGraph执行用户问题，可选择是否执行SQL查询。
    """
    try:
        result: GraphRunResult = await run_question(
            question=req.question,
            conversation_id=req.conversation_id or "default",
            # 如果调用方指定不执行SQL，一般只想要生成SQL文本
            # 跳过数据库查询，同时跳过report阶段LLM总结，
            # 即使缺少密钥，接口也能干净返回SQL结果
            skip_report=not req.execute,
            skip_execute=not req.execute,
        )
    except Exception as e:
        logger.exception("/api/sql graph invoke failed")
        return SqlResponse(
            question=req.question,
            sql="",
            reasoning="",
            executed=False,
            retrieved_chunks=[],
            error=f"graph invoke failed: {type(e).__name__}: {e}",
        )

    return _to_response(result, executed=req.execute)