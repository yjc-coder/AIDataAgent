"""
search_schema 工具（第7阶段）

作用：允许大模型按需查询数据表结构、字段定义与业务规则。
本工具只是对【第5阶段知识检索器 KnowledgeRetriever】的一层轻薄 @tool 包装；
自动RAG链路也复用同一个检索器，所以两条路径共享一套知识库、统一的返回格式。

工具返回JSON字符串：
    {"keyword": "...", "count": N, "chunks": [{"id","type","title","source","text"}]}

为什么返回JSON而不是纯文本段落？
因为 tool_call 节点会把这个JSON字符串反向解析成 KnowledgeChunk 对象，写入状态 state.retrieved_chunks。
下游的 sql_generate 节点消费这份工具返回结果时，和普通RAG检索结果完全一致。
一次检索结果，两处使用：大模型LLM读取 + 工作流Graph状态保存。
"""

from __future__ import annotations
# 前向类型注解，Python3.7+支持，可以在类定义前使用该类做类型标注

import json
import logging

from langchain_core.tools import tool
# LangChain工具装饰器：把普通函数包装成Agent可调用工具，自动生成工具描述Schema

from app.rag.documents import KnowledgeChunk
# 知识库切片实体类，代表一条检索出来的知识片段（表结构/业务说明）
from app.rag.retriever import get_retriever
# 获取检索器实例，模块级导入，方便单元测试mock打桩替换检索逻辑

logger = logging.getLogger(__name__)


@tool
def search_schema(keyword: str) -> str:
    """
    在业务数据库知识库中检索表结构与业务规则。

    适用场景：需要确认存在哪些表/字段、多表关联方式、指标口径定义（例如“销售额”怎么计算）时调用。
    Args:
        keyword: 表名、字段名或者业务术语，支持中文。
            示例："orders", "refund", "销售额"
    Returns:
        JSON字符串：{"keyword": 检索关键词, "count": 结果数量, "chunks": [知识片段列表]}
            每个知识片段包含 id、type、title、source、text
    """
    # 获取知识库检索器实例
    retriever = get_retriever()
    # 根据关键词召回相关知识片段，返回召回结果对象
    recalled = retriever.recall(keyword)

    # 组装返回给LLM的载荷
    payload = {
        "keyword": keyword,
        "count": len(recalled.chunks),  # 本次召回到的知识片段总数
        "chunks": [
            {
                "id": c.id,            # 知识片段唯一ID
                "type": c.type,        # 知识类型：例如 table_schema / business_rule / metric_def
                "title": c.title,      # 标题，如：订单表orders字段说明
                "source": c.source,    # 来源文档，如：数据库设计文档V2
                "text": c.text,        # 知识正文，表结构、业务规则文本
            }
            for c in recalled.chunks
        ],
    }
    logger.info("工具 search_schema 调用，关键词：%r，召回片段数量：%d", keyword, len(recalled.chunks))
    # 序列化为JSON，ensure_ascii=False保证中文不会被转义成\u编码
    return json.dumps(payload, ensure_ascii=False)


def chunks_from_tool_output(output: str) -> list[KnowledgeChunk]:
    """
    将 search_schema 的JSON输出反向解析为 KnowledgeChunk 对象列表。

    由 tool_call 节点调用，把工具返回结果合并写入工作流状态 state.retrieved_chunks。
    健壮性设计：如果输出不是合法JSON结构（例如报错文本），直接返回空列表，**不会抛出异常**。

    Args:
        output: search_schema 返回的JSON字符串
    Returns:
        list[KnowledgeChunk]: 解析成功返回知识切片列表；解析失败返回 []
    """
    try:
        # 把JSON字符串加载成python字典
        data = json.loads(output)
        # 取出chunks数组，不存在则赋值空列表
        raw_chunks = data.get("chunks") or []
        # 遍历原始chunk字典，逐个转为KnowledgeChunk实体对象
        return [
            KnowledgeChunk(
                id=str(c.get("id", "")),
                text=str(c.get("text", "")),
                type=str(c.get("type", "unknown")),
                source=str(c.get("source", "")),
                title=str(c.get("title", "")),
            )
            for c in raw_chunks
            if isinstance(c, dict)  # 增加类型判断，过滤掉非字典脏数据
        ]
    except (ValueError, TypeError, AttributeError):
        # JSON解析失败、字段不存在、属性访问异常时捕获全部异常，返回空列表，不中断工作流
        return []