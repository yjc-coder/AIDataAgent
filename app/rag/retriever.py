"""
知识库检索器（第5阶段）

封装向量库 + 轻量后处理层，完成三件事：
    1. 根据用户问题，从 Chroma 向量库召回 top-K 知识库片段(chunk)。
    2. 根据元数据 metadata.type，把召回结果分成两类：
       schema 表结构片段、business 业务规则片段。
    3. 将两类片段分别格式化，生成可以直接填充进 Prompt 占位符的字符串。

Text2SqlGenerator 会调用这个检索器，而不是一次性读取全部 md 知识库文件。

降级模式（没有可用 Embedding 密钥）：
    仍然返回 chunk，但基于哈希相似度排序。适合跑演示流程，生产环境不可用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rag.documents import KnowledgeChunk
from app.rag.vector_store import ensure_indexed, get_vector_store

# 模块自身引用，方便上层代码（如 app.agent.graph）在单元测试时
# 对 retriever_module.get_retriever 做猴子补丁（monkeypatch）
import sys as _sys
retriever_module = _sys.modules[__name__]

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """SQL 生成器填充 Prompt 占位符所需要的全部召回结果。

    三个字段分别是：
        schema_text:   把所有表结构片段拼接成的一段文本，直接塞进 Prompt 的 schema 槽位
        evidence_text: 把所有业务规则片段拼接成的一段文本，直接塞进 Prompt 的 evidence 槽位
        chunks:        原始召回 chunk 列表，保留以便上层做链路溯源、调试或二次处理
    """
    schema_text: str        # 拼接好的表结构文本
    evidence_text: str      # 拼接好的业务规则文本
    chunks: list[KnowledgeChunk]  # 原始召回 chunk 列表，用于链路溯源


class KnowledgeRetriever:
    """向量存储的轻薄封装门面类。

    职责仅限：调用向量库做检索 + 把结果分类格式化。
    不负责构建索引、不负责 Embedding，这些由 vector_store 模块处理。
    """
    def __init__(self, top_k: int = 4):
        # top_k：每次检索召回的片段数量，默认4
        # 取值过小可能漏召回关键表结构，过大则浪费 Prompt token
        self.top_k = top_k

    def recall(self, question: str) -> RecallResult:
        """根据用户问题召回 top-K 知识库片段，并按类型拆分。
        参数：
            question: 用户自然语言提问，例如"查询2026年销售额前十的城市"
        返回：
            RecallResult 三元组，schema_text 和 evidence_text 已格式化好可直接填 Prompt
        """
        # 获取向量存储单例（首次调用会触发索引构建，耗时约5~10秒）
        vs = get_vector_store()
        # 向量相似度检索：把 question 经 Embedding 转向量，再和库中所有向量做余弦相似度排序
        # 返回 top_k 个相似度最高的 KnowledgeChunk
        chunks = vs.query(question, top_k=self.top_k)

        # 按 chunk.id 升序排序，保证每次返回顺序稳定（不随相似度微小波动而变化）
        # 这样单元测试的断言可以写死顺序，调试日志也更易读
        chunks.sort(key=lambda c: c.id)

        # 把 chunk 列表拆分为 schema 和 business 两类，并拼接成 Prompt 文本
        schema_text, evidence_text = chunks_to_prompt_parts(chunks)

        return RecallResult(
            schema_text=schema_text,
            evidence_text=evidence_text,
            chunks=chunks,
        )


def _join_for_prompt(chunks: list[KnowledgeChunk], kind: str) -> str:
    """把同类型的 chunk 拼接成一段 Prompt 文本；无结果时返回兜底提示。
    参数：
        chunks: 同一类型（schema 或 business）的 chunk 列表
        kind:   类型标签字符串，仅用于兜底提示文案，如 "schema" / "business"
    返回的文本格式：
        [1] 标题1
        正文1
        ---
        [2] 标题2
        正文2

    多个 chunk 之间用分隔线 \n\n---\n\n 隔开，让 LLM 清楚边界。
    """
    if not chunks:
        # 没召回到任何同类型片段，给 LLM 一个明确提示，避免它编造
        return f"(no {kind} knowledge retrieved — answer from the question alone)"
    out: list[str] = []
    for i, c in enumerate(chunks, 1):
        # 构造 chunk 标题：有 title 字段就用 title，没有就只标序号
        header = f"[{i}] {c.title}".strip(" []") if c.title else f"[{i}]"
        out.append(f"{header}\n{c.text}")
    # 使用分隔线 "---" 分隔不同 chunk，让 LLM 清楚每段是独立的文档片段
    return "\n\n---\n\n".join(out)


def chunks_to_prompt_parts(chunks: list[KnowledgeChunk]) -> tuple[str, str]:
    """把 chunk 列表按 type 拆分为 schema / business 两类，分别格式化输出。

    第7阶段关键复用点：
        这个函数同时被【search_schema 工具】和【Text2SqlGenerator 检索器】共用。
        所以工具查到的 chunk 列表，可以渲染成和向量检索召回**完全一样格式**的 Prompt 文本，
        让 LLM 看到的格式统一，无需区分来源。

    参数：
        chunks: 待分类的 chunk 列表，可以是向量检索结果，也可以是工具调用结果

    返回：
        (schema_text, evidence_text) 二元组，分别对应 Prompt 里两个槽位
    """
    # 根据 chunk 元数据 type 字段过滤
    schema_chunks = [c for c in chunks if c.type == "schema"]
    business_chunks = [c for c in chunks if c.type == "business"]
    return (
        _join_for_prompt(schema_chunks, kind="schema"),
        _join_for_prompt(business_chunks, kind="business"),
    )


# ---------------------------------------------------------------------------
# 模块级懒加载单例
# ---------------------------------------------------------------------------
# 首次请求时才构造检索器，避免 FastAPI 启动时做耗时操作
_retriever: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    """懒加载单例：首次调用会执行知识库向量化索引（耗时5~10秒），之后复用缓存。

    索引构建失败时不抛异常，记一条 warning 后继续运行，
    检索器照常创建但召回结果会是空列表，Text2SQL 会降级到全量业务规则注入。
    """
    global _retriever
    if _retriever is None:
        from app.config import get_settings
        s = get_settings()
        try:
            # 首次调用确保向量索引已构建（空集合才会真正执行 Embedding）
            ensure_indexed()
        except Exception as e:  # pragma: no cover - 依赖环境配置，单元测试不覆盖
            logger.warning("RAG 索引构建失败 (%s); 继续运行，但无知识库检索能力。", e)
        # 使用配置文件里定义的 rag_top_k 参数创建检索器实例
        _retriever = KnowledgeRetriever(top_k=s.rag_top_k)
    return _retriever


def reset_retriever() -> None:
    """清空缓存的单例对象，单元测试专用。

    测试用例修改了 data/knowledge/ 内容或替换了 Embedding 函数后，
    调用本函数让下次 get_retriever() 重新构建索引。
    """
    global _retriever
    _retriever = None
