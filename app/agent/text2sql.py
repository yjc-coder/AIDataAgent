"""
Text-to-SQL 生成器模块

调用链路：
    ChatPromptTemplate("new_sql_generate")
        -> retriever.recall(question)         # 第5阶段
        -> ChatModel.with_structured_output(SqlGeneration)
        -> SqlGeneration(sql, reasoning) + 召回的知识库片段

使用结构化输出保证SQL字段永远是干净字符串：
没有markdown代码块标记、没有末尾分号、没有多余前置描述文本。
这样第4阶段的sqlglot校验器可以确定性地解析SQL。
"""
from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agent.business_knowledge import business_rules, schema_doc
from app.agent.prompts import load as load_prompt
from app.llm.client import get_chat_model
from app.rag.documents import KnowledgeChunk
from app.rag.retriever import KnowledgeRetriever, chunks_to_prompt_parts, get_retriever


class SqlGeneration(BaseModel):
    """LLM结构化输出模型

    reasoning 推理字段是非必填，但在SQL出错时非常有用：
    可以告诉我们（以及用户）模型为什么选择这样写SQL。
    """

    sql: str = Field(
        ...,
        description=(
            "单条 SELECT 语句（也支持 WITH ... SELECT）。"
            "禁止markdown代码块标记，末尾不要分号，不要在SQL前面附带解释文字。"
        ),
    )
    reasoning: str = Field(
        default="",
        description="一句话说明生成这条SQL的思路。",
    )


class Text2SqlResult(BaseModel):
    """返回给上层API层的完整结果对象

    同时携带LLM输出结果 + RAG检索证据，
    调用方（甚至用户）可以看到是哪些知识库片段支撑生成答案。
    """

    sql: str
    reasoning: str = ""
    retrieved_chunks: list[KnowledgeChunk] = Field(default_factory=list)


def _sql_from_content(content: str) -> str:
    """兜底兼容函数：尽力从纯文本中提取干净的 SELECT / WITH SQL"""
    text = content or ""
    # 匹配 ```sql ... ``` 代码块
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    # 找到第一个以 WITH / SELECT 开头的语句
    m = re.search(r"(?is)(WITH|SELECT)\b.*", text)
    if not m:
        return ""
    # 截断到第一个分号，去掉多余内容并去除首尾空格
    sql = m.group(0).split(";")[0].strip()
    # 校验开头必须是 SELECT 或 WITH，否则返回空字符串
    return sql if re.match(r"(?is)^(SELECT|WITH)\b", sql) else ""


def _extract_sql_generation(msg) -> SqlGeneration:
    """从LLM返回的AIMessage中解析出SqlGeneration对象

    兼容部分Qwen3模型的特殊行为：
    有些时候模型会一次性返回多个并行tool_call（多段partial payload），
    with_structured_output 原生会直接拒绝这种格式；
    还有时候模型不使用工具调用，直接在content里写SQL文本。

    处理逻辑：优先读取tool_call里的sql字段；如果没有，则手动解析content文本提取SQL。
    """
    # 遍历工具调用列表，找到第一个携带sql参数的工具返回结果
    for tc in getattr(msg, "tool_calls", None) or []:
        args = dict(tc.get("args") or {})
        if args.get("sql"):
            return SqlGeneration(sql=str(args["sql"]), reasoning=str(args.get("reasoning", "")))
    # 没有合法tool_call，则从消息文本里提取SQL
    sql = _sql_from_content(str(getattr(msg, "content", "") or ""))
    if sql:
        return SqlGeneration(sql=sql)
    # 两种方式都拿不到SQL，抛出异常
    raise ValueError("模型返回结果中没有可用的工具调用SQL，也没有纯文本SQL")


class Text2SqlGenerator:
    """无状态SQL生成器。每个请求新建实例，或者全局缓存复用都安全。"""

    def __init__(self, chat_model=None, *, max_rows: int = 1000, retriever: KnowledgeRetriever | None = None):
        self.chat_model = chat_model or get_chat_model()
        self.max_rows = max_rows
        # 加载提示词模板，自动识别 {变量名} 占位符
        self.prompt = ChatPromptTemplate.from_template(load_prompt("new_sql_generate"))
        # 绑定结构化输出工具，强制LLM调用SqlGeneration工具
        self.structured = self.chat_model.bind_tools([SqlGeneration], tool_choice="SqlGeneration")
        # 组装LangChain链：提示词模板 -> LLM结构化调用 -> 自定义解析函数
        self.chain = self.prompt | self.structured | _extract_sql_generation
        # 检索器懒加载：实例创建时不初始化，真正调用agenerate的时候才使用
        self._retriever = retriever

    async def agenerate(
        self,
        *,
        question: str,
        evidence: str = "",
        previous_steps: str = "(none)",
        execution_description: str = (
            "Generate a single SQL statement answering the user's question."
        ),
        chunks: list[KnowledgeChunk] | None = None,
    ) -> Text2SqlResult:
        # ------------------------------------------------------------------
        # 第7阶段新增逻辑：调用方可以直接传入知识库片段（例如search_schema工具查到的表结构）。
        # 如果传入chunks，则**跳过内部RAG召回**，工具获取的表结构直接送入提示词，不再执行向量检索。
        # ------------------------------------------------------------------
        if chunks:
            used_chunks = list(chunks)
            # 将知识库片段转换成提示词可用的表结构文本、业务说明文本
            schema_info, recalled_evidence = chunks_to_prompt_parts(used_chunks)
        else:
            # 第5阶段原始逻辑：针对用户问题召回top-K知识库片段（表结构+业务知识）
            # 如果检索器什么都没召回，会降级使用全量业务规则
            retriever = self._retriever or get_retriever()
            recalled = retriever.recall(question)
            used_chunks = recalled.chunks
            schema_info = recalled.schema_text
            recalled_evidence = recalled.evidence_text

        # 如果没有外部传入业务证据，则使用RAG召回的证据，兜底读取全局业务规则
        if not evidence:
            evidence = recalled_evidence or business_rules()

        # 极端兜底：如果schema_info为空（知识库完全没查到表结构），
        # 替换为预定义完整schema文档，保证传给大模型的schema占位永远不为空
        if "no schema knowledge retrieved" in schema_info:
            schema_info = schema_doc()

        # 执行LangChain调用链，传入所有prompt变量
        gen: SqlGeneration = await self.chain.ainvoke(
            {
                "dialect": "MySQL 8.0",
                "schema_info": schema_info,
                "evidence": evidence,
                "question": question,
                "previous_step_results": previous_steps,
                "execution_description": execution_description,
                "max_rows": self.max_rows,
            }
        )
        # 封装结果并返回
        return Text2SqlResult(
            sql=gen.sql,
            reasoning=gen.reasoning,
            retrieved_chunks=used_chunks,
        )


# ---------- 工厂函数 + 单例获取 -----------------------------------------


def get_text2sql_generator(
    *,
    max_rows: int = 1000,
    chat_model=None,
    retriever: "KnowledgeRetriever | None" = None,
) -> Text2SqlGenerator:
    """获取Text2SqlGenerator实例

    不带参数调用时，会返回懒加载的单例对象，
    LangGraph工作流可以直接无参调用 `get_text2sql_generator()` 获取生成器。
    """
    return Text2SqlGenerator(chat_model=chat_model, max_rows=max_rows, retriever=retriever)


# 模块引用，暴露为 `app.agent.text2sql.text2sql_module`
# 用于单元测试猴子补丁monkey patch，graph.py里会引用这个模块
import sys as _sys
text2sql_module = _sys.modules[__name__]