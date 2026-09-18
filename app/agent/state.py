"""
LangGraph 工作流第7阶段：智能体状态与核心常量定义

State（状态对象）是在图中各个节点之间传递的**全局共享数据载体**。
工作流里每个节点执行完成后，只需要返回需要修改的字段构成的局部字典；
LangGraph 会自动把返回的局部字典合并更新到全局状态对象中。

原先Java版本里用 Map<String,Object> 实现的 OverAllState，
这里改用 TypedDict 类型字典，方便面试官一眼看懂完整状态结构。

第7阶段新增：工具调用支持。增加4个状态字段，用于工具调用节点，
以及条件分支判断：当RAG检索置信度不足时，路由到工具节点执行。
"""

from __future__ import annotations

from typing import TypedDict

from app.rag.documents import KnowledgeChunk


# ===================== 状态字段常量名 =====================
# 单独抽成常量，方便API层、单元测试直接引用，
# 避免硬编码字符串，同时不暴露图内部实现细节

KEY_QUESTION = "question"                # 用户当前提问
KEY_CONVERSATION_ID = "conversation_id"  # 会话唯一ID，区分多轮对话
KEY_INTENT = "intent"                    # 用户意图标识
KEY_RETRIEVED_CHUNKS = "retrieved_chunks"# RAG检索出来的知识库片段列表
KEY_GENERATED_SQL = "generated_sql"      # 大模型生成的SQL语句
KEY_GENERATED_REASONING = "generated_reasoning" # 生成SQL时的思考过程
KEY_QUERY_RESULT = "query_result"        # SQL执行返回的数据集
KEY_ROW_COUNT = "row_count"              # SQL查询结果行数
KEY_FINAL_ANSWER = "final_answer"        # 最终整理好返回给用户的回答
KEY_ERROR = "error"                      # 全局错误信息
KEY_NODE_PATH = "node_path"              # 本次流程走过的节点路径记录
KEY_SHOULD_CONTINUE = "should_continue"  # 是否继续执行后续SQL流水线
KEY_SKIP_REPORT = "skip_report"          # 是否跳过结果总结节点
KEY_SKIP_EXECUTE = "skip_execute"        # 是否跳过数据库SQL执行节点

# ---------------- 第7阶段新增：工具调用相关状态常量 ----------------
KEY_SCHEMA_NEEDS_TOOL = "schema_needs_tool"
# bool：RAG检索结果为空/检索到的表结构信息不足，需要调用工具补充元数据
KEY_TOOL_CALLED = "tool_called"
# bool：标记本轮是否已经执行过工具调用节点，用来防止循环调用
KEY_TOOL_RESULTS = "tool_results"
# list[dict]：所有工具调用返回的结果集合，一次可以调用多个工具
KEY_TOOL_ERROR = "tool_error"
# str | None：工具执行层面的错误信息（业务安全错误，不会直接中断整个图）

# ---------------- 第8阶段新增：多轮对话上下文常量（由run_graph结合持久化检查点自动填充） ----------------
KEY_PREV_QUESTION = "prev_question" # str：上一轮用户提问，首轮为空字符串
KEY_PREV_SQL = "prev_sql"           # str：上一轮生成的SQL语句
KEY_PREV_ANSWER = "prev_answer"     # str：上一轮返回给用户的最终回答
KEY_HISTORY = "history"             # list[dict]：最近多轮对话（新→旧），支持"那前年呢"跨三轮指代


# ===================== 用户意图枚举常量 =====================

INTENT_DATA_QUERY = "DATA_QUERY"
# 用户意图：数据查询 → 完整走SQL生成、数据库查询流水线
INTENT_CHAT = "CHAT"
# 用户意图：闲聊对话 → 跳过SQL相关流程，直接进入结果总结节点


# ===================== 状态结构体 TypedDict =====================

class AgentState(TypedDict, total=False):
    """
    5节点LangGraph工作流的全局共享状态
    
    total=False：代表**输入时所有字段都不是必填项**；
    各个节点在运行过程中按需填充自己产出的字段。
    """

    # ---------- 输入字段：外部传入 ----------
    question: str                      # 用户当前问题
    conversation_id: str               # 会话ID

    # ---------- 意图识别节点 IntentNode 输出 ----------
    intent: str                        # 用户意图，取值为上面 INTENT_* 常量
    should_continue: bool              # 是否继续执行SQL流水线；闲聊场景为False

    # ---------- 知识库检索节点 SchemaRecallNode 输出 ----------
    retrieved_chunks: list[KnowledgeChunk] # RAG召回的知识库文本片段

    # ---------- SQL生成节点 SQLGenerateNode 输出 ----------
    generated_sql: str                 # 模型生成的SQL语句
    generated_reasoning: str           # 模型生成SQL的推理思考过程

    # ---------- SQL执行节点 SQLExecuteNode 输出 ----------
    query_result: list[dict] | None    # 数据库查询返回的结果集
    row_count: int | None              # 查询返回的数据行数

    # ---------- 结果总结节点 ReportNode 输出 ----------
    final_answer: str                  # 整理完成、返回给用户的最终回答

    # ---------- 表结构检索节点 / 工具调用节点（第7阶段）输出 ----------
    schema_needs_tool: bool
    # RAG检索结果为空、没有命中可用表结构 → 需要调用工具获取元数据
    tool_called: bool
    # 本轮是否已经执行工具节点，用作循环保护，防止反复调用工具
    tool_results: list[dict]
    # 每一次工具调用对应的返回结果，多个工具调用会存在列表多条记录
    tool_error: str | None
    # 工具调用失败时写入错误信息；属于工具层可捕获错误，不会终止整个图

    # ---------- 多轮对话上下文，由 run_graph 加载检查点自动注入（第8阶段） ----------
    prev_question: str                 # 上一轮用户提问，第一轮为空字符串
    prev_sql: str                      # 上一轮生成的SQL
    prev_answer: str                   # 上一轮输出给用户的最终答案
    history: list[dict]                # 最近多轮对话上下文（新→旧，最多3轮），支撑跨多轮指代

    # ---------- 流程日志/辅助控制字段 ----------
    node_path: list[str]
    # 按顺序记录本次执行经过的所有节点名称，用于调试、追踪流程路径
    error: str | None
    # 记录流程中第一个捕获到的错误；存在错误也不会直接终止图运行
    skip_report: bool
    # 标记为True时，总结节点不再调用大模型生成回答
    skip_execute: bool
    # 标记为True时，SQL执行节点不会去调用数据库执行SQL语句