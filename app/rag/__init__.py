"""
RAG（Retrieval-Augmented Generation）知识库检索模块（第5阶段引入）

本模块基于 Chroma 向量数据库，为 Text2SQL 流程提供【表结构 + 业务规则】的
知识召回能力，避免把全量知识一次性塞进 Prompt 造成 token 浪费和上下文污染。

三个子模块遵循单一职责原则，各司其职：

    documents.py     -> 知识库加载与切片
        从 data/knowledge/{schema,business}/ 目录读取所有 *.md 文件，
        按 Markdown 二级标题（##）切分成自包含的知识片段（chunk），
        每个片段携带 type/source/title 等元数据，便于检索后回流到 Prompt 对应槽位。

    vector_store.py  -> Chroma 向量库薄封装
        封装 chromadb.PersistentClient（本地嵌入式，无需 Docker/服务端），
        负责将 KnowledgeChunk 列表持久化为向量索引，并提供相似度检索接口。
        Embedding 函数根据 .env 配置自动选择：
          - 有真实 API Key → 调用 OpenAI/DashScope/DeepSeek 兼容的 Embedding 接口
          - 无 Key（如单元测试）→ 退化为基于 SHA-256 的确定性哈希 Embedding

    retriever.py     -> 检索器门面
        对外暴露 KnowledgeRetriever.recall(question) 接口，
        返回 RecallResult(schema_text, evidence_text, chunks) 三元组，
        其中 schema_text / evidence_text 已格式化为可直接填充 Prompt 占位符的字符串。

懒加载策略：
    检索器实例与向量索引在首次请求时才构造（耗时约5~10秒做 Embedding），
    这样 FastAPI 应用启动是即时的，即使 Embedding 服务不可用也不影响启动。
    不可用时记一条 warning 日志，Text2SQL 生成器会降级为全量业务规则注入。
"""

from app.rag.documents import KnowledgeChunk, load_knowledge_chunks
from app.rag.retriever import KnowledgeRetriever, get_retriever

__all__ = [
    "KnowledgeChunk",
    "KnowledgeRetriever",
    "get_retriever",
    "load_knowledge_chunks",
]
