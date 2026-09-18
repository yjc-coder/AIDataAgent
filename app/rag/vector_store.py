"""
基于 Chroma 的向量存储封装（第5阶段）

唯一职责：接收一个 KnowledgeChunk 列表，做两件事——
    (a) 把它们持久化到本地 Chroma 集合（collection）中
    (b) 基于查询文本做相似度检索（similarity_search）

设计说明：
    1. 使用 chromadb.PersistentClient，纯嵌入式、无 Docker、无服务端进程。
    2. Embedding 函数由 _make_embedding_function() 根据配置自动选择：
       - 当 EMBEDDING_API_KEY 是真实可用的 key 时，使用 Chroma 内置的
         OpenAI 兼容 Embedding 函数（同时兼容 DashScope / DeepSeek / OpenAI）；
       - 当 key 缺失（如单元测试、离线演示）时，退化为基于 SHA-256 的
         确定性哈希 Embedding，保证流程依然能跑通。
    3. 首次调用 add() 之前会先检查 is_empty()，所以 ensure_indexed() 第一次
       调用会触发 Embedding 网络往返；后续调用是 no-op，直到 force_rebuild=True。
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
from typing import Iterable, Protocol

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import get_settings
from app.rag.documents import KnowledgeChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding 函数协议定义——鸭子类型，对齐 chromadb 1.5.x 的调用约定
# ---------------------------------------------------------------------------


class _EmbeddingFunction(Protocol):
    """鸭子类型协议，匹配 chromadb 1.5.x 期望的 Embedding 函数接口。

    任何实现了 name() / embed_query() / embed_documents() 的对象
    都能被 Chroma 当作 Embedding 函数使用。
    """

    def name(self) -> str: ...
    def embed_query(self, input) -> list[list[float]]: ...
    def embed_documents(self, input) -> list[list[float]]: ...


class _HashEmbedding:
    """基于哈希的确定性 Embedding 函数，用于测试 / 离线场景。

    通过 SHA-256 把每段文本映射成固定长度向量，相似度分数稳定但不具备
    真实语义能力。仅用于跑通流程，生产环境必须配置真实 API Key。
    """

    _DIM = 64           # 哈希向量维度
    is_legacy = False    # Chroma 1.5.x 的兼容标记

    def name(self) -> str:
        return "hash-fallback-v1"

    def _vec(self, text: str) -> list[float]:
        """把一段文本转换成64维向量。

        用 SHA-256 前16位作为随机种子，再借 Python 的 random 生成确定性序列。
        相同文本永远得到相同向量；不同文本由于哈希散布也有一定区分度。
        """
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        return [rng.random() for _ in range(self._DIM)]

    def __call__(self, input) -> list[list[float]]:
        # Chroma 1.5.x 会校验 __call__ 签名，这里显式转发到 embed_documents
        return self.embed_documents(input)

    def embed_query(self, input) -> list[list[float]]:
        # 兼容单字符串和列表两种输入形式
        if isinstance(input, str):
            input = [input]
        return [self._vec(t) for t in input]

    def embed_documents(self, input) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        return [self._vec(t) for t in input]


def _make_embedding_function() -> _EmbeddingFunction:
    """根据配置选择合适的 Embedding 函数。

    刻意不在模块加载阶段导入 chromadb.utils.embedding_functions——
    它会传递性引入 openai SDK，拖慢启动速度。只有当 key 看起来可用时才导入。
    """
    settings = get_settings()
    key = (settings.embedding_api_key or "").strip()
    # 占位符值不算真实 key
    looks_real = bool(key) and key != "your-api-key-here"

    if not looks_real:
        logger.warning(
            "未配置真实的 Embedding API Key；回退到基于哈希的 Embedding（仅限离线）。"
            "如需真实检索能力，请设置 EMBEDDING_API_KEY。"
        )
        return _HashEmbedding()

    try:
        from chromadb.utils import embedding_functions

        return embedding_functions.OpenAIEmbeddingFunction(  # type: ignore[return-value]
            api_key=key,
            model_name=settings.embedding_model,
            api_base=settings.embedding_base_url,
        )
    except Exception as e:
        logger.warning(
            "构造 OpenAIEmbeddingFunction 失败 (%s)；回退到哈希 Embedding。", e
        )
        return _HashEmbedding()


# ---------------------------------------------------------------------------
# 向量存储封装类
# ---------------------------------------------------------------------------


class VectorStore:
    """对 Chroma 集合（collection）的轻薄封装。

    以 chunk.id 作为主键，所以重复调用 add() 是幂等的——
    已存在的 chunk 会被自动跳过，不会产生重复向量。
    """

    def __init__(self, persist_dir: str, name: str, embedding_fn: _EmbeddingFunction):
        # 先保证目录存在，Chroma 才能创建 sqlite 持久化文件
        os.makedirs(persist_dir, exist_ok=True)
        self._name = name
        self._embedding_fn = embedding_fn
        # PersistentClient：纯嵌入式，数据写入本地 sqlite 文件
        self._client = chromadb.PersistentClient(path=persist_dir)
        # 集合不存在则创建，存在则复用，embedding 函数必须和原来保持一致
        self._collection: Collection = self._client.get_or_create_collection(
            name=name, embedding_function=embedding_fn
        )

    def reset(self) -> None:
        """删除并重建集合。

        适用场景：Embedding 模型切换后，旧向量维度和新模型不兼容
        （例如64维哈希索引被1024维 text-embedding-v3 查询），
        必须先把旧集合删掉重建。
        """
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name, embedding_function=self._embedding_fn
        )

    # ---------- 写入侧 ----------------------------------------------

    # DashScope 的 text-embedding 模型单次请求最多接受10条输入；
    # chromadb 默认会把整批数据一次性发出，这里手动分批避免超限。
    EMBED_BATCH_SIZE = 10

    def add(self, chunks: Iterable[KnowledgeChunk]) -> None:
        """把一组 KnowledgeChunk 写入向量库。

        幂等设计：先读出已有 id 集合，只写入新 chunk，
        重复调用不会产生重复向量。
        """
        chunks = list(chunks)
        if not chunks:
            return
        # 取出已有 id，过滤掉重复 chunk
        existing = set(self._collection.get(include=[]).get("ids", []) or [])
        new = [c for c in chunks if c.id not in existing]
        if not new:
            logger.info("向量库已包含全部 %d 个 chunk——无需写入。", len(chunks))
            return
        # 分批写入，每批不超过 EMBED_BATCH_SIZE 条
        for i in range(0, len(new), self.EMBED_BATCH_SIZE):
            batch = new[i : i + self.EMBED_BATCH_SIZE]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
        logger.info("索引 %d 个新 chunk（当前总计 %d 个）。", len(new), self._collection.count())

    # ---------- 读取侧 -----------------------------------------------

    def query(self, text: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """基于查询文本做相似度检索，返回 top-K 个 KnowledgeChunk。"""
        # 空集合直接返回，避免 Chroma 报错
        if self._collection.count() == 0:
            return []
        # 防御性裁剪：Chroma 的 n_results 不允许超过集合当前大小
        k = min(top_k, self._collection.count())
        # query_texts 会先把文本经 Embedding 转向量，再做余弦相似度检索
        res = self._collection.query(query_texts=[text], n_results=k)
        # query 返回的是嵌套列表结构（外层是请求批次，这里只有一条查询）
        docs = (res.get("documents") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        out: list[KnowledgeChunk] = []
        # 把 Chroma 返回的三元组重新组装成 KnowledgeChunk
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            out.append(
                KnowledgeChunk(
                    id=ids[i] if i < len(ids) else f"unknown-{i}",
                    text=doc,
                    type=str(meta.get("type", "unknown")),
                    source=str(meta.get("source", "")),
                    title=str(meta.get("title", "")),
                    metadata=dict(meta),
                )
            )
        return out

    def count(self) -> int:
        """返回当前集合中向量数量"""
        return self._collection.count()

    def is_empty(self) -> bool:
        """集合是否为空（首次启动时用于判断是否需要构建索引）"""
        return self.count() == 0


# ---------------------------------------------------------------------------
# 懒加载单例
# ---------------------------------------------------------------------------

# 模块级单例缓存：首次构造后复用，避免反复创建 Chroma 客户端
_store: VectorStore | None = None

def get_vector_store() -> VectorStore:
    """懒加载单例：首次调用时才构造 VectorStore 实例。

    这样 FastAPI 启动是即时的，即使 Embedding 服务不可用也不阻塞启动。
    """
    global _store
    if _store is None:
        s = get_settings()
        ef = _make_embedding_function()
        _store = VectorStore(
            persist_dir=s.chroma_dir,
            name=s.chroma_collection,
            embedding_fn=ef,
        )
    return _store


def ensure_indexed(force: bool | None = None) -> None:
    """如果集合为空，则对全部知识库 chunk 执行 Embedding + 入库。

    参数：
        force: True 表示强制重建索引（用于编辑 data/knowledge/*.md 后重新索引）；
               None 表示读取 .env 里的 rag_force_rebuild 配置；
               False 表示按需索引（空才建）。
    """
    settings = get_settings()
    if force is None:
        force = settings.rag_force_rebuild

    vs = get_vector_store()
    if force:
        # 先清空旧集合，再重新写入
        vs.reset()
    elif not vs.is_empty():
        # 集合非空且不强制重建 → 直接返回，避免重复 Embedding
        return

    # 延迟导入避免循环依赖
    from app.rag.documents import load_knowledge_chunks
    from app.rag.pdf_converter import load_pdf_chunks

    # md 扫描与 PDF 解析是两条同级的知识源链路，在此汇合统一入库
    chunks = load_knowledge_chunks()
    pdf_chunks = load_pdf_chunks()
    if pdf_chunks:
        logger.info("从 data/knowledge/pdf/ 解析出 %d 个 chunk。", len(pdf_chunks))
    chunks.extend(pdf_chunks)
    logger.info(
        "构建向量索引：从 data/knowledge/ 加载 %d 个 chunk（force=%s）。",
        len(chunks),
        force,
    )
    vs.add(chunks)


def reset_vector_store() -> None:
    """清空缓存的单例对象，单元测试专用。"""
    global _store
    _store = None
