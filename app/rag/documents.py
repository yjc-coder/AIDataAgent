"""
知识库加载器 + 切片器（第5阶段）

功能概述：
    遍历 data/knowledge/{schema,business}/ 目录下所有 *.md 文件，
    按 Markdown 二级标题（## ）边界切片。每个切片是一个自包含的
    知识片段（KnowledgeChunk），携带如下元数据，方便检索器后续
    把它回流到 Prompt 对应的槽位：

    type     -> "schema" 或 "business"，由所在子目录推断
    source   -> 相对路径，例如 "schema/ecommerce.md"
    title    -> 切片首行（去掉 ## 前缀后的标题文本）
    section  -> H2 标题的 slug 形式，便于人类阅读的日志输出

为什么按二级标题（##）切片？
    因为我们的知识库文件按"每个 H2 对应一张表或一条业务规则"的约定组织。
    在自然边界上切片，能保证每个 chunk 语义自洽、上下文完整，
    相比固定字符数切片，召回准确率更高，也避免把一张表的字段切到两个 chunk 里。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings


# ---------------------------------------------------------------------------
# 知识库目录布局——相对项目根目录定位
# ---------------------------------------------------------------------------

# 本文件位于 app/rag/documents.py，parents[2] 即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
SCHEMA_SUBDIR = "schema"        # 表结构知识子目录名
BUSINESS_SUBDIR = "business"    # 业务规则知识子目录名
PDF_SUBDIR = "pdf"              # PDF 源文件子目录名（转换出的 md 也放在这里）


@dataclass
class KnowledgeChunk:
    """一个可被向量检索召回的知识单元。

    `id` 字段在重建索引时保持稳定——它由"文件相对路径 + 段落序号 + 标题 slug"
    组合而成，所以 Chroma 的 upsert 不会在重复索引时产生重复数据。
    """

    id: str
    text: str
    type: str  # "schema" 或 "business"，用于检索后分类回流到 Prompt
    source: str  # 相对路径，例如 "schema/ecommerce.md"
    title: str = ""  # 第一个 `## 标题` 行的文本，用于日志/调试
    metadata: dict = field(default_factory=dict)  # 附加元数据，写入 Chroma 供过滤


# ---------------------------------------------------------------------------
# 加载 + 切片
# ---------------------------------------------------------------------------


def _infer_type(rel_path: str) -> str:
    """根据文件所在子目录推断知识类型。

    路径里包含 "schema" → 表结构类型；
    路径里包含 "business" 或 "pdf" → 业务规则类型（PDF 转写文档默认
    归入业务知识——政策/报告类内容居多；表结构类知识仍建议手写 md）；
    都不匹配 → 标记为 "unknown"，后续检索时会被忽略。
    """
    parts = Path(rel_path).parts
    if SCHEMA_SUBDIR in parts:
        return "schema"
    if BUSINESS_SUBDIR in parts or PDF_SUBDIR in parts:
        return "business"
    return "unknown"


def _split_one_section(section_text: str, file_path: Path, section_idx: int) -> KnowledgeChunk:
    """把单个 Markdown 段落转换为 KnowledgeChunk。

    参数说明：
        section_text: 段落正文，**包含** `## 标题` 行本身。
        file_path:   该段落所属的 .md 文件完整路径。
        section_idx: 该段落在文件中的序号（从0开始），用于生成稳定 id。

    返回 None 表示该段落为空（既无标题也无正文），调用方应跳过。
    """
    lines = section_text.strip().splitlines()
    # 第一行是标题行；去掉 ## 前缀得到纯标题文本
    title_line = lines[0].lstrip("#").strip() if lines else ""
    # 剩余行作为正文
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    # 空段落直接跳过，不生成无意义的 chunk
    if not body and not title_line:
        return None  # type: ignore[return-value]

    # 计算相对 knowledge 根目录的路径，作为 source 字段
    rel = str(file_path.relative_to(KNOWLEDGE_DIR))
    # 拼接稳定的唯一 id：路径::序号::标题slug
    chunk_id = f"{rel}::{section_idx:03d}::{_slugify(title_line)}"
    return KnowledgeChunk(
        id=chunk_id,
        text=section_text.strip(),
        type=_infer_type(rel),
        source=rel,
        title=title_line,
        # metadata 会被原样写入 Chroma，query 返回时带回，方便上层溯源
        metadata={"source": rel, "type": _infer_type(rel), "title": title_line},
    )


def _slugify(text: str) -> str:
    """把任意标题文本转成只含字母数字和下划线的 slug。

    非字母数字字符全部替换为下划线，最后裁剪到60字符以内；
    如果结果为空，返回 "section" 兜底，保证 id 永远非空。
    """
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")[:60] or "section"


def _split_markdown_into_sections(md_text: str) -> list[str]:
    """按二级标题（## ）边界把 Markdown 文本切成段落列表。

    第一个 ## 之前的内容（通常是文件头/概述）作为独立的引导段落保留，
    不会丢失。每个段落保留它对应的 ## 标题行本身。
    """
    sections: list[str] = []
    current: list[str] = []
    for line in md_text.splitlines():
        # 遇到新的二级标题且当前已有内容 → 收尾前一段，开启新段落
        if line.startswith("## ") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    # 把最后一段也收进来
    if current:
        sections.append("\n".join(current))
    return sections


def load_knowledge_chunks() -> list[KnowledgeChunk]:
    """遍历 data/knowledge/ 目录，对每个 .md 文件按 H2 切片，输出 chunk 列表。"""
    settings = get_settings()
    max_chunk_size = settings.rag_chunk_size   # 单个 chunk 最大字符数，超过会二次切分
    overlap = settings.rag_chunk_overlap       # 二次切分时相邻片段的重叠字符数

    # 知识库目录不存在直接返回空列表（首次部署或测试环境可能没有数据）
    if not KNOWLEDGE_DIR.exists():
        return []

    chunks: list[KnowledgeChunk] = []
    # rglob 递归扫描所有子目录下的 .md 文件，sorted 保证每次构建索引顺序稳定
    for md_path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        rel = str(md_path.relative_to(KNOWLEDGE_DIR))
        text = md_path.read_text(encoding="utf-8")
        # 先按 H2 标题切大段
        sections = _split_markdown_into_sections(text)

        for idx, section_text in enumerate(sections):
            chunk = _split_one_section(section_text, md_path, idx)
            if chunk is None:
                continue

            # 如果某个段落超过最大字符限制，用字符级递归切片再切一次。
            # 实际上这里的知识库 .md 段落通常不超过30行，所以这个分支很少触发。
            if max_chunk_size and len(chunk.text) > max_chunk_size:
                chunks.extend(_resplit_large_chunk(chunk, max_chunk_size, overlap))
            else:
                chunks.append(chunk)

    return chunks


def _resplit_large_chunk(
    chunk: KnowledgeChunk, max_chunk_size: int, overlap: int
) -> list[KnowledgeChunk]:
    """超大段落兜底二次切片：用递归字符分割器再切一次。

    实现上只用标准库，不依赖 langchain 的 RecursiveCharacterTextSplitter，
    避免为了一个切片器引入整个 langchain 依赖。
    """
    text = chunk.text
    out: list[KnowledgeChunk] = []
    start = 0
    piece_idx = 0
    while start < len(text):
        end = min(len(text), start + max_chunk_size)
        piece = text[start:end]
        # 子片段继承原 chunk 的 type/source/title/metadata，并追加 subchunk 序号
        out.append(
            KnowledgeChunk(
                id=f"{chunk.id}#p{piece_idx:02d}",
                text=piece,
                type=chunk.type,
                source=chunk.source,
                title=chunk.title,
                metadata={**chunk.metadata, "subchunk": piece_idx},
            )
        )
        piece_idx += 1
        # 到文本末尾就结束
        if end >= len(text):
            break
        # 下一段起点 = 当前终点 - 重叠量，保证相邻片段有 overlap
        start = max(end - overlap, start + 1)
    return out
