"""PDF 知识源解析器（Phase 11）。

用户把 PDF 丢进 ``data/knowledge/pdf/``，``load_pdf_chunks()`` 在构建向量
索引时把它们**直接解析成 KnowledgeChunk**——与 ``documents.load_knowledge_chunks()``
扫描 md 文件是同级操作，两条链路在 ``ensure_indexed`` 汇合后统一入库。
不生成任何中间 md 文件。

切片策略（与 md 链路同构）：
    按字号检测出的标题把全文切成若干节，每节再用 md 链路现成的
    ``_split_one_section`` 生成 chunk（稳定 id、类型推断、metadata 全部
    复用同一套逻辑）。第一个标题之前的内容作为引导节保留。

纯扫描件（图片型 PDF）提取不到文本时跳过并告警——OCR 不在范围内。
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.rag.documents import (
    KNOWLEDGE_DIR,
    PDF_SUBDIR,
    KnowledgeChunk,
    _resplit_large_chunk,
    _split_one_section,
)

logger = logging.getLogger(__name__)


def pdf_dir() -> Path:
    """PDF 源文件目录：data/knowledge/pdf/（不存在则自动创建）。"""
    d = KNOWLEDGE_DIR / PDF_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collect_heading_sizes(doc) -> set[float]:
    """统计全文档出现过的去重字号，取最大的 1-3 档作为标题字号。

    阈值 1.5pt：正文通常只有一个字号，比它明显大的才算标题；
    少于两档字号时返回空集（解析退化为"仅文件标题 + 单节全文"）。
    """
    sizes: dict[float, int] = {}
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0=文本块；1=图片块
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "").strip()
                    if t:
                        s = round(span.get("size", 0.0), 1)
                        sizes[s] = sizes.get(s, 0) + len(t)
    if not sizes:
        return set()
    ranked = [s for s, _ in sorted(sizes.items(), key=lambda kv: -kv[0])]
    body = max(sizes.items(), key=lambda kv: kv[1])[0]  # 出现字符最多的字号≈正文
    headings = [s for s in ranked if s >= body + 1.5]
    return set(headings[:3])


def _page_lines(page, heading_sizes: set[float], doc_title: str) -> list[tuple[str, bool]]:
    """把一页的文本块拉平成 (文本, 是否标题) 行序列。

    与 _collect_heading_sizes 一致：字号按 0.1pt 精度归并，否则真实 PDF
    里 13.98pt 这类字号会匹配不上标题档位。与文档标题相同的行跳过
    （文件名已作为文档标题，避免重复）。
    """
    out: list[tuple[str, bool]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip().lstrip("#").strip()
            if not text or text == doc_title:
                continue
            size = round(max(s.get("size", 0.0) for s in spans), 1)
            out.append((text, size in heading_sizes))
    return out


def extract_sections(pdf_path: Path) -> list[str]:
    """PDF -> markdown 风格的节列表（每节以 ``## 标题`` 开头）。

    输出格式与 ``documents._split_markdown_into_sections`` 一致，从而
    复用同一套 chunk 生成逻辑。返回空列表表示无可提取文本。
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        heading_sizes = _collect_heading_sizes(doc)
        sections: list[str] = []
        current: list[str] = []
        for page in doc:
            for text, is_heading in _page_lines(page, heading_sizes, pdf_path.stem):
                if is_heading:
                    # 新标题 → 收尾上一节，以 markdown 标题行开启新节
                    if current:
                        sections.append("\n".join(current))
                    current = [f"## {text}"]
                else:
                    current.append(text)
        if current:
            sections.append("\n".join(current))
        return sections
    finally:
        doc.close()


def load_pdf_chunks() -> list[KnowledgeChunk]:
    """解析目录下全部 PDF 为 KnowledgeChunk 列表（与 md 扫描同级）。

    单个 PDF 解析失败只告警并跳过，绝不能拖垮整个索引构建。
    """
    settings = get_settings()
    max_chunk_size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap

    chunks: list[KnowledgeChunk] = []
    for pdf_path in sorted(pdf_dir().glob("*.pdf")):
        try:
            sections = extract_sections(pdf_path)
        except Exception as e:  # noqa: BLE001 — 单个坏文件不能拖垮索引构建
            logger.exception("PDF 解析失败 %s: %s", pdf_path.name, e)
            continue

        total_text = "".join(sections).strip()
        if len(total_text) < 20:
            logger.warning(
                "PDF %s 提取不到有效文本（可能是扫描件/图片型 PDF），已跳过。OCR 不在支持范围内。",
                pdf_path.name,
            )
            continue

        for idx, section_text in enumerate(sections):
            lines = section_text.strip().splitlines()
            if len(lines) <= 1:
                # 空节：只有标题行没有正文（如封面大标题），无信息量，跳过
                continue
            chunk = _split_one_section(section_text, pdf_path, idx)
            if chunk is None:
                continue
            if max_chunk_size and len(chunk.text) > max_chunk_size:
                chunks.extend(_resplit_large_chunk(chunk, max_chunk_size, overlap))
            else:
                chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for pdf in sorted(pdf_dir().glob("*.pdf")):
        print(f"===== {pdf.name}")
        for s in extract_sections(pdf):
            print(s[:400])
            print("-" * 30)
    if "--chunks" in sys.argv:
        for c in load_pdf_chunks():
            print(c.id, "|", c.title[:20])
