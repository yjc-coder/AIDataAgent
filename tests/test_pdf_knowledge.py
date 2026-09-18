"""PDF 知识源直连解析测试（Phase 11）。

PDF 与 md 是两条同级的知识源链路：PDF 直接解析成 KnowledgeChunk 进向
量库，不产生中间 md 文件。用 PyMuPDF 现场生成迷你 PDF（无需夹具文件）验证：
    1. load_pdf_chunks 能产出 type=business 的 chunk，id 稳定且含 pdf 路径
    2. 标题（大字号）正确成为切片边界，正文完整进入 chunk
    3. 坏文件 / 扫描件被安全跳过，不拖垮整体解析
    4. ensure_indexed 后 PDF chunk 与 md chunk 汇合入库（用假向量库隔离）
"""
from __future__ import annotations

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")


def _make_pdf(path: Path, title: str, sections: list[tuple[str, str]], title_size: float = 20.0,
              heading_size: float = 14.0, body_size: float = 10.0) -> None:
    """生成一个带字号层级的迷你 PDF（标题 > 小节标题 > 正文）。

    fontname="china-s" 是 PyMuPDF 内置的简体中文字体——默认的 helv 字体
    无法编码中文，写出来的 PDF 里中文会变成点号。
    """
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60.0
    page.insert_text((50, y), title, fontsize=title_size, fontname="china-s")
    for head, body in sections:
        y += 50
        page.insert_text((50, y), head, fontsize=heading_size, fontname="china-s")
        for line in body.split("\n"):
            y += 22
            page.insert_text((50, y), line, fontsize=body_size, fontname="china-s")
    doc.save(path)
    doc.close()


@pytest.fixture
def pdf_workspace(tmp_path: Path, monkeypatch):
    """镜像生产布局：tmp_path 当 knowledge 根，其下 pdf/ 子目录放源 PDF。

    chunk 的 type 推断依赖相对路径中的 pdf 目录名，所以目录结构必须与
    生产一致（KNOWLEDGE_DIR/pdf/*.pdf）。
    """
    from app.rag import documents as docs_mod
    from app.rag import pdf_converter

    knowledge_root = tmp_path
    pdf_root = knowledge_root / "pdf"
    pdf_root.mkdir()
    monkeypatch.setattr(pdf_converter, "pdf_dir", lambda: pdf_root)
    monkeypatch.setattr(docs_mod, "KNOWLEDGE_DIR", knowledge_root)
    return knowledge_root


def test_load_pdf_chunks_produces_business_chunks(pdf_workspace: Path):
    from app.rag.pdf_converter import load_pdf_chunks

    _make_pdf(
        pdf_workspace / "pdf" / "refund_policy.pdf",
        "会员退货政策",
        [
            ("退货时限", "签收后 7 天内可无理由退货。\n生鲜类商品不支持无理由退货。"),
            ("退款口径", "退款金额 = 实付金额 - 已使用的优惠券。"),
        ],
    )
    chunks = load_pdf_chunks()

    assert len(chunks) == 2  # 两个小节各一个 chunk
    assert all(c.type == "business" for c in chunks)
    assert all(Path(c.source).name == "refund_policy.pdf" for c in chunks)
    assert any("退货时限" in c.title for c in chunks)
    assert any("7 天内可无理由退货" in c.text for c in chunks)
    assert any("退款口径" in c.title for c in chunks)
    # chunk id 基于相对路径 + 序号 + 标题，跨次解析保持稳定
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all("refund_policy.pdf" in i for i in ids)


def test_broken_pdf_is_skipped_without_crashing(pdf_workspace: Path):
    from app.rag.pdf_converter import load_pdf_chunks

    _make_pdf(pdf_workspace / "pdf" / "good.pdf", "正常文档", [("小节", "这是一段足够长的正常内容。")])
    (pdf_workspace / "pdf" / "broken.pdf").write_bytes(b"not a real pdf")

    chunks = load_pdf_chunks()
    assert any("good.pdf" in c.source for c in chunks)
    assert not any("broken.pdf" in c.source for c in chunks)


def test_scanned_pdf_is_skipped_safely(pdf_workspace: Path):
    from app.rag.pdf_converter import load_pdf_chunks

    # 一页纯矢量图形、无任何文本的 PDF：提取不到文本
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect((50, 50, 200, 200))
    doc.save(pdf_workspace / "pdf" / "scanned.pdf")
    doc.close()

    assert load_pdf_chunks() == []


def test_ensure_indexed_merges_md_and_pdf_chunks(pdf_workspace: Path, monkeypatch):
    """ensure_indexed 应把 md 链路与 PDF 链路的 chunk 汇合入库。"""
    from app.rag import documents as docs_mod
    from app.rag.documents import KnowledgeChunk
    from app.rag.vector_store import ensure_indexed

    _make_pdf(pdf_workspace / "pdf" / "promo.pdf", "促销规则", [("满减", "满 300 减 50，可与会员折扣叠加。")])

    added: list[KnowledgeChunk] = []

    class _FakeVS:
        def add(self, chunks):
            added.extend(chunks)

        def reset(self):
            added.clear()

        def is_empty(self):
            return not added

    md_chunk = KnowledgeChunk(
        id="business/sales.md::001::sales",
        text="sales = sum(amount) where status='paid'",
        type="business",
        source="business/sales.md",
        title="sales",
        metadata={"type": "business"},
    )

    monkeypatch.setattr(docs_mod, "load_knowledge_chunks", lambda: [md_chunk])
    monkeypatch.setattr("app.rag.vector_store.get_vector_store", lambda: _FakeVS())

    ensure_indexed(force=True)

    assert any("sales.md" in c.source for c in added)      # md 链路
    assert any("promo.pdf" in c.source for c in added)     # PDF 链路
