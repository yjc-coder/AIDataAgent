"""Business + schema knowledge loader.

Phase 3 reads the rule markdowns from disk and concatenates them into
two Prompt fragments:
    - `business_rules()` -> injected as `{evidence}`
    - `schema_doc()`     -> injected as `{schema_info}`

Phase 5 will reuse the SAME .md files as Chroma documents — single
source of truth across both paths.
"""
from functools import lru_cache
from pathlib import Path

# data/knowledge/ is a sibling of app/, not a Python package — no __init__.py needed.
_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"


def _read_doc(rel: str) -> str:
    path = _KNOWLEDGE_DIR / rel
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def schema_doc() -> str:
    """Returns the schema knowledge Markdown for prompt injection.

    Today this is one Markdown summary table; Phase 5 RAG will swap
    in semantic-recall results, but the file remains canonical.
    """
    return _read_doc("schema/ecommerce.md")


@lru_cache(maxsize=None)
def business_rules() -> str:
    """Concatenates every `business/**/*.md` Markdown.

    Output is delimited by `### relative/path.md` headers so the LLM
    can pinpoint which rule it relied on when answering.
    """
    base = _KNOWLEDGE_DIR / "business"
    if not base.exists():
        return ""
    parts: list[str] = []
    for path in sorted(base.glob("**/*.md")):
        rel = path.relative_to(_KNOWLEDGE_DIR).as_posix()
        parts.append(f"### {rel}\n")
        parts.append(path.read_text(encoding="utf-8").strip())
        parts.append("")
    return "\n".join(parts).strip()
