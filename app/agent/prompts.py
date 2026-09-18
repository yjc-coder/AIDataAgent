"""Prompt template loader.

All prompt templates live as plain-text files in `app/prompts/`.
This mirrors the Java project's `resources/prompts/*.txt` design —
keeping prompts out of code makes them diff-friendly and tweakable
by non-coders.

LangChain's `ChatPromptTemplate.from_template` reads `{var}` placeholders,
so the on-disk syntax uses single-brace (NOT Jinja's `{{var}}`).
"""
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the raw template text for `name` (without `.txt` extension)."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
