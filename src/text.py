from __future__ import annotations


def load_text_files(path: str):
    """Loads .txt and .md files from a directory (recursively)."""
    from pathlib import Path

    p = Path(path)
    files = sorted(list(p.rglob("*.txt")) + list(p.rglob("*.md")))

    docs: list[dict] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        docs.append({"id": str(fp), "text": text})
    return docs

