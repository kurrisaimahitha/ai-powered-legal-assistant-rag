from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.text import load_text_files


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    """Simple chunker by characters (swap for tokenizer-based splitter if needed)."""
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


def build_faiss_index(docs_dir: str, out_dir: str, *, embed_model_name: str):
    """Build FAISS index from legal documents."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    docs = load_text_files(docs_dir)

    all_chunks: list[str] = []
    chunk_meta: list[dict] = []

    for d in docs:
        chunks = _chunk_text(d["text"])
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            chunk_meta.append({"source": d["id"], "chunk": i})

    model = SentenceTransformer(embed_model_name)

    # embeddings
    embeddings = []
    for i in tqdm(range(0, len(all_chunks), 32), desc="Embedding chunks"):
        batch = all_chunks[i : i + 32]
        emb = model.encode(batch, normalize_embeddings=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity when embeddings are normalized
    index.add(embeddings)

    faiss.write_index(index, str(out_path / "faiss.index"))

    with open(out_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "meta": chunk_meta}, f)

    # store embed model name for reproducibility
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"embed_model_name": embed_model_name}, f, indent=2)

    print(f"Built FAISS index: {out_path} | chunks={len(all_chunks)} | dim={dim}")

