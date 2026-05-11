from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    def __init__(self, index_dir: str, *, embed_model_name: str):
        index_path = Path(index_dir)
        self.index = faiss.read_index(str(index_path / "faiss.index"))

        with open(index_path / "chunks.json", "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.chunks: list[str] = payload["chunks"]
        self.meta: list[dict] = payload["meta"]

        self.model = SentenceTransformer(embed_model_name)

    def retrieve(self, question: str, *, top_k: int = 5) -> list[dict]:
        q_emb = self.model.encode([question], normalize_embeddings=True).astype("float32")

        # inner product on normalized vectors -> cosine similarity
        scores, ids = self.index.search(q_emb, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append({
                "score": float(score),
                "text": self.chunks[idx],
                "meta": self.meta[idx],
            })
        return results

