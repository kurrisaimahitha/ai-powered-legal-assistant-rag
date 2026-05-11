from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EvalResult:
    semantic_similarity: float


def evaluate_semantic_similarity(embed_model_name: str, predictions: list[str], references: list[str]) -> EvalResult:
    """Simple evaluation: cosine similarity between predicted and reference answers."""
    assert len(predictions) == len(references)

    model = SentenceTransformer(embed_model_name)
    pred_emb = model.encode(predictions, normalize_embeddings=True)
    ref_emb = model.encode(references, normalize_embeddings=True)

    sims = (pred_emb * ref_emb).sum(axis=1)
    return EvalResult(semantic_similarity=float(np.mean(sims)))

