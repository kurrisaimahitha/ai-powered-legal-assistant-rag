from __future__ import annotations

from src.llm import generate_answer, load_llm
from src.retriever import FaissRetriever


def answer_question(
    *,
    index_dir: str,
    question: str,
    llm_model_name: str,
    embed_model_name: str,
    top_k: int = 5,
) -> str:
    retriever = FaissRetriever(index_dir, embed_model_name=embed_model_name)
    retrieved = retriever.retrieve(question, top_k=top_k)

    context = "\n\n".join([f"[Score={r['score']:.4f}] {r['text']}" for r in retrieved])

    tokenizer, model = load_llm(llm_model_name)
    return generate_answer(tokenizer, model, question=question, context=context)

