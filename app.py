from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest import build_faiss_index
from src.rag import answer_question


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Legal Assistance (RAG)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents and build FAISS index")
    p_ingest.add_argument("--docs", type=str, required=True)
    p_ingest.add_argument("--out", type=str, required=True)
    p_ingest.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    p_ask = sub.add_parser("ask", help="Ask a question with RAG")
    p_ask.add_argument("--index", type=str, required=True)
    p_ask.add_argument("--question", type=str, required=True)
    p_ask.add_argument("--model", type=str, default="google/gemma-2b-it")
    p_ask.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_ask.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "ingest":
        build_faiss_index(docs_dir=args.docs, out_dir=args.out, embed_model_name=args.embed_model)
        return

    if args.cmd == "ask":
        out = answer_question(
            index_dir=args.index,
            question=args.question,
            llm_model_name=args.model,
            embed_model_name=args.embed_model,
            top_k=args.top_k,
        )
        print(out)
        return


if __name__ == "__main__":
    main()

