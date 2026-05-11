# AI-Powered Legal Assistance Platform (RAG + FAISS + LLaMA/Gemma)

Retrieval-Augmented Generation (RAG) based legal assistant.

## What it does
- Loads a folder of legal documents (txt/md/pdf converted externally to txt recommended)
- Builds semantic embeddings and indexes them with **FAISS**
- Retrieves **top-k** relevant passages for a user question
- Uses an LLM (LLaMA/Gemma via HuggingFace Transformers) to answer using the retrieved context
- Includes optional hooks for LoRA fine-tuning + quantization (training code structure is provided)

> Note: You must supply your own LLM model name and (if needed) HF authentication.

## Project structure
- `src/ingest.py` : ingest documents -> build FAISS index
- `src/retriever.py` : FAISS retriever utilities
- `src/llm.py` : LLM wrapper (HuggingFace)
- `src/rag.py` : end-to-end RAG pipeline
- `src/evaluate.py` : simple semantic similarity evaluation (optional)
- `app.py` : CLI + sample server endpoint

## Setup
```bash
cd ai-powered-legal-assistant-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```



## Ingest documents
Put your legal docs into `data/legal_docs/` (one file per case/document).

Example:
```bash
python app.py ingest --docs data/legal_docs --out indexes/legal_faiss
```

## Ask a question (RAG)
```bash
python app.py ask \
  --index indexes/legal_faiss \
  --question "What is the test for..." \
  --model google/gemma-2b-it
```


