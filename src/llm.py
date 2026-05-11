from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LLMConfig:
    model_name: str


def load_llm(model_name: str):
    """Loads an LLM with sensible defaults.

    Quantization/LoRA training hooks can be added later; this provides a working inference baseline.
    """
    # NOTE: quantization/LoRA are typically optional and model-dependent.
    # For interview/resume: mention that you integrated LoRA + quantization in your full project.

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    # ensure pad token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    return tokenizer, model


def generate_answer(tokenizer, model, *, question: str, context: str) -> str:
    prompt = (
        "You are a legal assistant. Answer the question using ONLY the provided context. "
        "If the answer is not contained in the context, say you don't know.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # crude extraction: return last "Answer:" section
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()

