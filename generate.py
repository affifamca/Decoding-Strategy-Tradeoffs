"""Wraps a single cached HF causal LM and runs it under each decoding config.

Model + tokenizer are loaded once per process and reused across every call,
same pattern as represent.py in the sibling project caching encoders.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

_MODEL = None
_TOKENIZER = None


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
        _MODEL = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
        _MODEL.eval()
    return _MODEL, _TOKENIZER


@dataclass
class GenerationResult:
    text: str
    prompt: str
    config_name: str
    latency_s: float
    n_new_tokens: int


def generate_one(prompt: str, config_name: str, max_new_tokens: int, seed: int | None = None) -> GenerationResult:
    model, tok = _load()
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tok(prompt, return_tensors="pt")
    gen_kwargs = dict(config.DECODING_CONFIGS[config_name])
    gen_kwargs.setdefault("pad_token_id", tok.eos_token_id)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, **gen_kwargs,
        )
    latency = time.perf_counter() - t0

    n_prompt_tokens = inputs["input_ids"].shape[1]
    new_tokens = out[0][n_prompt_tokens:]
    text = tok.decode(new_tokens, skip_special_tokens=True)

    return GenerationResult(
        text=text.strip(), prompt=prompt, config_name=config_name,
        latency_s=latency, n_new_tokens=int(new_tokens.shape[0]),
    )


def sequence_perplexity(text: str) -> float:
    """Reference-free fluency proxy: the model's own perplexity on its output."""
    model, tok = _load()
    if not text.strip():
        return float("nan")
    ids = tok(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return float(torch.exp(loss))
