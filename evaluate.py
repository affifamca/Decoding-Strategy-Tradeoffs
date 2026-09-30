"""Metrics: reference-based (ROUGE) and reference-free (perplexity, diversity)."""
from __future__ import annotations

import re

from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(hypothesis: str, reference: str) -> dict:
    if not hypothesis.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _ROUGE.score(reference, hypothesis)
    return {k: v.fmeasure for k, v in scores.items()}


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return list(zip(*[tokens[i:] for i in range(n)]))


def distinct_n(texts: list[str], n: int) -> float:
    """Fraction of unique n-grams across a set of generations. Higher = more diverse."""
    all_ngrams = []
    for t in texts:
        toks = t.split()
        all_ngrams.extend(_ngrams(toks, n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of n-grams in a single generation that are repeats of an earlier one."""
    toks = text.split()
    grams = _ngrams(toks, n)
    if not grams:
        return 0.0
    seen = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
