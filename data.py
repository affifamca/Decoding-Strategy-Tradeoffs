"""Load and cache CNN/DailyMail prompts for the two generation tasks.

Summarisation: (article, reference_highlight) pairs, for ROUGE against a
real reference. Continuation: the first CONTINUATION_PROMPT_WORDS words of
the article as an open-ended prompt, no reference needed.
"""
from __future__ import annotations

import json

from datasets import load_dataset

from . import config


def _cache_path(name: str, n: int) -> "config.Path":
    return config.DATA_CACHE / f"{name}-n{n}.json"


def load_summarization_prompts(n: int, split: str = "validation") -> list[dict]:
    cache = _cache_path("summarization", n)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    ds = load_dataset(
        config.DATASET_NAME, config.DATASET_VERSION, split=split,
        cache_dir=str(config.DATA_CACHE / "hf"),
    )
    ds = ds.shuffle(seed=config.SEED).select(range(n))

    rows = [
        {"id": ex["id"], "article": ex["article"], "reference": ex["highlights"]}
        for ex in ds
    ]
    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def load_continuation_prompts(n: int, split: str = "validation") -> list[dict]:
    cache = _cache_path("continuation", n)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    ds = load_dataset(
        config.DATASET_NAME, config.DATASET_VERSION, split=split,
        cache_dir=str(config.DATA_CACHE / "hf"),
    )
    # offset the shuffle so continuation prompts don't overlap summarization ones
    ds = ds.shuffle(seed=config.SEED + 1).select(range(n))

    rows = []
    for ex in ds:
        words = ex["article"].split()
        prompt = " ".join(words[: config.CONTINUATION_PROMPT_WORDS])
        rows.append({"id": ex["id"], "prompt": prompt})
    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows
