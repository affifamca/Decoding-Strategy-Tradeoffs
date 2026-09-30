"""H1 — Quality: ROUGE against a real reference, plus reference-free fluency
signals, for every decoding config on the summarisation task.

    python -m src.exp1_quality [--smoke]
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from . import config, data, evaluate, generate


def run(n: int) -> pd.DataFrame:
    prompts = data.load_summarization_prompts(n)
    rows = []
    for ex in tqdm(prompts, desc="exp1 quality"):
        for cfg_name in config.DECODING_CONFIGS:
            result = generate.generate_one(
                ex["article"][:2000],  # gpt2's 1024-token window; truncate long articles
                cfg_name, config.MAX_NEW_TOKENS_SUMMARY, seed=config.SEED,
            )
            r = evaluate.rouge(result.text, ex["reference"])
            rows.append({
                "id": ex["id"],
                "config": cfg_name,
                "rouge1": r["rouge1"],
                "rouge2": r["rouge2"],
                "rougeL": r["rougeL"],
                "perplexity": generate.sequence_perplexity(result.text),
                "repetition_rate": evaluate.repetition_rate(result.text),
                "n_words": evaluate.word_count(result.text),
                "latency_s": result.latency_s,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n = config.N_SUMMARIZATION_SAMPLES_SMOKE if args.smoke else config.N_SUMMARIZATION_SAMPLES
    df = run(n)
    out = config.RESULTS / ("exp1_quality_smoke.csv" if args.smoke else "exp1_quality.csv")
    df.to_csv(out, index=False)
    print(f"[exp1] wrote {out}  ({len(df)} rows)")
    print(df.groupby("config")[["rouge1", "rouge2", "rougeL", "perplexity", "repetition_rate"]].mean())


if __name__ == "__main__":
    main()
