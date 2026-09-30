"""H3 — Cost: wall-clock latency and throughput per decoding config, holding
the prompt and output length fixed so timing differences are attributable to
the decoding strategy alone (beam width, sampling overhead), not text length.

    python -m src.exp3_cost [--smoke]
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from . import config, data, generate

N_TIMING_PROMPTS = 10
N_TIMING_PROMPTS_SMOKE = 2
N_REPEATS = 3


def run(n: int) -> pd.DataFrame:
    prompts = data.load_continuation_prompts(n)
    rows = []
    for ex in tqdm(prompts, desc="exp3 cost"):
        for cfg_name in config.DECODING_CONFIGS:
            for r in range(N_REPEATS):
                result = generate.generate_one(
                    ex["prompt"], cfg_name, config.MAX_NEW_TOKENS_CONTINUATION,
                    seed=config.SEED + r,
                )
                rows.append({
                    "id": ex["id"],
                    "config": cfg_name,
                    "repeat": r,
                    "latency_s": result.latency_s,
                    "n_new_tokens": result.n_new_tokens,
                    "tokens_per_s": result.n_new_tokens / result.latency_s if result.latency_s > 0 else float("nan"),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n = N_TIMING_PROMPTS_SMOKE if args.smoke else N_TIMING_PROMPTS
    df = run(n)
    out = config.RESULTS / ("exp3_cost_smoke.csv" if args.smoke else "exp3_cost.csv")
    df.to_csv(out, index=False)
    print(f"[exp3] wrote {out}  ({len(df)} rows)")
    print(df.groupby("config")[["latency_s", "tokens_per_s"]].mean())


if __name__ == "__main__":
    main()
