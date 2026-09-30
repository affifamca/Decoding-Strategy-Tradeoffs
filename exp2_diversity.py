"""H2 — Diversity: draw several independent generations per prompt under each
decoding config and measure how much they differ (distinct-n), on the
open-ended continuation task where there's no single right answer.

    python -m src.exp2_diversity [--smoke]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

from . import config, data, evaluate, generate


def run(n: int):
    prompts = data.load_continuation_prompts(n)
    rows = []
    examples = []  # first prompt's outputs, every config — for the poster's example table

    for i, ex in enumerate(tqdm(prompts, desc="exp2 diversity")):
        for cfg_name in config.DECODING_CONFIGS:
            samples = [
                generate.generate_one(
                    ex["prompt"], cfg_name, config.MAX_NEW_TOKENS_CONTINUATION,
                    seed=config.SEED + s,
                ).text
                for s in range(config.N_SAMPLES_PER_PROMPT)
            ]
            rows.append({
                "id": ex["id"],
                "config": cfg_name,
                "distinct1": evaluate.distinct_n(samples, 1),
                "distinct2": evaluate.distinct_n(samples, 2),
                "mean_repetition_rate": sum(evaluate.repetition_rate(s) for s in samples) / len(samples),
            })
            if i == 0:
                examples.append({"config": cfg_name, "prompt": ex["prompt"], "samples": samples})

    return pd.DataFrame(rows), examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n = config.N_CONTINUATION_SAMPLES_SMOKE if args.smoke else config.N_CONTINUATION_SAMPLES
    df, examples = run(n)

    suffix = "_smoke" if args.smoke else ""
    out_csv = config.RESULTS / f"exp2_diversity{suffix}.csv"
    out_json = config.RESULTS / f"exp2_examples{suffix}.json"
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[exp2] wrote {out_csv}  ({len(df)} rows)")
    print(f"[exp2] wrote {out_json}  (worked example for the poster)")
    print(df.groupby("config")[["distinct1", "distinct2", "mean_repetition_rate"]].mean())


if __name__ == "__main__":
    main()
