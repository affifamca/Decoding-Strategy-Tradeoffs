"""Wiring check: downloads the model/dataset (if not cached) and runs one
generation per decoding config on a couple of prompts, end to end.

    python -m src.prepare --smoke
"""
from __future__ import annotations

import argparse

from . import config, data, generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    args = ap.parse_args()

    n = config.N_CONTINUATION_SAMPLES_SMOKE if args.smoke else config.N_CONTINUATION_SAMPLES
    prompts = data.load_continuation_prompts(n)
    print(f"[prepare] loaded {len(prompts)} continuation prompts")

    row = prompts[0]
    for name in config.DECODING_CONFIGS:
        result = generate.generate_one(
            row["prompt"], name, config.MAX_NEW_TOKENS_CONTINUATION, seed=config.SEED
        )
        ppl = generate.sequence_perplexity(result.text)
        print(f"[prepare] {name:22s} {result.latency_s:5.2f}s  ppl={ppl:7.1f}  "
              f"-> {result.text[:70]!r}")

    print("[prepare] OK — model, dataset and metrics wired correctly.")


if __name__ == "__main__":
    main()
