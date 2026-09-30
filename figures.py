"""Turns exp1/exp2/exp3 result CSVs into the three poster figures.

    python -m src.figures [--smoke]
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from . import config

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LABELS = {
    "greedy": "Greedy",
    "beam4": "Beam (k=4)",
    "sample_t0.7": "Sample T=0.7",
    "sample_t1.0_topp0.9": "Sample T=1.0, p=0.9",
    "sample_t1.2_topp0.95": "Sample T=1.2, p=0.95",
}


def fig1_quality_vs_diversity(suffix: str) -> None:
    q = pd.read_csv(config.RESULTS / f"exp1_quality{suffix}.csv")
    d = pd.read_csv(config.RESULTS / f"exp2_diversity{suffix}.csv")

    q_mean = q.groupby("config")["rougeL"].mean()
    d_mean = d.groupby("config")["distinct2"].mean()

    fig, ax = plt.subplots(figsize=(9.8, 2.7))
    for cfg_name in config.DECODING_CONFIGS:
        if cfg_name not in q_mean.index or cfg_name not in d_mean.index:
            continue
        ax.scatter(d_mean[cfg_name], q_mean[cfg_name], s=150,
                   color=config.FAMILY_COLORS[cfg_name], zorder=3)
        ax.annotate(LABELS[cfg_name], (d_mean[cfg_name], q_mean[cfg_name]),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.set_xlabel("Diversity — distinct-2 across resamples", fontsize=11)
    ax.set_ylabel("Quality — ROUGE-L", fontsize=11)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    out = config.FIGURES / f"fig1_quality_vs_diversity{suffix}.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def fig2_cost(suffix: str) -> None:
    c = pd.read_csv(config.RESULTS / f"exp3_cost{suffix}.csv")
    mean_latency = c.groupby("config")["latency_s"].mean().reindex(config.DECODING_CONFIGS)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    names = [LABELS[k] for k in mean_latency.index]
    colors = [config.FAMILY_COLORS[k] for k in mean_latency.index]
    ax.bar(names, mean_latency.values, color=colors)
    ax.set_ylabel("Mean latency per generation (s)", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = config.FIGURES / f"fig2_cost{suffix}.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def fig3_example_table(suffix: str) -> None:
    """Not a plot — a small HTML snippet the poster embeds directly, showing
    one prompt's output under every decoding strategy side by side."""
    examples = json.loads((config.RESULTS / f"exp2_examples{suffix}.json").read_text(encoding="utf-8"))
    prompt = examples[0]["prompt"] if examples else ""

    rows = []
    for ex in examples:
        sample = ex["samples"][0] if ex["samples"] else ""
        rows.append(f"<tr><td><strong>{LABELS[ex['config']]}</strong></td><td>{sample}</td></tr>")

    html = f"""<table>
  <tr><th>Strategy</th><th>Output</th></tr>
  {''.join(rows)}
</table>
<p class="prompt-caption">Prompt: <em>{prompt}</em></p>
"""
    out = config.FIGURES / f"fig3_example_table{suffix}.html"
    out.write_text(html, encoding="utf-8")
    print(f"[figures] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""

    fig1_quality_vs_diversity(suffix)
    fig2_cost(suffix)
    fig3_example_table(suffix)


if __name__ == "__main__":
    main()
