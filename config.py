"""Shared configuration: paths, model, dataset and decoding-strategy definitions."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = ROOT / "data_cache"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for d in (DATA_CACHE, ARTIFACTS, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Small, no-key, fully local model. gpt2-medium is a drop-in upgrade if you
# have the compute; both download once from Hugging Face and are cached.
MODEL_NAME = "gpt2"

# One dataset, two roles: the article lead sentence is a continuation prompt,
# the full article + its highlight is a summarisation pair.
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
N_SUMMARIZATION_SAMPLES = 60
N_CONTINUATION_SAMPLES = 60
N_SUMMARIZATION_SAMPLES_SMOKE = 4
N_CONTINUATION_SAMPLES_SMOKE = 4

MAX_NEW_TOKENS_SUMMARY = 60
MAX_NEW_TOKENS_CONTINUATION = 40
CONTINUATION_PROMPT_WORDS = 12  # how many words of the lead sentence seed the prompt

# Decoding configurations under comparison. Each is a kwargs dict passed
# straight to model.generate(); "label" is what shows up in every figure/table.
DECODING_CONFIGS = {
    "greedy": dict(do_sample=False, num_beams=1),
    "beam4": dict(do_sample=False, num_beams=4, early_stopping=True),
    "sample_t0.7": dict(do_sample=True, num_beams=1, temperature=0.7, top_p=1.0, top_k=0),
    "sample_t1.0_topp0.9": dict(do_sample=True, num_beams=1, temperature=1.0, top_p=0.9, top_k=0),
    "sample_t1.2_topp0.95": dict(do_sample=True, num_beams=1, temperature=1.2, top_p=0.95, top_k=0),
}

# exp2 (diversity) draws this many independent samples per prompt per config
N_SAMPLES_PER_PROMPT = 5

FAMILY_COLORS = {
    "greedy": "#1f6f8b",
    "beam4": "#2f9e63",
    "sample_t0.7": "#e08a3c",
    "sample_t1.0_topp0.9": "#c2542c",
    "sample_t1.2_topp0.95": "#8b3a62",
}
