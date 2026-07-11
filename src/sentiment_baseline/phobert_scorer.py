"""Transformer-based Vietnamese sentiment scorer (ISOLATED alternative to lexicon).

Uses a HuggingFace `transformers` sentiment-analysis pipeline. Default model is
a reliable multilingual sentiment model (XLM-RoBERTa) that covers Vietnamese;
swap to a PhoBERT-based sentiment head via --model.

Design notes:
- LAZY load: importing this module is free; the model only downloads/loads when
  `--scorer phobert` is selected. So lexicon runs are unaffected.
- Same `[-1, 1]` output scale as lexicon.py → downstream scaling in the dataset
  subclass still applies unchanged.
- Batch inference (`score_batch`) — much faster than one-by-one for ~11k titles.
- Does NOT modify lexicon.py, the existing processing flow (default), or any data.
  Existing lexicon sentiment files / running processes are untouched.

Requires: `pip install transformers` (torch is already a project dep).
First run downloads the model (~1 GB for the default) to the HF cache.

Example:
    python -m src.sentiment_baseline.process_news_to_sentiment \
        --scorer phobert --out_dir data/sentiment_baseline_phobert
    # or a specific PhoBERT sentiment model:
    python -m src.sentiment_baseline.process_news_to_sentiment \
        --scorer phobert --model 5cents/phobert-base-vn-sentiment \
        --out_dir data/sentiment_baseline_phobert
"""
import os

_PIPELINE = None
_MODEL_NAME = None

# Default: multilingual sentiment model (covers Vietnamese, reliable, 3-class).
DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Map common pipeline labels -> [-1, 1]
_LABEL_SCORE = {
    "very negative": -1.0, "negative": -0.6, "neutral": 0.0,
    "positive": 0.6, "very positive": 1.0,
    "neg": -0.6, "neu": 0.0, "pos": 0.6,
    "1 star": -1.0, "2 stars": -0.6, "3 stars": 0.0, "4 stars": 0.6, "5 stars": 1.0,
    "label_0": -0.6, "label_1": 0.0, "label_2": 0.6,  # generic fallback
}


def _load(model_name=None):
    """Lazy-load the HF pipeline on first use."""
    global _PIPELINE, _MODEL_NAME
    if _PIPELINE is not None:
        return
    from transformers import pipeline, AutoTokenizer  # imported lazily
    _MODEL_NAME = model_name or os.environ.get("SENTIMENT_MODEL", DEFAULT_MODEL)
    print(f"[phobert_scorer] loading model: {_MODEL_NAME} (first use, may download)...",
          flush=True)
    # use_fast=False: use the slow SentencePiece tokenizer directly (transformers 5.x
    # fast-tokenizer conversion mishandles some SentencePiece BPE models).
    try:
        tok = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=False)
    except Exception as e:
        print(f"[phobert_scorer] slow tokenizer failed ({e}); falling back to fast.")
        tok = _MODEL_NAME
    _PIPELINE = pipeline(
        "sentiment-analysis", model=_MODEL_NAME, tokenizer=tok,
        truncation=True, max_length=256,
    )
    print(f"[phobert_scorer] model ready.", flush=True)


def _label_to_score(label, raw_score):
    """Map a pipeline label (and its confidence) to [-1, 1]."""
    key = label.lower().strip()
    if key in _LABEL_SCORE:
        return _LABEL_SCORE[key]
    # heuristic fallback from label substring + confidence sign
    if "neg" in key:
        return -float(raw_score)
    if "pos" in key:
        return float(raw_score)
    return 0.0


def score(title, model_name=None):
    """Score a single title -> float in [-1, 1]."""
    return score_batch([title], model_name=model_name)[0]


def score_batch(titles, model_name=None, batch_size=32):
    """Score many titles in batch. Returns list[float] in [-1, 1]."""
    _load(model_name)
    safe = [ (t or "") for t in titles ]
    results = _PIPELINE(safe, batch_size=batch_size, truncation=True, max_length=256)
    out = []
    for r in results:
        s = _label_to_score(r.get("label", ""), float(r.get("score", 0.0)))
        out.append(max(-1.0, min(1.0, s)))
    return out
