"""PhoBERT (frozen) embedding extraction for news text.

Vendored verbatim from C:\\luanvan\\data_eda\\src\\nlp\\embeddings.py (2026-07-25) — copy only,
data_eda itself is not modified. Not expected to actually run in this baseline: every source's
per-article cache (data/external_news_embeddings/raw_cache/) already covers all current
crawl_data urls, so `_get_article_embeddings` should find 0 new rows. Kept importable so the
vendored aggregation code's import graph resolves. Requires ``transformers<5`` + ``sentencepiece``.
"""

from __future__ import annotations

import numpy as np


def extract_phobert_embeddings(
    texts: list[str],
    model: str = "vinai/phobert-base",
    max_len: int = 64,
    batch_size: int = 32,
) -> np.ndarray:
    """Frozen PhoBERT ``[CLS]`` embeddings for a list of texts. Returns (n, hidden_size)."""
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModel.from_pretrained(model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    raw_dim = net.config.hidden_size
    embs = np.zeros((len(texts), raw_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
            cls = net(**enc).last_hidden_state[:, 0, :]
            embs[i : i + len(batch)] = cls.cpu().numpy()

    if not np.isfinite(embs).all():
        raise ValueError(f"PhoBERT produced non-finite embeddings ({int(np.isnan(embs).sum())} NaN)")
    return embs
