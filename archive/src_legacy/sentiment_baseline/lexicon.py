"""Vietnamese financial sentiment lexicon (CRUDE - baseline only).

This is a simple keyword-counting scorer used only for the isolated 10-epoch
baseline to test end-to-end plumbing. It is NOT production quality.
Swap for PhoBERT / LLM later by replacing the `score()` function body
(keep the signature `score(title: str) -> float` in [-1, 1]).
"""

# Lowercased, diacritic Vietnamese financial keywords.
# Curated to reduce substring overlap. Some noise from overlap/false-matches
# is acceptable for a baseline signal.
POSITIVE = [
    'tăng trưởng', 'lợi nhuận', 'doanh thu', 'khuyến nghị mua',
    'vượt', 'đạt', 'kỷ lục', 'khả quan', 'thuận lợi', 'khởi sắc',
    'bứt phá', 'dẫn đầu', 'tích cực', 'hồi phục', 'thặng dư',
    'cao nhất', 'lớn nhất', 'mua', 'tăng',
]

NEGATIVE = [
    'giảm mạnh', 'thua lỗ', 'rủi ro', 'khởi tố', 'chỉ định',
    'cảnh báo', 'bất lợi', 'suy yếu', 'đình trệ', 'nợ xấu',
    'đình chỉ', 'thấp nhất', 'thiệt hại', 'phá vỡ', 'khuyến nghị bán',
    'bán', 'giảm', 'lỗ',
]


def score(title: str) -> float:
    """Return sentiment score in [-1, 1]. 0.0 = neutral / no keyword.

    Uses distinct keyword counting: (pos - neg) / (pos + neg + 1).
    """
    if not title:
        return 0.0
    t = title.lower()
    pos = sum(1 for w in POSITIVE if w in t)
    neg = sum(1 for w in NEGATIVE if w in t)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg + 1.0)
