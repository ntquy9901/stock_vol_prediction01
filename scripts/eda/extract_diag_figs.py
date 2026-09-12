"""Extract embedded base64 PNG figures from a self-contained diagnostic HTML report.

Reproducible extraction of the gamma-GBM S&P 500 diagnostic panels used as the earnings
motivation figures in docs/paper/2026-09-12_sp500_hose_gnnhar_final.md. Source HTML:
docs/reports/2026-09-09_gbm_sp500_diagnostic.html (built by build_gbm_diagnostic.py).

The HTML embeds four figures per horizon in order: the decile figure (panels A+B), the
forecast-bias figure (panel C), the storm-episode figure (panel D), and the permutation
feature-importance figure (panel E). We keep the three load-bearing motivation panels from
the h5 set: remaining error by decile (index 0), forecast bias (index 1), importance (index 3).
"""
import base64
import re
from pathlib import Path

_DATA_URI = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")

# index in HTML order (h5 set) -> output filename for the three motivation panels
SELECTION = {
    0: "fig_remaining_error_by_decile.png",
    1: "fig_forecast_bias_by_decile.png",
    3: "fig_feature_importance.png",
}
SRC = Path("docs/reports/2026-09-09_gbm_sp500_diagnostic.html")
OUT_DIR = Path("docs/paper/figures")


def extract_pngs(html_text):
    """Return decoded PNG byte strings, one per base64 data-URI, in document order."""
    return [base64.b64decode(m) for m in _DATA_URI.findall(html_text)]


def main():  # pragma: no cover
    pngs = extract_pngs(SRC.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, name in SELECTION.items():
        (OUT_DIR / name).write_bytes(pngs[idx])
        print("wrote", OUT_DIR / name, len(pngs[idx]), "bytes")


if __name__ == "__main__":  # pragma: no cover
    main()
