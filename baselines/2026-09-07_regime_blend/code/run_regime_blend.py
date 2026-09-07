"""Driver: run the regime-conditional blend over markets x horizons x deep bases; print a verdict table
and write results/qlike_anchor/regime_blend_result.json.

Run: .venv_gpu_encode/Scripts/python.exe baselines/2026-09-07_regime_blend/code/run_regime_blend.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regime_blend as RB  # noqa: E402

MARKETS = ["vn100", "vn30"]
HORIZONS = [1, 5, 10, 22]
DEEPS = ["VolGA", "LSTM"]
OUT = RB.REPO / "results" / "qlike_anchor" / "regime_blend_result.json"


def verdict(blend_q: float, harx_q: float, p: float) -> str:
    """Compact win/tie/loss vs HAR-X on QLIKE. ``p`` is the two-sided date-clustered DM p-value (HLN);
    combined with the directional ``blend_q < harx_q`` check it is a conservative-to-mildly-liberal WIN* gate."""
    if blend_q < harx_q and p < 0.05:
        return "WIN*"                                        # lower QLIKE and significant
    if blend_q < harx_q:
        return "win "                                        # lower but not significant
    return "tie " if abs(blend_q - harx_q) / harx_q < 0.005 else "LOSS"


def main() -> None:                                          # pragma: no cover - entry driver (I/O loop)
    results = []
    for deep in DEEPS:
        print(f"\n===== deep base = {deep} =====")
        print(f"{'market':7s} {'h':>3s} | {'deep':>7s} {'harx':>7s} {'blend':>7s} | "
              f"{'vs HARX':>8s} {'DMp':>6s} {'vs deep DMp':>11s}  meanW  verdict")
        for market in MARKETS:
            for h in HORIZONS:
                r = RB.evaluate(market, h, deep)
                if r is None:
                    print(f"{market:7s} {h:>3d} | cells absent"); continue
                results.append(r)
                q = r["test_metrics"]
                dq, hq, bq = q["deep"]["qlike"], q["harx"]["qlike"], q["blend"]["qlike"]
                pv = r["dm_blend_vs_harx"]["p"]; pd_ = r["dm_blend_vs_deep"]["p"]
                gain = (hq - bq) / hq * 100
                print(f"{market:7s} {h:>3d} | {dq:7.4f} {hq:7.4f} {bq:7.4f} | "
                      f"{gain:+7.2f}% {pv:6.3f} {pd_:11.3f}  {r['mean_w']:.2f}   {verdict(bq, hq, pv)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    assert results, "no results produced"
    print(f"\nwrote {OUT} | cells: {len(results)}")


if __name__ == "__main__":   # pragma: no cover
    main()
