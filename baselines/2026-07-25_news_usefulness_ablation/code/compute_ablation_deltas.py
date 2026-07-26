"""Compute per-ticker delta (all-ON dual-group minus HAR-only reference) and derive an ON/OFF
ticker classification for the next gated baseline.

QLIKE is the primary criterion (continuous, academic-standard for volatility, far less noisy
than per-ticker DirAcc over ~163 points — see requirements.md §1). delta_qlike < 0 means the
news model's QLIKE is LOWER (better) than HAR-only for that ticker -> news helps -> ON.
MSE delta and DirAcc delta are reported alongside for triangulation, not as the primary
classifier (DirAcc especially is known to be noisy at this sample size, per today's earlier
two experiments).

Run: python compute_ablation_deltas.py
"""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _latest_har_only_ref():
    candidates = sorted((_ROOT / "results").glob("har_only_ablation_ref_*/results.json"))
    if not candidates:
        raise FileNotFoundError("no har_only_ablation_ref_*/results.json found — run train_har_only_reference.py first")
    return candidates[-1]


def main():
    har_path = _latest_har_only_ref()
    all_on_path = _ROOT / "results" / "all_on_dual_group_per_ticker_eval.json"

    har = json.loads(har_path.read_text(encoding="utf-8"))
    all_on = json.loads(all_on_path.read_text(encoding="utf-8"))

    har_per_ticker = har["per_ticker_test_metrics"]
    on_per_ticker = all_on["per_ticker_test_metrics"]

    tickers = sorted(set(har_per_ticker) & set(on_per_ticker))
    missing = set(har_per_ticker) ^ set(on_per_ticker)
    if missing:
        print(f"[warn] tickers not in both models, excluded: {missing}")

    rows = []
    for t in tickers:
        h, o = har_per_ticker[t], on_per_ticker[t]
        delta_qlike = o["qlike"] - h["qlike"]     # negative = news improves QLIKE (better)
        delta_mse = o["mse"] - h["mse"]           # negative = news improves MSE (better)
        delta_diracc = (o["dir_acc"] or 0.0) - (h["dir_acc"] or 0.0)  # positive = news improves DirAcc
        rows.append({
            "ticker": t, "delta_qlike": delta_qlike, "delta_mse": delta_mse,
            "delta_diracc": delta_diracc,
            "har_qlike": h["qlike"], "on_qlike": o["qlike"],
        })

    rows.sort(key=lambda r: r["delta_qlike"])  # most negative (best improvement) first

    print(f"{'Ticker':<8}{'dQLIKE':>12}{'dMSE':>14}{'dDirAcc':>10}{'  Verdict'}")
    print("-" * 60)
    news_on, news_off = [], []
    for r in rows:
        verdict = "ON" if r["delta_qlike"] < 0 else "OFF"
        (news_on if verdict == "ON" else news_off).append(r["ticker"])
        print(f"{r['ticker']:<8}{r['delta_qlike']:>12.5f}{r['delta_mse']:>14.2e}"
              f"{r['delta_diracc']:>9.2f}%  {verdict}")

    print(f"\nNEWS_ON  ({len(news_on)}): {sorted(news_on)}")
    print(f"NEWS_OFF ({len(news_off)}): {sorted(news_off)}")

    # Triangulation: how many of the QLIKE-based ON tickers also improve on MSE and DirAcc?
    agree_mse = sum(1 for r in rows if r["ticker"] in news_on and r["delta_mse"] < 0)
    agree_diracc = sum(1 for r in rows if r["ticker"] in news_on and r["delta_diracc"] > 0)
    print(f"\nOf {len(news_on)} QLIKE-based NEWS_ON tickers: "
          f"{agree_mse} also improve on MSE, {agree_diracc} also improve on DirAcc "
          f"(agreement across all 3 metrics would be the most trustworthy signal)")

    out = {
        "har_only_source": str(har_path), "all_on_source": str(all_on_path),
        "per_ticker_deltas": rows,
        "news_on_tickers": sorted(news_on), "news_off_tickers": sorted(news_off),
    }
    out_path = _ROOT / "results" / "ablation_derived_ticker_classification.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] -> {out_path}")


if __name__ == "__main__":
    main()
