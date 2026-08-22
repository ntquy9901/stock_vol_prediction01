"""Combine the two S&P500 single-seed masked-rich runs (seed 42 + seed 123) into a 2-seed mean table.

HAR / HAR-X are deterministic OLS (identical across seeds); LSTM and LSTM+GAT are averaged over the two
seeds. Prints a plain table and writes a markdown fragment to results/_seed123_root/sp500_2seed_table.md.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S42 = REPO / "results" / "masked_rich_floor1e2"
S123 = REPO / "results" / "_seed123_root" / "results" / "masked_rich_floor1e2"
MODELS = ["HAR-X", "LSTM", "LSTM_wGAT_vol2pk"]
DISP = {"HAR-X": "HAR", "LSTM": "LSTM", "LSTM_wGAT_vol2pk": "LSTM+GAT"}
METS = ["mse", "rmse", "mae", "qlike", "r2"]


def load(root, h):
    return json.loads((root / f"sp500_h{h}" / "result.json").read_text())["metrics"]


def fmt(v, k):
    if k == "mse":
        return f"{v * 1e7:.3f}"
    if k in ("rmse", "mae"):
        return f"{v * 1e4:.3f}"
    return f"{v:.4f}"


def main():
    md = ["| h | Model | MSE (x1e-7) | RMSE (x1e-4) | MAE (x1e-4) | QLIKE | R2 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    print("S&P500 2-SEED MEAN (seeds 42 & 123); HAR = 5-feature model. * = best per horizon.")
    for h in (1, 5, 10, 22):
        a = load(S42, h)
        b = load(S123, h)
        mean = {m: {k: (a[m][k] + b[m][k]) / 2 for k in METS} for m in MODELS}
        best = {}
        for k in METS:
            vals = {m: mean[m][k] for m in MODELS}
            best[k] = max(vals, key=vals.get) if k == "r2" else min(vals, key=vals.get)
        for m in MODELS:
            cells_md, cells_txt = [], []
            for k in METS:
                s = fmt(mean[m][k], k)
                cells_md.append(f"**{s}**" if best[k] == m else s)
                cells_txt.append((s + "*") if best[k] == m else s)
            hc = str(h) if m == MODELS[0] else ""
            extra = "" if m == "HAR-X" else f"  (QLIKE s42={a[m]['qlike']:.4f} s123={b[m]['qlike']:.4f})"
            print(f"{hc:>3} {DISP[m]:<9}" + " ".join(f"{c:>10}" for c in cells_txt) + extra)
            md.append(f"| {hc} | {DISP[m]} | " + " | ".join(cells_md) + " |")
        print("   " + "-" * 60)
    out = REPO / "results" / "_seed123_root" / "sp500_2seed_table.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
