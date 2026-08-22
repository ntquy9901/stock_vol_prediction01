"""Throwaway: run the masked-rich S&P500 panel with a SECOND seed (123) for all horizons.

Writes to a separate results root (results/_seed123_root/...) so the seed-42 result.json files are NOT
clobbered. Reuses the delivered run_masked_rich.run unchanged (only cfg.seeds and the output REPO are
overridden). After both seeds exist, a 2-seed mean can be assembled from the two result trees.
"""
import glob
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"
SUB = REPO / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(CODE))

import run_masked_rich as RM  # noqa: E402
from config import Config  # noqa: E402

cfg = replace(Config(), seeds=(123,), batch_size=32)
files = glob.glob(str(REPO / "data" / "processed" / "sp500" / "*_processed.csv"))
if not files:
    raise SystemExit("no sp500 processed files found under data/processed/sp500")
price_dir = str(REPO / "data" / "raw" / "prices" / "sp500")

RM.REPO = REPO / "results" / "_seed123_root"   # redirect output so seed-42 files stay intact

# optional horizon args: `python run_sp500_seed123.py 5 10 22` runs only those (for parallel one-per-process)
_hz = tuple(int(a) for a in sys.argv[1:]) or (1, 5, 10, 22)
for h in _hz:
    res = RM.run("sp500", files, price_dir, h, cfg, lookback=10, with_corr=False)
    m = res["metrics"]
    print(f"[seed123] sp500 h{h} QLIKE HAR-X={m['HAR-X']['qlike']:.4f} LSTM={m['LSTM']['qlike']:.4f} "
          f"wGAT={m['LSTM_wGAT_vol2pk']['qlike']:.4f} | MAE LSTM={m['LSTM']['mae']:.6f}", flush=True)
print("SEED123 ALL DONE", flush=True)
