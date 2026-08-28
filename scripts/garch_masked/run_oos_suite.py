"""Out-of-sample suite: run the paper's masked-rich model (main config) on new market panels.

For each dataset in argv (default: hose hnx sp500), on the liquidity+history-screened universe, run the
delivered `run_masked_rich.run` (HAR + HAR-X + LSTM + LSTM+wGAT_vol2pk, 5 seeds, 4 horizons, main
output_param='zscore_floor', --no-corr) and then add a GARCH(1,1) row (validation-aligned) to each
result.json, matching the VN30/VN100 tables exactly. Resumable: a (dataset,horizon) whose result.json
already has a GARCH metric is skipped. Writes results/masked_rich_floor1e2/<ds>_h<h>/result.json.

Screen (data-quality audit 2026-08-23): keep tickers with >=250 rows and <=50% zero-variance (H==L)
days; illiquid/short tickers make QLIKE uninformative. HNX especially is illiquid.
"""
import glob
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"
SUB = REPO / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))

import run_masked_rich as RM  # noqa: E402
import masked_rich as MR  # noqa: E402
import compute_garch_masked as CG  # noqa: E402
import floor_sensitivity as FS  # noqa: E402
from config import Config  # noqa: E402

HORIZONS = (1, 5, 10, 22)
FILES = {
    "hose": REPO / "data" / "processed" / "hose" / "*_processed.csv",
    "hnx": REPO / "data" / "processed" / "hnx" / "*_processed.csv",
    "sp500": REPO / "data" / "processed" / "sp500" / "*_processed.csv",
}
PRICE = {
    "hose": REPO / "data" / "raw" / "prices" / "hose_vnstock",
    "hnx": REPO / "data" / "raw" / "prices" / "hnx_vnstock",
    "sp500": REPO / "data" / "raw" / "prices" / "sp500",
}


def _universe_fp(files):
    """Order-independent fingerprint of the screened ticker universe (external review M-06)."""
    stems = sorted(Path(f).name.replace("_processed.csv", "") for f in files)
    return {"n": len(stems), "hash": hashlib.sha1("|".join(stems).encode()).hexdigest()[:16]}


def _has_garch(rp, ds=None, h=None, files=None):
    """A result is complete for GARCH only if it carries a finite GARCH metric AND (when the current
    dataset/horizon/universe are known) its stored fingerprint matches -- so a stale/incomplete result
    from a different screened universe or horizon is recomputed instead of silently skipped (M-06)."""
    if not rp.exists():
        return False
    try:
        res = json.loads(rp.read_text(encoding="utf-8"))
        g = res.get("metrics", {}).get("GARCH")
        # R-04: any malformed/typed field (e.g. n="n/a", null qlike) must be treated as incomplete, not raise
        if not isinstance(g, dict) or not np.isfinite(float(g.get("qlike"))) or int(g.get("n", 0)) <= 0:
            return False
        meta = res.get("garch_meta")
        if not isinstance(meta, dict) or meta.get("schema") != 1:
            return False                              # pre-fingerprint result -> recompute with metadata
        if h is not None and int(res.get("horizon", -1)) != int(h):
            return False
        if files is not None and meta.get("universe_fp") != _universe_fp(files):
            return False                              # screened universe changed -> stale
        return True
    except (ValueError, TypeError, KeyError, OSError):
        return False                                  # malformed result -> recompute


def _add_garch(ds, files, price_dir, h, cfg, rp):
    """Rebuild the (deterministic) panel, compute validation-aligned GARCH, patch result.json in place
    with metrics['GARCH'], dm_date_clustered['GARCH_vs_HARX'] (basis guard: recomputed HAR-X QLIKE must
    match the stored value) and garch_meta (degradation + universe fingerprint + provenance)."""
    D = MR.build_masked_rich(files, price_dir, cfg.lookback, h)
    harx = CG._harx_pred(D, cfg)
    res = json.loads(rp.read_text(encoding="utf-8"))
    stored = res["metrics"]["HAR-X"]["qlike"]
    mh = CG._metrics(harx, cfg.qlike_floor)
    assert abs(mh["qlike"] - stored) < 1e-4, f"basis mismatch {ds} h{h}: {mh['qlike']} vs {stored}"
    st_list = []
    g = CG._garch_pred(D, h, cfg, status_out=st_list)
    res["metrics"]["GARCH"] = CG._metrics(g, cfg.qlike_floor)
    res["dm_date_clustered"]["GARCH_vs_HARX"] = CG._dm(g, harx, h, cfg.qlike_floor)
    meta = CG.garch_meta(st_list, h, cfg)
    meta["universe_fp"] = _universe_fp(files)         # M-06: pin the screened universe for resume validation
    res["garch_meta"] = meta
    rp.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")


def main():  # pragma: no cover - CLI entry driver; exercised by real OOS runs, not unit tests
    from config import SMOKE
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    smoke = "--smoke" in sys.argv
    datasets = args or ["hose", "hnx", "sp500"]
    cfg = replace(SMOKE if smoke else Config(), batch_size=32)   # 5 seeds, 20 epochs (Config defaults)
    horizons = (1,) if smoke else HORIZONS
    for ds in datasets:
        files = sorted(glob.glob(str(FILES[ds])))
        n_all = len(files)
        files = FS.screen_files(files)          # liquidity + history screen
        price_dir = str(PRICE[ds])
        print(f"===== {ds}: {len(files)}/{n_all} tickers after screen; price_dir={price_dir} =====", flush=True)
        for h in horizons:
            rp = REPO / "results" / "masked_rich_floor1e2" / f"{ds}_h{h}" / "result.json"
            if _has_garch(rp, ds, h, files):
                print(f"[oos] {ds} h{h} already complete (has GARCH) -> skip", flush=True)
                continue
            t0 = time.time()
            RM.run(ds, files, price_dir, h, cfg, with_corr=False, output_param="zscore_floor")
            _add_garch(ds, files, price_dir, h, cfg, rp)
            m = json.loads(rp.read_text(encoding="utf-8"))["metrics"]
            print(f"[oos] {ds} h{h} DONE {time.time()-t0:.0f}s | QLIKE HAR-X={m['HAR-X']['qlike']:.4f} "
                  f"LSTM={m['LSTM']['qlike']:.4f} wGAT={m['LSTM_wGAT_vol2pk']['qlike']:.4f} "
                  f"GARCH={m['GARCH']['qlike']:.4f}", flush=True)
    print("OOS SUITE DONE", flush=True)


if __name__ == "__main__":
    main()
