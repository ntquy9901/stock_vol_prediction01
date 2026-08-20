"""Run ONE experiment config (dataset, lookback, horizon): HAR-LSTM-GAT + LSTM(w/o GAT) + HAR + GARCH.

Seed-ensembles the deep models, evaluates all 5 metrics on the shared test observations, and runs
Diebold-Mariano (Ours vs HAR / GARCH / w/o-GAT) on QLIKE + SE. Designed to be launched many times in
parallel (one process per (dataset,lookback,horizon)) to exploit the otherwise-idle GPU.

CLI:  python run_all.py <dataset> <lookback> <horizon> [--smoke] [--data-root DIR] [--out DIR]
  dataset in {vn30, vn100, sp500}
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

import edges
import evaluate as E
from config import Config, SMOKE
from snapshots import build_snapshots
from train import train_pooled

HERE = Path(__file__).resolve().parent


def resolve_files(dataset: str, data_root: str) -> list[str]:
    root = Path(data_root)
    mapping = {
        "vn30": sorted(glob.glob(str(root / "vn30" / "*_processed.csv"))) or
                sorted(glob.glob(str(root / "*_processed.csv"))),
        "vn100": sorted(glob.glob(str(root / "vn100" / "*_processed.csv"))) or
                 sorted(glob.glob(str(root / "vn100_vnstock" / "*_processed.csv"))),
        "sp500": sorted(glob.glob(str(root / "sp500" / "*_processed.csv"))),
    }
    files = mapping[dataset]
    if not files:
        raise FileNotFoundError(f"no processed CSVs for {dataset} under {data_root}")
    return files


def run_experiment(dataset: str, files: list[str], lookback: int, horizon: int, cfg: Config,
                   out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train.log", "w", encoding="utf-8")
    t0 = time.time()
    snap = build_snapshots(files, lookback, horizon,
                           train_frac=cfg.train_frac, val_frac=cfg.val_frac)
    adj = edges.glasso_adjacency(snap.adj_pk_train, top_k=cfg.top_k)
    print(f"[basis] {dataset} lb{lookback} h{horizon} nodes={snap.num_nodes} "
          f"train_snaps={len(snap.train)} test_snaps={len(snap.test)} "
          f"edges={(adj > 0).sum() - snap.num_nodes}", flush=True)

    deep_preds = {}
    for use_graph, name in ((True, "HAR-LSTM-GAT"), (False, "LSTM (w/o GAT)")):
        per_seed = []
        for seed in cfg.seeds:
            tr = train_pooled(snap, adj, cfg, seed, use_graph, out_dir / f"{name}_seed{seed}", log=log)
            per_seed.append(E.predict_test(tr, snap))
        deep_preds[name] = E.ensemble(per_seed) if len(per_seed) > 1 else per_seed[0]

    preds = {
        "HAR": E.har_baseline(snap, cfg.qlike_floor),
        "GARCH": E.garch_baseline(snap, cfg.qlike_floor),
        **deep_preds,
    }
    metrics = {name: E.all_metrics(p, cfg.qlike_floor) for name, p in preds.items()}
    ours = preds["HAR-LSTM-GAT"]
    dm = {}
    for other in ("HAR", "GARCH", "LSTM (w/o GAT)"):
        dm[f"Ours_vs_{other}"] = {
            "qlike": E.dm_compare(ours, preds[other], horizon, "qlike", cfg.qlike_floor),
            "se": E.dm_compare(ours, preds[other], horizon, "se", cfg.qlike_floor),
        }
    log.close()
    result = {
        "dataset": dataset, "lookback": lookback, "horizon": horizon,
        "num_nodes": snap.num_nodes, "n_test": metrics["HAR"]["n"],
        "seeds": list(cfg.seeds), "metrics": metrics, "dm": dm,
        "row_order": ["HAR", "GARCH", "LSTM (w/o GAT)", "HAR-LSTM-GAT"],
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vn30", "vn100", "sp500"])
    ap.add_argument("lookback", type=int)
    ap.add_argument("horizon", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = SMOKE if a.smoke else Config()
    files = resolve_files(a.dataset, a.data_root)
    stamp = f"{a.dataset}_lb{a.lookback}_h{a.horizon}"
    out = Path(a.out) if a.out else (HERE.parents[1] / "results" / "soict" / stamp)
    res = run_experiment(a.dataset, files, a.lookback, a.horizon, cfg, out)
    ql = {k: round(v["qlike"], 4) for k, v in res["metrics"].items()}
    print(f"[done] {stamp} n_test={res['n_test']} QLIKE={ql} "
          f"DM(Ours_vs_HAR.qlike)={res['dm']['Ours_vs_HAR']['qlike']} {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
