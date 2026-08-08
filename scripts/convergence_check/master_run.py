"""Kill-resilient master driver for the 2026-08-06 convergence check (+20-epoch resume to 40).

Runs the 4 variants x 3 seeds. Each (variant, seed) is checked against disk first and SKIPPED
if a completed 40-epoch result already exists, so a session/kill interruption only costs the
single in-flight run -- re-running this driver picks up where it left off.

Sequential (CPU-only, concurrent horizon tasks already share the box). Logs to stdout.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

# This task began 2026-08-06. Only dirs stamped on/after this date belong to the convergence
# check; earlier per_ticker_gate/backbone runs (e.g. 2026-07-26, 2026-08-01) can also have 40/30
# epochs and would otherwise be misdetected as this task's completed runs. ISO dates compare
# correctly as strings.
TASK_START = "2026-08-06"


def _recent(d: Path) -> bool:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", d.name)
    return bool(m) and m.group(1) >= TASK_START

# 20-epoch source runs (checkpoint + results dir) per (variant, seed).
SRC = {
    "backbone": {
        42:   ("results/parallel_lstm_gnn_knn_2026-08-03_230722/best_parallel_model.pth",
               "results/parallel_lstm_gnn_knn_2026-08-03_230722"),
        123:  ("results/parallel_lstm_gnn_knn_seed123_2026-08-03_234613/best_parallel_model.pth",
               "results/parallel_lstm_gnn_knn_seed123_2026-08-03_234613"),
        2026: ("results/parallel_lstm_gnn_knn_seed2026_2026-08-04_000327/best_parallel_model.pth",
               "results/parallel_lstm_gnn_knn_seed2026_2026-08-04_000327"),
    },
    "no_graph": {
        42:   ("results/no_graph_ablation_seed42_2026-08-05_225806/best_no_graph_model.pth",
               "results/no_graph_ablation_seed42_2026-08-05_225806"),
        123:  ("results/no_graph_ablation_seed123_2026-08-05_231327/best_no_graph_model.pth",
               "results/no_graph_ablation_seed123_2026-08-05_231327"),
        2026: ("results/no_graph_ablation_seed2026_2026-08-05_232845/best_no_graph_model.pth",
               "results/no_graph_ablation_seed2026_2026-08-05_232845"),
    },
    "no_gate": {
        42:   ("models/dual_group_news_2026-08-05_230040/best.pt", None),
        123:  ("models/dual_group_news_2026-08-05_231746/best.pt", None),
        2026: ("models/dual_group_news_2026-08-05_233438/best.pt", None),
    },
    "full": {
        42:   ("models/per_ticker_gate_2026-08-03_230821/best.pt",
               "results/per_ticker_gate_2026-08-03_230821"),
        123:  ("models/per_ticker_gate_2026-08-04_000448/best.pt",
               "results/per_ticker_gate_2026-08-04_000448"),
        2026: ("models/per_ticker_gate_2026-08-04_002252/best.pt",
               "results/per_ticker_gate_2026-08-04_002252"),
    },
}

BACKBONE = "src/lstm_gat_hybrid/train_parallel_enhanced.py"
NOGRAPH = "scripts/ablation_no_graph/run_no_graph_ablation.py"
NOGATE = "baselines/2026-07-25_dual_group_news_embedding_baseline/code/train_dual_news.py"
FULL = "baselines/2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py"


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _num_epochs(rec):
    if not rec:
        return None
    ts = rec.get("training_summary")
    if ts and "num_epochs_trained" in ts:
        return ts["num_epochs_trained"]
    return rec.get("num_epochs_trained")


def find_completed(variant, seed, target=40):
    """Return results dir Path of a completed `target`-epoch run for (variant, seed), else None."""
    # Date-agnostic globs (the wall-clock date can roll over during a multi-hour run). Source
    # 20-epoch runs are excluded by the epoch-count / gate-history-max filters below; the
    # concurrent horizon tasks use the distinct `per_ticker_gate_h<H>_*` naming, so
    # `per_ticker_gate_2026-*` matches only this check's own runs (and the 20-ep sources).
    if variant == "backbone":
        for d in ROOT.glob(f"results/parallel_lstm_gnn_knn_seed{seed}_2026-*"):
            if _recent(d) and _num_epochs(_read_json(d / "training_results.json")) == target:
                return d
    elif variant == "no_graph":
        for d in ROOT.glob(f"results/no_graph_ablation_seed{seed}_2026-*"):
            if _recent(d) and _num_epochs(_read_json(d / "training_results.json")) == target:
                return d
    elif variant == "no_gate":
        for d in ROOT.glob("results/dual_group_news_2026-*"):
            if not _recent(d):
                continue
            r = _read_json(d / "results.json")
            if r and r.get("seed") == seed and not r.get("smoke") and _num_epochs(r) == target:
                return d
    elif variant == "full":
        for d in ROOT.glob("results/per_ticker_gate_2026-*"):
            if not _recent(d):
                continue
            r = _read_json(d / "results.json")
            gh = _read_json(d / "gate_history.json")
            if r and r.get("seed") == seed and gh:
                if max(int(e) for e in gh) == target:
                    return d
    return None


def find_full_stage(seed, target_max):
    """Newest per_ticker_gate 2026-08-06 dir for `seed` whose gate_history max == target_max."""
    best = None
    for d in ROOT.glob("results/per_ticker_gate_2026-*"):
        if not _recent(d):
            continue
        r = _read_json(d / "results.json")
        gh = _read_json(d / "gate_history.json")
        if r and r.get("seed") == seed and gh and max(int(e) for e in gh) == target_max:
            if best is None or d.stat().st_mtime > best.stat().st_mtime:
                best = d
    return best


def run(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(ROOT), check=True)


def do_seed(variant, seed):
    if find_completed(variant, seed):
        print(f"[skip] {variant} seed{seed}: 40-epoch result already on disk", flush=True)
        return
    ckpt, rdir = SRC[variant][seed]
    print(f"[run ] {variant} seed{seed}", flush=True)
    if variant == "backbone":
        run([PY, BACKBONE, "--graph_method", "knn", "--seed", seed, "--epochs", 20,
             "--resume_checkpoint", ckpt, "--resume_results_dir", rdir])
    elif variant == "no_graph":
        run([PY, NOGRAPH, "--seeds", seed,
             "--resume_checkpoint", ckpt, "--resume_results_dir", rdir])
    elif variant == "no_gate":
        run([PY, NOGATE, "--epochs", 20, "--seed", seed, "--resume_start_epoch", 20,
             "--resume_checkpoint", ckpt])
    elif variant == "full":
        # 20->30 (skip if a 30-ep stage already exists for this seed), then 30->40.
        if find_full_stage(seed, 30) is None:
            run([PY, FULL, "--epochs", 10, "--seed", seed,
                 "--resume_checkpoint", ckpt, "--resume_results_dir", rdir])
        stage30 = find_full_stage(seed, 30)
        if stage30 is None:
            raise RuntimeError(f"full seed{seed}: 30-epoch stage not found after step 1")
        mdir30 = ROOT / "models" / stage30.name
        run([PY, FULL, "--epochs", 10, "--seed", seed,
             "--resume_checkpoint", str(mdir30 / "best.pt"), "--resume_results_dir", str(stage30)])


def main():
    order = sys.argv[1:] or ["backbone", "full", "no_graph", "no_gate"]
    for variant in order:
        print(f"\n===== VARIANT {variant} =====", flush=True)
        for seed in (42, 123, 2026):
            do_seed(variant, seed)
        print(f"===== VARIANT {variant} DONE =====", flush=True)
    print("\nMASTER_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
