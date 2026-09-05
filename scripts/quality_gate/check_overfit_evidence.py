"""Pre-push gate: BLOCK a committed TRAINING result JSON (any driver) that lacks over/under-fit evidence or
whose learned models are over/under-fit. Invoked by scripts/git_hooks/pre-push on the result JSONs in the push
diff (results/** and baselines/**). Learned models are auto-detected by name (overfit_check.looks_learned), so
edge_hmatched (VolGA/VolGA_hm), masked_rich (LSTM/LSTM_wGAT_vol2pk), etc. are all covered. A file that is not a
training result (no learned-model test metrics) is skipped.

Usage: python check_overfit_evidence.py <result.json> [<result.json> ...]   (exit 1 if any fails)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overfit_check as OF  # noqa: E402


def _is_masked_rich_result(res: dict) -> bool:
    """Identify a masked-rich TRAINING result by SCHEMA, not by a literal learned-model key (F-04 v3): a partial
    artifact that lost its LSTM block must still be recognised as a training result so its missing evidence
    FAILS rather than being skipped. A GARCH-only / unrelated artifact (no design, no per-seed, no learned
    model at all) is genuinely not a training result and is skipped."""
    if not isinstance(res, dict):
        return False
    design = str(res.get("design", ""))
    if "masked" in design:                                      # the delivered runner stamps this design string
        return True
    if "metrics_per_seed" in res:                               # only training results carry per-seed stats
        return True
    metrics = res.get("metrics", {})
    if isinstance(metrics, dict) and any(OF.looks_learned(m) for m in metrics):
        return True                                             # any learned model (any driver) -> training result
    return False


def check_files(paths):
    """Return {path: [problems]} for every result.json that IS a training result and fails the evidence check."""
    problems = {}
    for p in paths:
        try:
            res = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:                                  # unreadable/corrupt -> a real problem
            problems[p] = [f"unreadable: {type(e).__name__}: {e}"]
            continue
        if not _is_masked_rich_result(res):
            continue                                            # genuinely not a training result -> skip
        # masked_rich (identified by design/per-seed) must carry the FULL expected learned set, so a partial
        # artifact that dropped a model still FAILS (F-04). Other drivers auto-detect learned from metrics keys.
        is_masked_rich = "masked" in str(res.get("design", "")) or "metrics_per_seed" in res
        ok, probs = OF.check_result_evidence(res, learned=(OF.LEARNED if is_masked_rich else None))
        if not ok:
            problems[p] = probs
    return problems


def main(argv):
    problems = check_files(argv)
    for p, probs in problems.items():
        print(f"[overfit-gate] FAIL {p}:")
        for pr in probs:
            print(f"    - {pr}")
    if not problems:
        print(f"[overfit-gate] OK: {len(argv)} result file(s) carry train/val/test fit evidence, no over/under-fit.")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry driver
    sys.exit(main(sys.argv[1:]))
