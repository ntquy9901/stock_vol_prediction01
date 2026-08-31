"""A4 data-pipeline runbook — one command runs P1->P6 crawl-to-ready for one market.

    python scripts/data_pipeline/run_pipeline.py --market {vn30|vn100|hose|hnx|sp500} [--incremental] [--dry-run]

Each phase is a small callable returning a status dict; the CLI runs them in order and emits a per-run
markdown report under ``docs/reports/``. Nothing is reimplemented: the ETL cleaners + enriched writer come
from ``baselines/2026-08-31_enriched_processed`` (which itself reuses ``scripts/etl_audit`` cleaners,
``scripts/eda`` estimators, and the canonical ``pipeline_config`` windows); the audit detectors come from
``scripts/etl_audit``; the data-quality gate from ``scripts/quality_gate``. See ``design_note.md``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
# Bootstrap sys.path for the reused modules (baseline folder name has a dash -> not importable as a package).
for _p in ("baselines/2026-08-31_enriched_processed/code", "scripts/etl_audit", "scripts/eda",
           "submission/soict_lstm_gat", "."):
    _ap = str((REPO / _p).resolve())
    if _ap not in sys.path:  # pragma: no cover - import bootstrap
        sys.path.insert(0, _ap)

import enrich  # noqa: E402  (baseline P3/P4 engine)
import dirty_data_detectors as detectors  # noqa: E402  (P2 detectors)
import pipeline_config as pc  # noqa: E402  (canonical windows)
from volatility_estimators import _YZ_N as _YZ_WINDOW  # noqa: E402  (Yang-Zhang window, published)
from scripts.quality_gate import data_schemas  # noqa: E402
from scripts.quality_gate import run_quality_gate as qg  # noqa: E402

# Statuses.
PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
DRY_RUN = "DRY_RUN"

# Causal lookback (days of context) for --incremental so trailing rolling windows on the recomputed tail
# match a full rebuild. All three windows are sourced from config (pc.) / the published estimator constant;
# no magic number. (line references pc. -> single-source-of-truth exempt)
INCREMENTAL_LOOKBACK = max(pc.HAR_MONTHLY_WINDOW, pc.VOLUME_ZSCORE_WINDOW, _YZ_WINDOW)

REPORT_DIR = REPO / "docs" / "reports"
AUDIT_DIR = REPO / "results" / "data_pipeline_audit"
P1_TESTS = ["tests/test_raw_prices_quality.py", "tests/test_processed_data_quality.py"]

PHASE_LABELS = {
    "P1": "P1 raw-quality tests", "P2": "P2 dirty-data audit", "P3": "P3 ETL clean",
    "P4": "P4 enrich (causal)", "P5": "P5 data-quality gate", "P6": "P6 freeze/version",
}


# --------------------------------------------------------------------------- helpers
def _run_pytest(paths, python_exe=None):
    """Run pytest on ``paths`` (subprocess). Returns ``(returncode, combined_output)``."""
    exe = python_exe or sys.executable
    proc = subprocess.run([exe, "-m", "pytest", "-q", *paths], cwd=str(REPO),
                          capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _pytest_tail(output):
    """Last non-empty line of pytest output (the summary line)."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001 - git absence is non-critical  # pragma: no cover
        return None


def _last_build_date(market_dir):
    """Max ``date`` across the enriched per-ticker files (ignores sidecars / rejection manifests)."""
    market_dir = Path(market_dir)
    csvs = sorted(p for p in market_dir.glob("*.csv") if not p.name.endswith("_rejections.csv"))
    maxd = None
    for p in csvs:
        d = pd.to_datetime(pd.read_csv(p, usecols=["date"])["date"], errors="coerce").max()
        if pd.notna(d) and (maxd is None or d > maxd):
            maxd = d
    return maxd


# --------------------------------------------------------------------------- P1
def phase_p1_raw_quality(dry_run=False, python_exe=None, runner=_run_pytest):
    """P1: run the raw + processed data-quality test suites to gate structural corruption."""
    if dry_run:
        return {"status": DRY_RUN, "detail": f"would run pytest {' '.join(P1_TESTS)}"}
    rc, out = runner([str(REPO / t) for t in P1_TESTS], python_exe)
    status = PASS if rc == 0 else FAIL
    return {"status": status, "detail": _pytest_tail(out), "returncode": rc}


# --------------------------------------------------------------------------- P2
def phase_p2_audit(market, price_dir=None, limit=None, out_dir=None, dry_run=False):
    """P2: run the dirty-data detectors per ticker; aggregate per-class counts + write an audit artifact."""
    price_dir = Path(price_dir) if price_dir is not None else enrich.PRICE_DIRS[market]
    files = sorted(Path(price_dir).glob("*_ohlcv.csv"))
    if limit is not None:
        files = files[:limit]
    if dry_run:
        return {"status": DRY_RUN, "detail": f"would audit {len(files)} ticker files under {price_dir}",
                "n_files": len(files)}

    counts: dict = {}
    n_dirty_tickers = 0
    for f in files:
        res = detectors.detect_all(enrich._read_ticker_csv(f))
        tick_dirty = 0
        for k, v in res["counts"].items():
            counts[k] = counts.get(k, 0) + int(v)
            tick_dirty += int(v)
        if tick_dirty:
            n_dirty_tickers += 1

    out_dir = Path(out_dir) if out_dir is not None else AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"{market}_audit.json"
    artifact.write_text(json.dumps({"market": market, "n_files": len(files),
                                    "n_dirty_tickers": n_dirty_tickers, "per_class_counts": counts},
                                   indent=2), encoding="utf-8")
    status = PASS if files else SKIPPED
    return {"status": status, "detail": f"audited {len(files)} tickers -> {artifact.name}",
            "per_class_counts": counts, "artifact": str(artifact), "n_files": len(files)}


# --------------------------------------------------------------------------- P3 + P4
def _recompute_tail(raw_prepared, last_date, lookback):
    """Recompute the enriched tail for dates > ``last_date`` from a slice with ``lookback`` warmup rows.

    The warmup guarantees every trailing rolling window on a kept row is fully populated, so the tail equals
    a full rebuild (no look-ahead)."""
    raw = raw_prepared.reset_index(drop=True)
    if last_date is None:
        slice_raw = raw
    else:
        dates = pd.to_datetime(raw["date"])
        new_idx = dates.index[dates > last_date]
        if len(new_idx) == 0:
            return raw.iloc[0:0]
        start = max(0, int(new_idx[0]) - lookback)
        slice_raw = raw.iloc[start:].reset_index(drop=True)
    out, _rej, _counts = enrich.build_ticker(slice_raw)
    if last_date is not None:
        out = out.loc[pd.to_datetime(out["date"]) > last_date].reset_index(drop=True)
    return out


def _append_csv(path, tail):
    """Append the enriched ``tail`` (column-ordered) to ``path``; create the file if absent."""
    tail = tail[enrich.ENRICHED_COLUMNS]
    if path.exists():
        tail.to_csv(path, mode="a", header=False, index=False)
    else:
        tail.to_csv(path, index=False)


def _incremental_build(market, price_dir=None, out_root=None, limit=None):
    """Append only new dates (with causal lookback) to an existing enriched build. Reuses the enrich engine."""
    price_dir = Path(price_dir) if price_dir is not None else enrich.PRICE_DIRS[market]
    out_root = Path(out_root) if out_root is not None else enrich.OUT_ROOT
    market_dir = out_root / market
    last_date = _last_build_date(market_dir)
    files = sorted(Path(price_dir).glob("*_ohlcv.csv"))
    if limit is not None:
        files = files[:limit]

    tails: dict = {}
    for f in files:
        ticker = f.stem.replace("_ohlcv", "")
        raw = enrich._prepare_raw(enrich._read_ticker_csv(f))
        tail = _recompute_tail(raw, last_date, INCREMENTAL_LOOKBACK)
        if len(tail):
            tails[ticker] = tail

    lbd = str(last_date.date()) if last_date is not None else None
    if not tails:
        return {"market": market, "mode": "incremental", "n_tickers": 0, "rows_out": 0, "rows_in": 0,
                "n_dropped": 0, "n_dirty_bars": 0, "dirty_by_class": {}, "cleaning_applied": {},
                "estimator_mean": {}, "last_build_date": lbd, "rows_appended": 0}

    market_pk = enrich.compute_market_pk(tails)  # same-day cross-sectional mean over the new-date slice
    rows_appended = 0
    n_dirty = 0
    for ticker, tail in tails.items():
        tail["market_pk"] = tail["date"].map(market_pk).to_numpy()
        n_dirty += int(tail["dirty_flag"].sum())
        _append_csv(market_dir / f"{ticker}.csv", tail)
        rows_appended += len(tail)
    return {"market": market, "mode": "incremental", "n_tickers": len(tails), "rows_out": rows_appended,
            "rows_in": rows_appended, "n_dropped": 0, "n_dirty_bars": n_dirty, "dirty_by_class": {},
            "cleaning_applied": {}, "estimator_mean": {}, "last_build_date": lbd,
            "rows_appended": rows_appended}


def phase_p3p4_enrich(market, out_root=None, price_dir=None, limit=None, incremental=False, dry_run=False):
    """P3 (ETL clean) + P4 (enrich causal columns) via the enriched writer. Returns split p3/p4 statuses."""
    out_root = Path(out_root) if out_root is not None else enrich.OUT_ROOT
    market_dir = out_root / market
    existing = (market_dir / "_schema_version.json").exists()

    if dry_run:
        mode = "incremental" if (incremental and existing) else "full"
        return {"status": DRY_RUN, "mode": mode,
                "detail": f"would {mode}-build {market} -> {market_dir}",
                "p3": {"status": DRY_RUN, "detail": "would apply ETL cleaners (priority order)"},
                "p4": {"status": DRY_RUN, "detail": "would write enriched causal columns"}}

    if incremental and existing:
        summary = _incremental_build(market, price_dir=price_dir, out_root=out_root, limit=limit)
        mode = "incremental"
    else:
        summary = enrich.build_market(market, price_dir=price_dir, out_root=out_root, limit=limit, write=True)
        mode = "full"

    p3 = {"status": PASS, "detail": f"dirty_bars={summary.get('n_dirty_bars')}, "
          f"dropped={summary.get('n_dropped')}"}
    p4 = {"status": PASS, "detail": f"rows_out={summary.get('rows_out')}, "
          f"tickers={summary.get('n_tickers')}"}
    return {"status": PASS, "mode": mode, "summary": summary, "p3": p3, "p4": p4}


# --------------------------------------------------------------------------- P5
def phase_p5_quality_gate(market, out_root=None, out_dir=None, dry_run=False,
                          schema_fn=None, drift_fn=None, enriched_validate=None):
    """P5: Pandera schema + Evidently drift (from run_quality_gate) + enriched-output schema validation."""
    if dry_run:
        return {"status": DRY_RUN, "detail": "would run Pandera check_schema + Evidently check_drift"}
    schema_fn = schema_fn or qg.check_schema
    drift_fn = drift_fn or qg.check_drift
    enriched_validate = enriched_validate or data_schemas.validate_enriched
    out_root = Path(out_root) if out_root is not None else enrich.OUT_ROOT
    out_dir = Path(out_dir) if out_dir is not None else \
        (REPO / "results" / "quality_gate" / datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    schema = schema_fn()
    drift = drift_fn(out_dir)
    # validate_enriched globs <root>/<market>/<ticker>.csv, so it takes the enriched ROOT (not a market dir).
    enr = enriched_validate(out_root)
    if any(r[1] == data_schemas.INVALID for r in enr):
        enr_status = FAIL
    elif all(r[1] == data_schemas.MISSING for r in enr):
        enr_status = SKIPPED
    else:
        enr_status = PASS

    status = FAIL if (schema.status == FAIL or enr_status == FAIL) else PASS
    return {"status": status, "schema": f"{schema.status}: {schema.detail}",
            "drift": f"{drift.status}: {drift.detail}", "enriched": f"{enr_status} ({len(enr)} files)",
            "detail": f"schema={schema.status}, enriched={enr_status}, drift={drift.status}"}


# --------------------------------------------------------------------------- P6
def phase_p6_freeze(market, out_root=None, mode="full", dry_run=False, last_build_date=None, git_sha=None):
    """P6: refresh ``_schema_version.json`` + write ``_provenance.json`` (additive, backward-compatible)."""
    out_root = Path(out_root) if out_root is not None else enrich.OUT_ROOT
    market_dir = out_root / market
    if dry_run:
        return {"status": DRY_RUN, "detail": f"would refresh schema_version + provenance in {market_dir}"}

    market_dir.mkdir(parents=True, exist_ok=True)
    csvs = [p for p in market_dir.glob("*.csv") if not p.name.endswith("_rejections.csv")]
    n_tickers = len(csvs)

    schema_path = market_dir / "_schema_version.json"
    if not schema_path.exists():
        schema_path.write_text(json.dumps({"schema_version": enrich.SCHEMA_VERSION,
                                           "columns": enrich.ENRICHED_COLUMNS, "market": market,
                                           "n_tickers": n_tickers}, indent=2), encoding="utf-8")

    prov = {"schema_version": enrich.SCHEMA_VERSION, "market": market, "mode": mode, "n_tickers": n_tickers,
            "build_timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_sha": git_sha if git_sha is not None else _git_sha(),
            "last_build_date": last_build_date, "columns": enrich.ENRICHED_COLUMNS}
    prov_path = market_dir / "_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return {"status": PASS, "detail": f"schema_version={enrich.SCHEMA_VERSION}, n_tickers={n_tickers}",
            "schema_version_path": str(schema_path), "provenance_path": str(prov_path)}


# --------------------------------------------------------------------------- report + orchestrator
def _write_report(market, report, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"2026-08-31_pipeline_run_{market}.md"
    lines = [f"# Data-pipeline run — {market}", "",
             f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"- Mode: {report['mode']} (incremental={report['incremental']})", "",
             "## Phase status", "", "| phase | status | detail |", "| --- | --- | --- |"]
    for key in ("P1", "P2", "P3", "P4", "P5", "P6"):
        ph = report["phases"][key]
        detail = str(ph.get("detail", "")).replace("|", "\\|")
        lines.append(f"| {PHASE_LABELS[key]} | {ph.get('status', '')} | {detail} |")

    summary = report.get("summary")
    if summary:
        lines += ["", "## Build summary", "",
                  f"- tickers: {summary.get('n_tickers')}", f"- rows_out: {summary.get('rows_out')}",
                  f"- dirty bars: {summary.get('n_dirty_bars')}", f"- dropped: {summary.get('n_dropped')}"]

    p2 = report["phases"]["P2"]
    if p2.get("per_class_counts"):
        lines += ["", "## Dirty-data audit (per-class ticker-day counts)", "", "| class | count |",
                  "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in p2["per_class_counts"].items()]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_pipeline(market, incremental=False, dry_run=False, out_root=None, price_dir=None, limit=None,
                 report_dir=None, python_exe=None):
    """Run P1->P6 for one market and (unless dry-run) write the per-run report. Returns the report dict."""
    if market not in enrich.PRICE_DIRS:
        raise ValueError(f"unknown market {market!r}; choose from {list(enrich.PRICE_DIRS)}")
    out_root = Path(out_root) if out_root is not None else enrich.OUT_ROOT
    price_dir = Path(price_dir) if price_dir is not None else enrich.PRICE_DIRS[market]
    market_dir = out_root / market
    existing = (market_dir / "_schema_version.json").exists()
    last_date = _last_build_date(market_dir) if (incremental and existing) else None
    lbd = str(last_date.date()) if last_date is not None else None

    p1 = phase_p1_raw_quality(dry_run=dry_run, python_exe=python_exe)
    p2 = phase_p2_audit(market, price_dir=price_dir, limit=limit, dry_run=dry_run)
    p34 = phase_p3p4_enrich(market, out_root=out_root, price_dir=price_dir, limit=limit,
                            incremental=incremental, dry_run=dry_run)
    mode = p34.get("mode", "full")
    p5 = phase_p5_quality_gate(market, out_root=out_root, dry_run=dry_run)
    p6 = phase_p6_freeze(market, out_root=out_root, mode=mode, dry_run=dry_run, last_build_date=lbd)

    report = {"market": market, "mode": mode, "incremental": incremental, "dry_run": dry_run,
              "phases": {"P1": p1, "P2": p2, "P3": p34["p3"], "P4": p34["p4"], "P5": p5, "P6": p6},
              "summary": p34.get("summary")}
    if not dry_run:
        report["report_path"] = str(_write_report(market, report,
                                                   report_dir if report_dir is not None else REPORT_DIR))
    return report


def main(argv=None):  # pragma: no cover - entry driver; run_pipeline + phases are unit-tested
    ap = argparse.ArgumentParser(description="Run the P1->P6 data pipeline for one market.")
    ap.add_argument("--market", required=True, choices=list(enrich.PRICE_DIRS))
    ap.add_argument("--incremental", action="store_true", help="append only new dates (causal lookback)")
    ap.add_argument("--dry-run", action="store_true", help="report what would run; write nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap tickers per market (smoke)")
    a = ap.parse_args(argv)
    report = run_pipeline(a.market, incremental=a.incremental, dry_run=a.dry_run, limit=a.limit)
    for key in ("P1", "P2", "P3", "P4", "P5", "P6"):
        ph = report["phases"][key]
        print(f"[pipeline] {key}: {ph.get('status')} - {ph.get('detail', '')}", flush=True)
    if not a.dry_run:
        print(f"[pipeline] report: {report.get('report_path')}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
