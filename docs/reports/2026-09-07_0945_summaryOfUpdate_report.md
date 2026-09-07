# Summary of update — 2026-09-07 (SonarScan, TimesFM archive, paper refresh, vnmarkets window)

Autonomous session. Four pieces of work, all completed; the substantive three are pushed to
`origin/master` (through `ea1171f..e9654ed`). The SonarScan report is kept local per request.

## 1. SonarScan (local report only)
SonarQube Community (Docker, localhost:9000) + sonar-scanner-cli over `baselines/ scripts/ src/
submission/` (archive/data/notebooks/vendored excluded). Result: 62,623 ncloc; 45 bugs; 46
vulnerabilities; 1,397 code smells; 9.3% duplication; 0 hotspots.
- Assessed every finding on the experiment-result path tận-dòng: **no issue affects experiment
  results.** All 41 `S1244` float-equality hits are intentional exact-zero/degenerate guards
  (std==0, ss_tot==0, denom==0, qlike_weight==0, zero-count); the 2 `S1764` are the NaN idiom
  `x != x` / `t == t`; the 45 CRITICAL vulns are `torch.load`/pickle on the project's own
  checkpoints (dual-use, low real risk). Recent deliverable code (qlike_anchor/edge_hmatched/2026-09)
  is clean (0 bugs/vulns).
- Report: `docs/reports/2026-09-07_sonarscan_local_report.md` (local, uncommitted per request).
- Server container `svp_sonar` stopped (RAM freed); restart with `docker start svp_sonar` to browse.

## 2. Archive the legacy TimesFM baseline (commit a0b18c3)
`git mv src/timesfm_baseline/ -> archive/src_legacy/timesfm_baseline/`, plus
`tests/test_timesfm_lora.py` and root `quick_test_training.py`. TimesFM never entered the VolGA
paper and its `peft` dep is uninstalled (tests uncollectable). Archiving takes it (and its cosmetic
SonarQube `S3923` dead-branch) out of scope for all audits. Scope verified via repo-wide grep: no
importer of `src.timesfm_baseline.*` outside the moved set; `_research/timesfm-google/` is the
separate vendored Google library, left in place. Documented in `archive/README.md`. The other
`S3923` (vnmarkets_eda) is an active script — not archived (resolved under item 4 instead).

## 3. New paper version + Drive upload (commit d639ec3)
`docs/paper/soict_harlstmgat_2026-09-07_final.tex` — refreshed onto the latest canonical VolGA run
(`results/edge_hmatched/edgehm_{vn100,vn30}_h*.json`, lookback 10, 7 folds, 5 seeds), replacing the
older 22-fold `walkforward_volga` run the 2026-09-05 version used.
- All four metric tables rebuilt from the JSONs with a deterministic extractor; **added R^2** as a
  fifth reported metric; dropped the per-seed QLIKE std (the latest run stores ensemble-aggregate
  metrics only).
- All DM tables refreshed with the latest VN100/VN30 date-clustered p-values.
- Captions updated (7 folds; VN30 N=33; 46,512 / 12,936 obs), lookback wording (ten days), and all
  abstract/intro/results/discussion/conclusion prose reconciled to the new numbers. Key shifts:
  VN100 h1 VolGA now significantly beats HAR-X on **QLIKE** (p=0.047) and the no-graph LSTM on QLIKE
  (p=0.006); the old "MAE p<0.001 vs HAR-X" claim is gone (now h5 AE p=0.001). VN30 VolGA leads all
  metrics at h1 but not significantly vs HAR-X; graph gives significant AE gains over LSTM at
  h1/h10/h22 (p<0.001).
- PDF compiled (pdflatex/MiKTeX, 11 pages, no undefined refs) and **uploaded to Drive**:
  `gdrive:luanvan_papers/soict_harlstmgat_2026-09-07_final.pdf` (via rclone). PDF is gitignored;
  the `.tex` is committed.
- Style: preserved the paper's existing voice (em-dashes etc.) rather than imposing the SNL
  paper-writing gate — the request was a numeric-table refresh, not a style rewrite (§3 Surgical).
  Numbers verified by deterministic generation from the JSONs.

## 4. vnmarkets EDA window: 20 -> 22, sourced from canonical config (commit e9654ed)
`scripts/eda/vnmarkets_eda.py`: the four hardcoded rolling windows (HAR weekly=5, monthly=22, volume
z-score mean/std) now import from `submission/soict_lstm_gat/pipeline_config` (single source of
truth). The volume z-score window moves **20 -> 22** to match the project monthly convention
(`VOLUME_ZSCORE_WINDOW=22`, per the documented root-cause); the stat key + chart title rename to
`volume_zscore_22`. Also collapsed the `S3923` dead if/else at `raw_n` (both branches equal since
`raw_files` is pre-capped to `--limit`).
- New `scripts/eda/test_vnmarkets_eda.py` pins the window to 22 both by config value and
  behaviourally (N-21 pooled z-scores) and covers the changed lines.
- Follow-up (not done): the committed EDA HTML reports predate the 22 window and would need
  regeneration to reflect it.

## Tests / gate
- `test_vnmarkets_eda.py` (3) + `test_vnmarkets_eda_smoke.py` / `_detectors.py` (18): pass.
- `test_timesfm_lora.py`: moved to archive (uncollectable without `peft`; out of scope).
- Pre-push gate on the push: PASS — changed-scope tests, **diff-cover C0 100% / C1 100%** on changed
  lines, data-quality (334) + delivered-baseline (69) tests, config-hardcode 0 BLOCK/0 WARN,
  pragma-guard clean, 7/7 checklist evidence complete. Evidence: `docs/reports/gate_logs/e9654ed.txt`.
- Code review: not run as a separate `/code-review` pass this session; the Stop-hook adversarial
  reminder + the config-hardcode/pragma/diff-cover gates covered the changed `.py`. The paper and
  archive changes are non-executable / file moves.

## Risks / follow-ups
- vnmarkets EDA HTML reports stale vs the 22 window (regenerate when convenient).
- sp500_eda.py / hnx_full_eda.py still use a 20-day volume window (different files, their own runs;
  out of this task's scope — left unchanged deliberately).
- The 2026-09-05 paper version is superseded by 2026-09-07 but left in place for history.
