# Reproduce — exact commands

Run everything with the GPU venv python: `.venv_gpu_encode/Scripts/python.exe`.
Folder names contain `-`, so scripts are run by path (not `python -m`); each bootstraps `sys.path`.
Seeds are fixed in `submission/soict_lstm_gat/pipeline_config.py` / `training_config`.

## 0. Environment
```
.venv_gpu_encode/Scripts/python.exe --version        # 3.10 (note: not 3.11)
.venv_gpu_encode/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1. Data pipeline (rebuild enriched panels) — optional, data is committed
```
.venv_gpu_encode/Scripts/python.exe scripts/data_pipeline/run_pipeline.py --market vn30 --dry-run
.venv_gpu_encode/Scripts/python.exe scripts/data_pipeline/run_pipeline.py --market vn30
.venv_gpu_encode/Scripts/python.exe scripts/data_pipeline/run_pipeline.py --market vn100
```

## 2. VolGA walk-forward (headline) — one JSON per market×horizon (~1.2h VN30, ~4h VN100 per horizon)
```
WF=baselines/2026-08-31_walkforward_volga/code/run_volga_walkforward.py
for M in vn30 vn100; do for H in 1 5 10 22; do
  .venv_gpu_encode/Scripts/python.exe -u "$WF" --market $M --horizon $H \
     --lookback 22 --folds-target 22 --epochs 16 --batch 32 --no-gpu-wait
done; done
# -> results/walkforward_volga/walkforward_volga_{vn30,vn100}_h{1,5,10,22}.json
```
Dashboards (CPU): `.venv_gpu_encode/Scripts/python.exe baselines/2026-08-31_walkforward_volga/code/build_dashboards.py`

## 3. Pooled/transfer VN30 ablation — both arms per horizon (~8h/horizon: Arm0≈Arm1)
```
ABL=baselines/2026-09-04_pooled_transfer_vn30/code/run_pooled_ablation.py
for H in 1 5 10 22; do
  .venv_gpu_encode/Scripts/python.exe -u "$ABL" --horizon $H --folds-target 22 --lookback 22 --epochs 16
done
# -> results/pooled_transfer_vn30/pooled_vn30_h{1,5,10,22}.json
.venv_gpu_encode/Scripts/python.exe baselines/2026-09-04_pooled_transfer_vn30/code/summarize_pooled.py
# -> docs/reports/2026-09-04_pooled_transfer_vn30_report.md
```

## 4. Tests + coverage (per baseline)
```
.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-08-31_walkforward_volga/code/tests -q
.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-09-04_pooled_transfer_vn30/code/tests \
   --cov=baselines/2026-09-04_pooled_transfer_vn30/code --cov-branch -q
ruff check --select F baselines/2026-08-31_walkforward_volga/code baselines/2026-09-04_pooled_transfer_vn30/code
```

## 5. Full quality gate (what CI runs on push)
```
QG=scripts/quality_gate/run_quality_gate.py      # LINT + TESTS + Pandera schema + Evidently drift
.venv_gpu_encode/Scripts/python.exe "$QG"
# pre-push hook: scripts/git_hooks/pre-push (6 steps; blocks on C0<100 / ruff-F / data-quality / config-hardcode)
```

## Spot-check a reported number
```
.venv_gpu_encode/Scripts/python.exe -c "import json; d=json.load(open('results/walkforward_volga/walkforward_volga_vn100_h1.json')); print({k:round(d['metrics'][k]['qlike'],4) for k in d['metrics']}); print('DM VolGA-vs-LSTM', d['dm_date_clustered']['VolGA_vs_LSTM'])"
# expect VolGA 0.4916 best; DM VolGA-vs-LSTM qlike p=0.0083
```
