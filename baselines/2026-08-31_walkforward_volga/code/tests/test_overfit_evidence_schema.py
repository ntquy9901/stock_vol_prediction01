"""The pooled result.json carries the over/under-fit evidence SCHEMA the pre-push gate checks:
train_metrics / val_metrics / metrics blocks + LSTM + LSTM_wGAT_vol2pk present (so it is recognised as a
training result and never fails for a MISSING-block reason)."""
from __future__ import annotations

import json

import check_overfit_evidence as CE

import run_volga_walkforward as WF


def test_result_json_matches_overfit_gate_schema(synth_files, tmp_path, train_cfg):
    files, tickers = synth_files
    cfg = train_cfg
    wf = WF.VolgaWFConfig(lookback=8, horizon=1, folds_target=1, val=25, test_frac=0.75)
    out = tmp_path / "result.json"
    WF.run_walkforward(files, wf, cfg, tickers, out_path=out)
    res = json.loads(out.read_text(encoding="utf-8"))

    # gate recognises this as a masked-rich TRAINING result (so missing evidence would FAIL, not skip)
    assert CE._is_masked_rich_result(res)
    # whether the tiny 2-epoch fit verdict is 'ok' is irrelevant; assert it never fails for a SCHEMA gap
    reasons = CE.check_files([str(out)]).get(str(out), [])
    assert all("missing" not in r for r in reasons)       # only a fit verdict may fail, never a missing block
    for m in ("LSTM", "LSTM_wGAT_vol2pk"):
        assert m in res["train_metrics"] and m in res["val_metrics"] and m in res["metrics"]
