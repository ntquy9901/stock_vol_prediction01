"""Leakage guard: the per-fold vol->PK graph AND the per-node scalers depend ONLY on train rows.
Perturbing future (post-train) rows must leave every train artifact bit-identical."""
from __future__ import annotations

import numpy as np

import wf_enriched_panel as EP
from wf_folds import make_folds

LB, H = 8, 1


def _first_fold(panel):
    n = len(panel.anchors)
    return make_folds(n, int(n * 0.7), 40, 25, H)[0]


def test_graph_and_scalers_use_train_rows_only(synth_files):
    files, tickers = synth_files
    panel = EP.build_enriched_panel(files, LB, H, tickers)
    fold = _first_fold(panel)
    D1 = EP.pack_fold(panel, fold, LB, H)
    adj1, tmean1, tstd1, xtr1 = D1.adj_vol2pk.copy(), D1.t_mean.copy(), D1.t_std.copy(), D1.X_tr.copy()

    # graph is a real (non-identity) directed edge with unit self-loops
    assert adj1.shape == (panel.N, panel.N)
    np.testing.assert_allclose(np.diag(adj1), 1.0)
    assert not np.allclose(adj1, np.eye(panel.N))

    # perturb every row STRICTLY AFTER the last train-target row (the forecast/future region)
    tr_anchor = panel.anchors[fold.train]
    last_tr_row = int(tr_anchor[-1]) + H
    panel.feats[last_tr_row + 1:] += 1e3
    panel.pk[last_tr_row + 1:] += 1e3

    D2 = EP.pack_fold(panel, fold, LB, H)
    np.testing.assert_array_equal(D2.adj_vol2pk, adj1)     # graph unchanged
    np.testing.assert_array_equal(D2.t_mean, tmean1)       # target scaler unchanged
    np.testing.assert_array_equal(D2.t_std, tstd1)
    np.testing.assert_array_equal(D2.X_tr, xtr1)           # scaled train windows unchanged
