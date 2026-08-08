# Pooled News and GNN Ablation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a leakage-controlled multi-horizon pilot comparing pooled HAR, Price LSTM,
Price+News, Price+News+Gate, and a matched GNN OFF/ON ablation. The forecast horizon is a run
parameter `h in {1, 5, 10, 22}` trading days (`--horizon`, default `h=5`, primary); one horizon per
run, with outputs namespaced by `h{N}`.

**Architecture:** P0-P3 share one deterministic per-ticker pooled sample manifest and train-only
preprocessing. G0-G1 reuse frozen P3 encoders on a separately split common-date graph manifest and
differ only by message passing. Existing baselines and shared `src/` behavior remain unchanged.

**Tech Stack:** Python 3.11 target, PyTorch, pandas, NumPy, scikit-learn, pytest, pytest-cov,
diff-cover, ruff, existing `src.common` evaluation/HAR utilities.

## Global Constraints

- Forecast horizon is a run parameter `h in {1, 5, 10, 22}` trading observations (default `h=5`,
  primary) and input length is 22 observations. The pooled and graph manifests within one run share
  the same horizon.
- Pooled data is split per ticker chronologically 70%/15%/15% before HAR/window generation.
- All loaders use `shuffle=False`; pooled samples sort by `(target_date, ticker_id)`.
- Price, target, outlier, and learned-news preprocessing parameters fit train data only.
- P0-P3 use one eligibility/sample manifest; G0-G1 use one matched graph manifest.
- Loss uses normalized targets; all six reported metrics use untouched raw targets and
  ticker-ID-selected inverse transforms.
- Screening is 5 epochs with seed 42. No run exceeds 10 epochs without separate user approval.
- Implementation stays under `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/` and imports
  shared `src/` modules read-only.
- Every behavior change follows test FAIL -> minimal implementation -> test PASS.
- Every task commit stages only files listed in that task; preserve unrelated worktree changes.

---

## File map

- `code/data.py` — typed sample records, price/news loading, split-first window generation,
  eligibility and manifest hashing, graph-compatible snapshots.
- `code/scaling.py` — per-ticker train-only winsor bounds, feature/target scalers, serialization,
  ticker-ID transform/inverse-transform.
- `code/models.py` — P1-P3 encoders/fusion/gate and matched G0/G1 graph model.
- `code/train.py` — loss, raw-scale evaluation, per-ticker DirAcc, one-config training, artifacts.
- `code/run_pilot.py` — P0 closed-form reference, screening orchestration, manifest gates, comparison.
- `test/test_data.py` — split, eligibility, manifests, news cutoff/mask, real-data slice.
- `test/test_scaling.py` — leakage isolation, zero variance, round-trip, raw-target preservation.
- `test/test_models.py` — shapes, shared encoders, ticker gate, GNN pairing.
- `test/test_train_smoke.py` — one-batch I/O runner, raw metrics, artifact and real-data smoke.

Every script inserts the project root and its baseline-local `code/` directory into `sys.path`.
Tests insert the baseline `code/` directory and import `data`, `scaling`, `models`, and `train`
directly; they never import `code.data`, which conflicts with Python's standard-library `code`
module.

### Task 1: Baseline package and validated raw split records

**Files:**
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/__init__.py`
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py`
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/__init__.py`
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py`

**Interfaces:**
- Produces: `SampleKey(ticker_id: int, ticker: str, target_date: str)`.
- Produces: `PooledSample(key, x_price_raw, x_news, news_mask, y_raw)`.
- Produces: `chronological_split(df, ratios=(0.7, 0.15, 0.15)) -> dict[str,pd.DataFrame]`.
- Produces: `load_and_split_price_data(data_dir, ratios=(0.7,0.15,0.15)) -> SplitFrames`.

- [ ] **Step 1: Write failing split and raw-input tests**

```python
def test_split_is_per_ticker_chronological_and_disjoint():
    parts = chronological_split(_frame(100))
    assert [len(parts[k]) for k in ("train", "val", "test")] == [70, 15, 15]
    assert parts["train"].date.max() < parts["val"].date.min()
    assert parts["val"].date.max() < parts["test"].date.min()

def test_invalid_duplicate_date_fails_before_split():
    with pytest.raises(ValueError, match="duplicate"):
        chronological_split(_frame_with_duplicate_date())
```

- [ ] **Step 2: Run the tests and confirm missing imports/functions fail**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py -v`

Expected: collection FAIL because `code.data` does not exist.

- [ ] **Step 3: Implement typed records, strict date validation, and raw splits**

Use frozen dataclasses. Parse dates with `errors="raise"`; reject duplicates and non-monotonic
input. Do not generate HAR features or final eligibility yet; Task 2 must fit train-only outlier
bounds first.

- [ ] **Step 4: Add deterministic vocabulary tests**

```python
def test_ticker_ids_are_sorted_and_stable(tmp_path):
    splits = load_and_split_price_data(tmp_path)
    assert splits.ticker_to_id == {"AAA": 0, "ZZZ": 1}
```

Persist raw-data validation failures. Final post-HAR exclusions are resolved once in Task 2.

- [ ] **Step 5: Run Task 1 tests**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/__init__.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/__init__.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py
git commit -m "Add deterministic pooled sample manifest"
```

### Task 2: Train-only preprocessing and scaler invariants

**Files:**
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/scaling.py`
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_scaling.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py`

**Interfaces:**
- Consumes: raw train/val/test ticker frames from Task 1.
- Produces: `TickerPreprocessor.fit(train_frame, train_features, train_targets)`.
- Produces: `PreprocessorStore.transform_features(ticker_id, x)` and
  `inverse_targets(ticker_ids, y_norm)`.
- Produces: `build_pooled_manifest(split_frames, preprocessors, seq_length=22,
  horizon=5) -> PooledManifest`.

- [ ] **Step 1: Write failing leakage and round-trip tests**

```python
def test_val_and_test_cannot_change_train_parameters():
    a = fit_store(train=_train(), val=_val(1.0), test=_test(2.0))
    b = fit_store(train=_train(), val=_val(1e9), test=_test(-1e9))
    assert a.to_dict() == b.to_dict()

def test_inverse_uses_explicit_ticker_id():
    store = _two_ticker_store(means=[10.0, 100.0], stds=[2.0, 5.0])
    got = store.inverse_targets(np.array([1, 0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(got, [100.0, 10.0])

def test_zero_variance_uses_unit_std_and_round_trips():
    scaler = ArrayStandardizer().fit(np.ones((4, 3)))
    np.testing.assert_allclose(scaler.std, np.ones(3))
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(np.ones((4, 3)))), 1.0)

def test_window_count_and_final_target_are_exact():
    samples = build_ticker_samples(_frame(70), "AAA", 0, seq_length=22, horizon=5)
    assert len(samples) == 70 - 22 - 5 + 1
    assert samples[-1].key.target_date == "2020-03-10"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_scaling.py -v`

Expected: FAIL because scaler classes are undefined.

- [ ] **Step 3: Write winsor-order/raw-target tests and confirm RED**

Test that validation/test extremes cannot change train bounds. Assert processing order is
`raw split -> fit train bounds -> clip model values -> generate split-local HAR -> fit feature
scalers -> build windows`. Samples store both `y_model_raw` for normalized loss and untouched
`y_eval_raw` for metrics.

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_scaling.py -v`

Expected: FAIL because winsor and manifest construction are undefined.

- [ ] **Step 4: Implement scalers, winsor bounds, HAR, eligibility, and windows**

Fit price/HAR `mean/std` along axis 0 for each ticker; fit one target mean/std per ticker. Replace
standard deviations below `1e-8` with `1.0`. Provide JSON-safe `to_dict()` and `from_dict()`.
Never expose a `fit` method accepting a split other than train. Fit `mean ± 3*std` on raw train,
clip model inputs/labels, generate HAR, and fit feature scalers. Define
`valid_count = post_har_rows - seq_length - horizon + 1`; require at least one valid window in every
split. Persist exclusions once, then sort the shared manifest by `(target_date,ticker_id)`.

- [ ] **Step 5: Run Task 1-2 tests**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_scaling.py -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/scaling.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_scaling.py
git commit -m "Add train-only pooled preprocessing"
```

### Task 3: Causal news alignment and manifest hashes

**Files:**
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py`

**Interfaces:**
- Produces: `load_effective_news_panel(path) -> NewsPanel`.
- Produces: `attach_news(samples, panel, feature_cols) -> list[PooledSample]`.
- Produces: `PooledManifest.content_hash(split) -> str`.

- [ ] **Step 1: Write failing effective-date, mask, and hash tests**

```python
def test_news_panel_uses_effective_trading_dates_and_masks_missing_days(tmp_path):
    panel = _write_news_panel(tmp_path, rows=[("AAA", "2020-01-02", [1.0, 2.0])])
    sample = _sample_with_dates("AAA", ["2020-01-01", "2020-01-02"])
    attached = attach_news([sample], load_effective_news_panel(panel), ["f0", "f1"])[0]
    np.testing.assert_array_equal(attached.news_mask, [0, 1])
    np.testing.assert_allclose(attached.x_news[0], [0, 0])

def test_real_source_timestamp_matches_panel_effective_date(real_news_record, real_panel):
    expected = effective_trading_date(real_news_record.pub_date,
                                      real_news_record.trading_dates)
    assert (real_news_record.ticker, expected) in real_panel.keys()

def test_manifest_hash_changes_when_target_mask_or_tensor_changes():
    assert manifest_hash([_sample(y=1.0)]) != manifest_hash([_sample(y=2.0)])
    assert manifest_hash([_sample(mask=[0])]) != manifest_hash([_sample(mask=[1])])
```

- [ ] **Step 2: Run the new tests and confirm failure**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py -v`

Expected: FAIL at undefined news functions.

- [ ] **Step 3: Implement fixed-width news lookup**

Read `data/features/dual_group_news_panel.parquet`, require unique `(ticker,date)`, stable sorted
feature columns, finite float32 values, and date normalization to `YYYY-MM-DD`. Treat panel `date`
as the effective trading date produced by the existing
`vendor_data_eda.phase04_news_helpers.effective_trading_date` 15:00 Asia/Ho_Chi_Minh rule. Add a
synthetic after-close test and independently join one bounded real source timestamp to its stored
panel effective date. Load provenance metadata and reject a learned PCA/news artifact whose fit
period exceeds any eligible pooled-training cutoff; do not silently refit during training.

- [ ] **Step 4: Implement content hashing and mapping-error guard**

Create separate SHA-256 hashes for eligibility keys/raw targets, price tensors, news tensors/masks,
and preprocessing. Canonicalize arrays as contiguous little-endian float32 and include dtype/shape;
serialize JSON with sorted keys and UTF-8. Zero coverage fails only if an independently
loaded effective panel contains an eligible `(ticker,date)` key that lookup failed to match.

- [ ] **Step 5: Run data tests including the real panel slice**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py -v`

Expected: PASS, including a test marked `smoke` that reads a small ACB slice and the real parquet.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py
git commit -m "Add causal news alignment to pooled samples"
```

### Task 4: P1-P3 shared models and ticker-indexed gate

**Files:**
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py`
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py`

**Interfaces:**
- Produces: `PooledPriceLSTM(price_dim, hidden_dim=64, dropout=0.2)`.
- Produces: `PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate,
  hidden_dim=64, news_hidden_dim=64, dropout=0.2)`.
- Both return normalized scalar predictions shaped `[B]` and reset LSTM state on every forward.

- [ ] **Step 1: Write failing shape, shared-weight, and gate tests**

```python
def test_pooled_models_return_one_prediction_per_independent_sample():
    model = PooledPriceNewsLSTM(3, 146, 33, use_gate=True)
    y = model(torch.randn(4, 22, 3), torch.randn(4, 22, 146),
              torch.ones(4, 22, dtype=torch.bool), torch.tensor([0, 5, 5, 32]))
    assert y.shape == (4,)

def test_gate_selection_uses_ticker_id_not_batch_position():
    model = _deterministic_gated_model(gates=[-10.0, 10.0])
    a = model(*_same_inputs(), ticker_ids=torch.tensor([1, 0]))
    b = model(*_same_inputs(), ticker_ids=torch.tensor([0, 1]))
    assert not torch.equal(a, b)

def test_all_missing_news_is_finite_and_input_independent():
    model = PooledPriceNewsLSTM(3, 4, 2, use_gate=False)
    y = model(_price(), torch.randn(2, 22, 4), torch.zeros(2, 22, dtype=torch.bool),
              torch.tensor([0, 1]))
    y2 = model(_price(), torch.randn(2, 22, 4), torch.zeros(2, 22, dtype=torch.bool),
               torch.tensor([0, 1]))
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, y2)
```

- [ ] **Step 2: Run model tests and confirm failure**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py -v`

Expected: FAIL because models are undefined.

- [ ] **Step 3: Implement the minimal models**

Use one shared Price LSTM and one shared News LSTM. Compact valid news observations in chronological
order and use packed lengths so internal/trailing missing timesteps do not update recurrent state;
an all-missing sequence receives a zero news representation. P2 omits gate
multiplication. P3 owns `gate_logits[num_tickers]`, initialized at zero, and indexes it only with
the batch's `ticker_ids`. Use direct concatenation and a two-layer shared head; add no registry.

- [ ] **Step 4: Add gradient-isolation and stateless-forward tests**

Verify changing ticker 1's target/news cannot change `gate_logits.grad[0]`, while ticker 1's own
gradient changes. Call the same model twice with identical inputs in different preceding-call
contexts and assert identical eval-mode outputs, proving no hidden state persists between batches.

- [ ] **Step 5: Run model tests**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```powershell
git add -- baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py
git commit -m "Add pooled price news and ticker gate models"
```

### Task 5: Raw-scale evaluation and one-configuration trainer

**Files:**
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/train.py`
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py`

**Interfaces:**
- Consumes: pooled samples, `PreprocessorStore`, P1-P3 models.
- Produces: `evaluate_by_ticker(model, loader, store, epsilon=1e-8) -> dict`.
- Produces: `run_training(config_name, loaders, store, output_dir, epochs, seed) -> Path`.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_metrics_use_raw_target_and_ticker_specific_inverse():
    result = evaluate_records(_records_with_clipped_y_norm_but_original_y_raw())
    assert result["targets_raw"] == [10.0, 100.0]
    assert set(result["metrics"]) >= {"mse", "rmse", "mae", "r2", "qlike",
                                      "directional_accuracy"}

def test_directional_accuracy_never_crosses_tickers():
    metrics = evaluate_records(_interleaved_two_ticker_records())
    assert metrics["directional_accuracy"] == pytest.approx(100.0)

def test_nonpositive_prediction_rate_above_one_percent_fails():
    with pytest.raises(ValueError, match="nonpositive prediction rate"):
        evaluate_records(_mostly_negative_predictions())
```

- [ ] **Step 2: Run evaluation tests and confirm failure**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py -v`

Expected: FAIL because training/evaluation functions are undefined.

- [ ] **Step 3: Implement evaluation**

Collect `ticker_id`, `target_date`, normalized predictions, and stored raw targets. Inverse by
ticker ID. Compute MSE/RMSE/MAE/R2/DirAcc on unfloored predictions; apply `1e-8` only inside QLIKE.
Save nonpositive rate and fail above 1%. Group/sort by ticker/date; exclude tickers with fewer than
two targets, report eligible count, fail if none remain, and report unweighted macro headline plus
observation-weighted diagnostic. Reuse
`src.common.evaluation.evaluate_predictions` for the other five metrics.

- [ ] **Step 4: Write failing one-batch I/O runner test**

The test builds a tiny on-disk price/news fixture, calls
`run_training("P1", loaders, store, tmp_path, epochs=1, seed=42)`, and asserts
that `results.json`, `sample_manifest.json`, `preprocessors.json`, checkpoint, and a partial learning
curve exist and contain finite JSON numbers.

- [ ] **Step 5: Implement the trainer**

Use MSE on normalized targets, Adam with `weight_decay=1e-5`, gradient clipping at 1.0, validation
each epoch, best-validation checkpointing, and epoch-5/10 curves. Set Python/NumPy/Torch/CUDA seeds.
Reject epochs outside 1-10. Do not evaluate test inside `run_training` during screening.

- [ ] **Step 6: Run Task 5 tests**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py -v`

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```powershell
git add -- baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/train.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py
git commit -m "Add leakage-safe pooled training and evaluation"
```

### Task 6: P0 HAR reference and P0-P3 screening runner

**Files:**
- Create: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py`

**Interfaces:**
- Produces: `run_har_reference(manifest, store, output_dir) -> Path`.
- Produces CLI flags `--phase`, `--epochs`, `--seed`, `--output-dir`, `--smoke`, and
  `--max-tickers`; smoke filtering happens before manifest construction and is recorded.

- [ ] **Step 1: Write failing HAR and orchestration tests**

```python
def test_har_reference_uses_exact_manifest_targets(tmp_path):
    result = run_har_reference(_manifest(), _store(), tmp_path)
    assert json.loads(result.read_text())["manifest_hash"] == _manifest().content_hash("val")

def test_runner_rejects_manifest_mismatch_before_training(monkeypatch):
    monkeypatch.setattr(run_pilot, "manifest_hashes", lambda: {"P0": "a", "P1": "b"})
    with pytest.raises(ValueError, match="P0-P3 manifest mismatch"):
        run_pilot.run_pooled_screening(_args())
```

- [ ] **Step 2: Run orchestration tests and confirm failure**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py -v`

Expected: FAIL at undefined runner functions.

- [ ] **Step 3: Implement P0 and sequential P0-P3 orchestration**

Fit one `sklearn.linear_model.LinearRegression` HAR model from the final three-element HAR vector at
each shared sample origin to the same normalized target as P1-P3. Do not add ticker fixed effects.
All P0-P3 windows have equal loss weight; report per-ticker metrics to expose long-history dominance.
P0 has no epochs or learning curve. Run P1-P3 sequentially with identical
loaders and seed. Before each run, assert the full content hash and ordered keys equal the shared
manifest. Save one validation-only comparison JSON/CSV. Promote only when QLIKE is lower than the
direct control, RMSE is no more than 1% worse, macro DirAcc is no more than one point worse, and the
epoch-5 curve is finite and non-divergent; do not invoke test evaluation.

- [ ] **Step 4: Run the full baseline-local test suite**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test -v`

Expected: PASS.

- [ ] **Step 5: Run pooled smoke on a bounded real-data slice**

Run: `python baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py --phase pooled --epochs 1 --seed 42 --max-tickers 3 --smoke`

Expected: P0-P3 complete, manifest hashes match, and all saved validation metrics are finite.

- [ ] **Step 6: Commit Task 6**

```powershell
git add -- baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py
git commit -m "Add pooled HAR through gate screening runner"
```

### Task 7: Matched G0/G1 graph ablation

**Files:**
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py`

**Interfaces:**
- Produces: `build_graph_manifest(price_frames: dict[str,pd.DataFrame], news_panel: NewsPanel,
  seq_length: int = 22, horizon: int = 5) -> GraphManifest` with global-date 70/15/15 splits.
- Produces: `GraphAblationModel.from_p3_checkpoint(path, use_gnn: bool)`.
- Produces: `build_graph_safe_p3_checkpoint(pooled_manifest: PooledManifest,
  graph_manifest: GraphManifest, output_dir: Path, seed: int) -> Path` and writes its path to
  `graph_safe_p3_checkpoint.txt`.
- Produces: CLI `--phase graph --p3-checkpoint PATH --epochs 5 --seed 42`.

- [ ] **Step 1: Write failing graph split and pairing tests**

```python
def test_graph_snapshot_never_mixes_split_labels():
    manifest = build_graph_manifest(_unaligned_frames())
    assert all(len({node.split for node in snap.nodes}) == 1 for snap in manifest.snapshots)

def test_graph_safe_p3_training_stops_at_graph_train_boundary():
    checkpoint = build_graph_safe_p3_checkpoint(_pooled_manifest(), _graph_manifest())
    assert checkpoint.max_training_target_date <= _graph_manifest().train_end_date

def test_g0_g1_start_with_identical_encoder_and_head_bytes(tmp_path):
    g0 = GraphAblationModel.from_p3_checkpoint(_checkpoint(tmp_path), use_gnn=False)
    g1 = GraphAblationModel.from_p3_checkpoint(_checkpoint(tmp_path), use_gnn=True)
    assert _state_bytes(g0.price_encoder) == _state_bytes(g1.price_encoder)
    assert _state_bytes(g0.head) == _state_bytes(g1.head)
    assert all(not p.requires_grad for p in g1.price_encoder.parameters())
```

- [ ] **Step 2: Run graph tests and confirm failure**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py -v`

Expected: FAIL at undefined graph interfaces.

- [ ] **Step 3: Implement global-date graph manifest**

Reuse the established common-date intersection and correlation adjacency implementation read-only.
Split the common global date axis 70/15/15 before graph windows. Persist ordered node vocabulary,
split label, adjacency bytes, input tensors, raw targets, and hash. Fit any topology threshold or
normalization using graph train only. Reject cross-boundary windows.

- [ ] **Step 4: Implement paired graph models**

Train a graph-safe P3 checkpoint using only pooled samples whose target dates do not exceed the
global graph-training boundary; reject the unrestricted pooled P3 checkpoint. Load its Price/News
encoders, gate/fusion, and matched head and freeze every pretrained component in evaluation mode.
Assert byte equality, `requires_grad=False`, and absent gradients for all frozen components in both
G0/G1. G0 sends `H_base` directly to the matched trainable head. G1 applies the existing GAT and
fixed residual `H_base + H_graph` (alpha 1.0) before an identically initialized trainable head.
Seed and batch order are identical; save paired per-seed metric deltas.

- [ ] **Step 5: Write failing G0/G1 runner/I/O smoke test and confirm RED**

Assert matching graph hashes, distinct output artifacts, complete frozen-component checks, and
rejection of a P3 checkpoint whose provenance crosses the graph-training boundary.

- [ ] **Step 6: Implement G0/G1 runner and rerun the smoke test**

Assert graph content hashes match before training. Run one-batch G0 and G1, check finite outputs,
frozen encoder gradients are absent, and message-passing parameters in G1 receive nonzero finite
gradients.

- [ ] **Step 7: Run graph tests**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 7**

```powershell
git add -- baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_data.py `
  baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_models.py
git commit -m "Add matched GNN off on ablation"
```

### Task 8: Quality gates before GPU pilot

**Files:**
- Modify only files in the new baseline when a gate finds a defect.
- Create/update: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code_review/code_review_2026-08-08.md`
- Create at execution time: the path emitted by
  `$reportPath = "docs/reports/$((Get-Date).ToString('yyyy-MM-dd_HHmm'))_summaryOfUpdate_report.md"`.

**Interfaces:**
- Consumes the complete baseline-local implementation.
- Produces a reviewed, tested implementation eligible for the 5-epoch screen.

- [ ] **Step 1: Run unit and integration tests**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test -v`

Expected: PASS with no skipped baseline-local tests except an explicitly unavailable real-data
fixture, which is a no-go for training until restored.

- [ ] **Step 2: Run smoke tests**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test -m smoke -v`

Expected: at least one real-data load and one end-to-end happy path PASS.

- [ ] **Step 3: Run lint**

Run: `ruff check baselines/2026-08-08_pooled_news_gnn_ablation_baseline`

Expected: PASS.

- [ ] **Step 4: Run changed-line coverage gates**

```powershell
python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test `
  --cov=baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code `
  --cov-branch --cov-report=xml -q
diff-cover coverage.xml --compare-branch 0edc36f --fail-under=100
```

Expected: C0 coverage against pre-implementation commit `0edc36f` is 100%. Emit XML and HTML
branch reports, map changed conditional lines manually, and record C1 >=80% in the review file.

- [ ] **Step 5: Run three-layer adversarial review**

Run Blind Hunter, Edge Case Hunter, and Acceptance Auditor. Fix all critical/high/medium findings
test-first, rerun Steps 1-4, and record findings/dispositions in the baseline code-review file.

- [ ] **Step 6: Write the implementation summary and commit quality-gate fixes**

Record commands actually run, test counts, C0/C1, smoke, lint, review dispositions, risks, and
exact files in `$reportPath`; use `Not run` only with a
specific reason.

### Task 9: Five-epoch screen and architecture recommendation

**Files:**
- Generated only: `results/pooled_news_gnn_pilot_2026-08-08_screen_seed42/`.
- Create at execution time: the path emitted by
  `$reportPath = "docs/reports/$((Get-Date).ToString('yyyy-MM-dd_HHmm'))_summaryOfUpdate_report.md"`.

**Interfaces:**
- Consumes the quality-gated pilot runner.
- Produces validation-only screening results and a decision on which configurations may proceed to
  10-epoch, three-seed confirmation.

- [ ] **Step 1: Run P0-P3 screening**

Run: `python baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py --phase pooled --epochs 5 --seed 42 --output-dir results/pooled_news_gnn_pilot_2026-08-08_screen_seed42`

Expected: P0-P3 artifacts, matching manifests, epoch-5 curves for P1-P3, and validation comparison.

- [ ] **Step 2: Run G0-G1 screening using the selected P3 checkpoint**

Run:

```powershell
$p3Checkpoint = Get-Content -Raw 'results/pooled_news_gnn_pilot_2026-08-08_screen_seed42/graph_safe_p3_checkpoint.txt'
$p3CheckpointPath = $p3Checkpoint.Trim()
python baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py --phase graph --p3-checkpoint $p3CheckpointPath --epochs 5 --seed 42 --output-dir results/pooled_news_gnn_pilot_2026-08-08_screen_seed42
```

Expected: paired G0/G1 artifacts and validation deltas. The graph phase writes and validates the
graph-safe checkpoint path; do not select a checkpoint using test metrics.

- [ ] **Step 3: Validate artifacts and decision rules**

Check finite JSON, nonpositive prediction rate <=1%, manifest hashes, learning curves, news coverage
by ticker, and the exact rules: lower QLIKE than the direct control, RMSE no more than 1% worse, and
macro DirAcc no more than one percentage point worse. Do not evaluate test.

- [ ] **Step 4: Report the screening result**

Write the exact validation table, curves, failures, promoted configurations, and recommendation.
Stop after 5 epochs unless the learning curves and decision rules justify the approved 10-epoch
confirmation round. Even if P3 is not promoted, build G0/G1 from the graph-safe P3 checkpoint to
answer the independent graph research question; label it non-promoted.

- [ ] **Step 5: Commit only source reports, not large generated checkpoints**

Stage the objective report and small machine-readable comparison artifacts according to existing
repository conventions. Do not stage unrelated results or user worktree changes.

### Task 10: Ten-epoch three-seed confirmation and one-time test evaluation

**Files:**
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py`
- Modify: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py`
- Generated: timestamped confirmation results under `results/`.
- Create at execution time: a timestamped `docs/reports/*_summaryOfUpdate_report.md`.

**Interfaces:**
- Produces CLI `--phase confirm --epochs 10 --seeds 42 123 2026`.
- Selects exactly one architecture from aggregate validation metrics before enabling test.

- [ ] **Step 1: Write failing confirmation-selection and test-lock tests**

Require lower QLIKE in at least two of three seeds, median RMSE degradation <=1%, and median macro
DirAcc degradation <=1 point. Assert test evaluation raises until one final architecture decision
is persisted, then allows exactly one evaluation of each of its three declared checkpoints.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_train_smoke.py -v`

Expected: FAIL because confirmation/test-lock behavior is undefined.

- [ ] **Step 3: Implement confirmation orchestration**

Run only promoted configurations for exactly 10 epochs at seeds 42, 123, and 2026. Aggregate
validation first, persist one selected architecture and its evidence, then unlock one test pass per
seed checkpoint. Test metrics cannot revise the selection.

- [ ] **Step 4: Repeat all quality gates**

Repeat Task 8 tests, smoke, lint, coverage, and three-layer review after confirmation code changes.

- [ ] **Step 5: Run confirmation after the 5-epoch report supports it**

Run: `python baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py --phase confirm --epochs 10 --seeds 42 123 2026`

Expected: aggregate validation selection followed by exactly three test evaluations for one final
architecture, with all mandatory metrics and across-seed uncertainty reported.
