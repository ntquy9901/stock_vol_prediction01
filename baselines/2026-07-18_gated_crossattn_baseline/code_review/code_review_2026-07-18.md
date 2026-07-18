# Code Review — Gated Cross-Attention Baseline (2026-07-18)

**Tool:** `/code-review`-style agent pass (1 agent, correctness focus across all 3 new
2026-07-18 baselines together) + self-fix, per CLAUDE.md DoD.

## Findings — 1 HIGH severity, confirmed and fixed

### [HIGH] Degenerate single-token cross-attention (query had no effect)

**Original code** collapsed the 22-day news sequence to ONE vector (via `NewsTemporalEncoder`'s
LSTM) *before* cross-attention, so both Q and K/V had `seq_len=1`:

```python
# BEFORE (buggy)
daily = self.news_pool(x_emb, mask)
news_rep = self.news_temporal(daily)                   # [B,S,64] — already collapsed to 1 vector
q = self.q_proj(har_embed).reshape(B * S, 1, -1)
kv = news_rep.reshape(B * S, 1, -1)                      # seq_len=1
attended, _ = self.cross_attn(q, kv, kv)
```

**Why this is wrong:** softmax over a single key is mathematically always exactly `1.0`,
regardless of the query's value. `attended` therefore reduces to `out_proj(V_proj(news_rep))` —
**completely independent of `har_embed`/`q_proj`.** The whole point of MSGCA-style gated
cross-attention (query-conditioned selection of which news content matters) was not implemented;
`q_proj` received no useful gradient (dead weights), and the model degenerated to "always attend
to the only available token," which is not attention at all.

**Fix:** attend over the **un-collapsed** 22-day sequence of daily-pooled news vectors (real
multi-token K/V, `seq_len=22`), removing `NewsTemporalEncoder` entirely from this baseline
(attention now performs both temporal aggregation and relevance selection):

```python
# AFTER (fixed)
daily = self.news_pool(x_emb, mask)                     # [B,seq,S,d_news] — NOT collapsed
kv = daily.permute(0, 2, 1, 3).reshape(B * S, seq_len, -1)  # [B*S, 22, d_news]
q = self.q_proj(har_embed).reshape(B * S, 1, -1)          # [B*S, 1, d_news]
attended, _ = self.cross_attn(q, kv, kv)                   # now genuinely query-dependent
```

**Regression test added:** `test_attended_output_depends_on_query` — feeds two very different
`x_har` inputs (different query) with the SAME news input, hooks `cross_attn`'s output, and
asserts the two attended outputs differ. This test **fails on the pre-fix code** (confirmed by
construction — the fix was implemented in response to this exact review finding) and passes on
the fixed code.

## Other findings

- **Test coverage gap (CONFIRMED):** no test exercised `train_gated_crossattn.py`'s real
  `train_epoch` (the `[B,S]→[B*S]` reshape into the MSE criterion). **Fixed:** added
  `test/test_train_loop.py`.
- **Informational (not a finding):** same dead-code `har.fusion` freeze as the other 2 baselines
  — not changed.

## Impact on results

The 15-epoch real-data training run reported in the final summary report used the **fixed**
code (rerun after this review). The original buggy run's numbers are superseded and not
reported as final.

## Final state

5/5 pytest pass (4 smoke incl. the new regression test + 1 train-loop integration).
