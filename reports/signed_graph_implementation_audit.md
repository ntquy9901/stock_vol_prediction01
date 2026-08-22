# Signed-Graph Implementation Audit (V2 / diagnosis doc §5)

Question: does the model consume the sign and weight of the graphical-lasso adjacency, or only its
support (a binary mask)? Answer: **only the support — the graph is effectively binary and unsigned.**

## Trace (file:line)
1. **Graphical lasso returns a signed weighted matrix.** `submission/soict_lstm_gat/edges.py` `glasso_adjacency`
   builds a signed Top-5 partial-correlation adjacency (negative edges retained).
2. **Top-k selection preserves sign/weight** in `edges.py` (values, not just presence).
3. **Batch broadcast** `model.py:73`: `adj = adjacency if adjacency.dim()==3 else adjacency.unsqueeze(0).expand(b,n,n)` — value preserved through broadcast (not consumed yet).
4. **Attention logit** `model.py:34-36`:
   `e_src=(wh*a_src).sum(-1)`, `e_dst=(wh*a_dst).sum(-1)`, `e=leaky(e_dst[:,:,None]+e_src[:,None,:])`
   — the logit depends ONLY on transformed node features `wh`; `adjacency` does not enter it.
5. **Edge sign/weight enters the logit?** NO. `A[i,j]` appears only at `model.py:40`
   `mask=(adjacency!=0)` then `model.py:41` `e=e.masked_fill(~mask, -inf)`.
6. **Edge sign/weight enters the message?** NO. `model.py:42-44`: `alpha=softmax(e,dim=2)`,
   `out=einsum("bijh,bjho->biho", alpha, wh)` — the message is `attention · wh`; `A[i,j]` is never
   multiplied into it.
7. **Separate parameters for negative relations?** NO — a single `W`, single attention.

Therefore the effective graph is `A_model = 1(A != 0)`: edges +0.6, −0.6, +0.1 are indistinguishable.

## Unit tests (characterization — `baselines/2026-08-21_har_anchored_residual/test/test_signed_graph_audit.py`, 3 pass)
- `test_current_gat_ignores_edge_weight_and_sign`: same support, different weights+signs ⇒ **identical output**.
- `test_current_gat_sign_flip_no_effect`: flipping an edge sign ⇒ **identical output**.
- `test_removing_nonself_edges_matches_identity`: self-only graph ⇒ finite self-message.
These pass, confirming weight/sign are discarded by the current layer.

## Implication
Every "graph" result in the study (E2 full-target; E6/E7 residual) tested a **binary equal-weight** graph,
not the signed/weighted graphical-lasso graph. The prior "graph adds no value" conclusion is valid only for
that binary family. Fixing this (a weighted or dual-relation signed layer where `A[i,j]` enters the logit
and/or message) is the architecturally-correct next step IF, and only if, the model-free screening finds
signal in the weighted/signed families. It does not (see `reports/model_free_graph_screening.md`): the
weighted (S1) and signed (S2) screens are ~ placebo and the innovation screen (S3) is ~0 across panels, so
the binary reduction is not what suppressed graph value — the value is absent from the data. A corrected
signed GAT is therefore not promoted at this stage.
