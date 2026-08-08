# Graph GPU Path Review

## Scope

Reviewed the graph-only device selection, graph-safe P3 transfers, G0/G1 transfers, provenance
checks, and CPU-only smoke coverage. P0-P3 screening paths were not changed.

## Blind Hunter

No critical or major finding. The graph runner validates required manifest hashes and the loaded
checkpoint's graph train boundary and train hash before calling `model.to(device)`.

## Edge Case Hunter

No critical or major finding. `auto` falls back to CPU only when CUDA is unavailable; explicit
`cuda` raises. CUDA seeding sets all CUDA seeds and deterministic cuDNN flags. Validation converts
predictions to CPU only for inverse-ticker metrics, outside the training inner loop.

## Acceptance Auditor

No critical or major finding. The path retains masked adjacency, graph-local normalized targets,
frozen P3 encoders, inverse-ticker metrics, and one update per snapshot. Runtime JSON records the
requested and selected device, PyTorch version, and CUDA version.

## Limitation

The graph runner remains a Python per-snapshot loop. It does not batch graph snapshots; this is
intentional for this bounded first GPU path.
