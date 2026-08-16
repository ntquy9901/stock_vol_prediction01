# Training acceleration evaluation — cloud GPU vs laptop upgrade

Date: 2026-08-16
Scope: read-only investigation. No code, config, or data modified.
Subject workload: `baselines/2026-08-15_trackA_gat_edge` (Track-A GAT: price LSTM + 2-layer
multi-head GAT over ~33-node graphs + news LSTM), trained one graph snapshot per gradient step.

---

## 1. Detected hardware

All values below are read directly from `nvidia-smi`, the GPU venv `torch`, and `wmic`. Nothing is
estimated in this table.

| Component | Value | Source |
|---|---|---|
| Machine | ASUS TUF Gaming F15 FX507VV (laptop; battery `GA50358` present) | `wmic computersystem`, `wmic path Win32_Battery` |
| Baseboard | ASUSTeK FX507VV | `wmic baseboard` |
| OS | Windows 11 Home Single Language, 10.0.26200 | `wmic os` |
| CPU | 13th Gen Intel Core i7-13620H, 10 cores / 16 threads | `wmic cpu` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | `nvidia-smi`, `torch.cuda.get_device_name` |
| GPU VRAM | 8188 MiB (~8.59 GB total), GDDR6 | `nvidia-smi`, `torch` |
| GPU SMs / compute cap | 24 SMs, capability 8.9 (Ada Lovelace, AD107) | `torch.cuda.get_device_properties` |
| GPU power cap | 80 W | `nvidia-smi` |
| Driver / CUDA | 566.07 / CUDA 12.7 (torch built on cu124, torch 2.6.0) | `nvidia-smi`, `torch` |
| RAM installed | 32 GB (2 × 16 GB, Samsung, DDR5-5600), ~31.6 GB usable | `wmic memorychip`, `wmic computersystem` |
| RAM slots | 2 total (SODIMM), **both populated** | `wmic memphysical MemoryDevices=2` |
| RAM max capacity | 64 GB (65,536 MB) | `wmic memphysical MaxCapacity=67108864 KB` |
| Thunderbolt | **Not detected** — no Thunderbolt PnP device present | `wmic path Win32_PnPEntity ...Thunderbolt` returned "No Instance(s) Available" |

GPU was idle at detection time (0% util, 0 MiB in use), so the utilization figures measured below
were not distorted by other running jobs.

---

## 2. Workload characterization — the crux

### 2.1 Model size and training structure
- `TrackAGatModel`: price LSTM (hidden 64, 2 layers), 2-layer GAT (4 heads, hidden 64, over a
  ~33-node graph), news LSTM (hidden 64). Tiny by deep-learning standards (~10^5 parameters range).
- Feature dims (from code): `price_dim=5`, `news_dim=768` (PhoBERT), sequence length `SEQ=22`,
  ~33 nodes/graph.
- Training loop (`train_resume.py::train_with_resume`): iterates snapshots one at a time,
  **batch = 1 graph per gradient step** (`snap["price"].unsqueeze(0)` → leading dim 1), ~6500
  snapshots per epoch. Each step moves several small tensors host→device individually inside
  `_forward_snap`.

### 2.2 Measured bottleneck: OVERHEAD-bound, not compute-bound
A micro-benchmark of the real `TrackAGatModel` (forward+backward+Adam step) on this RTX 4060, with
representative dims (N=33, SEQ=22, price_dim=5, news_dim=768), varying only the batch size:

| Batch (graphs/step) | ms / step | Throughput (graphs/s) | Peak VRAM |
|---|---|---|---|
| 1 (as trained) | 10.33 | 97 | 152 MB |
| 8 | 9.37 | 854 | 1066 MB |
| 32 | 10.13 | 3158 | 4018 MB |
| 64 | 223.8 (thrashing) | 286 | 8020 MB (hit 8 GB limit) |

Key reading: **batch 1 and batch 32 take essentially the same wall-clock time per step (~10 ms)**,
yet batch 32 does 32× the work. This means at batch 1 the GPU is roughly **97% idle** — the ~10 ms
is fixed per-step overhead (Python loop, ~dozens of tiny kernel launches, per-snapshot host→device
copies), not arithmetic. VRAM at batch 1 is only 152 MB of 8192 MB, so the 4060 is nowhere near
VRAM-limited at the current batch size.

Confirmed by live `nvidia-smi` sampling during a sustained batch=1 loop (as the real trainer runs):

```
0 %,  19 MiB, 16.84 W
0 %, 107 MiB, 18.86 W
12 %, 171 MiB, 19.35 W
18 %, 171 MiB, 20.58 W
```

GPU utilization stays 0–18%, power ~20 W of the 80 W cap, memory ~170 MB. The card spends almost
all its time waiting on the Python/launch pipeline. This is the definition of an overhead-bound
(CPU/launch-bound) workload.

The batch=64 row shows the only genuine hardware limit of the 4060: at 64 graphs the run needs
~8 GB and hits the VRAM ceiling, causing allocator thrashing. Everything up to batch 32 (~4 GB)
runs comfortably.

---

## 3. Question A — how much faster on a cloud A100 (~40 GB)?

### 3.1 Raw-spec comparison (context only)
| | RTX 4060 Laptop (detected) | A100 40 GB |
|---|---|---|
| SMs | 24 | 108 |
| FP32 peak | ~11–15 TFLOPS | ~19.5 TFLOPS (TF32 ~156 TFLOPS) |
| VRAM | 8 GB GDDR6 | 40 GB HBM2 |
| Mem bandwidth | ~256 GB/s | ~1555 GB/s |
| Power | 80 W | 400 W |

On paper the A100 is several times stronger. **That ratio does not transfer to this workload**,
because the workload is overhead-bound (Section 2.2): a faster GPU cannot accelerate Python loop
iterations, kernel-launch latency, or per-snapshot host→device copies, which are what actually
consume the 10 ms/step.

### 3.2 Realistic speedup
- **A100, code unchanged (still batch=1): ~1–1.5×, possibly no measurable gain.** The per-step time
  is dominated by launch/host overhead that the A100 does not reduce; on a shared cloud VM the
  effective figure can even be at parity. This is a fully overhead-bound regime — expect little.
- **The real lever is batching the snapshots, not the GPU.** On the *existing* 4060, batching 32
  graphs/step already yields ~3158 vs 97 graphs/s ≈ **~32× throughput at zero hardware cost** (fits
  in ~4 GB). This requires code changes: pad/stack snapshots to a common node count (or use a
  block-diagonal batched graph), batch the LSTM and GAT forward passes, and pre-move tensors. The
  current per-snapshot `.to(device)` and batch=1 loop would need restructuring.
- **A100 only pays off *after* batching.** With batching plus the A100's 40 GB, batch could grow to
  128–256 (the 4060 already OOMs at 64), and A100 compute would then be exercised — plausibly
  another ~2–5× on top of the batched 4060. Net vs today: high tens of × — but ~20–32× of that is
  obtainable locally for free by batching alone.

Bottom-line multiplier for the question as asked ("move to A100"): **~1–2× if nothing else changes;
the money is in code-level batching, which is free on the current GPU.**

### 3.3 Colab / cloud caveats
- **A100 is not the default.** Colab free tier gives T4 or (sometimes) L4, not A100. A100 requires
  Colab Pro+ / Colab Enterprise / GCP, and even paid A100 sessions are frequently "unavailable" at
  peak.
- **T4 can be slower than this laptop for FP32.** The free-tier T4 is Turing (~8 TFLOPS FP32), below
  the 4060 laptop's FP32; its advantage is 16 GB VRAM (more batching room), not raw speed. L4 (Ada,
  24 GB) is closer in generation to the 4060.
- **Session/runtime limits:** Colab disconnects on idle and caps total session length; long
  unattended multi-horizon runs risk interruption (the checkpoint/resume in `train_resume.py`
  mitigates but does not remove this).
- **Data/venv upload cost:** the repo's `data/`, feature parquets, and a CUDA venv must be uploaded
  or rebuilt per session; for a job whose GPU is 97% idle, upload + setup overhead can rival the
  training time saved.

---

## 4. Question B — laptop GPU / RAM upgrade feasibility

### 4.1 Discrete GPU (add or replace) — NOT feasible
The RTX 4060 Laptop GPU in the FX507VV is a soldered (BGA) mobile GPU on the mainboard, as in
essentially all modern gaming laptops. It cannot be removed, replaced, or added to. There is no
second internal GPU slot. Verdict: **not feasible.**

### 4.2 External GPU (eGPU) — NOT feasible on this unit
eGPU enclosures require Thunderbolt 3/4 (or OCuLink). `wmic` found **no Thunderbolt device** on this
machine; the FX507VV's USB-C is USB 3.2 with DisplayPort, not Thunderbolt. Without Thunderbolt there
is no standard eGPU path (an OCuLink mod is non-standard and out of scope). Even where Thunderbolt
exists, eGPU carries a PCIe-bandwidth penalty (~x4 link) — though notably, for *this* overhead-bound
workload that penalty would matter little; the blocker here is simply the absence of Thunderbolt.
Verdict: **not feasible on this unit.**

### 4.3 RAM upgrade — feasible
`wmic` reports 2 SODIMM slots, both populated with 16 GB DDR5-5600 (32 GB total), max capacity
64 GB. RAM is not soldered on the FX507VV (standard SODIMM). To increase capacity, both existing
16 GB modules would be **replaced** (no free slot to simply add to) with 2 × 32 GB DDR5 SODIMMs,
reaching the 64 GB ceiling. Verdict: **feasible** (replace both DIMMs).

However, RAM is **not the bottleneck** for this workload — training uses ~170 MB of GPU memory and
the model/data are small; 32 GB system RAM is already ample. A RAM upgrade would not speed up
training. It is only worth doing if system RAM is a constraint elsewhere (large in-memory feature
panels, many parallel jobs).

---

## 5. Recommendation

Priority order, most cost-effective first:

1. **Batch the snapshots in the training loop (free, largest win).** ~20–32× throughput on the
   existing RTX 4060 with no hardware spend, because the workload is overhead-bound at batch=1.
   This is the single highest-leverage change and directly attacks the measured bottleneck.
   Secondary free wins: keep tensors resident on-device / use pinned memory + non-blocking copies;
   reduce per-snapshot Python overhead.
2. **Only if still too slow after batching, rent a cloud A100 by the hour** (GCP / a GPU cloud),
   not for the raw GPU but because its 40 GB allows very large batches; expect a further ~2–5× over
   the batched 4060. Prefer hourly rental over Colab for reliability (no session caps, A100
   actually available).
3. **Do not upgrade the laptop for training.** Discrete GPU is soldered (not upgradable), eGPU is
   impossible (no Thunderbolt). A RAM bump to 64 GB is technically feasible (replace both SODIMMs)
   but will not speed up training and is only justified by non-GPU memory needs.

An A100 "moves the needle" far less than the raw spec sheet implies for this specific model, because
the code — not the GPU — is the current limiter.

---

## Notes / limitations
- Speedup figures for the batched and A100 cases are engineering estimates grounded in the measured
  batch-scaling table (Section 2.2); the batch-1 vs batch-32 wall-time equality and the live
  0–18% GPU utilization are direct measurements on this machine. A100 numbers are not measured here
  (no A100 available in this environment) and are presented as ranges with reasoning, per request.
- Micro-benchmark used representative synthetic tensors at the real model dims, not the real data
  loader; it isolates compute/overhead per step, which is exactly the quantity in question.
