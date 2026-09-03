# Phase 4 — Dynamic LRU Expert Cache: implementation plan

> Status: **approved to implement**. Handoff-grade: this doc + `HOT-EXPERT-PHASE0-RESULTS.md`
> + `BENCHMARKS.md` are all a fresh context needs. Repo: `ik_llama.cpp/` (CUDA build:
> `build-cuda/bin/llama-cli`, CPU-only: `build-novlk/bin/llama-cli`).

## Why dynamic (Phase 0 verdict, one paragraph)

GLM-5.3-Flash (321B, 42 MoE layers L3–44, 288 experts, top-8, IQ3_XXS) routing skew is real
but **session- and content-specific**, not model-global: top-12% pooled coverage 0.248
(static-cache gate was ≥0.35); cross-workload overlap 0.229 mean/0.053 min; even same-session
PP→TG transfer is 0.154 vs 0.456 oracle — a static hot list (or prompt-warmed cache) cannot
work. But within-session temporal locality is strong everywhere: an LRU cache tracks the
per-session oracle within a few points (story TG: prev-chunk 0.41–0.45 vs oracle 0.44–0.48;
creature greedy-TG 0.454 vs 0.456, warmed by ~200 tokens). Data/regenerate:
`HOT-EXPERT-PHASE0-RESULTS.md`, `run_glm_phase0_gen.sh`, `analyze_phase0.py`
(cached parses in `traces/*.ids.npz`).

**Conclusion:** a per-session dynamic cache captures ~40–45% of routed-expert reads at
H=32 slices/layer. Traffic model validated against champion #12: TG ≈ 9.01/(1−coverage).

## Target architecture

Per-MoE-layer LRU cache of expert slices in VRAM, capacity H slices/layer, with **two-path
masked execution**: hits computed on GPU from cache slots, misses on CPU from the full fused
tensors in pinned host RAM (today's exact path); contributions summed. A background thread
promotes missed slices HtoD for future tokens.

```
per MoE layer, per decode step (n_tokens=1):
  router (in-graph, UNCHANGED): gate_inp@cur → sigmoid → +bias → ggml_top_k → ids[8]
  classify ids vs remap[288] (orig id → slot | -1)         ← the only new sync point
  HOT  (GPU): fused up_gate + down on slot tensors, misses diverted to a trash slot,
              routing weights zeroed for misses → out_hot
  COLD (CPU): existing fused ops, ids masked -1 for hits   → out_cold   (native -1 skip)
  out = out_hot + out_cold   (cold path alone == today's result when cache empty)
promotion worker (async, dedicated CUDA copy stream):
  misses → admission filter → HtoD 8.6 MB slice copies into free/LRU slots;
  remap/slot metadata published only at step boundaries after copy events complete
```

- **Correctness is placement-independent**: cold-miss = today's behavior; never wrong output.
- GPU-vs-CPU numerics may differ slightly (M0 decides the validation standard).
- Also cover or explicitly skip the MTP tail layer's MoE (`build_glm5next.cpp:445-455`) —
  decide in M1; skipping is fine v1 (MTP off in champion below 40k anyway).

### Sizing (slice = up+gate+down for one expert in one layer ≈ 8.6 MB; 42 layers)

| VRAM budget | H/layer | notes |
|---|---|---|
| 3.5 GB (headroom only, keep 5-layer banks) | 9 | est. ~11 t/s — marginal |
| **12.3 GB (replace the 5-layer banks)** | **34** | **default target, est. 14–16 t/s** |
| 15.8 GB (replace + headroom) | 43 | est. 16–18 t/s; watch KV at 58k |

Default: replace the whole-layer banks (`-ot 'blk.(39|40|41|42|43).ffn_*=CUDA0'` lines in
the champion command go away when `--expert-cache` is on).

### The second bottleneck: PCIe promotion traffic (must-design-for)

Promotion demand at full tilt = (1−hit)·336·8.6 MB × TG ≈ **26 GB/s — at/over the gen4 x16
realized limit (~25 GB/s)**. Mandatory:
- **Admission filter**: promote only on 2nd miss inside a small window (LRU-K-ish); one-shot
  experts never enter. Expected 2–3× promotion-byte cut. A/B in M4.
- **Rate cap**: `--expert-cache-promote-gbps` (default ~8); degrades gracefully to lower hit
  rate, never below today's speed.
- **Read-only during PP** (n_tokens > 8: no promotion/eviction): PP ubatches touch ~all 288
  experts/layer — LRU would thrash 42× per PP for ~0.1 coverage gain. PP keeps today's path;
  TG warms its own cache in ~100–200 tokens.

### Masked-execution strategy (decide in M1; hazard flagged)

- **CPU cold path**: `ggml_compute_forward_mul_mat_id` natively skips `id < 0`
  (ggml.c:18286-18313) — mask hits to -1, done.
- **CUDA hot path hazard**: the fast-TG mmvq-id kernels have **no bounds/negative check**
  (mmvq-templates.cuh:295,319) — a -1 id is an OOB read. Use the **trash-slot pattern**:
  allocate H+1 slots; misses map to slot H (valid memory, garbage output); zero that row's
  routing weight so it contributes exactly 0.
- Weight zeroing, two viable routes (pick in M1):
  (a) **host classify + upload**: read ids back (per-layer sync), compute hot/cold id arrays
      and a binary weight mask on host, upload as input tensors; multiply routing weights by
      the mask in-graph. Simple; adds one DtoH+sync per MoE layer per step (~42 × ~10 µs).
  (b) **in-graph remap**: keep `remap[288]` as a small device tensor (updated per step),
      `remapped = get_rows(remap, ids)`; `mask = remapped >= 0`; `hot_weights = weights*mask`;
      `hot_ids = clamp(remapped, 0, H)`; cold ids = `mask ? -1 : id` likewise in-graph (tiny
      CPU-side split ops). No per-layer host sync for values; more graph surgery.
- **CUDA-graph capture constraint**: capture is per scheduler split
  (ggml-backend.cpp:1444, splits → `sched->splits`); `MUL_MAT_ID` breaks capture unless
  `ne[2]==1 && ids->ne[0]==1` (ggml-cuda.cu:4480) BUT `MOE_FUSED_UP_GATE` peeks ahead and
  skips a following quantized MUL_MAT_ID (ggml-cuda.cu:4487-4501) — **keep the hot path in
  that exact fused-op→down-mul_mat_id shape so decode stays capturable**. Note the generic
  (non-fast-TG) CUDA mul_mat_id does a D2H ids memcpy + sync (ggml-cuda.cu:2832-2835) —
  capture-incompatible; TG shape avoids it.
- Update-detection: `is_cuda_graph_update_required` memcmps node data pointers/params
  (ggml-cuda.cu:4554-4603); ≥4 consecutive changes disables graphs for that key
  (:4751-4758). Slot tensors are stable pointers; only id VALUES change — same as today.
  Kill-switch for A/B: `GGML_CUDA_DISABLE_GRAPHS=1`.

## Milestones (each gated; abort/document conditions explicit)

### M0 — feasibility gates (~0.5–1 day)
1. **Numeric parity**: same IQ3_XXS expert slice, CPU iqk vs CUDA mmvq — bitwise or
   max-abs-diff? Sets whether "expert-trace byte-identical" remains the standard.
   (Champion #12 already computes 5 layers' experts on GPU; PP traces were byte-identical.)
2. **PCIe microbenchmark**: pinned→VRAM 8.6 MB `cudaMemcpyAsync` rate; copy-stream vs
   compute-stream contention. Feeds the promotion-cap default.
3. **Capture-break cost**: champion TG with `GGML_CUDA_DISABLE_GRAPHS=1` at 19.7k —
   the worst-case price if the classify step breaks capture.
4. Confirm `--expert-cache`-off behavior plan: no flag = byte-identical (upstream guard).

### M1 — load-time split + two-path masked execution, all-CPU (~2–3 days)
Expert index is the last dim → split = contiguous memcpy, no dequant/requant.
1. Flags (`--expert-cache H` / `--expert-cache-gb B`): wire like `--expert-trace`
   (common.cpp:1227/4367 → cparams). Loader: glm5next section
   (`llama-load-tensors.cpp:4356`, MoE branch ~4463-4474, `create_std_ffn_exps` 4808-4836):
   allocate per-layer slot tensors `ffn_{up,gate,down}_exps.hot […, H+1]` (H experts + trash
   slot) + CPU-side `remap[288]`, `slot_expert[H+1]`, LRU state. Cold side = the existing
   full tensors initially (`.cold` repack only pays if the static path is later deleted).
2. Graph (`llm_build_moe_ffn` llama-build-context.cpp:1441; glm5next call site
   build_glm5next.cpp:629-639): when cache enabled, emit hot + cold variants with the
   masking strategy above, sum. Keep shapes static; masks are VALUES. Do **not** branch on
   `tensor->buffer` — both paths always exist when the flag is on; the scheduler assigns
   backends by weight-buffer ownership (ggml-backend.cpp:1314-1364).
3. Init: static dummy assignment (experts 0..H-1) — content irrelevant for correctness.
4. Gates: greedy TG text identical; expert-trace byte-identical; **TG regression at H=0 ≤
   noise; H=32-all-CPU ≤ 2%** — known overheads: extra thread-pool traversal + barriers per
   node per layer (ggml.c:29175, 4681), duplicated src1 Q8_K quantization per node
   (ggml.c:18262-18281 / 18588-18610), per-call iqk table setup (iqk_mul_mat.cpp:748).

### M2 — hot slots in VRAM, static content (~1–2 days)
1. Slot tensors on CUDA0 (new buft entry in `ctx_map`, ctor llama-load-tensors.cpp:380-392,
   or explicit alloc in the buffer-creation loop llama.cpp:4796-4885; carve from
   `device_mem[]`/`-gfm` margin, llama.cpp:4466-4682). Fill slices at load from the loaded
   host data (post-`load_all_data`, llama.cpp:4914-4926, before overrides freed at 5022).
2. Wire classify step into the decode loop (strategy (a): reuse the tracer's read-back
   pattern — `llama_expert_trace_mark_outputs` keeps topk tensors alive,
   llama-build-context.cpp:3076-3080 + llama.cpp:6552-6587; record-side readback at
   llama.cpp:6681-6768 shows the pitch honoring).
3. Gates: parity per M0; overhead measured (perf ≤ champion acceptable here — dynamism
   arrives in M3); capture status logged (75-segment baseline vs now).

### M3 — dynamic LRU + promotion (~3–5 days)
1. CPU-side per-layer state: `remap[288]`, `slot_expert[H+1]`, LRU list, pending-copy flags;
   mutations only at step boundaries; publish slot only after its copy event completes.
2. Promotion worker: new thread + **dedicated CUDA copy stream** (streams/events infra:
   common.cuh:71/862/879; `ggml_backend_cuda_event_*` ggml-cuda.cu:5224-5258; all existing
   HtoD uses the main compute stream, ggml-cuda.cu:4318 — add the copy-stream pattern).
   Closest existing mechanism to crib from: `only_active_experts` sched path
   (ggml-backend.cpp:2052-2143: ids read-back :2081-2083, unique-ids bitmap :2089-2096,
   per-contiguous-range `tensor_set_async` HtoD :2106-2120). Thread-pool/comms pattern:
   prefetch_pool (ggml-moe-prefetch.cpp:58-167) — but its workers do no CUDA calls.
3. Admission filter + rate cap + PP read-only policy (see above). Instrument hit/miss/
   promotion counters via the prefetch-pool pattern (atomics + env-gated dump,
   ggml-moe-prefetch.cpp:69-99) — e.g. `IK_EXP_CACHE_DEBUG=1`.
4. Debug verify: `IK_EXP_CACHE_VERIFY=1` runs both paths fully and compares per-layer
   outputs (slow; for tests). Failure mode must stay graceful: any cache bug → slower,
   never wrong.
5. Gates: parity per M0; **measured hit rate within 5 pts of Phase-0 prediction**
   (0.40–0.45 at H=32 on stories); **TG ≥ 12 t/s at 19.7k** before tuning (else investigate).

### M4 — benchmark, tune, document (~1–2 days)
1. Sweep H ∈ {16, 24, 32, 43} at 19.7k and 58k per `BENCHMARKS.md` recipes — sequential
   only (2026-06-09 contamination rule). Admission-policy A/B. Per-layer H budgets optional
   (Phase-0 TG spread: L18–27 ≈ 0.50+ vs L3–6 ≈ 0.19).
2. Pick default H (knee); update `BENCHMARKS.md`, `HOT-EXPERT-REPACK-PLAN.md`, this doc.
3. Success bar: **TG ≥ 14 t/s at 19.7k** (champion 10.16). Document either way.

## Validation & safety (every milestone)

- Parity: greedy TG text diff + expert-trace byte-compare vs champion config. H=0 must be
  byte-exact always. Cache-ON: exact iff M0 says GPU==CPU bitwise; else document the
  divergence budget (logit max-abs-diff on fixed PP + greedy text diff over the battery).
- Trace battery: `traces/glm53-*.bin` + `analyze_phase0.py` (needs zlib on LD_LIBRARY_PATH:
  `export LD_LIBRARY_PATH=/nix/store/78x9i5x1wpqw4kq0h39b8f35abcv156h-zlib-1.3.2/lib:$LD_LIBRARY_PATH`).
- CUDA builds need: `export LD_LIBRARY_PATH=/nix/store/pp1xkyx8s2i28x38ipp0775z6llqy9gj-cuda-merged-12.8/lib:/run/opengl-driver/lib`.
- Bench protocol per BENCHMARKS.md: `-t 32` always (SMT catastrophic), sequential runs,
  greedy `--temp 0 --seed 42`, TG = 128 tokens.

## Risks

| risk | mitigation |
|---|---|
| PCIe promotion saturates (~26 GB/s demand vs ~25 supply) | admission filter + rate cap + PP read-only; M0/M3 measurements |
| Per-layer classify sync breaks CUDA-graph capture / costs too much | M0 quantifies; keep hot path in the fused-op capture shape; worst case graphs off for MoE spans or strategy (b) |
| iqk two-call batching loss (extra pool traversal + requant) | M1 CPU-only profile at H=0/32 before any GPU placement |
| CUDA fast-TG kernel OOB on masked ids | trash-slot pattern (no -1 on GPU side); verified in M1 tests |
| GPU numerics ≠ CPU → trace/text drift | M0 gate sets validation standard; divergence budget documented |
| VRAM overcommit at 58k KV | hard budget flag; ≥1.5 GB KV margin; H uniform first |
| Promotion thread contends with 32 compute threads | DMA copy is host-cheap; keep worker off the compute pool's cores |
| MTP layer experts | v1 may skip MTP-layer cache; document choice |
| Divergence from upstream ik | everything behind `--expert-cache*`; no flag = byte-identical |

## Effort & payoff

M0 0.5–1 d · M1 2–3 d · M2 1–2 d · M3 3–5 d · M4 1–2 d → **7–12 days**.
Expected: TG 10.16 → **13–16 t/s** (PCIe-aware estimate; traffic model
TG = 9.01/(1−hit), validated), PP ~unchanged (77), graceful fallback everywhere.

## Key code facts (recon digest)

- MoE flow (`llm_build_moe_ffn`, llama-build-context.cpp:1441): gate mm :1470 → sigmoid
  :1489-1492 → +`exp_probs_b` :1510-1513 → `ggml_top_k` (argsort) :1531 → weights get_rows
  :1535 + norm/scale → `ggml_moe_up_gate` fused (-fmoe default on, :1605-1614; op ctor
  ggml.c:8090-8134) → down `ggml_mul_mat_id` :1655 → weight/mul_add :1675-1700.
- ids are I32 `[8, n_tokens]`, computed in-graph, data-dependent values, static shapes;
  CUDA decode fuses routing into `cuda_glm45moe_experts` (ggml-cuda.cu:3851-3860); fast-TG
  kernels read ids from device memory at kernel time.
- CPU IQ3_XXS TG: direct AVX2 `mul_mat_qX_K_q8_K_IQ<DequantizerIQ3XXS, ny>`
  (iqk_gemm_iquants.cpp:2683/2769); no repack at Ny=1 (iqk_mul_mat.cpp:245-326);
  expert-grouped batching + `GGML_EXPERT_CHUNKING` work-stealing ON (ggml.c:18327).
- CUDA IQ3_XXS: supported for both ops (:4838-4915); fast TG `ggml_cuda_mul_mat_id` :2874
  (needs F32 acts, ≤8 tok, on-device Q8_1 quant) and `ggml_cuda_moe_up_gate_unary` :3073
  (fuses down + ADD_ID); MMQ-id for larger batches (mmq_id.cu:499/518).
- Scheduler: splits at backend boundaries (ggml-backend.cpp:1444); CPU-resident MoE ops run
  on CPU (offload heuristic ggml-cuda.cu:5186-5222 false for decode shape).
- No precedent for expert-axis (ne[2]) splits or per-expert routing to different weight
  copies — novel graph work. Multi-GPU `llm_build_std_moe_ffn` (llama-build-context.cpp:1709+)
  is row-splits + `ggml_reduce` — not reusable here.
- `--defer-experts` (common.cpp:2184; llama-model-loader.cpp:599-662; mmap DONTNEED
  llama-mmap.cpp:406-545) and `--fit` (auto exps→CPU overrides, llama.cpp:4537-4658;
  mutually exclusive with `-ot`, :4352-4355) — document `--expert-cache` interactions.
- Pinned host RAM for `-ot exps=CPU` (use_mmap_buffer=false →
  `llama_default_buffer_type_cpu(true)`, llama.cpp:353-376; ggml-cuda.cu:1431-1522);
  champion pins ~93-105 GB. Without `-ot`: mmap views; `cudaHostRegister` needs
  `GGML_CUDA_REGISTER_HOST` (ggml-cuda.cu:5483, llama.cpp:4820-4826).
- Upstream analyzer: scripts/expert_trace_analyze.py; ours: analyze_phase0.py.
