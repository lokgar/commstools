# DD-03 — Native CUDA Sequential Equalizers (LMS/CMA/RDE, RLS, optional DD-PLL)

**Phase**: 4 (deliberately last — highest complexity, weakest honest ROI)
**Depends on**: DD-01 Part A (`_cuda` infrastructure), DD-00 (benchmark gate)
**Estimated effort**: ~2-3 weeks
**Code touched**: new `_cuda/src/seq_equalizers.cu`; backend plumbing in [equalization.py](../../commstools/equalization.py) (`lms` L3761, `cma` L6428, `rde` L6828, `rls` L4608); optionally [recovery.py](../../commstools/recovery.py) (`recover_carrier_phase_pll` L331)

---

## 1. Honest performance model (read first)

This section is binding: the implementation must not be sold on numbers it cannot hit.

- Per-symbol work for LMS at C=2, T=21 is **84 complex64 MACs + a slicer** — tiny. A warp-cooperative step costs ~100-200 cycles of latency per symbol ⇒ ~0.07-0.15 s per 10⁶ symbols. Numba CPU does the same loop in ~0.1-0.3 s plus ~2 ms of transfers.
- Expected outcome: **parity to ~3× vs. Numba CPU**. The PCIe round trip it eliminates is ~2 ms per 10⁶ symbols — negligible (review report §3.2).
- Versus the **JAX per-symbol `lax.scan` on GPU**: **10-50×** — XLA's per-iteration while-loop overhead is the real loser here.
- The honest value proposition is **GPU pipeline residency**: no sync stall, host freed for other work, equalizer composable inside an all-GPU receiver chain, and a grid that batches multiple independent signals at no extra cost.
- **RLS only**: the binding constraint is FP64 throughput (P must be `complex128` per CLAUDE.md). A100/H100 (FP64 1:2) → attractive; GeForce (1:64) → Numba CPU likely *wins*. The implementation must emit a one-time performance warning when `backend='cuda'` RLS runs on a device with FP64:FP32 < 1:8.

If after DD-00 benchmarking the LMS/CMA/RDE kernel cannot beat Numba CPU on an A100-class part, ship it anyway *only if* the pipeline-residency benchmark (equalizer inside a GPU-resident chain, no transfers) shows the end-to-end win; otherwise stop after RLS.

---

## 2. Design — warp-cooperative LMS/CMA/RDE kernel

### 2.1 Scope

**Plain `lms`, `cma`, `rde` only.** The inline-CPR variants (`cpr_type='pll'|'bps'` → `_get_numba_lms_cpr`/`_get_numba_rls_cpr` and JAX counterparts) stay Numba/JAX — their PLL/BPS/slip state machines are large and the port risk is high. Revisit only on demand. (Their JAX-side BPS inefficiency is fixed independently in DD-04 §3.4.)

**Pilot-aided `cma`/`rde`** (`pilot_ref`/`pilot_mask`, Numba `pa_cma`/`pa_rde` at [equalization.py:2910](../../commstools/equalization.py#L2910)/[3012](../../commstools/equalization.py#L3012)): cheap to include — the pilot logic is a per-symbol error-source switch (`mask[n] ? pilot error · gain : blind error`), two extra kernel inputs and one branch in step 4 below. Include in the first kernel version so the CUDA path covers the library's flagship blind-acquisition mode.

One templated kernel `seq_eq<ALGO, SLICER>` with `ALGO ∈ {LMS_TRAIN, LMS_DD, CMA, RDE}` and `SLICER ∈ {GRID, TABLE}` (same slicer modes as DD-01).

### 2.2 Thread mapping

- **One thread block of 64 threads (2 warps) per signal.** Grid: `(n_signals,)` — typically 1, but the design is batched for free (independent signals, e.g. parameter sweeps).
- **All C output channels stay in one block.** The butterfly couples channels through per-output errors and the shared input window every symbol — splitting channels across blocks would require global-memory synchronization per symbol, a non-starter.
- Lane mapping: flatten the (i, j, t) index space of the weight tensor W (C, C, T) — for C=2, T=21 that is 84 elements over 64 lanes (2 elements for some lanes). Each lane owns its W elements **in registers**.

### 2.3 Per-symbol step (sequential loop over n inside the kernel)

1. Cooperative load of the input window `x[:, n·sps : n·sps+T]` (C·T complex64) into **shared memory** (coalesced; double-buffer optional).
2. Each lane computes its partial products `conj(W[i,j,t]) · x[j, t]`; **accumulate in `double2` (complex128) registers** per the CLAUDE.md dot-product mandate.
3. Reduce partials to the C outputs `y[i]` via warp shuffle + a small shared-memory cross-warp reduction; one `__syncthreads()`.
4. Lane 0 of each warp (or thread i < C) computes the error:
   - `LMS_TRAIN`: `e = d[n] − y` (training ref, then switch to DD after `n_train` — same switch logic as the Numba kernel);
   - `LMS_DD`: slicer decision (GRID rounding or TABLE search over shared-memory constellation) then `e = d̂ − y`;
   - `CMA`: `e = y·(R₂ − |y|²)`;
   - `RDE`: nearest ring radius from a small shared table, `e = y·(R²_ring − |y|²)`.
   Broadcast `e` to all lanes via shared memory; `__syncthreads()`.
5. Each lane updates its register-resident W elements: `W += µ · e[i] · conj(x[j, t])` (accumulate the update in double, store back to the `complex64` register copy — mirror the exact promotion scheme used by the Numba kernels so trajectories are comparable).
6. Write `y[:, n]` (and `e[:, n]`) to global memory, coalesced.
7. After the loop: write final W once; optional `w_hist` decimated writes if `store_weights` (match existing semantics).

Register budget: 84 complex64 W + 84 complex128 accumulators ≈ well within 255 regs/thread at 64 threads/block. Occupancy is irrelevant — one signal occupies one SM; state that plainly in the docstring.

### 2.4 Divergence guard

Replicate the existing NaN/Inf divergence flag behavior: per-step check on `y` is too expensive; check W every K=1024 symbols, set a global flag, host reads it **once after the kernel** (same pattern as [equalization.py:6297](../../commstools/equalization.py#L6297)).

---

## 3. Design — block-parallel RLS kernel

### 3.1 Memory plan

- **One block, 128 threads** per signal. M = C·T (e.g. 42 for C=2, T=21).
- Shared memory: P (M×M `complex128`) = 16·M² B ≈ **28.2 KB at M=42**, plus x, u, k (M `complex128` each) and W (M·C `complex64`) ≈ **~31 KB total**. Fits everywhere (A100: 164 KB/SM, Ada: 100 KB/SM). Guard: refuse `backend='cuda'` (fall back with a warning) if 16·M² + vectors > available shared memory (configurable via `cudaFuncAttributeMaxDynamicSharedMemorySize`).
- Occupancy note: ~5 resident blocks/SM on A100 by shared memory — irrelevant for the single-signal case (one block total); the grid is batched like §2.2 for multi-signal use.

### 3.2 Per-symbol step

1. Cooperative load of regressor x (M `complex128` — promote from the `complex64` input on load, per the CLAUDE.md RLS mandate).
2. `u = P·x`: rows of P distributed over threads, FP64 complex accumulators.
3. Denominator `λ + xᴴ·u`: **exploit Hermitian symmetry, `xᴴP = (P·x)ᴴ = conj(u)ᵀ`** — exactly as the existing Numba kernel does (see the Hermitian shortcut at [equalization.py:586](../../commstools/equalization.py#L586) area); block-wide reduction (warp shuffle + shared).
4. Kalman gain `k = u / denom`: parallel over M.
5. Rank-1 update `P ← (P − k·conj(u)ᵀ)/λ`: each thread updates its assigned elements. Safe **after one `__syncthreads()`** because the update reads only the precomputed u/k and touches each P element exactly once.
6. **Hermitian re-symmetrization every step — required behavior, not optional**: update the upper triangle and mirror to the lower, matching the Numba kernel ([equalization.py:591-601](../../commstools/equalization.py#L591-L601) area) and the JAX `(P + Pᴴ)/2` ([equalization.py:1949](../../commstools/equalization.py#L1949) area). The CUDA kernel must replicate the *upper-triangle-then-mirror* scheme so semantics match the Numba reference.
7. Output `y = Wᴴ·x_block` per output channel, error, weight update `W += k·conj(e)` — parallel over M·C.

All P/k/u math in `complex128` throughout the loop. No exceptions (CLAUDE.md: single-precision P updates destroy positive-definiteness).

---

## 4. Optional — DD-PLL kernel

`recover_carrier_phase_pll` ([recovery.py:331](../../commstools/recovery.py#L331)) currently transfers the **full signal** to CPU ([recovery.py:474-477](../../commstools/recovery.py#L474-L477)) for sequential Numba kernels ([recovery.py:63-165](../../commstools/recovery.py#L63-L165), joint variant [187-268](../../commstools/recovery.py#L187-L268)). The state per channel is two scalars (φ, frequency integrator) and ~10 FLOPs/step + a slicer — trivially portable as **one thread per channel** (joint mode: C threads with a shared error reduction) reusing the §2 slicer machinery. Low effort once §2 exists; same fallback contract. Include in this DD as a final, separable PR.

---

## 5. API & integration

- Extend the existing backend enum: `backend='numba' | 'jax' | 'cuda'` on `lms`, `cma`, `rde`, `rls` (and `recover_carrier_phase_pll` if §4 lands).
- **No auto-dispatch.** Summation order differs from the Numba reference; silently changing numerics on GPU inputs is unacceptable for reproducibility. `'cuda'` is explicit opt-in; default remains `'numba'`.
- `backend='cuda'` requires a CuPy input array — raise `ValueError` otherwise (no implicit H2D). `get_kernel(...) is None` ⇒ raise with a clear message naming the fallback (`'numba'`), since the user explicitly requested CUDA.
- Warm-start (`w_init`), `EqualizerResult` fields, `store_weights` semantics: unchanged and shared with the other backends.
- All kernel sources in `_cuda/src/seq_equalizers.cu`, compiled via DD-01 `get_kernel` with `name_expressions` template instantiations.

## 6. Test & acceptance criteria

Workloads (DD-00): `eq/lms/16qam/N1e6/C2/sps2`, `eq/cma/qpsk/...`, `eq/rde/64qam/...`, `eq/rls/16qam/N1e5/...` (PMD-mixed channels).

1. **Convergence equivalence vs. Numba reference** (statistical, never sample-exact): final weights `rtol ≤ 1e-4`; steady-state EVM within **0.1 dB** on the canonical workloads; error-trajectory decay envelopes visually/statistically consistent (compare windowed MSE).
2. **RLS robustness**: 10⁶-symbol run divergence-free; periodic spot checks that P stays Hermitian (max |P − Pᴴ| below tolerance) and positive-definite (Cholesky succeeds on sampled snapshots).
3. **Performance**: documented honest numbers per device class — gate: ≥ 1× (parity) vs. Numba CPU wall time on an A100-class part for LMS/CMA/RDE, ≥ 2× for RLS; GeForce numbers measured and documented (no gate); ≥ 10× vs. JAX GPU scan. FP64-cliff warning fires on GeForce.
4. **Zero behavior change** for `backend='numba'` and `'jax'` (existing tests untouched and green).
5. Batched grid smoke test: 4 independent signals in one launch match 4 sequential single-signal runs.

## 7. Out of scope

- `lms_cpr`/`rls_cpr` inline-CPR variants (review report §7).
- Auto-selection heuristics between backends.
- Multi-GPU.
