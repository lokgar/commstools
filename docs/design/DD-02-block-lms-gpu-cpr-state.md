# DD-02 — GPU-Resident CPR State in `block_lms` (+ CUDA Graph Capture)

**Phase**: 2 — three steps, each its own PR
**Depends on**: DD-01 Part A (`_cuda` infrastructure), DD-00 (benchmark gate)
**Estimated effort**: ~1 week (steps 1-2) + optional step 3
**Code touched**: [equalization.py](../../commstools/equalization.py) `block_lms` (L5536-6357), new `_cuda/src/cs_block.cu`

---

## 1. Motivation

`block_lms` is the library's GPU-committed equalizer, but its per-block loop ([equalization.py:6005](../../commstools/equalization.py#L6005)) crosses the PCIe bus every iteration when CPR is active:

1. **Unwrap carry** — `bps_prev4`/`bps_offset4` are *always* CPU `np.float64` ([equalization.py:5909-5923](../../commstools/equalization.py#L5909-L5923)). On GPU, each block does a synchronous D2H (`_delta = to_device(_cumul_dev[:, -1], "cpu")`, [equalization.py:6113-6115](../../commstools/equalization.py#L6113-L6115)) **and** two H2D uploads (`xp.asarray(bps_prev4)` / `xp.asarray(bps_offset4)`, [equalization.py:6104](../../commstools/equalization.py#L6104), [6110](../../commstools/equalization.py#L6110)). The bytes are trivial (16 B); the **synchronization stall** per block is the cost — ~390 stalls per 10⁵ symbols at `block_size=256`.
2. **Cycle-slip correction** (`cpr_cycle_slip_correction=True`) — full `(C, B)` float64 D2H at [equalization.py:6124](../../commstools/equalization.py#L6124), CPU Numba kernel `cs_block` ([equalization.py:5443-5531](../../commstools/equalization.py#L5443-L5531)), H2D write-back at [equalization.py:6207](../../commstools/equalization.py#L6207), plus the per-channel net-slip carry loop at [equalization.py:6214-6217](../../commstools/equalization.py#L6214-L6217) reading host arrays.
3. **Kernel-launch overhead** (found during review; not in the original analysis) — each block iteration issues ~30-50 small CuPy kernels (FFTs, einsum, elementwise chains) at ~5-10 µs Python/driver overhead each ⇒ **~200-400 µs/block of pure launch overhead**. For small/medium blocks this dominates the actual GPU work. Removing items 1-2 makes the loop body host-sync-free, which is the prerequisite for capturing it as a **CUDA Graph** and replaying it with a single launch per block.

The CPR estimate feeds back into the error computation that updates the weights ([equalization.py:6228](../../commstools/equalization.py#L6228) onward), so **cross-block pipelining is impossible by construction** — the only levers are (a) removing the syncs and (b) collapsing launch overhead. That is exactly steps 1-3.

---

## 2. Design

### Step 1 — Device-resident unwrap carry (PR 1)

- Inside the block loop, `bps_prev4` and `bps_offset4` become `xp` arrays on the active device (i.e. `xp.asarray(...)` **once** before the loop; all per-block updates on-device).
- The GPU branch ([equalization.py:6101-6115](../../commstools/equalization.py#L6101-L6115)) loses the `to_device` and the per-block `xp.asarray` calls: `bps_prev4 += _cumul_dev[:, -1]` etc. run as device ops. The CPU/NumPy branch ([equalization.py:6091-6100](../../commstools/equalization.py#L6091-L6100)) is unchanged (np arrays are already "device-resident" there).
- The net-slip carry loop ([equalization.py:6214-6217](../../commstools/equalization.py#L6214-L6217)) becomes a vectorized device op: `bps_offset4 += (phi_corr_last − phi_blk_last) * 4.0` computed from device arrays (no `float()`/`!= 0.0` host tests — the `+= 0` no-op is free).
- **CPRState contract preserved**: `CPRState` ([equalization.py:78](../../commstools/equalization.py#L78)) documents "all arrays are CPU NumPy". Convert *at function entry* (warm-start ingest) and *at function exit* (state export) only. Add a round-trip warm-start test on GPU.

### Step 2 — GPU cycle-slip kernel (PR 2)

**Decision: port the slip detector to a tiny sequential GPU kernel** (`_cuda/src/cs_block.cu`), keeping the Numba kernel verbatim as the CPU fallback.

Rejected alternatives, recorded for the implementer:
- *Pinned-memory async transfer*: similar per-block wall time, but keeps the host in the loop and **forecloses CUDA Graph capture (step 3)** — the bigger prize.
- *Per-block (instead of per-symbol) regression*: changes detection behavior; not acceptable as a default. (Possible future `cs_mode='block'` fast variant — out of scope.)

Kernel design:
- Launch: **one block, C threads** (one thread per channel). Channels are independent; within a channel the algorithm is strictly sequential — that's fine, the total work is C×B ≈ 512 trivial scalar iterations (~20-40 µs even single-threaded, vs. two PCIe syncs + host loop today).
- Faithful port of `cs_block` ([equalization.py:5443-5531](../../commstools/equalization.py#L5443-L5531)) — **including** the rolling-stats identities, exactly:
  - prediction: n<10 → last value; n≥10 → online OLS with closed-form `Sx`, `Sxx` (relative coordinates), `slope/intercept` from `Sy`, `Sxy`;
  - slip test: `k_slip = round(diff/quantum)`; snap if `|diff| > threshold && k_slip != 0`;
  - circular-buffer update with the O(1) full-buffer identity `Sxy_new = Sxy_old − Sy_old + y_old + (H−1)·y_new`.
- State buffers stay **persistent in global device memory** across blocks (allocated once before the loop): `cs_buf_y` (C, H=100) float64, `cs_buf_ptr`/`cs_buf_n` (C,) int64, `cs_stats` (C, 4) float64 — ~1.6 KB for C=2; latency is hidden by L2 at this size. (`cs_buf_x` is unused in the current kernel — drop it from the CUDA signature; keep it in the Numba call for compatibility.)
- All scalar math in **float64** (matches the Numba kernel; the FP64 throughput cliff is irrelevant at 512 iterations/block).
- Input/output: operate in-place on the device `_phi_f64` (C, B) float64 (or `phi_corr` device copy) — the D2H at [6124](../../commstools/equalization.py#L6124) and H2D at [6207](../../commstools/equalization.py#L6207) are deleted on the GPU path.
- Dispatch via `get_kernel("cs_block")` (DD-01 Part A); `None` ⇒ existing D2H + Numba path.
- CPRState export at function exit transfers the cycle-slip state buffers back to CPU NumPy (entry: upload once).

### Step 3 — CUDA Graph capture (PR 3, optional, separate)

Prerequisites established by steps 1-2: **zero host syncs in the block-loop body**. Remaining prerequisite to engineer here: **stable buffer pointers** — graph replay re-executes recorded pointers, so all per-block temporaries (`x_win`, FFT outputs, `rotated`, `min_d2`, `cat_d2`/`cs_d2`, `e_scatter`, …) must become **pre-allocated workspaces** reused every iteration, not fresh CuPy pool allocations. This is a mechanical but invasive refactor of the loop body; do it behind a `use_cuda_graph=True`-style internal flag.

- Capture once on the first full block via `cp.cuda.Stream.begin_capture()` / `end_capture()` (CuPy's graph API), then `graph.launch()` per block. The final partial block (B′ < B) runs eagerly through the normal path.
- Caveats to verify during implementation: cuFFT plan callbacks inside capture (CuPy's FFT is capturable on modern versions — verify against the pinned CuPy version); no Python-side branching inside the captured region (the `cpr_type`/flags branches resolve *before* capture since they're loop-invariant).
- Expected effect: ~30-50 launches/block → 1; saves the ~200-400 µs/block overhead — at `block_size=256` and 10⁶ symbols (~3900 blocks) that is **0.8-1.6 s of pure overhead removed**, likely the largest single win in this DD.

## 3. Test & acceptance criteria

Workload (DD-00): `block_lms/bps+cs/16qam/N1e6/C2`, GPU.

1. **Numerical**: `phase_trajectory` and `y_hat` match `main` within `rtol=1e-6` (per-channel float64 phase math is order-preserved ⇒ near-bit-exact expected); weights at exit `rtol=1e-6`.
2. **Sync elimination**: nsys capture of the block loop shows **zero D2H memcpy operations inside the loop** (allowed: entry/exit conversions, final divergence-flag check at [equalization.py:6297](../../commstools/equalization.py#L6297)). Alternative CI-friendly check: monkeypatch-spy on `to_device` asserting call count is loop-count-independent.
3. **CPRState**: warm-start round-trip test — run N symbols, export state, resume, compare against single uninterrupted run (existing pattern; must pass on GPU).
4. **Performance**: ≥ 2× wall-time vs. DD-00 baseline after steps 1-2; ≥ 5× with step 3 graph capture (gate step 3 on its own number).
5. CPU path (`--device=cpu`) byte-identical to `main` throughout.

## 4. Out of scope

- `cs_mode='block'` approximate slip detection.
- Pinned-memory/async generalization of `to_device` (review report §7).
- Graph capture for the non-CPR `block_lms` configuration (possible follow-up; same workspace refactor applies).
