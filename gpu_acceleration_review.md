# GPU Acceleration Analysis — Review, Verdict & Revised Roadmap

**Status**: Supersedes `gpu_acceleration_analysis.md` (kept for traceability).
**Scope**: Claim-by-claim verification of the original analysis against the codebase (commit `0c110e7`), corrected quantitative estimates, additional bottlenecks found during review, and a revised implementation roadmap. Detailed technical-task documents live in [docs/design/](docs/design/).

---

## 1. Executive Verdict

The original analysis is **directionally sound**: every bottleneck it names exists in the code, and the three core recommendations (fused BPS kernel, device-resident `block_lms` CPR state, GPU paths for sequential equalizers) survive review. It is approved as a basis for implementation **with the corrections below**, which change priorities, headline numbers, and one algorithm's name.

| # | Correction | Consequence |
| --- | --- | --- |
| **P1** | The headline BPS example (256-QAM) is wrong: 256-QAM is *square*, and both BPS implementations already have an O(1) grid-rounding fast path that never allocates the distance tensor ([recovery.py:969-985](commstools/recovery.py#L969-L985), [equalization.py:6033-6050](commstools/equalization.py#L6033-L6050)). The `(CHUNK, B, M)` tensor path only fires for **non-square constellations**: cross-QAM (32/128), APSK, and arbitrary/geometrically-shaped sets. | Fused kernel (DD-01) is still justified — those constellations are real use cases, and the kernel also collapses the ~8 elementwise launches of the square path — but the beneficiary population and "20× for 256-QAM" claims must be restated. |
| **P2** | §2.4 transfer cost is overstated. A 1M-symbol dual-pol `complex64` signal is 16 MB; D2H+H2D over PCIe 4 is ~1.5-2 ms round trip vs. 50-300 ms of Numba loop compute. | The honest cost of the CPU offload is the **synchronization stall / broken async pipeline**, not bandwidth. This also answers the inline question in the original doc: *the round-trips in CPU-backed CMA/RDE/LMS are not a big speed bottleneck per se* — the sequential CPU loop itself dominates. Reframes DD-03's value proposition. |
| **P3** | §3.2 RLS shared-memory sizing ("≈14 KB") assumes `complex64` — which violates this project's own precision mandate (CLAUDE.md: P matrix **must** be `complex128`). Correct figure: 42² × 16 B ≈ **28.2 KB** (~31 KB with x/u/k/W vectors). Still fits in shared memory, but the binding constraint is the **FP64 throughput cliff**: 1:2 on A100/H100, **1:64 on GeForce RTX 30/40-series**. | A CUDA RLS kernel is attractive on data-center GPUs and roughly pointless on GeForce, where Numba CPU stays the right answer. DD-03 gates its recommendation on detected compute capability. |
| **P4** | §3.3/§4.1 "Delayed LMS" pseudocode is not classic DLMS (which updates *every* symbol with a D-delayed error). It freezes weights over D symbols and applies one aggregated gradient — that is **block-update LMS**. | DD-04 names the mode honestly (`update_mode='block'`, `block_len=D`) and documents the convergence ceiling (~1/D, reduce μ accordingly). |
| **P5** | §5 Phase-3 "Block-CMA via FDAF structure" is over-scoped: `block_lms` *already is* a full overlap-save FDAF engine. Block-CMA is a different per-block error function (`e = y·(R₂ − |y|²)`) in an otherwise identical loop — roughly 50 lines. | Folded into DD-04 as an `error_fn='cma'|'rde'` parameter on `block_lms`; no new engine. |
| **P6** | §3.1 single-thread "near-register-speed" kernel oversells. A single GPU thread runs serial scalar code far slower than a modern CPU core; the design must be **warp-cooperative** (lanes share the (C,C,T) dot products via shuffle/shared-memory reductions). Honest expectation: **parity to ~3× vs. Numba CPU** — but **10-50× vs. the JAX per-symbol `lax.scan` on GPU**, whose per-iteration XLA overhead is the real loser. | DD-03 carries a mandatory honest-performance-model section and is deliberately scheduled **last**. |
| **P7** | §4.2 Pallas/Triton: dropped to future work (§7). CuPy `RawKernel` covers the CUDA path with a much smaller dependency/maintenance surface; the realistic JAX-side win (chunked scan) needs no custom kernels. | One custom-kernel mechanism (NVRTC C++), not two. |
| **P8** | Minor staleness: line numbers drifted (`block_lms` 5556→[5536](commstools/equalization.py#L5536); unwrap copy 6141→[6113](commstools/equalization.py#L6113); cycle-slip copy 6152→[6124](commstools/equalization.py#L6124); inline BPS 6080→[6052](commstools/equalization.py#L6052)); and the cycle-slip loop is already Numba-compiled (`_get_numba_cs_block`, [equalization.py:5429](commstools/equalization.py#L5429)), not plain Python — the residual cost is the per-block D2H/H2D + sync, not loop speed. | Design docs cite the verified current locations. |

---

## 2. Claim-by-Claim Verification

| Original claim (§) | Disposition | Evidence |
| --- | --- | --- |
| §2.1 BPS allocates `(CHUNK, B, M)` distance tensor | **PARTIALLY TRUE** | Tensor allocated only on the non-square branch at [recovery.py:986-989](commstools/recovery.py#L986-L989); square QAM takes the O(1) rounding path at [recovery.py:969-985](commstools/recovery.py#L969-L985). `CHUNK_N = max(block_size, round_up(1024, block_size))` ([recovery.py:954](commstools/recovery.py#L954)), i.e. ~1024 symbols. Metrics `float32` for `complex64` input; blockwise metric is a non-overlapping `reshape(n_b, block_size, B).sum(axis=1)` ([recovery.py:993-995](commstools/recovery.py#L993-L995)). |
| §2.2-B1 `block_lms` per-block synchronous D2H for unwrap carry | **VERIFIED** | `bps_prev4`/`bps_offset4` are always CPU `np.float64` ([equalization.py:5909-5923](commstools/equalization.py#L5909-L5923)); per-block `_delta = to_device(_cumul_dev[:, -1], "cpu")` at [equalization.py:6113-6115](commstools/equalization.py#L6113-L6115). **Additional finding**: the same path also does per-block *H2D* uploads — `xp.asarray(bps_prev4)` / `xp.asarray(bps_offset4)` at [equalization.py:6104](commstools/equalization.py#L6104) and [6110](commstools/equalization.py#L6110) — so device-resident state removes transfers in both directions. Note the unwrap arithmetic itself is already on-device via `cumsum` ([equalization.py:6102-6109](commstools/equalization.py#L6102-L6109)); only the (C,) carry crosses. |
| §2.2-B2 cycle-slip correction full (C, B) D2H + CPU loop + H2D per block | **VERIFIED** | D2H at [equalization.py:6124](commstools/equalization.py#L6124), Numba kernel `cs_block` ([equalization.py:5443-5531](commstools/equalization.py#L5443-L5531); sequential per-symbol online-OLS slip detector with circular buffer + O(1) rolling-stats identities), H2D write-back at [equalization.py:6207](commstools/equalization.py#L6207). Loop is compiled, not plain Python (correction P8). |
| §2.2-B3 inline BPS 4D `(P, C, B, M)` tensor per block | **PARTIALLY TRUE** | Only on the non-square branch ([equalization.py:6052-6055](commstools/equalization.py#L6052-L6055)); square QAM fast path at [equalization.py:6033-6050](commstools/equalization.py#L6033-L6050). The DD slicer has an analogous (C, B, M) tensor at [equalization.py:6263-6266](commstools/equalization.py#L6263-L6266). |
| §2.3 DD-PLL copies full signal to CPU for Numba loop | **VERIFIED** | D2H at [recovery.py:474-477](commstools/recovery.py#L474-L477); sequential Numba kernels at [recovery.py:63-165](commstools/recovery.py#L63-L165) (single-channel) and [187-268](commstools/recovery.py#L187-L268) (joint). No JAX/GPU path exists. |
| §2.4 standalone equalizers round-trip via CPU under `backend='numba'` | **VERIFIED, cost overstated** | Full-array D2H (`np.ascontiguousarray(to_device(samples, "cpu"))`, e.g. [equalization.py:4149-4150](commstools/equalization.py#L4149-L4150)) + H2D return. See correction P2: transfers ~1-3 % of runtime; the sequential loop dominates. JAX path is per-symbol `lax.scan` carrying W (C,C,T) with `Precision.HIGHEST` set ([equalization.py:1762](commstools/equalization.py#L1762), [1885](commstools/equalization.py#L1885)) — slow on GPU due to per-step XLA overhead, which is the stronger argument for DD-03/DD-04. |
| §3.1 register-resident LMS/CMA kernel | **APPROVED, redesigned** | Warp-cooperative, not single-thread (correction P6). Design in DD-03. |
| §3.2 shared-memory RLS kernel | **APPROVED, corrected** | `complex128` mandate → ~31 KB shared; FP64-cliff gating (correction P3). Design in DD-03. |
| §3.3/§4.1 Delayed LMS / chunked JAX scan | **APPROVED, renamed** | Block-update formulation (correction P4). Design in DD-04. |
| §4.2 Pallas/Triton | **DEFERRED** | Future work (§7). |
| §5 roadmap | **REVISED** | See §6 — ordering changed: benchmark harness first, sequential kernels last. |

### 2.1 Full equalizer inventory (variants the original analysis did not disposition)

| Equalizer / variant | GPU profile | Disposition |
| --- | --- | --- |
| `lms`/`rls` with `cpr_type='pll'\|'bps'` (inline CPR; Numba [equalization.py:615](commstools/equalization.py#L615)/[1022](commstools/equalization.py#L1022), JAX [1970](commstools/equalization.py#L1970)/[2376](commstools/equalization.py#L2376)) | Numba: same CPU-offload story as the plain variants (P2 framing applies). JAX: per-symbol scan, **plus the full-buffer BPS recompute defect** (finding 4 below). | CUDA port stays scope-cut (DD-03 §2.1); the JAX BPS defect gets a dedicated fix in DD-04 §3.4 — a clear win independent of any CUDA work. |
| Pilot-aided `cma`/`rde` (`pilot_ref`/`pilot_mask`; Numba `pa_cma`/`pa_rde` [equalization.py:2910](commstools/equalization.py#L2910)/[3012](commstools/equalization.py#L3012), JAX [3123](commstools/equalization.py#L3123)/[3190](commstools/equalization.py#L3190)) | Identical backend pattern to plain `cma`/`rde` (CPU offload under `'numba'`, per-symbol scan under `'jax'`); the pilot logic is a per-symbol error-source switch. | Covered by the same designs at marginal cost: DD-03 kernel takes an optional `pilot_ref`/`pilot_mask` input (§2.1 note), DD-04 block-update handles pilots as masked per-symbol error selection within the chunk. |
| `zf_equalizer` ([equalization.py:7226](commstools/equalization.py#L7226)) | Healthy: overlap-save forward/backward, batched GEMM across bins, explicit Cramer 2×2 fast path avoiding cuSOLVER, no host syncs (debug plot aside). | No action. |
| `apply_taps` ([equalization.py:7341](commstools/equalization.py#L7341)) | Healthy on the default path (zero-copy strided windows + one batched einsum/cuBLAS GEMM). **Exception**: `samples_prefix`/`pad_mode≠'zeros'` routes the *full signal* through a CPU helper — D2H+H2D round trip at [equalization.py:7437](commstools/equalization.py#L7437). | One-shot, not per-loop; DD-05 item 12. |

---

## 3. Corrected Quantitative Analysis

### 3.1 BPS memory traffic (non-square path)

Per 1024-symbol chunk, per channel, with B = 64 test phases and M = 256 points
(`x_rot[:, :, None] - const` → `abs` → `**2` → `min`, none fused by CuPy):

| Intermediate | Shape | Bytes |
| --- | --- | --- |
| complex diff | (1024, 64, 256) `complex64` | 134.2 MB |
| `abs` result | (1024, 64, 256) `float32` | 67.1 MB |
| `d_sq` | (1024, 64, 256) `float32` | 67.1 MB |
| `chunk_min_d` | (1024, 64) `float32` | 0.26 MB |

≈ 270 MB written + ≈ 270 MB re-read ⇒ **~0.55 GB of DRAM traffic per 1024 symbols per channel**, i.e. **~0.55 TB per 10⁶ symbols** — ~0.5 s on an A100 (≈1 TB/s) and >1 s on consumer GPUs, purely memory-bound. (The original report's "~131 GB" figure counted only one intermediate.)

A fused kernel reads 8 KB of symbols and writes 262 KB of `min_d2` per chunk per channel: **~2000× less traffic**. The stage becomes compute-bound at ~N·B·M·8 FLOPs ≈ 1.3×10¹¹ for 10⁶ symbols → **~7-10 ms** at 15-20 TFLOP/s FP32. Realistic end-to-end speedup of the BPS *distance stage*: **one to two orders of magnitude on non-square constellations**; for the full BPS call, 5-20× depending on the share of unwrap/argmin. The same kernel removes the (P, C, B, M) per-block tensor in `block_lms` (e.g. P=64, C=2, B=256, M=128 → 33.5 MB complex diff per block, ~390 blocks per 10⁵ symbols).

### 3.2 Transfer-cost reality check (§2.4)

16 MB round trip (1M dual-pol symbols) ≈ 1.5-2 ms incl. sync, vs. 50-300 ms Numba loop runtime. **Bandwidth is irrelevant; the stall and pipeline break are the cost.** DD-03 is therefore sold as *GPU pipeline residency* (no sync points, host freed, composable with GPU-resident chains), not as a transfer-elimination speedup.

### 3.3 FP64 throughput cliff (binding constraint for the RLS kernel)

| Device class | FP64:FP32 rate | CUDA RLS verdict |
| --- | --- | --- |
| A100 / H100 / V100 | 1:2 | Attractive |
| RTX 30/40-series (GeForce) | 1:64 | Numba CPU likely wins; warn at runtime |

The `complex128` P-matrix mandate (CLAUDE.md) is non-negotiable — single-precision P updates destroy positive-definiteness. The kernel therefore cannot dodge FP64.

---

## 4. Additional Findings (not in the original analysis)

Ranked by expected impact.

1. **`block_lms` kernel-launch overhead → CUDA Graph opportunity (likely the largest `block_lms` win).** Each block iteration issues ~30-50 small CuPy kernel launches (~5-10 µs Python/driver overhead each ⇒ ~200-400 µs/block). Once DD-02 removes all host syncs from the loop, the per-block body can be captured once and replayed as a single graph launch. Bigger than the 16-byte D2H itself. → DD-02 step 3.
2. **Per-channel host syncs in VV/Tikhonov MIMO alignment**: `float(xp.mean(phi_u[ch] - phi_u[0]))` + `round(...)` inside per-channel loops at [recovery.py:734-735](commstools/recovery.py#L734-L735) and [recovery.py:1338-1339](commstools/recovery.py#L1338-L1339). → DD-05.
3. **Pilot CPR interpolates per channel** (`xp.interp` is 1D-only; per-channel `CubicSpline` too) at [recovery.py:1520-1547](commstools/recovery.py#L1520-L1547). Vectorizable via shared `searchsorted` + gather. → DD-05.
4. **JAX inline-BPS recomputes the full circular buffer every symbol.** In `_get_jax_lms_cpr` ([equalization.py:2123-2160](commstools/equalization.py#L2123-L2160)) and `_get_jax_rls_cpr` ([equalization.py:2492-2522](commstools/equalization.py#L2492-L2522)) with `cpr_type='bps'`, each scan step rebuilds `rotated` (B, KB, C) and re-evaluates distances for **all KB buffer slots** — O(B·KB·M) work per symbol on the non-square branch — where the Numba kernel maintains an O(B·M) incremental running sum (add new slot, subtract evicted slot). A ~KB× (32-64×) per-symbol compute waste, fixable by carrying the (B, C) running metric plus a (B, KB, C) per-slot distance buffer in the scan state. → DD-04 §3.4.
5. **Scalar-extraction list comprehensions** that force one GPU sync per element: [timing.py:549](commstools/timing.py#L549)/[564](commstools/timing.py#L564) (`int(best_tx[rx])` per RX channel), [frequency.py:345](commstools/frequency.py#L345)/[509](commstools/frequency.py#L509)/[688](commstools/frequency.py#L688)/[711](commstools/frequency.py#L711) (`float(slopes[c])` per channel). → DD-05.
6. **`resolve_phase_ambiguity`** per-channel `float(xp.angle(...))` + per-candidate SER evaluations ([recovery.py:2237-2250](commstools/recovery.py#L2237-L2250)). → DD-05.
7. **Batched-fit D2H in analysis.py** ([analysis.py:402](commstools/analysis.py#L402), [585-587](commstools/analysis.py#L585-L587), [665](commstools/analysis.py#L665)) and **per-channel `float()` logging loops in metrics.py** (~[metrics.py:186](commstools/metrics.py#L186), [278](commstools/metrics.py#L278), [330](commstools/metrics.py#L330), [426](commstools/metrics.py#L426)). → DD-05.
8. **(N, M) broadcast distance matrices** in LLR demapping ([mapping.py:65](commstools/mapping.py#L65), [88](commstools/mapping.py#L88)) and metrics ([metrics.py:605-606](commstools/metrics.py#L605-L606)) — same pathology family as BPS; the DD-01 kernel (TABLE/GRID modes, argmin variant) is directly reusable. → DD-05 / DD-01 follow-up.
9. **`to_device(..., "cpu")` is always a synchronous `cp.ndarray.get()`** ([backend.py](commstools/backend.py)) — no pinned-memory or stream-async variant exists. Future work (§7); DD-02's device-resident state makes it moot on the hottest path.
10. **User guidance gap**: Tikhonov `method='exact'` (RTS) forces a full D2H + CPU-Numba smoother, while `method='sskf'` is GPU-native (`filtfilt`). Worth a docstring note steering GPU users to `'sskf'`.

Confirmed healthy (no action): `shift_frequency`, `fft_fractional_delay`, AWGN/phase-noise/CD impairments, `correct_carrier_phase` — fully vectorized, device-resident; `to_jax` uses zero-copy DLPack; float64 promotion before `xp.unwrap` is intentional and mandated.

---

## 5. Revised Roadmap

| Phase | Doc | Work | Depends on | Risk / note |
| --- | --- | --- | --- | --- |
| 0 | [DD-00](docs/design/DD-00-benchmark-harness.md) | Profiling & benchmark harness (`benchmarks/`, pytest-benchmark, CUDA events, NVTX, checked-in baselines) | — | Prerequisite: every later phase merges against its numbers |
| 1a | [DD-05](docs/design/DD-05-host-sync-hygiene.md) | Host-sync hygiene quick wins (≤ ~20-line diffs, behavior-identical) | DD-00 | Lowest risk |
| 1b | [DD-01](docs/design/DD-01-bps-fused-kernel.md) | `commstools/_cuda` infrastructure + fused BPS distance kernel | DD-00 | Attacks the one verified memory-bandwidth pathology; infra reused by DD-02/03 |
| 2 | [DD-02](docs/design/DD-02-block-lms-gpu-cpr-state.md) | `block_lms` device-resident CPR state → GPU cycle-slip kernel → optional CUDA Graph capture | DD-01 (infra) | Graph capture is the big win; staged in three PRs |
| 3 | [DD-04](docs/design/DD-04-block-update-equalizers.md) | Block-update LMS/CMA (CuPy + JAX chunked scan) + `error_fn='cma'|'rde'` for `block_lms` | DD-00 only | Backend-portable; can run parallel to Phase 2 |
| 4 | [DD-03](docs/design/DD-03-cuda-sequential-equalizers.md) | Warp-cooperative LMS/CMA/RDE kernel; block-parallel FP64 RLS kernel; optional DD-PLL kernel | DD-01 (infra) | Highest complexity, weakest honest ROI — deliberately last |

---

## 6. Out of Scope / Rejected

- **New FDAF "Block-CMA engine"** — folded into `block_lms` as an error-function parameter (P5).
- **Auto-dispatch to CUDA kernels when input is CuPy** — summation order differs from the Numba reference; numeric changes must be opt-in (`backend='cuda'`).
- **GPU offload of inherently host-bound steps**: `linear_sum_assignment` in `resolve_channel_permutation` (CPU LP solver, one small matrix), filter-tap design (`firwin` etc.), scalar root-finding (`brentq`) — transfers are small and one-shot.

## 7. Future Work (not designed here)

- **Pallas/Triton kernels for the JAX backend** — revisit only if profiling shows JAX-backend users genuinely bottlenecked on BPS/RLS (P7).
- **Pinned-memory / stream-async `to_device`** — general utility, superseded on the hottest paths by device-resident state (DD-02).
- **Fused windowed-sum + argmin BPS kernel** — stretch goal; only if DD-00 profiling shows the cumsum/argmin chain matters after DD-01 (it carries ~1 % of the traffic).
- **CUDA kernels for `lms_cpr`/`rls_cpr` inline-CPR variants** — large PLL/slip state machines; revisit on demand (DD-03 scope cut).

---

## 8. Design Document Index

| Doc | Title |
| --- | --- |
| [DD-00](docs/design/DD-00-benchmark-harness.md) | Profiling & benchmark harness |
| [DD-01](docs/design/DD-01-bps-fused-kernel.md) | Fused BPS distance kernel + `commstools/_cuda` infrastructure |
| [DD-02](docs/design/DD-02-block-lms-gpu-cpr-state.md) | GPU-resident CPR state in `block_lms` (+ CUDA Graph capture) |
| [DD-03](docs/design/DD-03-cuda-sequential-equalizers.md) | Native CUDA sequential equalizers (LMS/CMA/RDE, RLS, DD-PLL) |
| [DD-04](docs/design/DD-04-block-update-equalizers.md) | Block-update LMS/CMA + Block-CMA/RDE error modes for `block_lms` |
| [DD-05](docs/design/DD-05-host-sync-hygiene.md) | Host-sync hygiene checklist |
