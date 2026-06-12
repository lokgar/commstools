# DD-04 — Block-Update LMS/CMA (Time-Domain) + Block-CMA/RDE Error Modes for `block_lms`

**Phase**: 3 (independent of the CUDA infrastructure — can run parallel to Phase 2)
**Depends on**: DD-00 (benchmark gate) only — pure CuPy/JAX, no custom kernels
**Estimated effort**: ~1 week
**Code touched**: [equalization.py](../../commstools/equalization.py) (`lms` L3761 / `cma` L6428 JAX+CuPy paths; `block_lms` L5536 error stage)

---

## 1. Motivation & positioning

The JAX GPU path for the sequential equalizers is a **per-symbol `lax.scan`** carrying W (C, C, T) ([equalization.py:1802/1943/2159](../../commstools/equalization.py#L1802) area). Per-step arithmetic intensity is far too low to occupy a GPU, and XLA's while-loop per-iteration overhead dominates — the GPU path is currently slower than Numba CPU for single streams.

Breaking the per-symbol dependency with a small update block D turns D dot products into one matrix product per chunk, which XLA/CuPy execute on the wide units. **Positioning against the existing `block_lms`** (mandatory user-facing doc table):

| Mode | Domain | Adaptation lag | Use case |
| --- | --- | --- | --- |
| per-symbol `lms`/`cma` | time | 1 symbol | fastest dynamics, CPU (Numba) |
| **this DD: `update_mode='block'`, D ∈ [8, 32]** | time | D symbols | fast dynamics (polarization rotation) **on GPU** |
| `block_lms` | frequency (overlap-save) | `block_size` (≈256) | throughput king, slow/static channels |

### Naming honesty (correction P4 of the review report)

The original analysis called this "Delayed LMS (DLMS)". Classic DLMS updates **every** symbol with a D-delayed error; what is proposed (and what vectorizes) is **block-update LMS**: weights frozen over D symbols, one aggregated gradient applied per chunk. Expose it as `update_mode='block'`, `block_len=D`. Do not use the word "delayed" in the API.

## 2. Mathematics

For chunk k covering symbols n ∈ [kD, (k+1)D):

```text
Y_k = X_k · W_k            # forward: (D, C·T) @ (C·T, C) — frozen weights
E_k = err(Y_k)             # per-symbol errors, elementwise (training / DD / CMA / RDE)
W_{k+1} = W_k + µ · X_kᴴ · E_k    # one aggregated gradient matmul
```

- **Convergence/misadjustment**: the aggregated gradient effectively scales the step by D ⇒ stability bound shrinks ~1/D and µ must be reduced accordingly (document: "µ_block ≈ µ_per-symbol / D" as the starting heuristic, with the trade-off note that tracking bandwidth drops by the same factor). D ∈ [8, 32]; default 16.
- Per CLAUDE.md: the two matmuls **must** run with float64/complex128 accumulation semantics — JAX: `precision=jax.lax.Precision.HIGHEST` (TF32 mitigation, same as existing code at [equalization.py:1762](../../commstools/equalization.py#L1762)); CuPy: promote the gradient matmul operands to `complex128` (weights stay `complex64` storage).
- Remainder: pad the final partial chunk to D and **mask its gradient contribution** (zero-weight the padded rows); outputs of padded symbols are discarded.

## 3. Implementation

### 3.1 JAX path

Replace the per-symbol scan with a **chunked `lax.scan`**: reshape the regressor stream into (N/D, D, C·T) (build per-symbol windows with the existing window-gather machinery, then reshape), carry W, body = the three lines of §2 with `jnp.dot(..., precision=Precision.HIGHEST)`. Error functions reuse the existing JAX slicer/CMA/RDE error code. This is a new code path selected by `update_mode='block'` — the existing per-symbol scan remains the `update_mode='sequential'` default.

### 3.2 CuPy path

Same chunk loop as a plain Python loop over N/D chunks (the per-chunk work is now matmul-sized, so Python overhead is amortized):

- window gather via `cupy.lib.stride_tricks.sliding_window_view` over the (C, N·sps) input → per-chunk (D, C, T) view → reshape (D, C·T); **views, not copies**;
- `Y = X @ W` (cast to complex128 for the products per §2), error elementwise, `W += µ · X.conj().T @ E`.
This gives `backend='numba'`-independent GPU execution for CuPy inputs without any custom kernel (and also works on NumPy/CPU).

### 3.3 Block-CMA / Block-RDE for `block_lms` (closes original report Phase 3)

`block_lms` already has the complete overlap-save FDAF engine; only the error stage is training/DD-specific. Add `error_fn='lms' (default) | 'cma' | 'rde'`:

- at the error-computation stage (after [equalization.py:6234](../../commstools/equalization.py#L6234) `y_rot`), substitute `e = y·(R₂ − |y|²)` (CMA, R₂ from the constellation) or the ring-radius RDE error;
- CMA/RDE are phase-blind ⇒ when `error_fn != 'lms'`, CPR (`cpr_type`) interactions must be validated: CMA ignores the rotation, so either forbid `cpr_type` with CMA (clean) or document the supported combination. Recommend **forbid in v1** (raise) — pilot/DD-based CPR with CMA error is a research feature, not a port.
- ~50 lines; no structural change to the engine.

### 3.4 JAX inline-BPS incremental running sum (independent quick win, own PR)

Found during review (report §4 finding 4): the JAX inline-CPR kernels `_get_jax_lms_cpr` ([equalization.py:2123-2160](../../commstools/equalization.py#L2123-L2160)) and `_get_jax_rls_cpr` ([equalization.py:2492-2522](../../commstools/equalization.py#L2492-L2522)) with `cpr_type='bps'` recompute the BPS metric over the **entire KB-slot circular buffer every symbol** — `rotated` (B, KB, C) rebuilt and (for non-square constellations) distances re-evaluated for all slots: O(B·KB·M) per step. The Numba reference maintains an O(B·M) incremental running sum.

Fix, preserving exact semantics:

- add to the scan carry: `bps_d2_slots` (B, KB, C) float64 per-slot distances and `bps_metric` (B, C) float64 running sum (carry growth ~64 KB for B=KB=64, C=2 — negligible);
- per step: compute distances for the **new slot only** (B, C) — O(B·M); update `bps_metric += d2_new − bps_d2_slots[:, slot, :]`; `dynamic_update_slice` the slot;
- the warm-up masking (`fill < KB`) is reproduced by initializing evicted-slot distances to 0 — identical to the current `slot_mask` semantics since empty slots contribute 0 today;
- `_cpr_state_to_jax_inits` ([equalization.py:1670](../../commstools/equalization.py#L1670)) gains the per-slot distance buffer reconstruction for warm-start (recompute from the stored `bps_buf` once at entry — cheap).

Expected: ~KB× (32-64×) reduction in per-symbol BPS compute on the JAX path; output must match the current JAX implementation to float64 reduction tolerance (`rtol=1e-12` class — same values summed in different order) and the Numba reference within existing cross-backend test tolerances.

### 3.5 Pilot-aided variants

The pilot-aided `cma`/`rde` modes (`pilot_ref`/`pilot_mask`, kernels at [equalization.py:2910](../../commstools/equalization.py#L2910)/[3012](../../commstools/equalization.py#L3012)/[3123](../../commstools/equalization.py#L3123)/[3190](../../commstools/equalization.py#L3190)) are a per-symbol error-source switch and carry over to `update_mode='block'` directly: compute both error variants elementwise over the chunk and `where(mask, e_pilot·gain, e_blind)` before the aggregated gradient. Include from the start — it is a one-line selection, and pilot-aided CMA is the library's flagship blind-acquisition mode.

## 4. API

- `lms(..., update_mode='sequential'|'block', block_len=16)`; same for `cma` (and `rde` if trivial). Valid with `backend='jax'` (chunked scan) and the CuPy/NumPy array path; `backend='numba'` + `update_mode='block'` → error (pointless combination).
- `block_lms(..., error_fn='lms'|'cma'|'rde')`.
- Docstrings carry the positioning table (§1) and the µ-scaling heuristic (§2).

## 5. Test & acceptance criteria

Workloads (DD-00): `eq/lms-block/16qam/N1e6/C2/sps2/D16`, CMA variant on QPSK, `block_lms/error_fn=cma`.

1. **Convergence (statistical, never exact-equality)**: steady-state EVM/MSE floor within **0.2 dB** of the per-symbol reference after the transient, on the canonical workloads, for D ∈ {8, 16, 32} with µ scaled per §2. Parametrized over `backend_device`/`xp` fixtures, `xpt` assertions.
2. **Performance**: `update_mode='block'` on JAX-GPU ≥ **10×** wall time vs. the per-symbol JAX scan at D=16 (DD-00 baseline).
3. **Equivalence in the static limit**: with µ=0, block and sequential modes produce identical outputs (forward path sanity).
4. `error_fn='cma'` in `block_lms`: converges blind on the PMD-mixed QPSK workload (no training symbols supplied); existing `error_fn='lms'` behavior byte-identical to `main`.
5. Remainder handling: N not divisible by D produces the same output length and matching tail values vs. a truncated-N run.

## 6. Out of scope

- Classic per-symbol-delayed DLMS (no vectorization benefit).
- Block-update RLS (the P recursion does not block-aggregate without changing the algorithm).
- CMA+CPR combined modes in `block_lms` (§3.3 — forbidden in v1).
