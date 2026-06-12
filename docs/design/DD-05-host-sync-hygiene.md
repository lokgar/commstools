# DD-05 — Host-Sync Hygiene (Quick-Win Checklist)

**Phase**: 1a (lowest risk, do first alongside DD-01)
**Depends on**: DD-00 (for before/after D2H counts) — no other DD
**Estimated effort**: ~2-3 days total
**Code touched**: small, behavior-identical diffs (≤ ~20 lines each) across timing.py, frequency.py, recovery.py, analysis.py, metrics.py, mapping.py

---

## 1. Motivation

Beyond the headline kernels, the review found a population of small host-synchronization points: per-channel `float()`/`int()` extraction inside loops, list comprehensions over GPU scalars, and 1-D-only APIs forcing per-channel loops. Each costs one full GPU pipeline flush (~5-50 µs + stall). Individually minor; collectively they serialize otherwise-async GPU receiver chains, and they are all fixable with mechanical, behavior-identical diffs covered by existing tests.

**Rule being enforced** (add to the contributing notes if desired): *inside library code, never extract a scalar from a possibly-GPU array inside a loop — compute the full per-channel vector on device, transfer once, then loop on the host copy.*

## 2. Checklist

Each row is one independent commit. "Fix pattern" abbreviations: **V** = vectorize on device; **B** = batch one D2H transfer of the (C,)-or-similar vector, then loop on the host copy; **G** = gate behind `logger.isEnabledFor(logging.INFO)`.

| # | Location | Current pattern | Fix | Notes |
| --- | --- | --- | --- | --- |
| 1 | [timing.py:549](../../commstools/timing.py#L549), [564](../../commstools/timing.py#L564) | `[corr_all[rx, int(best_tx[rx])] for rx in range(num_sig_ch)]` then `xp.stack` — one sync per RX channel | **V**: `xp.take_along_axis(corr_all, best_tx[:, None, ...], axis=1)` (single gather, no host scalars) | Hot path (MIMO timing estimate) |
| 2 | [timing.py:593-623](../../commstools/timing.py#L593-L623) | per-channel `int(peak_indices[ch])`, `float(coherence[ch])`, `float(metrics[ch])` for logging | **B** + **G**: one `to_device((peaks, coherence, metrics), "cpu")` group, loop over NumPy | |
| 3 | [frequency.py:688](../../commstools/frequency.py#L688), [711](../../commstools/frequency.py#L711) | `[float(slopes[c]) / (2π) for c in range(C)]`, `[float(pwr_per_ch[c]) ...]` | **B**: `slopes_np = to_device(slopes, "cpu")` once; arithmetic on the NumPy vector | Hot path (pilot FOE) |
| 4 | [frequency.py:345](../../commstools/frequency.py#L345), [509](../../commstools/frequency.py#L509) | nested `float(...)` per channel in m-th-power / Mengali-Morelli estimate assembly | **B**: single (C,) transfer, then host loop | |
| 5 | [recovery.py:734-735](../../commstools/recovery.py#L734-L735) (Viterbi-Viterbi), [1338-1339](../../commstools/recovery.py#L1338-L1339) (Tikhonov) | per-channel `float(xp.mean(phi_u[ch] - phi_u[0]))` + `round(...)` in MIMO alignment loops | **V**+**B**: `diffs = xp.mean(phi_u - phi_u[0:1], axis=-1)` on device (all C at once), one (C,) D2H, host `np.round` | Behavior-identical: same means, same rounding |
| 6 | [recovery.py:1520-1547](../../commstools/recovery.py#L1520-L1547) (pilot CPR) | per-channel `xp.interp` loop (1-D-only API); per-channel `CubicSpline` | **V** (linear case): shared pilot grid ⇒ one `xp.searchsorted` on the common x-axis + vectorized gather/lerp over all C rows. Cubic: keep per-channel loop (CubicSpline is cheap; pilot counts are small) but hoist any per-iteration transfers out | Linear interp is the default/hot path |
| 7 | [recovery.py:2229-2256](../../commstools/recovery.py#L2229-L2256) (`resolve_phase_ambiguity`) | per-channel `float(xp.angle(corr))` + `int(round(...))` + per-candidate `_ser` each forcing `float(xp.mean(...))` | **V**+**B**: compute all per-channel correlations as one (C,) device vector, all candidate SER metrics as one (C, n_cand) device matrix, single D2H, decide on host | n_cand = symmetry_order (≤8) — trivially batchable |
| 8 | [analysis.py:402](../../commstools/analysis.py#L402), [585-587](../../commstools/analysis.py#L585-L587), [665](../../commstools/analysis.py#L665) | multiple separate D2H transfers feeding CPU `scipy.optimize` fits | **B**: keep all lag/PSD computation on device; one grouped transfer of the small fit inputs | Transfers themselves are small; this collapses 3+ syncs to 1 per call |
| 9 | [metrics.py:186-188](../../commstools/metrics.py#L186-L188), [278-280](../../commstools/metrics.py#L278-L280), [330-332](../../commstools/metrics.py#L330-L332), [426-428](../../commstools/metrics.py#L426-L428) | per-channel `float(metric[ch])` logging loops in `evm`/`snr`/`ber`/`ser` | **B**+**G**: one `to_device(metric, "cpu")` per call site, loop over NumPy, all gated by `isEnabledFor` | |
| 10 | [mapping.py:65](../../commstools/mapping.py#L65), [88](../../commstools/mapping.py#L88); [metrics.py:605-606](../../commstools/metrics.py#L605-L606) | `(N, M)` broadcast distance matrices (`symbols[:, None] − constellation[None, :]`) | Two options: chunk over N (bounded peak memory, xp-only, safe default), or reuse the DD-01 `TABLE`/`RETURN_ARGMIN` kernel once it lands | JAX path (mapping.py) is JIT-fused already — touch only the CuPy/NumPy paths. Do the chunking now; kernel reuse is a DD-01 follow-up |
| 11 | [recovery.py](../../commstools/recovery.py) `recover_carrier_phase_tikhonov` docstring | `method='exact'` silently forces full-signal D2H + CPU-Numba RTS; `'sskf'` is GPU-native (`filtfilt`) | Docs only: steer GPU users to `'sskf'`; note the transfer in the `'exact'` docstring | No code-path change |
| 12 | [equalization.py:7437](../../commstools/equalization.py#L7437) (`apply_taps`) | `samples_prefix`/`pad_mode≠'zeros'` path routes the **full signal** through CPU (`to_device(samples, "cpu")` → `_build_padded_samples` → `xp.asarray`) | **V**: perform prefix/edge padding on-device with `xp` ops (the helper is plain pad/concat logic); keep CPU helper for NumPy inputs | Default zeros path already clean; one-shot transfer, low priority |

Out-of-scope syncs (justified, do not "fix"): `linear_sum_assignment` transfer in `resolve_channel_permutation` (CPU LP solver, one small matrix); filter-design (`firwin`) and scalar root-finding (`brentq`) on CPU; intentional float64 promotion before every `xp.unwrap` (precision-mandated).

## 2.1 Implementation outcome (2026-06-12)

| # | Outcome |
| --- | --- |
| 1, 2 | Done — `take_along_axis` gather + batched peak/coherence/metric transfers in `estimate_timing` |
| 3, 4 | Partially pre-existing: m-th-power (one scalar argmax sync + batched `mu`), Mengali-Morelli, and pilot-power weights were already batched in current code; the remaining `slopes` per-channel comprehension fixed |
| 5 | Done — vectorized M-fold alignment in both VV and Tikhonov (one (C-1,) D2H) |
| 6 | **No-op by design**: the per-channel `xp.interp` loop contains no host syncs (kernel launches only), and a code comment documents that a `searchsorted` form was deliberately replaced for boundary safety. Left as is. |
| 7 | Done — batched (C,) angle transfer + vectorized rotation; SER gated behind INFO |
| 8 | L402 already batched; the three same-shape linewidth metrics packed into one transfer. PSD arrays (f/S/β) stay as separate transfers (returned to the caller anyway). |
| 9 | Done — all four metrics loops gated behind `isEnabledFor(INFO)` with batched transfers |
| 10 | mapping.py sites are `@jax.jit` (XLA-fused) — skipped per design. metrics.py `mi()` chunked at 64k symbols with on-device accumulator (peak memory ~4 GB → ~270 MB at N=1e6, M=256). |
| 11 | Done — Tikhonov docstring steers GPU users to `'sskf'` |
| 12 | Done — `_build_padded_samples` made backend-generic; `apply_taps` pads on-device |

## 3. Test & acceptance criteria

1. **Numerics**: all existing tests green on `--device=cpu` and `--device=gpu`; the touched functions produce identical results (same dtype, `xpt.assert_allclose` at tight tolerance — these are reorderings of transfers, not of arithmetic, so exact equality is expected for items 1-9; item 10 chunking changes no summation order either).
2. **Sync counts**: for items 1, 3, 5, 7 — nsys (or a `to_device`/`.get()` spy fixture) shows D2H count per function call is **independent of C** after the fix (was: proportional to C).
3. **Benchmarks**: DD-00 `bench_sync_misc.py` before/after numbers recorded per item (expected single-call gains 5-50 %, chain-level gains larger; no hard gate — these are hygiene fixes justified by the sync-count criterion).
4. Each item lands as its own commit with the table row referenced in the message.
