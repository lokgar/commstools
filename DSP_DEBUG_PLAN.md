# Intradyne DP Coherent DSP — Debug & Remediation Plan

## Context — why this plan exists

Captured intradyne dual-pol 16-QAM waveforms run through the CommsTools offline DSP
pipeline (FOE → resample → timing → matched filter → iterative `block_lms` with internal
BPS CPR → standalone BPS → `correct_carrier_phase` → `resolve_phase_ambiguity` → demap)
produce a **visually clean constellation** (blind EVM ≈ −18 dB) but **catastrophic SER over
the first ~16 k symbols**:

| Discard count | Errors    | Window | SER     |
| ------------- | --------- | ------ | ------- |
| 8 192         | 992       | 40 960 | 2.4e-2  |
| 16 384        | 18        | 32 768 | 5.5e-4  |

→ 974 of the 992 errors fall in symbols 8 192–16 384 (~12 % SER over that window);
underlying steady-state SER is ~5 e-4. The "fix" of doubling `num_train_symbols` is a
symptom mask. The homodyne-loopback path (no FO, no independent phase noise) works fine
on the same code, so the regression is confined to stages that activate under non-zero
FO and independent TX/LO phase noise. The user separately observed that **MSE jumps at
the start of every warm-started `block_lms` iteration even with CPR disabled** — so at
least one issue is independent of CPR.

Goal of this plan: replace symptom-masking with a small, additive set of library changes
that (a) preserve the homodyne-loopback path byte-for-byte, (b) fix the warm-start
inconsistencies, (c) restore the JAX backend with CPR, and (d) eliminate the SER cliff
in the user's pipeline. The diagnostic-experiments phase has been skipped at user
request; the plan jumps directly to fixes, ordered so that the smallest/independent
landings come first and the multi-feature warm-start work follows.

## Issues — root causes, locations, severity

### Issue A — CPR state silently zero-initialized on every call  *(severity: critical)*

Even when `w_init` warm-starts the taps, the BPS 4-fold unwrap accumulator
(`bps_prev4`, `bps_offset4`), the BPS sliding-window distance history (`bps_d2_hist`),
the PLL integrators (`pll_phi`, `pll_freq`), and the cycle-slip regression buffers
(`cs_buf_x/y/ptr/n/stats`) are zeroed on entry. The final full-signal DD-only call
therefore re-locks CPR from φ = 0 while the taps are already converged → ~5–10 k
symbols of wrong rotation → wrong DD decisions → LMS gradients corrupt the taps → the
MSE plot's slow −11 → −14 dB roll-off across 20 k symbols, exactly matching the
observed SER region.

Locations: [equalization.py](commstools/equalization.py)
- `lms` (CPR branch): lines [3444–3451](commstools/equalization.py#L3444-L3451);
  `bps_prev4`/`bps_offset4` reset inside the JIT shim near 597–598.
- `block_lms`: lines [4604–4617](commstools/equalization.py#L4604-L4617).
- `EqualizerResult`: lines [76–144](commstools/equalization.py#L76-L144).

### Issue B — Zero-padding warmstart transient (independent of CPR)  *(severity: high)*

Every call left-pads samples with `pad_left = num_taps // 2` zeros
([lms 3345–3348](commstools/equalization.py#L3345-L3348),
[block_lms 4634–4640](commstools/equalization.py#L4634-L4640)). The first output
symbol's tap window is half zeros even with perfectly trained warm-start weights, so
`y[0] = W·window` is wrong, the slicer/training error spikes, LMS pushes weights to
compensate, and after approximately `num_taps` samples things resettle. Inside one
long call this is invisible; under the user's iterative warm-start pattern it produces
a visible MSE jump at the start of every iteration. The user confirmed this is present
even with `cpr_type=None`.

### Issue C — `_normalize_inputs` re-normalizes per call  *(severity: medium)*

[`_normalize_inputs` (lines 2583–2631)](commstools/equalization.py#L2583-L2631) computes
`norm_vec = rms(samples) · √sps` and divides. Warm-starting weights trained against one
input scale and applying them to a different-length sample buffer produces a small but
non-zero scale mismatch (~0.5 % for tiled data; larger for diverse captures) that the
LMS update has to chase. Reported in `EqualizerResult.input_norm_factor` but not
pluggable.

### Issue D — `resolve_phase_ambiguity` scores on the full sequence  *(severity: high for diagnosis)*

[sync.py:4012–4093](commstools/sync.py#L4012-L4093) tests `k ∈ {0,1,2,3}` rotations and
picks the one with lowest SER over the **entire** symbol stream. When the front of the
stream is junk, the chosen rotation is biased; the user observed `best_k` flipping
non-deterministically (k=3 vs k=1) between traces. With a `num_skip_symbols`
parameter the choice becomes deterministic.

### Issue E — `lms()` step-size sits at the stability boundary  *(severity: low; docs only)*

User's `lms()` test uses `step_size=8e-3` with `num_taps=75`, `C=2`. The complex-LMS
stability bound is μ < 2/(C · T · P_x) ≈ 0.0133 with P_x ≈ 1 after internal
normalization — the 8e-3 setting is uncomfortably close, so P_x fluctuations push the
loop transiently unstable, producing the visible MSE spikes in the page 16 plot.

### Issue F — User-side FOE precision  *(severity: low; user-side)*

User correction noted: M-th-power FOE doesn't help here. The user's PSD already shows a
DC-bias leakage tone; raising the signal to the M-th power just produces another tone
at M × the offset frequency, with the same bin-quantization. And on high-order QAM the
M-th-power signal-to-noise is poor anyway (the constellation symbols don't add
coherently in the M-th-power spectrum the way a clean tone does). So the real fix is
**not switching algorithms** — it's adding sub-bin interpolation around whatever PSD
peak the user is already locating.

The user's existing pipeline does `f[Pxx[0].argmax()]`, which is bin-quantized to
`fs / nperseg` ≈ 15 kHz at 4 GSps. The 700°-over-5000-symbol residual ramp in BPS is
consistent with ≈ 10 kHz of leftover FO and is what forces the CPR transient to be as
long as it is.

The minimal user-side fix is a 3-point parabolic (Quinn / Jacobsen) interpolation
around the peak bin in their own notebook code — five lines. Optionally we can ship a
small helper `peak_frequency_subbin(Pxx, f, bin_idx)` in `commstools/sync.py` so other
captures benefit. Both options are documentation-grade work; not on the critical path
for the SER cliff.

### Issue G — JAX backend: complex128 promotion crashes joint LMS+CPR  *(severity: high)*

User report: launching joint `lms(backend="jax", cpr_type=...)` raises a JAX precision
inconsistency error (complex64 vs complex128). Root cause is a Python complex literal
in two `jnp.exp(...)` calls inside the scan body:

- [equalization.py:1786](commstools/equalization.py#L1786) — `phasor = jnp.exp(-1j * phi_wrapped.astype(jnp.float32))`
- [equalization.py:1806](commstools/equalization.py#L1806) — `phasor_inv = jnp.exp(1j * phi_wrapped.astype(jnp.float32))`
- And the identical pair in `_get_jax_rls_cpr` at [2019](commstools/equalization.py#L2019)
  and [2039](commstools/equalization.py#L2039).

With `jax_enable_x64=True`, the Python `-1j` literal is `complex128`; multiplying it by
a `float32` array promotes to `complex128`, so `phasor` is `complex128`. The downstream
`y_fin = y_raw * phasor` promotes `y_raw` (complex64) → `complex128`, contaminating
`e_clean`, `e_eq`, and the einsum gradient. `W_new = W + step_size * einsum(...)` then
tries to add a `complex128` array into a `complex64` carry — `lax.scan` is strict about
carry dtype invariance, so the trace fails.

The Numba/CuPy paths have the same `-1j * float` shape (e.g.
[equalization.py:4799](commstools/equalization.py#L4799),
[4827](commstools/equalization.py#L4827)) but explicitly cast the phase to `complex64`
inside the multiply: `xp.exp(-1j * phi_c.astype(xp.complex64))`. NumPy is lenient
enough that the broadcast result is `complex128` but it gets implicitly downcast on
multiply with `y_block` (also `complex64`); JAX is not.

**Precise fix.** In all four sites, replace
```python
phasor = jnp.exp(-1j * phi_wrapped.astype(jnp.float32))
```
with
```python
_phi32 = phi_wrapped.astype(jnp.float32)
phasor = jnp.exp(_phi32 * jnp.array(-1j, dtype=jnp.complex64))
```
`phi_wrapped` is float64 (derived from float64 wrap arithmetic), so it must be
cast to float32 *before* the multiply: float32 × complex64 → complex64.
Without the cast, float64 × complex64 → complex128, re-introducing the promotion
inside XLA. Same pattern for `+1j` in the `phasor_inv` line.

**Defensive sweep.** Add a one-shot assertion at the top of `_get_jax_lms_cpr` /
`_get_jax_rls_cpr` (after `jax, jnp, _ = _get_jax()`):
```python
assert jnp.array(-1j * jnp.array(1.0, dtype=jnp.float32)).dtype == jnp.complex64, (
    "Python complex literal is promoting to complex128 — set jax_enable_x64=False "
    "or cast the literal explicitly."
)
```
Off by default; only ship if we hit a regression. Better: write the casts correctly
and document why.

**Tests.** Add `test_jax_lms_cpr_dtype_invariant`,
`test_jax_rls_cpr_dtype_invariant` to [tests/test_cpr_equalizer.py](tests/test_cpr_equalizer.py):
- Run `lms(backend="jax", cpr_type="bps")` and assert
  `result.y_hat.dtype == jnp.complex64` and `result.weights.dtype == jnp.complex64`.
- Run under both `jax.config.update("jax_enable_x64", True)` and
  `update("jax_enable_x64", False)` parametrizations.
- Match Numba output within `1e-3` complex-magnitude tolerance on a small fixed-seed
  capture to guard against any silent precision regression in either backend.

**Backwards compatibility.** Pure bug fix. Existing callers that "worked" only because
they had `jax_enable_x64=False` keep working; callers that crashed now succeed; output
shape and dtype are unchanged.

**Severity:** High — the JAX backend is currently unusable with CPR. Land independently
of Phases 2–4 below.

## Phase 1 — JAX `lms_cpr` / `rls_cpr` dtype fix  *(Issue G, can land first)*

Fix in [equalization.py](commstools/equalization.py) lines 1786, 1806, 2019, 2039 as
detailed in Issue G above. Independent of the warm-start work — purely unblocks the JAX
backend with CPR enabled. Tests as listed in Issue G.

## Phase 2 — `resolve_phase_ambiguity` adds `num_skip_symbols`  *(smallest landing)*

- Add `num_skip_symbols: int = 0` to the signature ([sync.py:4017](commstools/sync.py#L4017)).
- Inside the per-channel loop at [4076–4085](commstools/sync.py#L4076-L4085), score on
  the trimmed slices `rotated[num_skip_symbols:]` vs `ref[ch, num_skip_symbols:]`.
  **The applied rotation still covers the full input — only the scoring is trimmed.**
- Guard `num_skip_symbols >= N` with a `ValueError`.
- Default 0 reproduces current behaviour exactly.
- **Tests** in [tests/test_sync.py](tests/test_sync.py) (or `test_carrier_sync.py` —
  whichever already covers this function):
  - `test_resolve_phase_ambiguity_skip`: construct a sequence whose first 500 symbols are
    sign-flipped so the full-sequence best_k is wrong; verify `num_skip_symbols=500`
    recovers the correct k.
  - `test_resolve_phase_ambiguity_skip_zero_is_baseline`: byte-exact regression check.

## Phase 3 — Warm-start state plumbing: `CPRState` + `input_norm_factor`

Land Issues A and C together — the tests overlap and Issue C is required for the
warm-start pattern to be internally consistent.

**`CPRState` dataclass.** Add to [equalization.py](commstools/equalization.py) just
above `EqualizerResult` (~line 75). Frozen, all-`None`-by-default fields, all stored as
CPU NumPy arrays (document this — the rest of `EqualizerResult` can be CuPy but CPR
state is small and stays on CPU):
- `bps_prev4 (C,) float64`, `bps_offset4 (C,) float64`,
  `bps_d2_hist (P, C, K-1) float32` (or None when K==1)
- `pll_phi (C,) float64`, `pll_freq (C,) float64`
- `cs_buf_x (C, H) float64`, `cs_buf_y (C, H) float64`,
  `cs_buf_ptr (C,) int64`, `cs_buf_n (C,) int64`, `cs_stats (C, 4) float64`
- Identity tags: `cpr_type str`, `symmetry int`, `P int`, `K int`, `H int`

Add `cpr_state: Optional[CPRState] = None` field to
[`EqualizerResult` (line 145)](commstools/equalization.py#L145).

**Parameter additions.** Add to `lms()`, `rls()`, and `block_lms()`:
- `cpr_state: Optional[CPRState] = None`
- `input_norm_factor: Optional[Union[float, np.ndarray]] = None`

Add `input_norm_factor` (no `cpr_state`) to `cma()`, `rde()`, and `apply_taps()`.

**Wiring.**
- Modify [`_normalize_inputs` (line 2583)](commstools/equalization.py#L2583) to accept
  `input_norm_factor` and skip the RMS computation when supplied. Validate shape.
- `_normalize_inputs` is called at **10 sites** — all must be updated:
  - `lms` Numba (3358), `lms` JAX (3508)
  - `rls` Numba (3976), `rls` JAX (4148)
  - `block_lms` (4544)
  - `cma` Numba (5171), `cma` JAX (5256)
  - `rde` Numba (5552), `rde` JAX (5640)
  - `apply_taps` (5900)
- In `lms` CPR setup ([3444–3451](commstools/equalization.py#L3444-L3451)),
  `rls` CPR setup, and `block_lms` setup ([4604–4617](commstools/equalization.py#L4604-L4617)),
  replace the unconditional `np.zeros(...)` initializations with `if cpr_state is not None and
  shapes_match: copy from cpr_state else zeros`. Reject mismatched `(C, num_taps,
  cpr_type, P, K, H, symmetry)` with a clear `ValueError`.
- After the kernel returns, pack the in-place-mutated state arrays into a new
  `CPRState` and attach to `result.cpr_state` before exit.
- Default `None` for all new parameters reproduces current behaviour exactly.

**JAX backend.** If wiring JAX scan kernels' carry is too invasive for the same PR,
gate at the top with `if backend == "jax" and cpr_state is not None: raise
NotImplementedError(...)`. Don't ship silently-wrong JAX behaviour.

**Tests** in [tests/test_cpr_equalizer.py](tests/test_cpr_equalizer.py) and
[tests/test_block_lms.py](tests/test_block_lms.py) (or current closest match — verify
filenames first):
1. `test_cpr_state_warmstart_bps_lms` — split a Wiener-phase capture in two; second call
   uses `w_init=r1.weights` + `cpr_state=r1.cpr_state`. Assert first 200 symbols of
   second run have MSE within 1 dB of first run's steady state.
2. `test_cpr_state_warmstart_pll_lms` — same for PLL; check `phase_trajectory` continuity.
3. `test_cpr_state_warmstart_block_lms_bps` — same for `block_lms`.
4. `test_cpr_state_none_is_baseline` — byte-exact regression with `cpr_state=None`.
5. `test_cpr_state_shape_mismatch_raises` — wrong `(C, P, K, cpr_type)` raises.
6. `test_input_norm_factor_passthrough` — provide nf1, verify `result.input_norm_factor
   == nf1` and y_hat matches manual pre-scaled call.
7. `test_input_norm_factor_none_is_baseline` — byte-exact regression.

**Manual verification.** End-to-end pipeline: pass `cpr_state=res_prev.cpr_state` and
`input_norm_factor=res_prev.input_norm_factor` into the final `block_lms` call.
Expected: SER over symbols [0, 8192] drops by > 2 orders of magnitude; the
`discard=8192` and `discard=16384` SER values converge.

## Phase 4 — Zero-padding warmstart fix

After Phase 3 has landed (Phase 3 sets the canonical warm-start API which Phase 4
extends).

- Add `samples_prefix: Optional[ArrayType] = None` to `lms`, `block_lms`, `rls`, `cma`,
  `rde`, `apply_taps`. Shape `(C, ≥ pad_left)` MIMO, `(≥ pad_left,)` SISO where
  `pad_left = min(center_tap, pad_total)` is computed inside the function body; in raw
  pre-normalization units. Runtime validation: `assert samples_prefix.shape[-1] >= pad_left`
  after `pad_left` is known. `pad_left` is at most `num_taps - 1` but may be smaller when
  `center_tap < num_taps - 1`; the docstring should note this explicitly.
- When supplied: normalize the prefix by the same `input_norm_factor` selected for the
  main batch (this is why Phase 3 lands first), then replace the
  `np.pad(samples_np, (pad_left, pad_right))` call with
  `np.concatenate([prefix[..., -pad_left:], samples_np, np.zeros(..., pad_right)], axis=-1)`.
  Tail is still zero-padded.
- Add a secondary lever `pad_mode: str = "zeros"` accepting `{"zeros", "edge", "reflect"}`,
  used only when `samples_prefix` is None. Default `"zeros"` reproduces current behaviour
  exactly — **critical** because changing the default would alter the transient on
  every cold-start call.
- Touch padding hunks: `lms` [3345–3348, 3361–3365, 3511–3516](commstools/equalization.py#L3345),
  `block_lms` [4634–4640](commstools/equalization.py#L4634-L4640),
  `apply_taps` [5903–5907](commstools/equalization.py#L5903-L5907).

**Tests** in [tests/test_equalization.py](tests/test_equalization.py):
1. `test_samples_prefix_eliminates_warmstart_transient` — train on first half, warm-start
   on second with `samples_prefix=samples[:, N/2 - (num_taps-1):N/2]`. Assert
   `mean(|error[:num_taps]|²)` within 1 dB of steady-state.
2. `test_pad_mode_edge_equivalent_to_prefix_for_constant_signal`.
3. `test_pad_mode_zeros_is_baseline` — byte-exact regression.
4. `test_samples_prefix_shape_validation` — wrong shapes raise.
5. Parametrize at least one non-default `center_tap` (the prefix length must use
   `pad_left = min(center_tap, pad_total)`, not the simpler `num_taps - 1`).

## Phase 5 — Documentation

- [`lms()` docstring (3145–3153)](commstools/equalization.py#L3145-L3153): replace
  "1e-4 to 1e-2" with a `C · num_taps`-pivoted range — for SISO/16-QAM use 1e-3–5e-3;
  for 2×2 MIMO with `num_taps ≥ 50` use 5e-4–2e-3. State the bound
  `μ < 2 / (C · num_taps · P_x)` and remind that `P_x ≈ 1` after normalization.
- FOE precision note: do **not** rebrand M-th power as a recommended replacement for
  PSD-argmax on bias-leakage-tone captures (user pushback: a tone is a tone — raising
  to the M-th power just creates another tone at M× the offset with the same bin
  resolution, and on high-order QAM the in-band M-th-power signal is noisy). Instead
  add either (a) a short user-side recipe in `CLAUDE.md` showing 3-point parabolic
  peak interpolation, or (b) a small helper `peak_frequency_subbin(Pxx, f, bin_idx)`
  in `commstools/sync.py`. Pick (b) only if a second internal user asks for it.
- [CLAUDE.md](CLAUDE.md): add a short "Iterative warm-start training" recipe showing
  the canonical sequence threading `w_init`, `cpr_state`, `input_norm_factor`,
  `samples_prefix`; and the `resolve_phase_ambiguity(num_skip_symbols=...)` pattern.

## End-to-end verification

After Phases 1–4:
- JAX backend with `cpr_type="bps"` or `"pll"` runs without dtype errors and produces
  output within tight tolerance of the Numba reference.
- The user's pipeline with `num_train_symbols=8192` (no doubling) reports SER ≤ 1e-3.
- No MSE jump at the start of warm-started iterations.
- `resolve_phase_ambiguity` returns a deterministic `best_k` across captures.
- Every `EqualizerResult` carries `cpr_state` and `input_norm_factor` for the next call.

**Regression guardrails:**
```
uv run pytest tests/test_cpr_equalizer.py tests/test_block_lms.py \
              tests/test_equalization.py tests/test_carrier_sync.py \
              tests/test_sync.py -q
```
All `*_is_baseline` tests must pass byte-exactly — these protect the
homodyne-loopback path.

## Risks

- **CPRState on GPU.** State arrays stay on CPU intentionally; document on the
  dataclass to prevent confused `cp.asarray(res.cpr_state)` calls.
- **JAX asymmetry.** If JAX `cpr_state` is deferred, the `NotImplementedError` gate
  must land in the same PR as the Numba support — silent wrong-result is unacceptable.
- **Numba dispatcher cache.** Adding ndarray args to the CPR kernels invalidates the
  cache; smoke-run the compilation locally before pushing.
- **`samples_prefix` with non-default `center_tap`.** Prefix length must use the
  computed `pad_left`, not `num_taps - 1`. Parametrized test required.
- **Don't change cold-start padding default.** `pad_mode="zeros"` must remain the
  default so cold-start gradient trajectories are byte-exact.
- **MIMO/SISO norm-factor coercion.** Internally returns `float` for SISO, `(C,)` for
  MIMO. Validator must accept both; a SISO call receiving a `(C,)` factor must raise.
- **`num_skip_symbols >= N`.** Guard with `ValueError` (otherwise the trimmed SER mean
  is over an empty slice → `nan` → `best_k = 0` silently).
- **Issue C is *required* once Issue B lands.** Without consistent
  `input_norm_factor` pass-through, the `samples_prefix` is normalized by the new
  call's RMS — slightly mis-aligned with the warm-started weights' tap-plane scale.
  Land C in Phase 3, B in Phase 4.

## Critical files

- [commstools/equalization.py](commstools/equalization.py) — primary
- [commstools/sync.py](commstools/sync.py)
- [tests/test_cpr_equalizer.py](tests/test_cpr_equalizer.py)
- [tests/test_block_lms.py](tests/test_block_lms.py)
- [tests/test_equalization.py](tests/test_equalization.py)
- [tests/test_sync.py](tests/test_sync.py) or [tests/test_carrier_sync.py](tests/test_carrier_sync.py)

## Delivery

The user asked for the final plan at the project root. On approval I will save this
plan to `/home/lokgar/commstools/DSP_DEBUG_PLAN.md` (one Write call) before any code
changes begin.
