# Block-wise FOE Tracking and Joint Equalization & CPR

This document details the architectural plan to add drifting Carrier Frequency Offset (FOE) compensation and intrinsic Carrier Phase Recovery (CPR) to the adaptive equalizers (LMS/RLS), encompassing both CPU (Numba) and GPU (JAX) backends.

## Proposed Changes

---

### 1. Block-wise FOE Pre-compensation

**Target File:** `commstools/sync.py`

A new top-level estimator will be added to handle drifting frequency. It uses a sliding-window approach over the waveform.

#### [NEW] `estimate_frequency_offset_blockwise`

**Arguments:**
`samples`, `sampling_rate`, `block_size=4096`, `overlap=0.5`, `method="mth_power"`,
`sps=2`, `modulation=None`, `order=None`.

> **Correction:** `sampling_rate` (Hz) is a mandatory argument. Both underlying
> estimators (`estimate_frequency_offset_mth_power` and
> `estimate_frequency_offset_mengali_morelli`) require it. Omitting it from the
> signature was a critical oversight.

**Logic Pipeline:**

1. Slice the input signal into overlapping frames based on `block_size` and
   `overlap`. Block centers are placed at strides of `step = round(block_size *
   (1 - overlap))` samples. Overlap should be kept to ≥ 0.5 to avoid
   under-sampling fast frequency drifts.
2. Invoke the requested string-mapped algorithm on each frame:
   - `"mth_power"` → `estimate_frequency_offset_mth_power`
   - `"mengali_morelli"` → `estimate_frequency_offset_mengali_morelli`
3. Form an array of per-block scalar frequency estimates `Δf[k]` (in Hz) with
   associated block-center time indices `t_k` (in samples).
4. Interpolate `Δf[k]` to a dense per-sample grid using
   `scipy.interpolate.interp1d(..., kind='cubic', fill_value='extrapolate')`.
5. Integrate the dense frequency trajectory to obtain the phase trajectory:
   `θ(n) = 2π / fs * cumsum(Δf_dense(n))`.

**Return value — phase trajectory `(N,) float64`:**
The output is a per-sample phase array in radians, consumed by the already-existing
`correct_carrier_phase(samples, phase_vector)`.

The intended pipeline:

```python
theta = estimate_frequency_offset_blockwise(samples, fs, ...)
corrected = correct_carrier_phase(samples, theta)
```

> **Correction — do not extend `correct_frequency_offset`:** The original plan
> stated the output was "ready to be passed into `correct_frequency_offset`".
> This is wrong, and extending that function is the wrong fix. `correct_frequency_offset`
> takes `offset` in **Hz** and distinguishes a per-channel `(C,)` offset from a
> scalar by shape — a `(N,)` phase vector is ambiguous when `C == N`, and
> `sampling_rate` would silently become meaningless. The correct downstream
> function is `correct_carrier_phase`, which already accepts a per-sample
> phase vector and is the standard consumer for BPS/PLL/VV phase estimates.

---

### 2. Joint LMS/RLS + CPR

**Target File:** `commstools/equalization.py`

We will integrate strictly causal phase trackers directly inside the LMS and RLS update loops.

#### [MODIFY] `lms` and `rls` API

The main equalizer functions will accept new arguments consistent with the standalone trackers in `sync.py`:

```python
    cpr_type: Optional[str] = None,               # "pll", "bps", or None
    cpr_pll_bandwidth: float = 1e-3,              # Normalized loop bandwidth (used to derive mu, beta)
    cpr_bps_test_phases: int = 64,                # Number of test angles for BPS
    cpr_cycle_slip_correction: bool = True,       # Enables causal slip tracking
    cpr_cycle_slip_history: int = 1000,           # Extrapolation history length (symbols)
    cpr_cycle_slip_threshold: float = np.pi/4,    # Slip detection threshold (radians)
```

> **Correction — removed `cpr_pll_mu` and `cpr_pll_beta`:** The original plan
> proposed three PLL parameters: `cpr_pll_bandwidth`, `cpr_pll_mu`, and
> `cpr_pll_beta`. This is redundant and inconsistent with the existing
> `recover_carrier_phase_pll` API which converts a single
> `loop_bandwidth_normalized` into `mu` and `beta` internally. A single
> `cpr_pll_bandwidth` is sufficient; `mu` and `beta` are derived from it
> identically to the standalone function.

**Parameter Parsing / Initialization:**

- For `cpr_type="pll"`, convert `cpr_pll_bandwidth` into discrete `K_p`, `K_i`
  coefficients identically to the standalone DPLL solver.
- For `cpr_type="bps"`, pre-compute the complex unit vectors for the
  `cpr_bps_test_phases` search space over `[0, π/2)` (QAM 4-fold symmetry).
- The constellation's rotational symmetry order (`symmetry = 4` for QAM, `= M`
  for M-PSK) is derived from the existing `modulation`/`order` arguments and
  used to set the cycle-slip quantum `2π/symmetry`.

**Return value:**
Extend the existing `(y_hat, errors, W_final, w_hist)` tuple to
`(y_hat, errors, W_final, w_hist, phase_trajectory)` when `cpr_type is not
None`. `phase_trajectory` is shape `(N_sym, C) float32`, the per-symbol phase
estimates used for de-rotation.

---

#### [MODIFY] Numba Compiler Kernels (`_get_numba_lms`, `_get_numba_rls`)

The high-speed `@njit` kernels will ingest CPR state variables. To avoid
proliferating kernel variants, a **unified kernel** approach is used: a single
kernel factory accepts an integer `cpr_mode` (0=none, 1=pll, 2=bps) and the
associated state arrays. Numba compiles conditional branches on integer
arguments efficiently.

**New arguments to the kernel:**

```text
cpr_mode        : int32     — 0=none, 1=pll, 2=bps
pll_mu          : float32   — proportional gain K_p
pll_beta        : float32   — integral gain K_i
bps_phases      : (B,)      complex64  — unit vectors exp(j·θ_k) for k=0..B-1
constellation   : (M,)      complex64  — already present; also used by BPS
cs_enabled      : bool
cs_threshold    : float32
cs_history_len  : int32
phase_out       : (N_sym, C) float32  — pre-allocated output for phase trajectory
```

**Per-Symbol Kernel Logic Execution Flow:**

1. **Forward Filter:** Compute the raw filtered output: `y_raw = W^H X`.

2. **Phase Estimation:**
   - **PLL (`cpr_mode=1`):** Set `φ_hat` from the current integrator state
     (stored per-channel in a small carry array).
   - **BPS (`cpr_mode=2`):** Rotate `y_raw` by each of the `B` pre-computed
     test phases. Select `φ_hat = argmin_k |y_raw · exp(-j·θ_k) - nearest_constellation_point|²`.
     This is a **single-symbol (zero-latency) causal BPS**, as a windowed BPS
     would introduce non-causal data access incompatible with the online loop.
     > **Note:** Single-symbol BPS is noisier than block-averaged BPS. For
     > high-order QAM (≥ 64-QAM) consider using PLL mode instead, which
     > benefits from the integral term and is less sensitive to per-symbol noise.

3. **Causal Cycle Slip Correction (Intrinsic Unwrapper):**
   - Maintain a per-channel `phase_history_buffer` as a circular buffer of
     `cs_history_len` entries.
   - Linearly extrapolate the expected phase from the buffer using O(1)
     incremental Welford-style sufficient statistics (identical algorithm to
     the existing `_cycle_slip_loop` in `sync.py`, adapted for per-symbol
     granularity).
   - If `|φ_hat - φ_expected| > cs_threshold`, snap `φ_hat` to the nearest
     valid quantum multiple of `2π/symmetry` that aligns with the historical
     slope.
   - Append the corrected `φ_corrected` to the circular buffer.

4. **Output Rotation:** `y_final = y_raw · exp(j · φ_corrected)`.

5. **Slicer:** Hard decision `d` from the rotated symbol; compute clean error
   `e_clean = d - y_final`.

6. **PLL State Update (PLL mode only):**
   Update the discrete phase and frequency integrators using the
   **cross-product phase detector**:

   ```text
   e_phase = Im(y_final · conj(d))
   ```

   > **Correction:** The original plan said "using the angle of the clean
   > error `e_clean`". This is wrong. `angle(d - y_final)` is not a standard
   > phase detector output and has poor behaviour for large errors. The correct
   > expression is `Im(y_final · conj(d))` — the cross-product detector used
   > in the existing `_dd_pll_loop` — which equals `|y||d| sin(Δφ)` and is
   > linear for small phase errors.

7. **De-rotate Error (weight update domain):**

   ```text
   e_eq = e_clean · exp(-j · φ_corrected)
   ```

   This maps the error back to the pre-rotation domain. Derivation via
   Wirtinger calculus: minimising `|e_clean|² = |d - exp(jφ) W^H X|²` w.r.t.
   `W*` gives gradient `-exp(jφ) X conj(e_clean)`, which is equivalent to the
   standard update with `e_eq = e_clean · exp(-jφ)`.
   *(The plan's original derivation was correct here — confirmed.)*

8. **Weight Update:**

   ```text
   W[i,j,t] += μ · conj(e_eq[i]) · X_wins[j,t]   (LMS)
   ```

   or the corresponding RLS Riccati update with `e_eq` in place of `e`.

**Kernel Cache Key:**
The `_NUMBA_KERNELS` dict uses string keys. New keys: `"lms_cpr"`, `"rls_cpr"`
for the CPR-enabled variants to avoid polluting the existing baseline kernels.

---

#### [MODIFY] JAX Compiler Kernels (`_get_jax_lms`, `_get_jax_rls`)

Functionality is equivalently written inside the XLA functional loops.

**`lax.scan` carry tuple** expanded from `(W)` / `(W, P)` to:

```text
(W, pll_phase, pll_freq, phase_hist_buf, phase_hist_ptr)     # LMS
(W, P, pll_phase, pll_freq, phase_hist_buf, phase_hist_ptr)  # RLS
```

where:

- `pll_phase` : `(C,) float32` — per-channel integrator state
- `pll_freq` : `(C,) float32` — per-channel frequency integrator
- `phase_hist_buf` : `(C, cs_history_len) float32` — circular buffer of past
  phase corrections (fixed static shape; `cs_history_len` is a closure var)
- `phase_hist_ptr` : `(C,) int32` — circular write pointer per channel

**Branching strategy — static Python if/else at trace time:**

> **Correction:** The original plan proposed `jax.lax.switch` for CPR mode
> selection. This is architecturally wrong. `cpr_type` is a **static Python
> string** (a closure variable captured at `@jax.jit` trace time). XLA sees
> only one branch — the dead branches are pruned at compile time. Plain Python
> `if/else` inside the `step` function body achieves zero-overhead branching
> at zero runtime cost and avoids the carry-shape complexity of `lax.switch`.
> `jax.lax.switch` is only necessary when the branch selector is a
> **dynamic runtime integer**, which is not the case here.

**BPS implementation in JAX:**
Pre-compute test-phase vectors `bps_phases : (B,) complex64` as a closure
variable. Inside `step`:

```python
rotated = jnp.conj(bps_phases) * y_raw  # (B, C) broadcast
# min-dist to constellation for each candidate
dists = jnp.min(jnp.abs(rotated[:, :, None] - const[None, None, :]) ** 2, axis=-1)  # (B, C)
best_k = jnp.argmin(dists.sum(axis=-1) if joint_channels else dists, axis=0)  # (C,)
phi_hat = bps_phases_angle[best_k]  # (C,)
```

**JAX kernel cache key:**
Extend `_JITTED_EQ` key from `("lms", num_taps, stride, const_size, num_ch)` to
`("lms", num_taps, stride, const_size, num_ch, cpr_type, cpr_bps_test_phases, cs_history_len)`.
This ensures a new XLA compilation is triggered for each distinct CPR
configuration.

---

## Verification Plan

### Automated/Mathematical Verification

1. **Zero-Deviation Baseline:** When `cpr_type=None`, verify bit-exact
   (machine precision) output equality against the unmodified algorithm for
   both LMS and RLS, SISO and MIMO, Numba and JAX backends.

2. **Numba / JAX Backend Parity:** With `cpr_type="pll"` and `cpr_type="bps"`,
   verify that the Numba and JAX kernels produce identical `y_hat` outputs
   (within `float32` round-off, `atol=1e-5`) for the same input and initial
   state.

3. **Cycle Slip Stress Test:** Inject a step-wise phase sequence with
   deliberate `π/2` jumps into a DD LMS block simulation; verify the
   equalizer filters maintain continuous convergence through each artificial
   slip and that the output `phase_trajectory` correctly tracks the jumps.

4. **PLL Convergence / Phase Noise:** Drive LMS + `cpr_type="pll"` through a
   channel with additive Wiener-process phase noise of known linewidth
   `Δν · T_s`. Confirm the per-symbol phase RMSE is within the theoretical
   PLL tracking jitter `σ_φ² ≈ B_L / (Δν · T_s)` bound.

5. **Block-wise Phase Coherence:** Test `estimate_frequency_offset_blockwise`
   on a signal with an induced frequency chirp (quadratic phase), ensuring the
   output phase trajectory effectively linearises the varying frequency and
   that post-correction EVM is within 0.5 dB of the no-chirp reference.

6. **MIMO Coverage:** Run a 2×2 butterfly equalizer with `cpr_type="pll"` and
   confirm all four sub-filter taps converge independently and that both output
   channel phase trajectories are tracked correctly.
