# JAX CPR Warm-Start Implementation Plan

## Context

The Numba path in `lms()` and `rls()` already supports CPR state warm-start via
`CPRState` (merged in Phase 3 of DSP_DEBUG_PLAN.md).  The JAX path
(`_get_jax_lms_cpr`, `_get_jax_rls_cpr`) currently raises `NotImplementedError`
when `cpr_state is not None`.

The JAX path uses `jax.lax.scan` with a structured carry dict.  Threading CPR
state across block boundaries requires injecting the initial carry values from
`CPRState` rather than zero-initializing them.

---

## Carry structure (current)

Inside `_get_jax_lms_cpr` the scan carry is a dict with at minimum:

| Key | dtype | shape | Description |
|-----|-------|-------|-------------|
| `W` | complex64 | `(C, C, T)` | tap weights |
| `pll_phi` | float32 | `(C,)` | PLL phase accumulator |
| `pll_freq` | float64 | `(C,)` | PLL frequency integrator |
| `bps_prev4` | float32 | `(C,)` | BPS 4-fold unwrap previous argmin |
| `bps_offset4` | float32 | `(C,)` | BPS 4-fold unwrap offset |
| `bps_d2_buf` | float32 | `(B, C, K-1)` | BPS rolling distance-history buffer |
| `cs_buf_x` | float64 | `(C, H)` | Cycle-slip regression x-buffer |
| `cs_buf_y` | float64 | `(C, H)` | Cycle-slip regression y-buffer |
| `cs_buf_ptr` | int32 | `(C,)` | Cycle-slip buffer write pointer |
| `cs_buf_n` | int32 | `(C,)` | Cycle-slip buffer fill count |
| `cs_stats` | float64 | `(C, 4)` | Cycle-slip regression accumulators |

**JAX constraint:** `lax.scan` enforces carry dtype invariance — the input and
output carry must have identical dtypes at every step.  Any warm-start
initialization must therefore match the traced carry dtype exactly.

**Critical dtype note (Phase 1 fix):** `pll_phi` in the scan is kept as **float64**
(the dual-path design).  Only the rotation phasor is cast to float32:
```python
_phi32 = pll_phi.astype(jnp.float32)
phasor = jnp.exp(_phi32 * jnp.array(-1j, dtype=jnp.complex64))
```
The warm-start value for `pll_phi` must therefore be float64 (same as `CPRState.pll_phi`
which is stored as float64 by the Numba path).

---

## Implementation plan

### Step 1 — Build initial carry from CPRState

In `_get_jax_lms_cpr` (and `_get_jax_rls_cpr`), after the existing
`w_init` warm-start, add:

```python
def _cpr_state_to_jax_carry(st: Optional[CPRState], num_ch, bps_B, bps_K, cs_H):
    """Convert a CPRState snapshot to JAX-typed initial carry values."""
    import jax.numpy as jnp

    def _or_zeros(arr, shape, dtype):
        if arr is not None:
            return jnp.array(arr, dtype=dtype)
        return jnp.zeros(shape, dtype=dtype)

    pll_phi   = _or_zeros(st.pll_phi if st else None,   (num_ch,), jnp.float64)
    pll_freq  = _or_zeros(st.pll_freq if st else None,  (num_ch,), jnp.float64)
    bps_prev4 = _or_zeros(st.bps_prev4 if st else None, (num_ch,), jnp.float32)
    bps_off4  = _or_zeros(st.bps_offset4 if st else None,(num_ch,),jnp.float32)
    bps_d2    = _or_zeros(
        st.bps_d2_hist if st else None,
        (bps_B, num_ch, bps_K - 1), jnp.float32
    )
    cs_bx = _or_zeros(st.cs_buf_x if st else None,   (num_ch, cs_H), jnp.float64)
    cs_by = _or_zeros(st.cs_buf_y if st else None,   (num_ch, cs_H), jnp.float64)
    cs_ptr= _or_zeros(st.cs_buf_ptr if st else None, (num_ch,),       jnp.int32)
    cs_n  = _or_zeros(st.cs_buf_n if st else None,   (num_ch,),       jnp.int32)
    cs_st = _or_zeros(st.cs_stats if st else None,   (num_ch, 4),     jnp.float64)

    return pll_phi, pll_freq, bps_prev4, bps_off4, bps_d2, cs_bx, cs_by, cs_ptr, cs_n, cs_st
```

Merge these into the existing carry dict before calling `lax.scan`.

### Step 2 — Extract final carry and build output CPRState

After `lax.scan` returns the final carry, extract and store the CPR fields:

```python
final_carry = scan_result[1]   # the last carry from lax.scan
cpr_state_out = CPRState(
    pll_phi     = np.asarray(final_carry["pll_phi"]),
    pll_freq    = np.asarray(final_carry["pll_freq"]),
    bps_prev4   = np.asarray(final_carry["bps_prev4"]),
    bps_offset4 = np.asarray(final_carry["bps_off4"]),
    bps_d2_hist = np.asarray(final_carry["bps_d2"]),
    cs_buf_x    = np.asarray(final_carry["cs_buf_x"]),
    cs_buf_y    = np.asarray(final_carry["cs_buf_y"]),
    cs_buf_ptr  = np.asarray(final_carry["cs_buf_ptr"]),
    cs_buf_n    = np.asarray(final_carry["cs_buf_n"]),
    cs_stats    = np.asarray(final_carry["cs_stats"]),
    cpr_type    = cpr_type,
    num_ch      = num_ch,
    symmetry    = 4,
    bps_P       = bps_B,
    bps_K       = bps_K,
    cs_H        = cs_H,
)
result.cpr_state = cpr_state_out
```

### Step 3 — jax.config.update for float64

The `pll_phi` / `pll_freq` fields are float64.  Ensure `jax.config.update("jax_enable_x64", True)` is called before the scan in the CPR path (it already is in the current JAX RLS path, but verify it is also done for JAX LMS CPR).

### Step 4 — Phase accumulator precision contract

The dual-path design must be preserved:

- **Accumulator update** (`pll_phi += mu * e_ph + pll_freq`): runs at float64.
  JAX standard promotion: `float64 + float32 → float64` ✓
- **Phasor computation** (rotation): `_phi32 = pll_phi.astype(float32)` then
  `exp(_phi32 * jnp.array(-1j, dtype=complex64))` → complex64 ✓

Do not change these promotions; they are the Phase 1 fix.

---

## Tracing constraint: static BPS buffer shape

`bps_d2_buf` is a rolling circular buffer of shape `(B, C, K-1)`.  For
`lax.scan`, all carry shapes must be static (known at trace time).  `bps_B`
(`cpr_bps_test_phases`) and `bps_K` (`cpr_bps_block_size`) are already
runtime constants captured in the closure — no change needed.

---

## Test plan

1. **JAX LMS + PLL warm-start parity** — run two consecutive blocks on CPU with
   JAX backend; verify that the initial MSE on the second block with warm `cpr_state`
   is ≤ cold restart + 3 dB (same assertion as `test_cpr_state_warmstart_lms`).

2. **JAX LMS + BPS warm-start parity** — same as above with `cpr_type='bps'`.

3. **JAX RLS + PLL warm-start parity** — same pattern for `rls()` JAX backend.

4. **Carry dtype invariance** — assert that `final_carry["pll_phi"].dtype == jnp.float64`
   and `final_carry["pll_phi"].shape == (num_ch,)` after scan to catch any
   unintended promotion.

5. **Cold-start equivalence** — `cpr_state=None` on JAX must produce byte-exact
   output identical to omitting the parameter (regression against current behavior).

6. **Remove `pytest.skip` guards** — the three tests currently skipping on GPU
   (`test_cpr_state_warmstart_lms[gpu-*]`, `test_cpr_state_none_is_baseline_lms[gpu]`,
   `test_cpr_state_warmstart_rls[gpu]`) and the analogous block_lms tests should
   pass after this implementation without the skip guards.

---

## Files to modify

| File | Change |
|------|--------|
| `commstools/equalization.py` | Add `_cpr_state_to_jax_carry()` helper; update `_get_jax_lms_cpr` and `_get_jax_rls_cpr` to inject/extract carry; remove `NotImplementedError` guards |
| `tests/test_cpr_equalizer.py` | Remove `pytest.skip("... JAX/GPU path")` guards from Tests 12 and 13 |
| `tests/test_block_lms.py` | Remove `pytest.skip` guards from tests 12 and 13 |
| `tests/test_equalization.py` | (No change — `TestPrefixPadNormPhase4` already backend-agnostic) |

---

## Prerequisites

- Phase 1 dtype fix must remain intact (float64 accumulator + float32 phasor).
- `jax_enable_x64` must be set before JAX CPR tracing — already done for RLS, verify for LMS.
- Do not implement in the same session as this plan was written — see DSP_DEBUG_PLAN.md Phase notes.
