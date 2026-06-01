# Plan: Unify the inline equalizer PLL and standalone DD-PLL parameterization

> **Goal:** both entry points accept **both** parameterizations (raw `mu`/`beta` **and** a
> `bandwidth` shortcut) through **one shared gain resolver**, so the bandwidth→gain mapping
> is defined exactly once and cannot drift apart. Part A adds raw gains to the inline PLL;
> Part C adds the bandwidth shortcut to the standalone PI. Defaults on both sides are
> preserved (capability becomes symmetric, behavior does not change).

## Context

The decision-directed PLL exists in two places with **inconsistent control surfaces**:

- **Standalone** [`recover_carrier_phase_pll`](commstools/recovery.py#L497): exposes raw
  `mu`, `beta` (1st-order when `beta=0`, 2nd-order when `beta>0`), **and** a
  `loop_filter ∈ {"pi","butterworth"}` choice with `loop_bandwidth_normalized` for the
  Butterworth IIR.
- **Inline** (`lms`/`rls`, `cpr_type='pll'`): exposes **only** `cpr_pll_bandwidth`
  ([equalization.py:3703](commstools/equalization.py#L3703)). The PI gains are derived
  internally by [`cpr_pll_gains`](commstools/equalization.py#L3613) as `μ=4B, β=4B²`
  (fixed damping; the formula gives **ζ=1**, not the ζ=1/√2 the docstring claims).

Consequences of the inline limitation:
- No **1st-order** loop (β is always `4B²>0`) — the natural choice for a
  frequency-stable / high-phase-noise laser (e.g. LXM).
- No **independent frequency-tracking vs phase-bandwidth** — to track faster drift you
  must widen `μ` too (more phase noise). Bad for the drift-dominated case (IDP).
- No **damping** control (locked to the critical-damping curve).
- No Butterworth loop filter.

Both loops use the **identical** update and detector — `e = Im(y·d*) = yi·d_r − yr·d_i`,
`φ += μe + ν`, `ν += βe` — so `(mu, beta)` are **directly portable** between them
(assuming the same ~unit-power symbol scaling, which the equalizer enforces via
`_normalize_inputs`). That makes the PI half of this a clean, low-risk change.

**Recommendation:** implement **Part A** (independent `mu`/`beta` inline) now; **defer
Part B** (inline Butterworth) — analysis below shows the cost/benefit doesn't justify it.

---

## Part A — Independent `mu`/`beta` on the inline PLL  ✅ recommended

### Why it's cheap
All four inline kernels **already receive `pll_mu`/`pll_beta` as scalar arguments** and
implement the PI update verbatim:

- numba `lms_cpr` — update at [equalization.py:936-959](commstools/equalization.py#L936)
- numba `rls_cpr` — update at [equalization.py:1289-1310](commstools/equalization.py#L1289)
- jax `lms_cpr` — update at [equalization.py:2232-2233](commstools/equalization.py#L2232)
- jax `rls_cpr` — update at [equalization.py:2578-2579](commstools/equalization.py#L2578)

The gains are computed at the **four call sites** that wrap these kernels:
[4163](commstools/equalization.py#L4163), [4397](commstools/equalization.py#L4397)
(`lms` numba/jax) and [4938](commstools/equalization.py#L4938),
[5199](commstools/equalization.py#L5199) (`rls` numba/jax), each as:

```python
pll_mu, pll_beta = cpr_pll_gains(cpr_pll_bandwidth)
```

**So the kernels do not change at all.** We only change how `pll_mu`/`pll_beta` are
produced at those four sites.

### Changes

1. **Add two optional params** to `lms` and `rls` signatures (default `None`):
   ```python
   cpr_pll_mu: Optional[float] = None,
   cpr_pll_beta: Optional[float] = None,
   ```
   Keep `cpr_pll_bandwidth` for backward compatibility.

2. **Add a single SHARED resolver** (used by *both* `lms`/`rls` **and** the standalone
   `recover_carrier_phase_pll` — see Part C). Unambiguous precedence with `mu is None`
   as the bandwidth trigger:
   ```python
   def resolve_pll_gains(bandwidth, mu, beta):
       """mu given → raw PI gains (beta defaults 0.0 → 1st-order).
       mu is None → derive critically-damped (mu=4B, beta=4B²) from bandwidth."""
       if mu is not None:
           return float(mu), float(beta if beta is not None else 0.0)
       if beta is not None:                      # beta without mu is ambiguous
           raise ValueError("beta requires mu to be set (or use bandwidth).")
       return cpr_pll_gains(bandwidth)          # the single 4B/4B² mapping
   ```

   **Placement:** must be importable by both `equalization.py` and `recovery.py` without a
   circular import — verify the import direction first; if there's any risk, host
   `cpr_pll_gains` + `resolve_pll_gains` in a small neutral helper rather than
   duplicating the `4B/4B²` formula (duplication is exactly what would let the two drift
   back apart).

3. **Swap the four call sites** to use `resolve_pll_gains(cpr_pll_bandwidth,
   cpr_pll_mu, cpr_pll_beta)`. Default `cpr_pll_mu=None` ⇒ bandwidth path ⇒ **inline
   output unchanged**.

4. **Docstrings** (`lms`, `rls`): document `cpr_pll_mu`/`cpr_pll_beta`, the precedence
   rule, that `beta=0` ⇒ 1st-order, the `(μ,β)↔(B_L,ζ)` mapping
   (`ωₙT=√β`, `ζ=μ/(2√β)`), and that the values are interchangeable with
   `recover_carrier_phase_pll`'s `mu`/`beta`. Fix the stale "ζ=1/√2" note on
   `cpr_pll_gains` (actual ζ=1).

### State / warm-start
No change. `CPRState` already carries `pll_phi`, `pll_freq`
([equalization.py:89-90](commstools/equalization.py#L89)); gains are not state.

### Precision
`pll_phi`/`pll_freq` are already `float64` in the kernels. Keep gains as `float32` to
match the existing path (the standalone uses `float64`; the difference is immaterial at
these gain magnitudes, but if exact parity with the standalone is wanted, promote the
inline gains and `e_ph` accumulation to `float64` — optional, low priority).

### Tests
- **Parity:** inline `lms(cpr_type='pll', cpr_pll_mu=m, cpr_pll_beta=b, step_size=0,
  w_init=identity)` on a phase-rotated tone vs `recover_carrier_phase_pll(mu=m, beta=b)`
  → phase trajectories match to tight tolerance (confirms identical loop).
- **Backward compat:** `cpr_pll_bandwidth=1e-3` with no mu/beta reproduces current output
  bit-for-bit.
- **1st-order:** `cpr_pll_beta=0` tracks a phase step but leaves a constant error under a
  frequency offset (and a 2nd-order loop nulls it) — validates loop order behaviour.
- numba and jax backends agree.

**Effort:** small (1 helper + 2 signatures + 4 one-line call-site edits + docs/tests).
**Risk:** low (kernels untouched, default path unchanged).

---

## Part B — Inline Butterworth loop filter  ⚠️ analyze → recommend defer

### What it would require
The Butterworth loop is a 2nd-order IIR biquad on the phase error (Direct-Form-II
transposed), per [`_dd_pll_bw_loop`](commstools/recovery.py#L168), with coefficients from
`scipy.signal.butter(2, 2·lbw)` ([recovery.py:702-711](commstools/recovery.py#L702)) and
**two extra state variables `w1, w2`** replacing the single `freq` integrator. Porting it
inline means, for **each of the four kernels**:

1. A `loop_filter`/`cpr_pll_mode` flag to branch PI vs biquad in the hot loop.
2. Pass `b0,b1,b2,a1,a2` into every kernel.
3. Add `w1,w2` biquad state per channel (numba: extra arrays; **jax: extra `lax.scan`
   carry entries** in both `lms_cpr` and `rls_cpr`).
4. Extend **`CPRState`** with `pll_bw_w1`, `pll_bw_w2` and update
   `_cpr_state_to_jax_inits` / serialization
   ([equalization.py:1607](commstools/equalization.py#L1607)) so warm-start across
   blocks/iterations stays continuous.
5. SciPy coefficient design at the wrapper (one-time, cheap).

That's **4 kernels + CPRState + jax scan carries + warm-start plumbing** — the
high-surface-area, high-risk part of the codebase (the same kernels that carry BPS +
cycle-slip state).

### Benefit
Marginal in this setting. The Butterworth filter's advantage is a flatter passband /
sharper roll-off vs the PI's response. But a **DD-PLL's** performance is dominated by the
**phase-detector nonlinearity and cycle slips**, not the loop-filter shape; a
well-tuned 2nd-order PI is within a fraction of a dB of the Butterworth for realistic
phase-noise + frequency-drift mixes. The cases where the filter shape matters (very narrow
loop, aggressive roll-off requirements) are better served by the **decoupled** path, where
the standalone `recover_carrier_phase_pll(loop_filter='butterworth')` **already exists**.

### Recommendation
**Do not implement inline Butterworth now.** Instead:
- Ship Part A (covers the real gaps: 1st-order, independent μ/β, damping).
- Document the deliberate asymmetry: *"Inline PLL = PI only (1st/2nd order via
  `cpr_pll_mu`/`cpr_pll_beta`); for a Butterworth loop filter, use the decoupled
  `recover_carrier_phase_pll`."*
- Revisit only if a measured case shows PI-inline leaving ≥ ~0.5 dB on the table that
  Butterworth-inline recovers. If so, implement behind a `cpr_pll_loop_filter='pi'`
  default flag, kernels first (numba), then jax + CPRState.

---

## Part C — Bandwidth shortcut on the standalone PI loop  ✅ recommended (pairs with A)

The symmetric half: let `recover_carrier_phase_pll`'s **PI** path also accept a bandwidth
shortcut, via the **same `resolve_pll_gains`** helper from Part A. This gives
`loop_bandwidth_normalized` a coherent meaning for `loop_filter='pi'` (critically-damped
gains), not just for Butterworth.

### Changes (standalone PI)

1. **Change `mu`/`beta` defaults to `None`** in the signature (the bandwidth trigger).
   Resolve at the top of the function:

   ```python
   if loop_filter == "pi":
       mu, beta = resolve_pll_gains(loop_bandwidth_normalized, mu, beta)
   ```

2. **Preserve current default behavior:** the historical default was `mu=1e-2, beta=0`
   (1st-order). To keep that exactly, either (a) keep `mu=1e-2` as the default and treat
   `mu=None` as the explicit opt-in to the bandwidth path, **or** (b) move the `1e-2/0`
   default *inside* the resolver's "both None" branch. Option (a) is the least surprising
   and is what Part A already assumes. Decide once and apply to both functions.
3. **Validation / precedence** mirroring the existing Butterworth guard
   ([recovery.py:646-649](commstools/recovery.py#L646)): if the user passes both explicit
   `mu` and a non-default bandwidth for `loop_filter='pi'`, warn that `mu`/`beta` win;
   `loop_filter='butterworth'` continues to ignore `mu`/`beta` and use bandwidth.
4. **Docstring:** state that for `'pi'`, `mu=None` derives critically-damped gains from
   `loop_bandwidth_normalized`, and that those gains equal the inline
   `cpr_pll_bandwidth` path.

**Effort:** small. **Risk:** low *if* the default-preservation rule (step 2) is chosen
carefully — this is the only behavior-compat trap on the standalone side.

---

## Consistency summary (target state)

| Control | standalone `recover_carrier_phase_pll` | inline `lms`/`rls` |
|---|---|---|
| `mu`, `beta` (raw PI gains) | ✅ (existing) | ✅ (new `cpr_pll_mu`/`cpr_pll_beta`) |
| bandwidth shortcut (critical damping) | ✅ (new, Part C, via `loop_bandwidth_normalized`) | ✅ (existing `cpr_pll_bandwidth`) |
| shared `resolve_pll_gains` mapping | ✅ | ✅ (same helper) |
| 1st-order (`beta=0`) | ✅ | ✅ |
| damping control | ✅ (via μ,β) | ✅ (via μ,β) |
| Butterworth IIR | ✅ | ❌ (documented; use standalone — Part B) |
| cycle-slip correction | ✅ | ✅ (already) |
| joint channels | ✅ | ✅ (already) |

Both entry points accept both parameterizations through the same resolver; `mu`/`beta` and
the bandwidth shortcut are numerically interchangeable between them.

---

## Verification plan

1. `uv run pytest tests/ -k "pll or cpr or equaliz"` — existing suite green (backward compat).
2. New parity + 1st-order tests (Part A) pass on numba **and** jax; CPU and GPU.
3. `uv run ruff check commstools/equalization.py` and `uv run mypy commstools/`.
4. Real-data sanity: on the IDP capture (frequency-drift dominated), a high-β / modest-μ
   inline loop should track the residual drift with less phase-noise let-through than the
   critical-damping `cpr_pll_bandwidth` curve at equal frequency-tracking — confirm via
   output SNR / residual-phase std.
