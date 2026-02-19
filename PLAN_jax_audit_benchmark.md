# Plan: JAX Audit, Equalizer Cache Fix, PMD Impairment, and Benchmark

## Context

The JAX-based DSP functions (`compute_llr`, adaptive equalizers) have several performance and correctness issues: a critical cache key bug causing unbounded JIT recompilation, redundant intermediate allocations in device transfers, and no dtype safety at the CuPy↔JAX boundary. Additionally, no channel impairments beyond AWGN exist, and the `_benchmarks/` folder is empty.

---

## Part 1: Backend Precision Guard (HIGH PRIORITY)

**File**: [backend.py](commstools/backend.py)

Seamless CuPy↔JAX interoperability requires ironclad dtype guarantees at the transfer boundary.

### Problem
`to_jax()` (lines 296-421) auto-downcasts float64→float32 and complex128→complex64 when JAX x64 is disabled, but:
1. **No post-conversion validation** — if DLPack or `device_put` silently preserves the original dtype (e.g. x64 enabled upstream), downstream JIT kernels receive unexpected precision
2. **No explicit dtype contract** — callers like `compute_llr` pass `dtype=np.complex64` but never verify the result
3. The CuPy→JAX DLPack path (lines 381-399) casts before transfer but doesn't validate after

### Fix
Add a `_ensure_dtype` post-condition at the end of `to_jax()`. Refactor to funnel all return paths through a single exit point:

```python
# Before every return, wrap result:
def to_jax(data, device=None, dtype=None):
    ...
    # At end, before returning `result`:
    if target_dtype is not None:
        jax_target = jnp.dtype(target_dtype)
        if result.dtype != jax_target:
            logger.debug(f"to_jax: casting {result.dtype} → {jax_target}")
            result = result.astype(jax_target)
    return result
```

This ensures every caller (equalizers, compute_llr, user code) gets exactly the dtype they requested, regardless of JAX x64 mode or DLPack quirks.

---

## Part 2: Equalizer Cache Key Fix (CRITICAL)

**File**: [equalizers.py](commstools/equalizers.py)

### CMA (line 193-228)
- `n_sym` is in the cache key → **new JIT compilation for every different signal length**
- **Fix**: Remove `n_sym` from `_get_jitted_cma` signature/key. Add `n_sym` as a parameter to `cma_scan` with `@functools.partial(jax.jit, static_argnums=(4,))`. JAX's internal bounded LRU cache handles retracing when `n_sym` changes (unavoidable with `lax.scan` static length), but the manual `_JITTED_EQ` dict no longer grows unboundedly
- Update call site (line 667) to pass `n_sym` as 5th arg

### LMS (line 88-136)
- `n_train` in cache key → recompiles when training length changes
- **Fix**: Remove `n_train` from key. Pass as dynamic JAX scalar (`jnp.int32`) to `lms_scan`. It's used only in `jnp.where(idx < n_train, ...)` which accepts dynamic values — no retracing needed
- Update call site (line 456)

### RLS (line 139-190)
- Same `n_train` issue as LMS
- **Fix**: Same approach — remove from key, pass as dynamic arg
- Update call site (line 558-559)

---

## Part 3: compute_llr Precision Fixes

**File**: [mapping.py](commstools/mapping.py)

### Remove redundant `jnp.asarray()` wrappers
- **Line 775**: `jax.device_put(jnp.asarray(const), device)` → `jax.device_put(const, device)` — `const` is already `np.ndarray`, `device_put` accepts it directly and avoids an intermediate JAX allocation on the default device
- **Line 788**: Same fix for `bits_table_np`

### Add dtype validation (defensive)
- After line 763 (`jax_symbols_flat = to_jax(...)` in NumPy/CuPy path), add assertion that resulting dtype is complex64/float32

---

## Part 4: JAX Async Dispatch — `block_until_ready()` Commentary

### How JAX async dispatch works
JAX operations return immediately — they enqueue work on the XLA runtime and return a "future" array. The computation hasn't completed when the Python call returns. This means:
- **`time.perf_counter()` around JAX calls measures dispatch overhead, NOT computation time**
- Without `block_until_ready()`, benchmarks report ~0.01ms for operations that actually take 50ms

### Where it matters in this codebase
1. **Adaptive equalizers**: The `_unpack_result()` helper calls `from_jax()` → `np.asarray()`, which **implicitly blocks** until the JAX computation completes. So equalizer timing is already correct when measured end-to-end.
2. **`compute_llr()`**: Returns a **JAX array directly** (by design, for autograd). Any timing around `compute_llr()` calls MUST use `result.block_until_ready()` before stopping the timer.

### What we'll do
- **No changes to library code** — adding `block_until_ready()` inside library functions would break JAX pipelining (the whole point of async dispatch is to overlap host/device work)
- **In the benchmark script**: Use `block_until_ready()` explicitly after any JAX-returning function
- **Add a note in `compute_llr` docstring** warning that the return value is async and `block_until_ready()` is needed for timing

---

## Part 5: PMD Impairment in impairments.py

**File**: [impairments.py](commstools/impairments.py)

Add `apply_pmd()` — static first-order PMD model for dual-polarization signals.

### Model
Jones matrix in frequency domain:
```
H(f) = R(θ)^T · diag(e^{-jπf·τ}, e^{+jπf·τ}) · R(θ)
```
Where τ = DGD (differential group delay), θ = polarization rotation angle.

### Implementation
- Fully vectorized: FFT both pols → apply 2×2 Jones matrix per freq bin via broadcasting (no Python loop) → IFFT
- Parameters: `signal` (2-pol array or Signal), `dgd` (seconds), `theta` (radians), `sampling_rate` (Hz, optional — extracted from Signal)
- Signal object support: extract `sampling_rate`, return Signal with updated samples
- Backend-agnostic via `dispatch()` — works on both NumPy and CuPy

### Tests (append to `tests/test_impairments.py`)
- Identity: `dgd=0, theta=0` → output ≈ input
- Energy conservation: `||output||² ≈ ||input||²`
- Pure rotation: `dgd=0, theta=π/4` → verify 45° polarization coupling
- Shape preservation: MIMO (2, N) in → (2, N) out
- Signal object round-trip

---

## Part 6: Benchmark Script

**File**: `_benchmarks/bench_equalizers.py` (new)

### Pipeline
1. Generate dual-pol 16-QAM at 2 SPS via `Signal.qam(num_streams=2)`
2. Apply PMD via `impairments.apply_pmd()` (DGD ~ 0.3 Tsym, θ = π/5)
3. Apply AWGN via `impairments.add_awgn()` at 20 dB Es/N0
4. Run CMA, LMS (data-aided), RLS (data-aided) with timing
5. Compute EVM/SNR for each equalizer output
6. Plot: input constellation, output constellations (3 equalizers), convergence curves, timing bar chart

### Timing methodology
- Warmup: 3 iterations (JIT compilation + XLA optimization)
- Measurement: 10 iterations with `time.perf_counter()`
- Equalizer `_unpack_result` → `from_jax` → `np.asarray` already blocks, so timing is correct
- For GPU: if CuPy available, re-run full pipeline on CuPy arrays and compare
- Report: mean ± std (ms), min (ms)

### Output
- Console: timing table + per-equalizer EVM
- Saved figure: `_benchmarks/equalizer_benchmark.png`

---

## Implementation Order
1. **backend.py** — dtype guard in `to_jax()` (foundation for all JAX interop)
2. **equalizers.py** — cache key fixes (critical performance bug)
3. **mapping.py** — redundant allocation removal + dtype assertion + docstring note on async
4. **impairments.py** — PMD function + tests
5. **_benchmarks/** — benchmark script

## Verification
```bash
# 1. Run existing tests (regression check)
uv run pytest tests/test_equalizers.py tests/test_mapping.py -v

# 2. Run impairment tests (new PMD tests)
uv run pytest tests/test_impairments.py -v

# 3. Run full test suite
uv run pytest -v

# 4. Run benchmark
uv run python _benchmarks/bench_equalizers.py
```
