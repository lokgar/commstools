# Plan: Equalizer Convergence Fixes + Block-Adaptive GPU Acceleration

## Context

The user ran `bench_equalizers.py` and found: (1) LMS and CMA don't converge, (2) GPU is much slower than CPU, (3) DA equalizers lack partial training support, (4) `normalize` option needs audit. Root cause analysis revealed a power mismatch between samples and training symbols, center-tap misalignment for low-ISI scenarios, CPU round-trips killing GPU perf, and a fundamental architecture problem (sample-by-sample `lax.scan` underutilizes GPU).

---

## Part 1: Convergence Fixes

### Empirical Analysis

Ran 5 experiments testing LMS/CMA/RLS across signal types, tap counts, and step sizes.

**Key findings:**

| Signal type | Even/odd sample power | Center-tap matters? | Convergence |
|---|---|---|---|
| Zero-stuffed (`pulse_shape="none"`) | 2.0 / 0.0 | Yes (CMA locks on zero) | Narrow step-size range |
| Zero-stuffed + ISI channel | ~1.3 / ~0.3 | Somewhat | Better than no ISI |
| RRC pulse-shaped | ~0.50 / ~0.49 | No (equal power) | Reliable with enough taps |
| RRC + PMD (MIMO) | ~0.50 / ~0.49 | No | LMS/CMA/RLS all converge |

**Root cause — power mismatch**: `pulse_shape="none"` calls `normalize(expand(symbols, sps), "peak")` in [filtering.py:554](commstools/filtering.py#L554), creating samples with per-symbol power 2.0. But `source_symbols` is stored before pulse shaping at power 1.0. This 2x mismatch amplifies the LMS error gradient, cutting the stability margin in half and making step-size selection extremely fragile.

**Secondary issue — center-tap alignment**: When `num_taps // 2` is odd and `sps=2`, the center tap falls between symbol instants. For zero-stuffed signals this means zero energy → CMA stuck. For RRC with minimal ISI (mild channel, good timing), inter-symbol samples can have low energy, causing slow CMA startup. Even though RRC experiments showed equal even/odd power (~0.50/0.49), this depends on channel conditions — with minimal dispersion the eye diagram closes less, and inter-symbol samples approach zero crossings.

### Fix A: Center-tap alignment for sps=2

**File**: [equalizers.py](commstools/equalizers.py)

Ensure center tap aligns with a symbol-bearing sample position. Add `_get_center_tap(num_taps, sps)`:

```python
def _get_center_tap(num_taps, sps):
    center = num_taps // 2
    if sps == 2 and center % 2 != 0:
        center -= 1  # align with data-bearing (even) sample position
    return center
```

Update `_init_butterfly_weights` to accept `sps` and use `_get_center_tap`. Update delay calculation in `_prepare_training` to use the same center. This is a robustness fix — ensures reliable convergence even for low-ISI scenarios.

### Fix B: RMS-normalize input samples in equalizers

**File**: [equalizers.py](commstools/equalizers.py)

Add input power normalization at entry to each equalizer (`lms`, `rls`, `cma`). Before converting to JAX, normalize both samples and training symbols to unit average power independently. This makes the equalizers robust to any input power convention and eliminates the step-size sensitivity caused by power mismatch.

```python
# In lms(), rls(), cma() — after dispatch(samples):
sig_power = xp.mean(xp.abs(samples) ** 2)
samples = samples / xp.sqrt(sig_power + 1e-20)
if training_symbols is not None:
    train_power = xp.mean(xp.abs(training_symbols) ** 2)
    training_symbols = training_symbols / xp.sqrt(train_power + 1e-20)
```

### Fix C: `_prepare_training` forces CPU round-trip

**File**: [equalizers.py](commstools/equalizers.py)

`np.asarray(to_device(training_symbols, "cpu"))` copies GPU data to CPU unnecessarily. Refactor to keep training symbols on-device using `dispatch()`. Only the small constellation array (for `np.unique`) touches CPU.

### Fix D: Benchmark should use realistic signals

**File**: [_benchmarks/bench_equalizers.py](_benchmarks/bench_equalizers.py)

Use `pulse_shape="rrc"` (realistic Rx signal) with adequate `num_taps` (>= 21). Use `normalize=True`.

### Fix E: `normalize` option — keep as-is

NLMS (`normalize=True`) is the correct default. No changes needed.

---

## Part 2: Block-Adaptive Equalizers (GPU Acceleration)

**File**: [equalizers.py](commstools/equalizers.py)

### Architecture

Instead of `lax.scan` over individual symbols, scan over **blocks** of `B` symbols. Within each block:

1. **Parallel forward pass** (vmap): Apply frozen weights to B symbols simultaneously
2. **Error computation** (vmap): Compute B errors in parallel
3. **Weight update**: Average gradient across block, update weights once

When `block_size=1`, this is identical to current sample-by-sample behavior.

### API Change

Add `block_size: int = 1` parameter to `lms()`, `rls()`, `cma()`. When `block_size > 1`, route to new `_get_jitted_block_*` functions. Backward compatible — default `block_size=1` uses existing kernels unchanged.

### Block LMS

Full block parallelism. Inner block is entirely vmap'd:

```python
def block_step(W, block_idx):
    # 1. Extract B windows in parallel (vmap over dynamic_slice)
    X_block = vmap(extract_one)(sym_indices)  # (B, C, num_taps)
    # 2. Apply weights in parallel
    Y_block = vmap(apply_weights)(X_block)     # (B, C)
    # 3. Compute errors (training or DD)
    E_block = D_block - Y_block                # (B, C)
    # 4. Average gradient + single weight update
    avg_grad = mean(vmap(grad_one)(E_block, X_block), axis=0)
    W_new = W + mu_eff * avg_grad
```

### Block CMA

Same as block LMS but with CMA error `e = y*(|y|^2 - R2)`. All B errors computed in parallel since they use the same frozen weights.

### Block RLS (Hybrid)

- **Phase 1** (parallel): Apply frozen weights to B symbols via vmap → Y_block, E_block
- **Phase 2** (sequential): Update (W, P) via inner `lax.scan` over B symbols using pre-computed errors

RLS's inverse correlation matrix P requires sequential updates (matrix inversion lemma recurrence). The benefit comes from the parallel forward pass and reduced outer-scan overhead.

### Expected GPU Speedup

| Algorithm | block_size=1 | block_size=8 | block_size=32 |
|-----------|-------------|-------------|---------------|
| LMS/CMA   | 1x          | ~4-6x       | ~8-15x        |
| RLS       | 1x          | ~1.5-2x     | ~2-3x         |

### Remainder Handling

Pad `n_sym` to next multiple of `block_size`. Truncate output after scan.

### Convergence vs Block Size Trade-off

Larger blocks = fewer weight updates per symbol = slower adaptation. Guidance:

- Slowly varying channels / high SNR: `block_size=16-64`
- Fast fading / low SNR: `block_size=4-8`
- CMA is more sensitive (non-convex cost surface): recommend `block_size <= 32`

---

## Part 3: Test Updates

**File**: [tests/test_equalizers.py](tests/test_equalizers.py)

1. Add `test_center_tap_alignment` — verify convergence with various `num_taps` values
2. Add block equivalence test: `block_size=1` output matches original per-sample output
3. Add block convergence tests for LMS, CMA, RLS with `block_size=8`
4. Add block MIMO test: 2x2 butterfly with `block_size=8`
5. Add remainder handling test: `n_sym % block_size != 0`

---

## Part 4: Benchmark Update

**File**: [_benchmarks/bench_equalizers.py](_benchmarks/bench_equalizers.py)

- Use `pulse_shape="rrc"`, `normalize=True`, reasonable step sizes
- Test multiple `block_size` values (1, 8, 32) to show GPU scaling
- Document GPU vs CPU expectations in script header

---

## Implementation Order

1. Convergence fixes: `_get_center_tap`, `_init_butterfly_weights`, `_prepare_training`, input RMS normalization
2. Block LMS kernel + `lms()` routing
3. Block CMA kernel + `cma()` routing
4. Block RLS kernel + `rls()` routing
5. `_unpack_result` update for block weight history
6. Tests
7. Benchmark update

## Verification

```bash
# Run equalizer tests (existing + new)
uv run pytest tests/test_equalizers.py -v

# Full regression
uv run pytest -v

# Benchmark with block sizes
uv run python _benchmarks/bench_equalizers.py
```
