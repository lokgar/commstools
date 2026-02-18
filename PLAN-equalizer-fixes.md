# Plan: Equalizer Fixes — core.py cleanup, SPS enforcement, debug plots

## Context

Follow-up fixes to the equalizers module based on user feedback:

1. `Signal.equalize()` in core.py is messy — uses kwargs for things it should read directly from `self`
2. SPS should default to 2 (industry standard for fractionally-spaced equalization) and error if input is not 2 SPS
3. Add debug plotting for equalizer diagnostics (convergence curve + tap weights)

## Changes

### 1. Clean up `Signal.equalize()` — `core.py:1185`

**Problem**: The method pops `sps` from kwargs and uses kwargs to pass constellation/modulation — but it has direct access to `self.sps`, `self.mod_scheme`, `self.mod_order`.

**Fix**: Read metadata directly from `self`, pass explicitly to equalizer functions. Only use `**kwargs` for equalizer-specific params (`num_taps`, `step_size`, `normalize`, `forgetting_factor`, `store_weights`, etc.).

```python
def equalize(self, method="lms", training_symbols=None, **kwargs) -> "Signal":
    from . import equalizers
    from .mapping import gray_constellation

    sps = int(self.sps) if self.sps else 2
    if method != "zf" and sps != 2:
        raise ValueError(
            f"Signal is at {sps} SPS. Adaptive equalizers require 2 SPS "
            f"(T/2-spaced input) — resample first."
        )

    # Build constellation from signal metadata for DD slicing
    constellation = None
    if self.mod_scheme and self.mod_order:
        constellation = gray_constellation(self.mod_scheme, self.mod_order)

    if method == "lms":
        kwargs.setdefault("reference_constellation", constellation)
        result = equalizers.lms(self.samples, training_symbols, sps=sps, **kwargs)
    elif method == "rls":
        kwargs.setdefault("reference_constellation", constellation)
        result = equalizers.rls(self.samples, training_symbols, sps=sps, **kwargs)
    elif method == "cma":
        kwargs.setdefault("modulation", self.mod_scheme)
        kwargs.setdefault("order", self.mod_order)
        result = equalizers.cma(self.samples, sps=sps, **kwargs)
    elif method == "zf":
        channel_estimate = kwargs.pop("channel_estimate")
        noise_variance = kwargs.pop("noise_variance", 0.0)
        self.samples = equalizers.zf_equalizer(
            self.samples, channel_estimate, noise_variance
        )
        return self
    else:
        raise ValueError(f"Unknown equalization method: {method}")

    self.samples = result.y_hat
    self._equalizer_result = result
    return self
```

### 2. SPS enforcement — `equalizers.py` + `core.py`

**Principle**: Equalizers require T/2-spaced input (2 SPS). This is validated strictly — `sps != 2` is an error, not a silent default.

**Standalone functions** (`lms`, `rls`, `cma`):

- Change default: `sps=2`
- `_validate_sps`: raise `ValueError` if `sps != 2` with message: "Adaptive equalizers require 2 samples/symbol (T/2-spaced input). Got sps={sps}."
- Keep `num_taps < 4*sps` warning

**`Signal.equalize()`**:

- Compute `sps = int(self.sps)` from signal metadata
- If `sps != 2`, raise `ValueError` before calling any equalizer: "Signal is at {sps} SPS. Equalizers require 2 SPS — resample first."
- This catches wrong-SPS signals early with an actionable message

**Tests**: All adaptive equalizer tests must provide 2 SPS input (upsample via `np.repeat(tx, 2)` or similar). Add a test that verifies `sps != 2` raises `ValueError`.

### 3. Debug plots — `plotting.py` + `core.py`

**Add to `plotting.py`**: New `equalizer_result(result, ...)` standalone function.

2-panel layout:

- **Panel 1: MSE convergence** — `10*log10(|error|^2)` vs symbol index, smoothed with a moving average. Shows convergence dynamics.
- **Panel 2: Final tap weights** — stem plot of `|weights|` for SISO. For MIMO butterfly `(C, C, num_taps)`, create a `(C, C)` subplot grid with one stem plot per (i, j) pair.

Follows existing plotting conventions:

- Accept optional `ax` (list of 2 axes), `title`, `show` params
- Use `dispatch()` + `to_device(..., "cpu")` for GPU compat
- Return `(fig, axes)` when `show=False`, `None` when `show=True`
- Default `figsize=(10, 4)` for 2-panel side-by-side

**Add to `core.py` Signal class**: `plot_equalizer(ax=None, title=None, show=False)` thin wrapper that reads `self._equalizer_result` and calls `plotting.equalizer_result()`. Raises `ValueError` if `_equalizer_result is None`.

## Files

| File                       | Changes                                                        |
| -------------------------- | -------------------------------------------------------------- |
| `commstools/core.py`       | Rewrite `equalize()` body, add `plot_equalizer()` method       |
| `commstools/equalizers.py` | Change `sps` default to 2, update `_validate_sps` to require 2 |
| `commstools/plotting.py`   | Add `equalizer_result()` function                              |
| `tests/test_equalizers.py` | Update all tests for sps=2 or explicit sps=1                   |

## Verification

```bash
uv run pytest tests/test_equalizers.py -v
uv run pytest  # full regression
```
