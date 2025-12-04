## Plan: Backend-Aware Refactoring for GPU Acceleration

This plan addresses hardcoded numpy/scipy usage throughout the codebase and refactors to use the global backend abstraction, enabling proper GPU acceleration when available.

### Steps

1. **Add missing backend methods to `backend.py`** — Implement `sin()`, `cos()`, `sinc()`, `pi` property, `isclose()`, `full()`, `clip()`, `freqz()`, `gaussian_filter()`, `firwin`, and `ones_like()`, etc. in both `NumpyBackend` and `CupyBackend` classes to support filter computation

2. **Refactor filter tap functions in `filtering.py`** — Replace hardcoded numpy operations in `gaussian_taps()`, `rrc_taps()`, and `sinc_taps()` with backend equivalents; replace `np.pi` with `backend.pi`; use `backend.sinc()`, `backend.cos()`, `backend.sin()` throughout

3. **Refactor `sequences.py` `prbs()` function** — Either implement backend-aware bitwise operations for GPU support, or explicitly document as CPU-only with `ensure_on_backend()` call after generation; remove justification comment

4. **Refactor `plotting.py` scipy usage** — Use proper backend for calculations, especially in eye_diagram.

5. **Update `mapping.py` and other DSP modules** — Audit for any remaining numpy hardcoding; ensure all computational functions use `get_backend()` and backend methods consistently

### Further Considerations

1. **scipy.signal.firwin alternative?** — Current custom `sinc_taps()` works well, but consider adding optional `firwin`-based implementation for consistency with scipy ecosystem --- make only `firwin` implementation for concise code that reuses existing scipy functionality.

### Detailed Findings from Code Review

#### 1. Core Architecture Review

**Strengths:**
- Clean Protocol-based backend abstraction allowing multiple backends
- Global backend state with `get_backend()` and `set_backend()` API
- Proper data transfer utilities (`ensure_on_backend()`, `to_host()`)
- Signal class design with immutable metadata and fluent API

**Issues:**
- Inconsistent backend usage in DSP functions (compute with numpy, convert at end)
- Missing critical backend methods (trig functions, utilities)
- scipy.signal.freqz not backend-aware in plotting
- Private API import (`scipy.ndimage._filters`)

#### 2. Hardcoded Numpy/Scipy Locations

**filtering.py - CRITICAL:**
- `gaussian_taps()` (lines 38-62): All `np.exp()`, `np.arange()`, `np.sqrt()` hardcoded
- `rrc_taps()` (lines 65-119): Uses `np.cos()`, `np.sin()`, `np.isclose()`, `np.pi` throughout
- `sinc_taps()` (lines 122-160): Uses `np.arange()`, `np.sinc()`, backend methods exist but not used
- `normalized_taps()` (lines 163-189): Calls sinc_taps which has numpy hardcoding

**sequences.py - MODERATE:**
- `prbs()` (lines 5-58): All PRBS generation uses numpy arrays with justification comment about "bitwise ops are standard"

**plotting.py - MULTIPLE:**
- Line 6: `from scipy.ndimage._filters import gaussian_filter` - private API
- `filter_response()` line 435: `signal.freqz()` not backend-aware
- Other functions properly use `to_host()` for data transfer

**Other modules:**
- `mapping.py`: Clean, uses backend properly
- `multirate.py`: Clean, uses backend properly
- `signal.py`: Clean, uses backend properly
- `utils.py`: Clean, uses backend properly

#### 3. Refactoring Patterns

**Pattern 1: Numpy Computation → Backend Conversion (WRONG)**
```python
def some_function():
    backend = get_backend()
    # Compute with numpy
    result = np.exp(np.arange(10))
    # Convert at end
    return backend.array(result)
```

**Should Be (RIGHT):**
```python
def some_function():
    backend = get_backend()
    # Compute with backend
    result = backend.exp(backend.arange(10))
    return result
```

**Pattern 2: Mathematical Constants**
Replace `np.pi` with either:
- `math.pi` (backend-agnostic scalar)
- `backend.pi` property (if implemented)

**Pattern 3: scipy.signal Usage**
There is full compatibility between scipy.signal and cupyx.scipy.signal, use when possible.
