# Project Architecture Review

## Executive Summary
The current architecture provides a solid foundation with a clear separation of concerns (Core, DSP, Channel). The use of Protocols for the Backend is a good design choice for extensibility. However, the heavy reliance on global state (both for Backend and Config) and the inconsistency between data locality and processing logic present significant risks for scalability, testing, and correctness.

## Answers to Specific Questions

### 1. Backend Implementation & Context Manager
**Is the current backend implementation logical?**
The `Backend` protocol pattern is logical, but the *usage* of the global backend in processing functions is problematic. Currently, functions in `dsp/functions.py` fetch the global backend (`get_backend()`) regardless of where the input `Signal`'s data actually resides. This can lead to performance issues (implicit copying) or crashes if the global backend doesn't support the input array type.

**Do we need the context manager?**
**Yes, keep it, but redefine its scope.** The context manager (`using_backend`) is valuable for controlling *where new data is created* (e.g., `zeros`, `arange`). However, it should not dictate how *existing* data is processed. Processing should always happen where the data lives.

**Can Signal automatically switch backends?**
Yes, this is a desirable feature. You can implement a mechanism where processing functions automatically move the `Signal` to the active global backend if desired, or (better) have the function adapt to the `Signal`'s backend.
*   **Option A (Auto-move):** Functions check if `signal.backend` matches `get_backend()`. If not, they call `signal.to(get_backend().name)`. This ensures all ops happen on the active backend but incurs data transfer costs.
*   **Option B (Dispatch):** Functions use `signal.backend` for operations. This avoids transfer but requires the backend to handle the data type.

### 2. Parameter Validation
**Does it make sense to use parameter validation?**
**Absolutely.**
*   **SystemConfig:** Invalid configurations (e.g., negative sampling rates, invalid modulation formats) should be caught immediately at initialization.
*   **Signal:** Ensuring `samples` dimensions match `sampling_rate` expectations or that complex samples are indeed complex is crucial for debugging.
*   **Recommendation:** Use **Pydantic** for `SystemConfig` to get robust validation, type coercion, and serialization for free.

## Critical Architecture Review

### 1. Backend Consistency & Global State
**Issue:** `dsp/functions.py` functions like `add_awgn` instantiate `backend = get_backend()`.
*   If `signal.samples` is a NumPy array but `set_backend('jax')` was called, `backend` is `JaxBackend`.
*   `backend.mean(signal.samples)` will attempt to run JAX operations on a NumPy array. While JAX handles this, the reverse (NumPy backend on JAX array) might fail or force eager evaluation.
*   **Fix:** Functions should derive the backend from the input `Signal`.
    ```python
    # In Signal class
    @property
    def backend(self) -> Backend:
        return self._get_backend()

    # In dsp/functions.py
    def add_awgn(signal, ...):
        backend = signal.backend  # Use the data's backend
        ...
    ```

### 2. Configuration Management
**Issue:** Functions mix explicit arguments with global config fallbacks inside the function body.
*   This makes the function signature deceptive and the logic hard to test (you must mock global state).
*   **Fix:** Use the global config only as a default value in the signature or via a decorator, keeping the function body pure.
    ```python
    # Better pattern
    def add_awgn(signal, snr_db=None):
        if snr_db is None:
             snr_db = require_config().snr_db
    ```
    (The current code does this, which is acceptable, but `require_config` is safer than `get_config` which might return None).

### 3. Signal Class Design
**Issue:** `Signal` is a dataclass but has significant logic (`spectrum`, `to`).
*   It's good, but `_get_backend` is heuristic.
*   **Fix:** Store the backend explicitly or make the heuristic robust.
*   **Issue:** `update` method manually reconstructs the class.
*   **Fix:** Use `dataclasses.replace(self, **changes)` for cleaner code.

## Recommendations

1.  **Expose `Signal.backend`:** Make `_get_backend` a public property `backend`.
2.  **Data-Local Processing:** Update DSP functions to use `signal.backend` instead of `get_backend()`.
3.  **Auto-Switching Logic:** If you want the "Auto-switch" behavior, implement a `ensure_backend` method:
    ```python
    def ensure_backend(self, backend_name: str = None) -> 'Signal':
        target = backend_name or get_backend().name
        if self.backend.name != target:
            return self.to(target)
        return self
    ```
    Then use it in functions: `signal = signal.ensure_backend()` at the start of DSP functions.
4.  **Adopt Pydantic:** Replace standard dataclasses for `SystemConfig` with Pydantic models for built-in validation.
