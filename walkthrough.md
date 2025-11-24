# Refactoring Walkthrough

## Changes Accomplished

### 1. Dependency Management
- Added `pydantic>=2.0.0` to `pyproject.toml`.

### 2. Configuration Management (`commstools/core/config.py`)
- Replaced `dataclasses` with `pydantic.BaseModel`.
- Added validation for `sampling_rate` (must be > 0) and `filter_roll_off` (0-1).
- Implemented `model_post_init` logic using `@model_validator(mode='after')`.
- Updated `to_yaml` and `from_yaml` to leverage Pydantic's serialization.

### 3. Signal Architecture (`commstools/core/signal.py`)
- Added `backend` property to `Signal`.
- Added `ensure_backend(backend_name)` method to auto-align signal data with the global backend.
- Updated `update` method to use `dataclasses.replace` (preserving immutability).
- **Optimization:** Reordered `_get_backend` checks to prioritize NumPy arrays, preventing unnecessary JAX initialization (and warnings) when using the NumPy backend.

### 4. Backend Consistency (`commstools/core/backend.py`)
- Removed `using_backend` context manager to discourage mixed-state execution.

### 5. DSP Functions (`commstools/dsp/functions.py`)
- Updated `matched_filter` and `add_awgn` to call `signal.ensure_backend()` at the start.
- This ensures that setting the global backend (`set_backend('jax')`) effectively controls where processing happens, even if the input signal was created on NumPy.

## Verification Results

### Automated Tests
Ran `uv run pytest tests` with 100% pass rate (28/28 tests).
- Verified `SystemConfig` validation raises `ValidationError` for invalid inputs.
- Verified `Signal` backend switching works correctly.
- Verified DSP functions operate correctly on both backends.

### Manual Verification
- Verified `examples/jit_demo.py` and `examples/demo_usage.py` run without `using_backend`.
