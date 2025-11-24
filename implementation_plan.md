# Architecture Refactoring Plan

## Goal Description
Refactor the `commstools` library to improve backend consistency (NumPy/JAX) and robust configuration management. This addresses the issues identified in the architecture review regarding global state reliance and lack of parameter validation.

## User Review Required
[IMPORTANT]
> This refactoring introduces **Pydantic** as a core dependency.
> Breaking changes:
> - `SystemConfig` will now be a Pydantic model, not a standard dataclass.
> - `Signal.update` will use `dataclasses.replace` semantics internally. **Note:** `dataclasses.replace` creates a new instance, preserving immutability required for JAX.
> - `using_backend` context manager will be **removed**.
> - DSP functions will **automatically align** the input signal to the active global backend.

## Proposed Changes

### Dependencies
#### [MODIFY] [pyproject.toml](file:///home/lokgar/commstools/pyproject.toml)
- Add `pydantic>=2.0.0` to dependencies.

### Core Components

#### [MODIFY] [commstools/core/backend.py](file:///home/lokgar/commstools/commstools/core/backend.py)
- Remove `using_backend` context manager.

#### [MODIFY] [commstools/core/config.py](file:///home/lokgar/commstools/commstools/core/config.py)
- Replace `dataclasses` with `pydantic.BaseModel`.
- Implement field validation (e.g., `sampling_rate > 0`).
- Update `from_yaml` and `to_yaml` to use Pydantic's built-in serialization (or keep wrappers for compatibility).

#### [MODIFY] [commstools/core/signal.py](file:///home/lokgar/commstools/commstools/core/signal.py)
- Expose `backend` property (public wrapper for `_get_backend`).
- Add `ensure_backend(self, backend_name: str = None) -> "Signal"` method.
- Refactor `update` method to use `dataclasses.replace`.

### DSP Functions

#### [MODIFY] [commstools/dsp/functions.py](file:///home/lokgar/commstools/commstools/dsp/functions.py)
- Update `matched_filter`, `add_awgn` to call `signal.ensure_backend()` at the start.
- This ensures processing happens on the user-selected global backend, converting data if necessary.

### Examples & Tests
#### [MODIFY] [tests/test_core_architecture.py](file:///home/lokgar/commstools/tests/test_core_architecture.py)
- Remove tests for `using_backend`.

#### [MODIFY] [examples/jit_demo.py](file:///home/lokgar/commstools/examples/jit_demo.py)
- Replace `using_backend` with explicit `set_backend` or `signal.to()`.

#### [MODIFY] [examples/demo_usage.py](file:///home/lokgar/commstools/examples/demo_usage.py)
- Replace `using_backend` with explicit `set_backend`.

## Verification Plan

### Automated Tests
- Run existing tests: `uv run pytest tests`
- Create new tests for backend switching:
    - Verify `Signal.ensure_backend` moves data correctly.
    - Verify DSP functions work on both NumPy and JAX backends without implicit global switching.
- Verify Config validation:
    - Test that invalid config raises `ValidationError`.

### Manual Verification
- Verify that `uv sync` works after adding dependency.
