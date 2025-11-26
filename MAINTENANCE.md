# CommsTools Maintenance & Extension Guide

This guide outlines the best practices for extending `commstools`, specifically for adding new DSP functions and signal generation logic. It ensures consistency with the dual-backend architecture (NumPy/JAX) and the global configuration system.

## Architecture Principles

*   **Signal-First**: The `Signal` class is the primary data carrier. It encapsulates samples and metadata.
*   **Backend-Agnostic**: Code should run on both NumPy and JAX without modification. Use the `Backend` protocol methods.
*   **Config-Aware**: Functions should respect `SystemConfig` defaults when parameters are not explicitly provided.
*   **Modular DSP**: The DSP module is split into `sequences`, `mapping`, `filters`, and `waveforms`.

## Adding New DSP Functions

DSP functions should generally take a `Signal` as input and return a modified `Signal`.

### Pattern

```python
from typing import Optional
from commstools import Signal, get_config

def my_dsp_function(signal: Signal, param: Optional[float] = None) -> Signal:
    # 1. Access Backend
    backend = signal.backend
    
    # 2. Resolve Parameters (Explicit > Config > Default)
    config = get_config()
    if param is None:
        if config is not None:
            # Assuming 'my_param' exists in config or config.extra
            param = config.get("my_param", default=1.0)
        else:
            param = 1.0
            
    # 3. Perform Computation using Backend
    # Use backend methods (backend.exp, backend.fft, etc.) instead of np/jnp directly
    # to ensure compatibility with the signal's current location (CPU/GPU).
    new_samples = signal.samples * param 
    
    # 4. Return Updated Signal
    # Use .update() to preserve other metadata (sampling_rate, etc.)
    return signal.update(samples=new_samples)
```

## Implementing Signal Generation

Signal generation is split into four stages:
1.  **Sequences**: Generate raw bits/symbols (e.g., `commstools.dsp.sequences`).
2.  **Mapping**: Map bits to constellation symbols (e.g., `commstools.dsp.mapping`).
3.  **Pulse Shaping**: Upsample and shape pulses (e.g., `commstools.dsp.filters`).
4.  **Waveforms**: Orchestrate the above to create a `Signal` (e.g., `commstools.dsp.waveforms`).

### 1. Adding a New Filter/Pulse Shape (`dsp/filters.py`)

Add a new tap generator function ending in `_taps`.

```python
def my_pulse_taps(samples_per_symbol: int, param: float) -> ArrayType:
    backend = get_backend()
    # ... calculate taps ...
    return backend.array(taps)
```

Update `get_taps` factory if necessary.

### 2. Adding a New Waveform (`dsp/waveforms.py`)

```python
from commstools.dsp import mapping, filters

def my_waveform(bits, samples_per_symbol, pulse_type='rect') -> Signal:
    # 1. Map
    symbols = mapping.my_map(bits)
    
    # 2. Shape
    taps = filters.get_taps(pulse_type, samples_per_symbol)
    samples = filters.shape_pulse(symbols, taps, samples_per_symbol)
    
    # 3. Return Signal
    return Signal(samples=samples, ...)
```

## Key Rules
*   **Do not import `numpy` or `jax` directly for computation** inside the function logic if you want to support both. Use `signal.backend` or `get_backend()`.
*   **Use `signal.update()`** instead of creating a new `Signal` from scratch when modifying an existing signal.
