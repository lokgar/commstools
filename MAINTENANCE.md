# CommsTools Maintenance & Extension Guide

This guide outlines the best practices for extending `commstools`, specifically for adding new DSP functions and signal generation logic. It ensures consistency with the dual-backend architecture (NumPy/JAX) and the global configuration system.

## Architecture Principles

*   **Signal-First**: The `Signal` class is the primary data carrier. It encapsulates samples and metadata.
*   **Backend-Agnostic**: Code should run on both NumPy and JAX without modification. Use the `Backend` protocol methods.
*   **Config-Aware**: Functions should respect `SystemConfig` defaults when parameters are not explicitly provided.

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

### Key Rules
*   **Do not import `numpy` or `jax` directly for computation** inside the function logic if you want to support both. Use `signal.backend`.
*   **Use `signal.update()`** instead of creating a new `Signal` from scratch to preserve metadata.

## Implementing Signal Generation

Generators should create and return a `Signal` object, populating metadata from `SystemConfig`.

### Pattern

```python
from typing import Optional
import numpy as np # Safe to use for initial generation if converting later, 
                   # but prefer backend.array if possible for JAX-native generation.
from commstools import Signal, get_config, get_backend

def generate_custom_waveform(length: int = 1000) -> Signal:
    config = get_config()
    backend = get_backend() # Use current global backend
    
    # 1. Get Metadata from Config
    sr = config.sampling_rate if config else 1e6
    fc = config.center_freq if config else 0.0
    
    # 2. Generate Data
    # If logic is complex, you might generate in numpy and convert, 
    # or use backend methods for JIT compatibility.
    t = backend.arange(length) / sr
    samples = backend.exp(1j * 2 * backend.array(np.pi) * fc * t)
    
    # 3. Return Signal
    # Pass use_config=True to auto-fill metadata from global config
    if config:
        return Signal(samples=samples, use_config=True)
    else:
        return Signal(samples=samples, sampling_rate=sr, center_freq=fc)
```
