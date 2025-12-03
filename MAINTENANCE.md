# CommsTools Maintenance & Extension Guide

This guide outlines the best practices for extending `commstools`, specifically for adding new DSP functions and signal generation logic. It ensures consistency with the dual-backend architecture (NumPy/JAX).

## Architecture Principles

* **Signal-First**: The `Signal` class is the primary data carrier. It encapsulates samples and metadata.
* **Backend-Agnostic**: Code should run on both NumPy and JAX without modification. Use the `Backend` protocol methods.

## Working with Signals

The `Signal` class is **immutable**. This design choice is critical for JAX performance and functional programming patterns.

* **Immutability**: Methods like `upsample`, `fir_filter`, or `to` do **not** modify the signal in place. They return a **new** `Signal` instance.
* **Chaining**: You can chain methods for concise code.

```python
# Correct usage
sig = sig.to("jax")
sig = sig.upsample(2)
sig = sig.fir_filter(taps)

# OR chained
sig = sig.to("jax").upsample(2).fir_filter(taps)

# Incorrect usage (will not change 'sig')
sig.to("jax") 
sig.upsample(2)
```

### Backend Transfer

Use the `.to(backend_name)` method to explicitly move data between backends (e.g., CPU/NumPy to GPU/JAX).

```python
# Move to JAX (e.g., for GPU acceleration)
sig_jax = sig.to("jax")

# Move back to NumPy (e.g., for plotting or I/O)
sig_numpy = sig_jax.to("numpy")
```

## Adding New DSP Functions

DSP functions, if they are directly related to the signal, should generally take a `Signal` as input and return a modified `Signal`.

### Pattern

```python
from typing import Optional
from commstools import Signal

def my_dsp_function(signal: Signal, param: Optional[float] = None) -> Signal:
    # 1. Access Backend
    backend = signal.backend
    
    # 2. Resolve Parameters (Explicit > Default)
    if param is None:
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

1. **Sequences**: Generate raw bits/symbols (e.g., `commstools.dsp.sequences`).
2. **Mapping**: Map bits to constellation symbols (e.g., `commstools.dsp.mapping`).
3. **Pulse Shaping**: Apply DSP building blocks and filters (e.g., `commstools.dsp.filtering`).
4. **Waveforms**: Orchestrate the above to create complete waveforms (e.g., `commstools.waveforms`).

**Note:** Waveforms are now in the top-level `commstools.waveforms` module, not in `commstools.dsp.waveforms`.

### 1. Adding a New Filter/Pulse Shape (`dsp/filtering.py`)

The filters module is organized into four sections:

#### Filter Design (Tap Generation)

Add a new tap generator function ending in `_taps`:

```python
def my_pulse_taps(sps: int | float, param: float, span: int = 4) -> ArrayType:
    """
    Generate custom pulse filter taps.
    
    Args:
        sps: Samples per symbol
        param: Custom parameter
        span: Number of symbols to span
    
    Returns:
        Filter taps (length = int(sps * span))
    """
    backend = get_backend()
    # ... calculate taps ...
    return backend.array(taps)
```

#### Core Building Blocks

Building blocks operate at the sample level:

* `fir_filter(samples, taps, mode)` - Generic FIR filtering (convolution)
* `expand(symbols, factor)` - Zero-insertion upsampling
* `upsample(samples, factor, filter_type)` - Complete upsampling (expand + fir_filter)
* `decimate(samples, factor)` - Filtered downsampling  
* `resample(samples, up, down)` - Rational rate conversion

The `fir_filter()` function is the fundamental operation used by `upsample`, `matched_filter`, and `shape_pulse`. Use it directly when you have custom taps or need specific convolution modes.

#### Matched Filtering

Matched filtering uses the `matched_filter` function (or generic `fir_filter` with custom taps):

```python
# Using matched_filter (automatically does conj + time-reverse)
pulse_taps = my_pulse_taps(sps, **kwargs)
filtered = matched_filter(samples, pulse_taps, mode="same")

# Or using generic fir_filter with manual matched taps
backend = get_backend()
matched_taps = backend.conj(pulse_taps[::-1])
filtered = fir_filter(samples, matched_taps, mode="same")
```

#### High-Level Operations

Compose building blocks into complete operations:

```python
def my_shaping_operation(symbols, sps: int | float, **kwargs) -> ArrayType:
    """Complete shaping pipeline for custom modulation."""
    # Expand to sample rate
    expanded = expand(symbols, int(sps))
    
    # Apply pulse shaping
    taps = my_pulse_taps(sps, **kwargs)
    backend = get_backend()
    shaped = backend.convolve(expanded, taps, mode="same")
    
    return shaped
```

### 2. Adding a New Waveform (`commstools/waveforms.py`)

**Important:** Waveforms are now in the top-level `commstools.waveforms`, not `commstools.dsp.waveforms`.

```python
from .dsp import mapping, filtering
from .core import Signal

def my_waveform(
    bits,
    sps: int | float,
    pulse_shape: str = "rrc",
    symbol_rate: float = 1e9,
    **kwargs
) -> Signal:
    """
    Generate custom modulation waveform.
    
    Args:
        bits: Input bit sequence
        sps: Samples per symbol
        pulse_shape: Pulse shape type ("rrc", "gaussian", etc.)
        symbol_rate: Symbol rate in Hz
        **kwargs: Additional pulse shaping parameters
    
    Returns:
        Signal object with waveform samples
    """
    # 1. Map bits to symbols
    symbols = mapping.my_map(bits)
    
    # 2. Apply pulse shaping
    samples = filtering.shape_pulse(symbols, sps, pulse_shape, **kwargs)
    
    # 3. Return Signal with metadata
    sampling_rate = symbol_rate * sps
    return Signal(samples=samples, sampling_rate=sampling_rate)
```

## Key Rules

* **Do not import `numpy` or `jax` directly for computation** inside the function logic if you want to support both. Use `signal.backend` or `get_backend()`.
* **Use `signal.update()`** instead of creating a new `Signal` from scratch when modifying an existing signal.
