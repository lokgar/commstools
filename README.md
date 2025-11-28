# CommsTools

CommsTools is a modern, modular Python library for high-performance digital communications research.

Built with a **dual-backend architecture**, it allows researchers to seamlessly switch between **NumPy** for ease of debugging and compatibility, and **JAX** for GPU acceleration and Just-In-Time (JIT) compilation—without changing their high-level code.

## 🚀 Key Features

* **Dual-Backend Architecture**:
  * **NumPy**: Standard CPU execution using scipy.signal, perfect for debugging and small-scale simulations.
  * **JAX**: High-performance GPU/TPU acceleration with automatic differentiation and JIT compilation.
* **Unified Signal Abstraction**: The core `Signal` class encapsulates complex IQ samples with critical physical metadata (sampling rate, modulation format, etc.), abstracting away the underlying array implementation.
* **Building-Block DSP**: Modular DSP primitives (`expand`, `interpolate`, `decimate`, `resample`, `matched_filter`) allow flexible signal processing pipelines.
* **Functional API & JIT**: Standalone processing functions can be decorated with `@jit` to automatically leverage JAX's Just-In-Time compilation when running on the JAX backend, while remaining standard Python functions on NumPy.

## 📦 Installation

Requires Python 3.12+.

```bash
# Install from source
git clone https://github.com/yourusername/commstools.git
cd commstools
uv pip install -e .
```

*Note: For GPU support with JAX, ensure you have the appropriate CUDA drivers installed and `jax[cuda]` configured.*

## ⚡ Quick Start

### Basic Signal Processing

```python
import numpy as np
from commstools import Signal, set_backend
from commstools.dsp import sequences, mapping, filters
from commstools import waveforms

# Generate PRBS sequence
bits = sequences.prbs(order=7, length=100)

# Map to OOK symbols
symbols = mapping.ook(bits)

# Generate waveform with pulse shaping
sps = 8  # samples per symbol
waveform = waveforms.ook(symbols, sps=sps, pulse_shape="rrc", rolloff=0.35)

print(f"Generated {len(waveform)} samples from {len(symbols)} symbols")
```

### Building-Block DSP Operations

```python
from commstools.dsp import filters

# Generic FIR filtering
taps = filters.rrc_taps(sps=8, rolloff=0.35)
filtered = filters.fir_filter(signal, taps)

# Upsample with pulse shaping
tx_signal = filters.upsample(symbols, factor=sps, filter_type="rrc", rolloff=0.35)

# At receiver: matched filtering
pulse_taps = filters.rrc_taps(sps, rolloff=0.35)
rx_signal = filters.matched_filter(tx_signal, pulse_taps)

# Decimate back to symbol rate
decimated = filters.decimate(rx_signal, factor=sps)
```

### Backend Switching for Performance

```python
# 1. Define a processing function (JIT-enabled)
from commstools import jit

@jit
def apply_gain(signal: Signal, gain: float) -> Signal:
    return signal.update(signal.samples * gain)

# 2. Create a Signal (defaults to NumPy)
set_backend("numpy")
sig = Signal(
    samples=np.exp(1j * 2 * np.pi * np.arange(1000) * 0.01), 
    sampling_rate=1e6
)

# 3. Process on CPU (NumPy)
out_numpy = apply_gain(sig, gain=2.0)
print(f"NumPy Output: {type(out_numpy.samples)}")

# 4. Process on GPU (JAX) - Seamless Switch
try:
    set_backend("jax")
    
    # Apply same function (now runs on JAX and is JIT compiled!)
    out_jax = apply_gain(sig, gain=2.0)
    
    print(f"JAX Output:   {type(out_jax.samples)}")
    
    # Compute Spectrum (accelerated)
    freqs, psd = out_jax.spectrum()
except ImportError:
    print("JAX not installed, skipping GPU demo.")
```

## 🏗️ Architecture

The library is built around three core concepts:

1. **`Backend` Protocol**: Defines the interface for array operations (`fft`, `exp`, `sum`, etc.). Implementations exist for `NumpyBackend` and `JaxBackend`.
2. **`Signal` Class**: The primary data carrier. It holds the data array (agnostic of backend) and provides utility methods like `plot_psd()` and `time_axis()` that delegate to the active backend.
3. **Functional API (`@jit`)**: Processing logic is implemented as pure functions. The `@jit` decorator ensures they are compiled when running on JAX, while remaining standard Python functions on NumPy.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
