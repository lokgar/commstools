# CommsTools

CommsTools is a modern, modular Python library for high-performance digital communications research.

Built with a **dual-backend architecture**, it allows researchers to seamlessly switch between **NumPy** for ease of debugging and compatibility, and **JAX** for GPU acceleration and Just-In-Time (JIT) compilation—without changing their high-level code.

## 🚀 Key Features

*   **Dual-Backend Architecture**:
    *   **NumPy**: Standard CPU execution, perfect for debugging and small-scale simulations.
    *   **JAX**: High-performance GPU/TPU acceleration with automatic differentiation and JIT compilation.
*   **Unified Signal Abstraction**: The core `Signal` class encapsulates complex IQ samples with critical physical metadata (sampling rate, center frequency, modulation format), abstracting away the underlying array implementation.
*   **Functional API & JIT**: Standalone processing functions can be decorated with `@jit` to automatically leverage JAX's Just-In-Time compilation when running on the JAX backend, while remaining standard Python functions on NumPy.
*   **Developer Ergonomics**: Type-safe design with modern Python hints, making it easy to build robust and scalable communication systems.

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

Here is a simple example demonstrating how to generate a signal, apply a processing block, and switch backends.

```python
import numpy as np
from commstools import Signal, jit, set_backend, using_backend

# 1. Define a processing function (JIT-enabled)
@jit
def apply_gain(signal: Signal, gain: float) -> Signal:
    # Operations are backend-agnostic
    # .like() returns a new Signal with same metadata but new samples
    return signal.like(signal.samples * gain)

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
    with using_backend("jax"):
        # Move signal to JAX backend
        sig_jax = sig.to("jax")
        
        # Apply same function (now runs on JAX and is JIT compiled!)
        out_jax = apply_gain(sig_jax, gain=2.0)
        
        print(f"JAX Output:   {type(out_jax.samples)}")
        
        # Compute Spectrum (accelerated)
        freqs, psd = out_jax.spectrum()
except ImportError:
    print("JAX not installed, skipping GPU demo.")
```

## 🏗️ Architecture

The library is built around three core concepts:

1.  **`Backend` Protocol**: Defines the interface for array operations (`fft`, `exp`, `sum`, etc.). Implementations exist for `NumpyBackend` and `JaxBackend`.
2.  **`Signal` Class**: The primary data carrier. It holds the data array (agnostic of backend) and provides utility methods like `spectrum()` and `time_axis()` that delegate to the active backend.
3.  **Functional API (`@jit`)**: Processing logic is implemented as pure functions. The `@jit` decorator ensures they are compiled when running on JAX, while remaining standard Python functions on NumPy.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
