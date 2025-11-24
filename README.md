# commstools

**High-Performance Digital Communications Research Framework**

`commstools` is a modern, modular Python framework designed to bridge the gap between theoretical simulation and hardware-in-the-loop experimentation for digital signal processing (DSP) and telecommunications.

Built with a **dual-backend architecture**, it allows researchers to seamlessly switch between **NumPy** for ease of debugging and compatibility, and **JAX** for GPU acceleration and Just-In-Time (JIT) compilation—without changing their high-level code.

## 🚀 Key Features

*   **Dual-Backend Architecture**: Write your DSP logic once and run it on:
    *   **NumPy**: Standard CPU execution, perfect for debugging and small-scale simulations.
    *   **JAX**: High-performance GPU/TPU acceleration with automatic differentiation and JIT compilation.
*   **Unified Signal Abstraction**: The core `Signal` class encapsulates complex IQ samples with critical physical metadata (sample rate, center frequency, modulation format), abstracting away the underlying array implementation.
*   **Modular Processing Blocks**: A standardized `ProcessingBlock` protocol ensures that filters, channel models, and DSP algorithms are reusable and composable.
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
from commstools import Signal, ProcessingBlock, set_backend, using_backend

# 1. Define a custom Processing Block
class GainBlock(ProcessingBlock):
    def __init__(self, gain: float):
        self.gain = gain

    def process(self, signal: Signal) -> Signal:
        # Operations are backend-agnostic
        return Signal(
            samples=signal.samples * self.gain,
            sample_rate=signal.sample_rate
        )

# 2. Create a Signal (defaults to NumPy)
set_backend("numpy")
sig = Signal(
    samples=np.exp(1j * 2 * np.pi * np.arange(1000) * 0.01), 
    sample_rate=1e6
)

# 3. Process on CPU (NumPy)
processor = GainBlock(gain=2.0)
out_numpy = processor(sig)
print(f"NumPy Output: {type(out_numpy.samples)}")

# 4. Process on GPU (JAX) - Seamless Switch
try:
    with using_backend("jax"):
        # Move signal to JAX backend
        sig_jax = sig.to("jax")
        
        # Apply same processor (now runs on JAX)
        out_jax = processor(sig_jax)
        
        print(f"JAX Output:   {type(out_jax.samples)}")
        
        # Compute Spectrum (accelerated)
        freqs, psd = out_jax.spectrum()
except ImportError:
    print("JAX not installed, skipping GPU demo.")
```

## 🏗️ Architecture

The framework is built around three core concepts:

1.  **`Backend` Protocol**: Defines the interface for array operations (`fft`, `exp`, `sum`, etc.). Implementations exist for `NumpyBackend` and `JaxBackend`.
2.  **`Signal` Class**: The primary data carrier. It holds the data array (agnostic of backend) and provides utility methods like `spectrum()` and `time_axis()` that delegate to the active backend.
3.  **`ProcessingBlock`**: A protocol for any transformation applied to a signal. This enforces a consistent API (`process(signal) -> signal`) across the library.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
