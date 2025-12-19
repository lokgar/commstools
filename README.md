# CommsTools

CommsTools is a modern, modular Python library for high-performance digital communications research.

## 🚀 Key Features

* **Hardware-First Architecture**:
  * **CPU Mode**: Standard execution built on NumPy. Default behavior when GPU is not available, fine for debugging and small-scale simulations.
  * **GPU Mode**: High-performance execution built on CuPy (JAX is also supported). Requires CUDA-compatible GPU and appropriate drivers.
* **Unified Signal Abstraction**: The core `Signal` class encapsulates complex IQ samples with critical physical metadata (sampling rate, modulation format, etc.) and provides methods for DSP operations and visualization.
* **JAX Interoperability**: Explicitly export data to JAX for specialized research tasks like gradient calculation or machine learning integration.

## 📦 Installation

Requires Python 3.12+.

```bash
# Install from source
git clone https://github.com/yourusername/commstools.git
cd commstools
uv pip install -e .
```

*Note: For GPU support, ensure you have a CUDA-compatible GPU and the appropriate drivers installed. The library depends on `cupy-cuda13x` by default.*

## 🛠️ Usage

### Device Placement

The library adopts a data-driven approach. The `Signal` object manages data placement. If CuPy is installed and functional, new `Signal` objects default to the GPU.

```python
from commstools import Signal
import numpy as np

samples = np.random.randn(1000) + 1j * np.random.randn(1000)

# Create a signal
# Defaults to GPU if CuPy is available, otherwise CPU.
sig = Signal(samples=samples, sampling_rate=1e6, symbol_rate=1e5)

# Verify backend
print(sig.backend)  # 'GPU' or 'CPU'

# Apply DSP
# Operations are automatically dispatched to the correct backend (NumPy or CuPy)
sig.fir_filter(taps=np.ones(10))
```

### Explicit Device Transfer

For specific scenarios, you can explicitly move data between devices.

```python
# Move to CPU
sig.to("cpu")

# Move back to GPU
sig.to("gpu")
```

### JAX Interoperability

```python
# Export samples to JAX array (Zero-copy on GPU via DLPack)
jax_array = sig.export_samples_to_jax()

# Now you can use JAX transformations
import jax
grad = jax.grad(some_loss_function)(jax_array)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
