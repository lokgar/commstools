# CommsTools

CommsTools is a modern, modular Python library for high-performance digital communications research.

It adopts a **"Hardware-First" architecture**, allowing researchers to seamlessly switch between **CPU (NumPy)** for ease of debugging and **GPU (CuPy)** for massive parallel acceleration—without changing their high-level code.

## 🚀 Key Features

* **Hardware-First Architecture**:
  * **CPU Mode**: Standard execution built on NumPy. Default behavior, perfect for debugging and small-scale simulations.
  * **GPU Mode**: High-performance execution built on CuPy. Simply move your signals to the GPU to accelerate heavy DSP operations.
* **Unified Signal Abstraction**: The core `Signal` class encapsulates complex IQ samples with critical physical metadata (sampling rate, modulation format, etc.).
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

### Global Backend Selection

The library is designed to run on a specific backend (CPU or GPU) chosen at the start of your program.

```python
from commstools import set_backend, Signal
import numpy as np

# Select Hardware: 'cpu' (default) or 'gpu'
set_backend("gpu") 

# Create a signal (automatically placed on the selected backend)
# If backend is 'gpu', this creates a CuPy array on the device.
samples = np.random.randn(1000) + 1j * np.random.randn(1000)
sig = Signal(samples=samples, sampling_rate=1e6, symbol_rate=1e5)

# Apply DSP
# Operations are automatically dispatched to optimized CUDA kernels if on GPU
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
jax_array = sig.samples_to_jax()

# Now you can use JAX transformations
import jax
grad = jax.grad(some_loss_function)(jax_array)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
