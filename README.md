# CommsTools

CommsTools is a modern, modular Python library for high-performance digital communications research.

Built with a **dual-backend architecture**, it allows researchers to seamlessly switch between **NumPy** for ease of debugging and compatibility, and **JAX** for GPU acceleration and Just-In-Time (JIT) compilation—without changing their high-level code.

## 🚀 Key Features

* **Dual-Backend Architecture**:
  * **NumPy**: Standard CPU execution built on NumPy, perfect for debugging and small-scale simulations.
  * **JAX**: High-performance GPU/TPU acceleration with automatic differentiation and JIT compilation.
* **Unified Signal Abstraction**: The core `Signal` class encapsulates complex IQ samples with critical physical metadata (sampling rate, modulation format, etc.), abstracting away the underlying array implementation.

## 📦 Installation

Requires Python 3.12+.

```bash
# Install from source
git clone https://github.com/yourusername/commstools.git
cd commstools
uv pip install -e .
```

*Note: For GPU support with JAX, ensure you have the appropriate CUDA drivers installed and `jax[cuda]` configured.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
