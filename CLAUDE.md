# CLAUDE.md

This file provides guidance and reference commands for developers and AI agents (such as Claude Code) working in the **CommsTools** repository.

---

## 1. Project Overview

CommsTools is a Python library for high-performance digital communications research. It provides a unified `Signal` abstraction over CPU (NumPy), GPU (CuPy), and JAX backends, with automatic runtime dispatch based on where data resides.

---

## 2. Command Reference

All environment and script executions **must** use `uv` (Astral's Python package manager). Python 3.12+ is required.

### Environment Management

```bash
# Sync environment (core dependencies only)
uv sync

# Sync environment with all extras (includes local GPU development packages)
uv sync --all-extras

# Re-resolve and upgrade all packages in the lockfile to their latest matched versions
uv lock --upgrade

# Add a dependency / dev dependency
uv add <package>
uv add --dev <package>

# Run a Python script or console command
uv run <script.py>
```

### Testing (pytest)

```bash
# Run tests on CPU only (default)
uv run pytest --device=cpu

# Run tests on GPU (requires CuPy + CUDA)
uv run pytest --device=gpu

# Run tests on all backends (CPU and GPU)
uv run pytest --device=all

# Run a single test file
uv run pytest tests/test_signal.py

# Run a specific test case
uv run pytest tests/test_signal.py::test_signal_creation -v

# Run with test coverage
uv run pytest --cov=commstools
```

### Linting & Formatting

```bash
# Run Ruff linter
uv run ruff check .

# Format code with Ruff
uv run ruff format .

# Run static type checking
uv run mypy commstools/
```

### Version Management & Release Workflow

Since `bump-my-version` is defined in the project's development dependencies, always use `uv run bump-my-version` to run it directly from your local virtual environment (it is faster and avoids re-downloading packages compared to `uvx`).

#### Release Step-by-Step Guide

1. **Commit your active code changes**:
   Make sure all your actual development features or bugfixes are committed in Git first:

   ```bash
   git add .
   git commit -m "feat: add digital transceiver enhancement"
   ```

2. **Verify tests & package build correctness**:
   Ensure all tests are passing and the library compiles cleanly:

   ```bash
   uv run pytest
   uv build
   ```

3. **Bump the version locally**:
   Because `[tool.bumpversion]` in `pyproject.toml` is configured with `commit = true` and `tag = true` by default, running the bump command **automatically** updates version strings, commits the version changes to Git, and generates the Git release tag in a single atomic transaction:

   ```bash
   uv run bump-my-version bump patch   # e.g., 3.4.1 → 3.4.2
   uv run bump-my-version bump minor   # e.g., 3.4.1 → 3.5.0
   uv run bump-my-version bump major   # e.g., 3.4.1 → 4.0.0
   ```

4. **Push the release commits and tags to GitHub**:
   Push the feature commits, version bump commit, and release tags to your origin repository:

   ```bash
   git push origin main --tags
   ```

*Note: You only need to run `uv sync --all-extras` during release preparation if you explicitly added or modified package dependencies in `pyproject.toml`.*

---

## 3. Reference Implementation Files

For complete end-to-end digital transceiver pipelines and visualization setups, refer to the high-quality Python simulation scripts under the `examples/` directory:

* **Single-Carrier Transmission & Recovery**:
  * [single_carrier_transmission.py](file:///home/lokgar/commstools/examples/single_carrier_transmission.py): End-to-end timing synchronization (coarse/fractional), frequency offset recovery (Mengali-Morelli), matched filtering, carrier phase recovery (Viterbi-Viterbi), phase ambiguity resolution, and performance analysis (EVM, BER) on a 16-QAM baseband signal.
* **Multi-Stage & Cascaded Equalization**:
  * [multi_stage_equalization.py](file:///home/lokgar/commstools/examples/multi_stage_equalization.py): Detailed coherent receiver DSP chain illustrating cascaded fractionally-spaced blind CMA, pilot-aided LMS, frequency/phase synchronization, and residual symbol-spaced LMS equalization.
* **End-to-End Coherent Optical Pipeline**:
  * [psqam_intradyne_pipeline.py](file:///home/lokgar/commstools/examples/psqam_intradyne_pipeline.py): High-fidelity simulation of a coherent optical intradyne transceiver system modeling IQ imbalance, PMD polarization mixing, chromatic dispersion, frequency/phase recovery, and butterfly MIMO equalization with probabilistic constellation shaping.
* **Visualization and Diagnostic Plots**:
  * [generate_plots.py](file:///home/lokgar/commstools/examples/generate_plots.py): Script utilizing the library's styling and plotting theme to generate constellation density plots, PSD spectra, and high-density 2D eye diagrams.

---

## 4. DSP & Coding Guidelines

### Multi-Backend Dispatching

Always utilize the `backend.dispatch(samples)` helper in DSP functions. It dynamically returns the raw device array, the corresponding array module (`xp`), and the signal processing module (`sp`) based on the location of the data:

```python
from commstools.backend import dispatch

def my_dsp_function(samples):
    x, xp, sp = dispatch(samples)
    # xp is numpy or cupy; sp is scipy or cupyx.scipy
    return xp.fft.fft(x)  # transparent CPU/GPU execution
```

### Data Types & Precision

To maximize GPU throughput and minimize memory footprint, CommsTools utilizes mixed-precision layouts. Adhere strictly to the following dtype conventions:

* **Default Storage**: Use `complex64` (`np.complex64` / `cp.complex64`) for raw IQ samples and `float32` (`np.float32` / `cp.float32`) for real-valued signals.
* **Filter Dot-Product Accumulators**: In sequential adaptive filtering loops (LMS, CMA), inputs and weights are stored in `complex64` to save bandwidth. However, inside the hot loop, intermediate multiply-accumulate operations (dot-products for `y_out` and gradient weight adjustments) **must** promote variables to double precision (`float64` / `complex128`) to prevent catastrophic round-off cancellation over long symbol durations.
* **RLS Matrix Inversions**: The Recursive Least Squares (RLS) algorithm is highly sensitive to numerical accumulation. The inverse correlation matrix $P$, Kalman gain vector $k$, and regressor buffers must be maintained in **double precision** (`complex128` / `float64`) throughout the sequential loop. Updating $P$ in single-precision `complex64` quickly leads to loss of its positive-definite Hermitian properties, resulting in catastrophic filter divergence.
* **JAX TensorFloat-32 (TF32) Mitigation**: On modern NVIDIA GPUs (Ampere+), JAX defaults to TensorFloat-32 (TF32) for fast matrix multiplications, which truncates the mantissa from 23 bits (FP32) to 10 bits. For sequential gradient-accumulation loops like LMS/RLS weight updates, TF32 truncation causes severe drift. You **must** explicitly specify `Precision.HIGHEST` for JAX matrix products in equalizers to force true FP32.
* **Phase Unwrapping & Kalman Smoothers (CPR)**: In carrier phase recovery (e.g., Viterbi-Viterbi, BPS, pilot-aided), phase angle arrays must be promoted to double precision (`float64`) before calling `xp.unwrap()`. Unwrapping is extremely sensitive to $\pm\pi/M$ boundaries; single-precision `float32` rounding error can trigger spurious quadrant wrap-around slips. Tikhonov Kalman smoothers must also compute block transitions in `float64` to prevent underflow in small noise covariance variables.

### Signal Normalisation Invariant

To prevent scaling issues across cascaded DSP operations, CommsTools maintains a strict power normalisation invariant:

* **Symbol power representation**: A signal oversampled at `sps` has average sample energy `E[|x|²] = 1 / sps`.
* **Symbol-rate representation**: A signal at `sps=1` has average sample energy `E[|x|²] = 1`.
* Any new DSP blocks (e.g. filters, upsamplers, decimators) that alter the rate **must** apply the exact deterministic gain corrections (e.g., `sps_before / sps_after` scaling) to preserve this invariant.

### Reproducibility & Randomness

Never call `np.random` directly inside library code. Every function performing stochastic modeling (e.g. noise injection, impairments) must:

1. Accept an optional `seed: Optional[int] = None` parameter.
2. Initialize an active-backend-aware random number generator:

   ```python
   rng = xp.random.RandomState(seed) if seed is not None else xp.random
   noise = rng.normal(0, std, samples.shape)
   ```

### Performance JIT Compilation

* **Numba**: Use `@numba.njit(cache=True, fastmath=True, nogil=True)` for serial loops (like sequential LMS/RLS adaptive updates on CPU). Keep kernels compiled lazily and cached.
* **JAX**: Use `jax.lax.scan` for compiling sequential weight updates to GPU.

### Array Shapes

* **SISO**: 1-D array: `(N_samples,)`
* **MIMO**: 2-D array: `(N_channels, N_samples)` — **time is always on the last axis**.

---

## 5. Testing Conventions

* **Parametrization**: Test cases must utilize `backend_device` and `xp` fixtures from `conftest.py` to automatically validate code correctness on both CPU and GPU backends.
* **Assertions**: Standard `numpy.testing` assertions raise `TypeError` when evaluated on GPU arrays. Always use the `xpt` helper assertion module. Use `xp.asarray(expected)` to cast expectation variables to the active backend, and cast reductions to standard Python scalars before comparison:

  ```python
  from commstools.testing import xpt
  # ...
  xpt.assert_allclose(result, expected, rtol=1e-5)
  assert float(xp.mean(xp.abs(result))) > 0.0
  ```
