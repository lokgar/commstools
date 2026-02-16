# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CommsTools is a Python library for high-performance digital communications research. It provides a unified Signal abstraction over CPU (NumPy), GPU (CuPy), and JAX backends with automatic dispatch based on where data resides.

## Commands

Package manager is `uv`. Python 3.12+ required.

```bash
# Install (editable)
uv pip install -e .

# Run python script
uv run python <script.py>

# Run all tests (CPU only, default)
uv run pytest

# Run tests on GPU
uv run pytest --device=gpu

# Run tests on both backends
uv run pytest --device=all

# Run a single test file
uv run pytest tests/test_signal.py

# Run a specific test
uv run pytest tests/test_signal.py::test_name -v

# Run with coverage
uv run pytest --cov=commstools
```

## Architecture

### Multi-Backend Dispatch

`backend.py` contains the `dispatch()` function which returns `(array, xp_module, scipy_module)` based on whether data is on CPU or GPU. All DSP functions use this to work transparently on NumPy or CuPy arrays. `backend.use_cpu_only(True/False)` forces CPU mode globally.

### Core Data Model (core.py)

All core classes use **Pydantic BaseModel** for validation.

- **Signal**: Primary container for complex IQ samples + metadata (sampling_rate, symbol_rate, modulation_scheme, modulation_order, pulse_shape, source_bits, source_symbols). Factory methods: `Signal.pam()`, `Signal.psk()`, `Signal.qam()`, `Signal.generate()`. Supports method chaining (e.g., `sig.to("gpu").fir_filter(taps)`).
- **Preamble**: Synchronization sequences (Barker, Zadoff-Chu). Converts to Signal via `to_signal()`.
- **SingleCarrierFrame**: Frame structure with preamble, payload, pilots (block/comb), and guard intervals. Supports MIMO. `get_structure_map()` returns indices for receiver parsing.

### Bit-First Convention

Generation flows: bits → symbols → samples. `Signal.source_bits` is the ground truth; symbols and samples derive from it when modulation metadata is present.

### Array Shape Convention

- SISO: `(N_samples,)`
- MIMO: `(N_channels, N_samples)` — time on last axis

### Module Roles

| Module | Role |
|---|---|
| `mapping.py` | Bits↔symbols, Gray coding, constellation generation, LLR soft demapping |
| `filtering.py` | Pulse shaping (RRC, RC, Gaussian, smoothrect), FIR filter application |
| `multirate.py` | Resampling, decimation, polyphase filtering |
| `sync.py` | Barker/ZC sequences, cross-correlation frame detection |
| `metrics.py` | EVM, SNR, BER computation |
| `impairments.py` | AWGN channel (Es/N0 aware with SPS handling) |
| `spectral.py` | Frequency shifting, Welch PSD |
| `plotting.py` | Constellation, eye diagram, PSD, symbol plots |
| `helpers.py` | Random generation, normalization (average_power/peak/unity_gain), interpolation |

### Testing Conventions

Tests use `backend_device` and `xp` fixtures from `conftest.py` to parametrize CPU/GPU execution. GPU tests are skipped automatically if CuPy is unavailable. When writing tests that need backend parametrization, include `backend_device` in the fixture list.
