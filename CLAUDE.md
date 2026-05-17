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

# Bump project version
uvx bump-my-version bump patch   # 0.1.0 → 0.1.1
uvx bump-my-version bump minor   # 0.1.0 → 0.2.0
uvx bump-my-version bump major   # 0.1.0 → 1.0.0
```

## Architecture

### Multi-Backend Dispatch

`backend.py` contains the `dispatch()` function which returns `(array, xp_module, scipy_module)` based on whether data is on CPU or GPU. All DSP functions use this to work transparently on NumPy or CuPy arrays. `backend.use_cpu_only(True/False)` forces CPU mode globally.

### Core Data Model (core.py)

All core classes use **Pydantic BaseModel** for validation.

- **Signal**: Primary container for complex IQ samples + metadata (sampling_rate, symbol_rate, modulation_scheme, modulation_order, pulse_shape, source_bits, source_symbols). Factory methods: `Signal.pam()`, `Signal.psk()`, `Signal.qam()`, `Signal.generate()`. Supports method chaining (e.g., `sig.to("gpu").fir_filter(taps)`). `matched_filter(taps=None)` accepts optional explicit taps; if omitted, taps are derived from `pulse_shape` metadata.
- **Preamble**: Synchronization sequences (Barker, Zadoff-Chu). Converts to Signal via `to_signal()`.
- **SingleCarrierFrame**: Frame structure with preamble, payload, pilots (block/comb), and guard intervals. Supports MIMO. `get_structure_map()` returns indices for receiver parsing.

### Bit-First Convention

Generation flows: bits → symbols → samples. `Signal.source_bits` is the ground truth; symbols and samples derive from it when modulation metadata is present.

### Array Shape Convention

- SISO: `(N_samples,)`
- MIMO: `(N_channels, N_samples)` — time on last axis

### Module Roles

| Module | Role |
| --- | --- |
| `mapping.py` | Bits↔symbols, Gray coding, constellation generation, LLR soft demapping |
| `filtering.py` | Pulse shaping (RRC, RC, Gaussian, smoothrect), FIR filter application |
| `multirate.py` | Resampling, decimation, polyphase filtering |
| `sync.py` | Barker/ZC sequences, cross-correlation timing; frequency offset estimation (M-th power spectral with Jacobsen sub-bin interpolation, Mengali-Morelli multi-lag autocorrelation, pilot-aided WLSQ); carrier phase recovery (Viterbi-Viterbi, BPS, pilot-aided, DD-PLL) |
| `equalization.py` | LMS, RLS, CMA (blind), RDE (blind), ZF/MMSE block equalizers; butterfly MIMO topology; Numba + JAX backends |
| `metrics.py` | EVM, SNR, BER computation |
| `impairments.py` | AWGN channel (Es/N0 aware with SPS handling), PMD (differential group delay, Jones matrix) |
| `spectral.py` | Frequency shifting, Welch PSD |
| `plotting.py` | Constellation, eye diagram, PSD, waveform, filter response, equalizer convergence |
| `helpers.py` | Random generation, normalization (average_power/peak/unity_gain/unit_energy), cross-correlation, SI formatting |

### Testing Conventions

Tests use `backend_device` and `xp` fixtures from `conftest.py` to parametrize CPU/GPU execution. GPU tests are skipped automatically if CuPy is unavailable. When writing tests that need backend parametrization, include `backend_device` in the fixture list.

Use `xpt` (backend-aware testing module) for assertions involving GPU arrays — `np.testing.assert_*` raises `TypeError` on cupy arrays. Use `xp.asarray(result)` to convert results to the current backend before comparison, and `float(xp.mean(...))` for scalar reductions.

### Iterative Warm-Start Recipe (LMS / RLS)

For streaming pipelines where the same signal is processed in consecutive blocks, thread four state fields across calls to eliminate inter-block transients:

```python
from commstools.equalization import lms, CPRState
from commstools.sync import resolve_phase_ambiguity

# Block 0 — cold start (preamble / first training block)
r0 = lms(
    block0_samples, training_symbols,
    num_taps=31, sps=2,
    step_size=1e-3, modulation="qam", order=16,
    cpr_type="pll",  # or "bps"
)

# Block k — warm start from previous result
# 1. weights:           eliminates gradient re-convergence transient
# 2. input_norm_factor: keeps gradient scale consistent across blocks
# 3. cpr_state:         resumes PLL/BPS integrators, no CPR re-locking transient
# 4. samples_prefix:    replaces leading zero-pad with real signal history
prefix_len = 31 - 1  # num_taps - 1 (conservative; actual pad_left = num_taps // 2)
rk = lms(
    blockk_samples, training_symbols_k,
    num_taps=31, sps=2,
    step_size=1e-3, modulation="qam", order=16,
    cpr_type="pll",
    w_init=r0.weights,
    input_norm_factor=r0.input_norm_factor,
    cpr_state=r0.cpr_state,
    samples_prefix=prev_block_samples[-prefix_len:],
)

# Phase ambiguity resolution — skip symbols that are still converging
y_resolved = resolve_phase_ambiguity(
    rk.y_hat, ref_symbols, "qam", 16,
    num_skip_symbols=len(training_symbols_k),  # skip DA region
)
```

**CPRState note:** JAX backend does not yet support `cpr_state` warm-start (raises `NotImplementedError`). Use `backend='numba'` for streaming pipelines. See `JAX_CPR_WARMSTART_PLAN.md` for the planned implementation.
