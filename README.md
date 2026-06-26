# CommsTools

**High-performance digital communications research library for Python.**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Backends](https://img.shields.io/badge/backend-NumPy%20%7C%20CuPy%20%7C%20JAX-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia)

---

CommsTools is a Python library for digital communications research that treats hardware as a first-class concern. A single `Signal` object carries IQ samples, physical metadata, and modulation context — and every DSP operation dispatches automatically to NumPy, CuPy, or JAX depending on where the data lives. No rewriting code for GPU. No losing metadata in function calls.

---

## Why CommsTools?

Most research codebases accumulate loose arrays with ad-hoc metadata dictionaries. CommsTools enforces a different model:

- **One object, complete context.** Sampling rate, symbol rate, modulation format, pulse shape — all travel with the signal through the chain.
- **Backend-transparent DSP.** `dispatch()` resolves the right NumPy/CuPy/SciPy modules at runtime based on where the data lives. The same function body runs on CPU or GPU.
- **Method chaining.** `sig.to("gpu").fir_filter(taps).resample(sps_out=2).plot_psd()` just works.
- **JAX escape hatch.** Zero-copy DLPack export on GPU lets you call JAX transforms (gradients, vmaps, scans) on live signal data without leaving the research loop.

---

## Features

| Area | Capability |
| --- | --- |
| **Signal Generation** | PAM, PSK, QAM factory methods with Gray-coded constellations and configurable pulse shaping |
| **Pulse Shaping** | RRC, RC, Gaussian, Smooth-Rectangle — tap generators + filtering pipeline |
| **Filtering** | FFT-based FIR convolution, matched filter, Overlap-Save, polyphase multirate |
| **Synchronization** | Barker / Zadoff-Chu sequences, cross-correlation timing, M-th power (Jacobsen) & Mengali-Morelli & pilot-aided FOE, Viterbi-Viterbi / BPS / DD-PLL carrier phase recovery |
| **Equalization** | LMS, RLS, CMA (blind), ZF / MMSE block — butterfly MIMO topology, Numba + JAX backends |
| **Channel Models** | AWGN (Es/N0 with SPS correction), PMD (differential group delay, Jones matrix) |
| **Soft Demapping** | LLR computation via max-log and exact log-sum-exp (JAX JIT compiled) |
| **Metrics** | EVM (%, dB), data-aided SNR, BER |
| **Analysis** | Data-aided carrier-phase trajectory, zero-phase drift detrending, AWGN-free lag-slope linewidth fit, Di Domenico $\beta$-separation line FWHM, overlapping Allan deviation |
| **Spectral** | Welch PSD, frequency shift with bin-quantized mixing |
| **Frames** | `SingleCarrierFrame` with block/comb pilots, guard intervals, structure map for receiver parsing |
| **MIMO** | All operations shape-aware for `(N_channels, N_samples)` arrays |
| **Visualization** | Constellation, eye diagram, PSD, waveform, filter response, equalizer convergence |

---

## Installation

**Requires Python 3.12+** and [`uv`](https://github.com/astral-sh/uv).

**Core dependencies** (installed automatically): NumPy, SciPy, Numba, Matplotlib, PyYAML, Pydantic, and **JAX (CPU)**. JAX is a **mandatory** core dependency — it powers the soft-demapper (`compute_llr` / `gmi`) and the equalizers' `jax` backend, both of which run on CPU. The optional `gpu` extra layers the CUDA stacks (`jax[cuda13]`, `cupy-cuda13x`) on top.

#### A. Direct Installation (For Users)

Since the package is not yet registered on PyPI, you can install it directly from GitHub:

##### Using `uv` (Recommended)

To install the core package (CPU backends — NumPy, Numba, and JAX on CPU):
```bash
uv pip install git+https://github.com/lokgar/commstools.git
# or to add as a project dependency:
uv add "commstools @ git+https://github.com/lokgar/commstools.git"
```

To install with GPU acceleration (includes CUDA 13 JAX and CuPy stacks):
```bash
uv pip install "commstools[gpu] @ git+https://github.com/lokgar/commstools.git"
# or to add as a project dependency:
uv add "commstools[gpu] @ git+https://github.com/lokgar/commstools.git"
```

##### Using standard `pip`

To install the core CPU-only package:
```bash
pip install git+https://github.com/lokgar/commstools.git
```

To install with GPU acceleration:
```bash
pip install "commstools[gpu] @ git+https://github.com/lokgar/commstools.git"
```

*(Note: In the future, once registered on PyPI, you will be able to run `uv add commstools` or `pip install commstools` directly.)*

#### B. Development Installation (For Contributors)
```bash
git clone https://github.com/lokgar/commstools.git
cd commstools

# Sync environment with core dependencies only
uv sync

# Sync environment with all dependencies including GPU development packages:
uv sync --all-extras
```

### GPU Support

GPU execution requires CUDA 13.x drivers. The library supports hardware acceleration via `cupy-cuda13x` and the CUDA plugins for the already-required core JAX (`jax[cuda13]`), defined as **optional dependencies** under the `gpu` extra. If these are installed and a CUDA-capable GPU is detected at import time, new `Signal` objects will default to GPU placement.

To force CPU-only mode regardless of hardware:

```python
from commstools import backend
backend.use_cpu_only(True)
```

---

## Core Concepts: The Signal Object

`Signal` is a Pydantic model that binds IQ samples to the metadata needed for every downstream operation. You never pass sampling rate separately again.

```python
import numpy as np
from commstools import Signal

# Wrap existing samples
sig = Signal(
    samples=np.random.randn(1000) + 1j * np.random.randn(1000),
    sampling_rate=1e9,
    symbol_rate=250e6,
)

print(sig.sps)       # 4.0 — derived from rates
print(sig.backend)   # 'CPU' or 'GPU'
print(sig.duration)  # seconds
```

---

## Other Key Capabilities

### Timing & Carrier Synchronization

Since CommsTools organizes sync algorithms into dedicated, granular modules, imports match their actual roles:

```python
from commstools.timing import barker_sequence, estimate_timing, correct_timing
from commstools.frequency import estimate_frequency_offset_mengali_morelli, correct_frequency_offset
from commstools.recovery import recover_carrier_phase_viterbi_viterbi, correct_carrier_phase

# Coarse/Fine timing synchronization using preambles
preamble_seq = barker_sequence(length=13)
delay = estimate_timing(rx_samples, preamble_seq)
aligned = correct_timing(rx_samples, delay)

# Fine data-aided frequency offset correction
fo_hz = estimate_frequency_offset_mengali_morelli(aligned, fs=sampling_rate, modulation="qam", order=16)
cfo_corrected = correct_frequency_offset(aligned, fo_hz, fs=sampling_rate)

# Carrier phase recovery (Viterbi-Viterbi algorithm)
phase_est = recover_carrier_phase_viterbi_viterbi(cfo_corrected, modulation="qam", order=16)
recovered_symbols = correct_carrier_phase(cfo_corrected, phase_est)
```

### Adaptive Equalization (MIMO-Aware)

Adaptive equalization modules handle butterfly MIMO topologies out of the box with Numba-accelerated loops:

```python
from commstools.equalization import lms, cma

# Supervised LMS (T/2-spaced, butterfly 2x2 MIMO)
result = lms(
    x=rx_samples,          # (N_rx, N_samples) oversampled at sps=2
    training_symbols=tx_symbols,  # (N_tx, N_train) training symbols
    num_taps=31,
    step_size=1e-3,
)
print(result.y_hat.shape)  # equalized output symbols: (N_tx, N_symbols)

# Blind CMA (no training data required)
result_cma = cma(rx_samples, num_taps=31, step_size=1e-4)
```

---

## Module Reference

| Module | Contents |
| --- | --- |
| `commstools.core` | `Signal`, `Preamble`, `SingleCarrierFrame` |
| `commstools.backend` | `dispatch()`, `to_device()`, `to_jax()`, `from_jax()`, `use_cpu_only()` |
| `commstools.mapping` | `gray_constellation()`, `map_bits()`, `demap_symbols_hard()`, `compute_llr()` |
| `commstools.filtering` | `rrc_taps()`, `rc_taps()`, `gaussian_taps()`, `smoothrect_taps()`, `fir_filter()`, `shape_pulse()` |
| `commstools.multirate` | `resample()`, `decimate()`, `upsample()`, `decimate_to_symbol_rate()` |
| `commstools.timing` | `barker_sequence()`, `zadoff_chu_sequence()`, delay estimation & timing alignment |
| `commstools.frequency` | Frequency offset estimation (FOE) & correction (M-th power, Mengali-Morelli) |
| `commstools.recovery` | Carrier phase recovery (CPR) & cycle-slip correction (Viterbi-Viterbi, BPS, DD-PLL) |
| `commstools.equalization` | `lms()`, `rls()`, `cma()`, `rde()`, `zf_equalizer()`, `EqualizerResult` |
| `commstools.impairments` | `apply_awgn()`, `apply_pmd()`, `apply_phase_noise()`, `apply_iq_imbalance()`, `apply_chromatic_dispersion()` |
| `commstools.metrics` | `evm()`, `snr()`, `ber()` |
| `commstools.analysis` | Laser phase characterization (`characterize_carrier_phase()`, `carrier_phase_trajectory()`, `separate_drift_phase_noise()`, `linewidth_increment()`, `linewidth_beta_separation()`, `allan_deviation()`) |
| `commstools.spectral` | `welch_psd()`, `shift_frequency()` |
| `commstools.plotting` | `constellation()`, `eye_diagram()`, `psd()`, `time_domain()`, `filter_response()`, `equalizer_result()` |
| `commstools.helpers` | `random_bits()`, `random_symbols()`, `normalize()`, `rms()`, `format_si()` |

---

## Running Tests

```bash
# CPU (default)
uv run pytest --device=cpu

# GPU (requires CuPy + CUDA)
uv run pytest --device=gpu

# Both backends
uv run pytest --device=all

# Single module / directory
uv run pytest tests/equalization/ -v

# With coverage
uv run pytest --cov=commstools
```

GPU tests skip automatically when CuPy is unavailable.

---

## Requirements

| Dependency | Role |
| --- | --- |
| `numpy >= 2.2` | CPU array operations |
| `scipy >= 1.15` | CPU signal processing |
| `pydantic >= 2.0` | Signal model validation |
| `numba >= 0.64` | JIT-compiled equalizer loops |
| `matplotlib >= 3.10` | Visualization |
| `cupy-cuda13x` | GPU arrays (optional) |
| `jax[cuda13]` | JAX transforms (optional) |

---

## Contributing

Contributions are welcome. Please open an issue to discuss large changes before submitting a PR. For bug fixes and small improvements, PRs are fine directly.

The project uses `uv` for environment management:

```bash
uv sync --all-extras  # install all dependencies including GPU optional packages
uv run pytest         # run the test suite
uv run ruff check .   # lint
```

---

## License

MIT
