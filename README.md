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
- **Method chaining.** `sig.to("gpu").fir_filter(taps).resample(2).plot_psd()` just works.
- **JAX escape hatch.** Zero-copy DLPack export on GPU lets you call JAX transforms (gradients, vmaps, scans) on live signal data without leaving the research loop.

---

## Features

| Area | Capability |
| --- | --- |
| **Signal Generation** | PAM, PSK, QAM factory methods with Gray-coded constellations and configurable pulse shaping |
| **Pulse Shaping** | RRC, RC, Gaussian, Smooth-Rectangle — tap generators + filtering pipeline |
| **Filtering** | FFT-based FIR convolution, matched filter, Overlap-Save, polyphase multirate |
| **Synchronization** | Barker / Zadoff-Chu sequences, cross-correlation timing, M-th power & Kay FOE, Viterbi-Viterbi & BPS carrier phase recovery |
| **Equalization** | LMS / NLMS, RLS, CMA (blind), ZF / MMSE block — butterfly MIMO topology, Numba + JAX backends |
| **Channel Models** | AWGN (Es/N0 with SPS correction), PMD (differential group delay, Jones matrix) |
| **Soft Demapping** | LLR computation via max-log and exact log-sum-exp (JAX JIT compiled) |
| **Metrics** | EVM (%, dB), data-aided SNR, BER |
| **Spectral** | Welch PSD, frequency shift with bin-quantized mixing |
| **Frames** | `SingleCarrierFrame` with block/comb pilots, guard intervals, structure map for receiver parsing |
| **MIMO** | All operations shape-aware for `(N_channels, N_samples)` arrays |
| **Visualization** | Constellation, eye diagram, PSD, waveform, filter response, equalizer convergence |

---

## Installation

**Requires Python 3.12+** and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yourusername/commstools.git
cd commstools
uv pip install -e .
```

### GPU Support

GPU execution requires CUDA 13.x drivers. The library depends on `cupy-cuda13x` and `jax[cuda13]`, which are listed as standard dependencies. If a CUDA-capable GPU is detected at import time, new Signal objects will default to GPU placement.

On **WSL2**, CUDA libraries installed inside the venv may not be on the dynamic linker path. If CuPy or JAX fails to find `libcuda` / `libnvrtc`, add this to your `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/commstools/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
```

To force CPU-only mode regardless of hardware:

```python
from commstools import backend
backend.use_cpu_only(True)
```

---

## Quick Start

```python
from commstools import Signal
from commstools.impairments import apply_awgn

# Generate a 16-QAM signal — pulse-shaped, ready to process
# Defaults to GPU if CuPy is available, otherwise CPU
sig = Signal.qam(
    order=16,
    num_symbols=10_000,
    sps=4,
    symbol_rate=100e9,
    pulse_shape="rrc",
    rrc_rolloff=0.2,
    seed=42,
)

sig.print_info()
# Signal @ GPU | 16-QAM | sps=4 | fs=400.00 GHz | Rs=100.00 GHz | 40000 samples

# Add noise at 20 dB Es/N0
noisy = apply_awgn(sig, esn0_db=20.0)

# Visualize
noisy.plot_constellation(title="16-QAM at 20 dB Es/N0")
noisy.plot_psd()
```

---

## Core Concepts

### The Signal Object

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

Key properties automatically derived from metadata:

| Property | Derived from |
| --- | --- |
| `sps` | `sampling_rate / symbol_rate` |
| `bits_per_symbol` | `log2(mod_order)` |
| `duration` | `len(samples) / sampling_rate` |
| `num_streams` | first axis of `samples` if MIMO |

### Signal Generation

Factory methods cover the common modulation families:

```python
# PAM-4
sig_pam = Signal.pam(order=4, num_symbols=1000, sps=2, symbol_rate=50e9)

# BPSK / QPSK / 8-PSK
sig_psk = Signal.psk(order=8, num_symbols=1000, sps=4, symbol_rate=25e9, pulse_shape="rrc")

# 64-QAM with Gaussian pulse
sig_qam = Signal.qam(order=64, num_symbols=5000, sps=4, symbol_rate=50e9,
                     pulse_shape="gaussian", gaussian_bt=0.35)
```

### Multi-Backend Dispatch

Every DSP function calls `backend.dispatch(data)` which returns `(array, xp, sp)` — the data, the array module (NumPy or CuPy), and the scipy-equivalent module:

```python
from commstools.backend import dispatch

def my_dsp(samples):
    x, xp, sp = dispatch(samples)
    return xp.fft.fft(x)   # runs on CPU or GPU, same code
```

Transfer data between devices at any time:

```python
sig.to("cpu")   # returns self — chains work
sig.to("gpu")
```

### JAX Interoperability

Export samples for JAX transforms with zero-copy on GPU (DLPack):

```python
# Export (zero-copy on GPU)
jax_arr = sig.export_samples_to_jax()

# Apply any JAX transform
import jax
import jax.numpy as jnp
processed = jax.vmap(jnp.fft.fft)(jax_arr)

# Write results back — device is preserved
sig.update_samples_from_jax(processed)
```

### MIMO Support

All operations are MIMO-aware. Samples of shape `(N_channels, N_samples)` are handled everywhere:

```python
from commstools import Signal
from commstools.impairments import apply_awgn
import numpy as np

# 2×2 MIMO signal
mimo_samples = np.random.randn(2, 10000) + 1j * np.random.randn(2, 10000)
sig = Signal(samples=mimo_samples, sampling_rate=100e9, symbol_rate=25e9)

print(sig.num_streams)  # 2

noisy = apply_awgn(sig, esn0_db=15.0)
```

---

## Examples

### Pulse Shaping and Matched Filtering

```python
from commstools import Signal

sig = Signal.qam(order=16, num_symbols=2000, sps=4, symbol_rate=25e9,
                 pulse_shape="rrc", rrc_rolloff=0.2)

# Taps derived automatically from the signal's pulse_shape metadata
sig.matched_filter()

# Or supply explicit taps when needed
from commstools.filtering import rrc_taps
sig.matched_filter(taps=rrc_taps(sps=4, rolloff=0.2, span=16))
```

### Channel Simulation and Metrics

```python
from commstools import Signal
from commstools.impairments import apply_awgn
from commstools.metrics import evm, snr, ber
from commstools.mapping import demap_symbols_hard
import numpy as np

sig = Signal.qam(order=16, num_symbols=10_000, sps=1, symbol_rate=100e9, seed=0)
noisy = apply_awgn(sig, esn0_db=25.0)

evm_pct, evm_db = evm(noisy.samples, sig.samples)
snr_lin, snr_db  = snr(noisy.samples, sig.samples)
print(f"EVM: {evm_pct:.2f}%  SNR: {snr_db:.1f} dB")
```

### Synchronization

```python
from commstools.sync import (
    barker_sequence,
    estimate_timing,
    estimate_frequency_offset_mth_power,
    recover_carrier_phase_viterbi_viterbi,
    correct_frequency_offset,
    correct_carrier_phase,
)

# Frequency offset estimation and correction (blind, M-th power law)
fo_hz = estimate_frequency_offset_mth_power(rx_symbols, modulation_order=4)
corrected = correct_frequency_offset(samples, fo_hz, fs=sampling_rate)

# Carrier phase recovery (Viterbi & Viterbi)
phase_est = recover_carrier_phase_viterbi_viterbi(rx_symbols, modulation_order=4)
recovered  = correct_carrier_phase(rx_symbols, phase_est)
```

### Adaptive Equalization

```python
from commstools.equalization import lms, cma

# Supervised LMS (T/2-spaced, butterfly MIMO)
result = lms(
    x=rx_samples,          # (N_rx, N_samples) oversampled at sps=2
    train_y=tx_symbols,    # (N_tx, N_train) training symbols
    num_taps=31,
    mu=1e-3,
    mode="nlms",
)

print(result.symbols.shape)  # equalized output symbols
result.plot()                # convergence curves + tap evolution

# Blind CMA (no training data required)
result_cma = cma(rx_samples, num_taps=31, mu=1e-4, radius=1.0)
```

### Constellation Visualization

```python
from commstools.plotting import constellation, ideal_constellation

# Plot received vs ideal
constellation(rx_symbols, title="RX 16-QAM")
ideal_constellation(modulation="qam", order=16)
```

### Frame Structures

```python
from commstools import Preamble, SingleCarrierFrame

preamble = Preamble.barker(length=13)
frame = SingleCarrierFrame(
    preamble=preamble,
    num_payload_symbols=1000,
    pilot_type="comb",
    pilot_spacing=16,
    modulation="qam",
    mod_order=16,
    sps=2,
    symbol_rate=50e9,
)

# Get structure map for receiver parsing
structure = frame.get_structure_map()
print(structure.pilot_indices)
print(structure.payload_indices)
```

---

## Module Reference

| Module | Contents |
| --- | --- |
| `commstools.core` | `Signal`, `Preamble`, `SingleCarrierFrame`, `SignalInfo` |
| `commstools.backend` | `dispatch()`, `to_device()`, `to_jax()`, `from_jax()`, `use_cpu_only()` |
| `commstools.mapping` | `gray_constellation()`, `map_bits()`, `demap_symbols_hard()`, `compute_llr()` |
| `commstools.filtering` | `rrc_taps()`, `rc_taps()`, `gaussian_taps()`, `smoothrect_taps()`, `fir_filter()`, `shape_pulse()` |
| `commstools.multirate` | `resample()`, `decimate()`, `upsample()`, `decimate_to_symbol_rate()` |
| `commstools.sync` | `barker_sequence()`, `zadoff_chu_sequence()`, timing & FOE & CPR functions |
| `commstools.equalization` | `lms()`, `rls()`, `cma()`, `zf_equalizer()`, `EqualizerResult` |
| `commstools.impairments` | `apply_awgn()`, `apply_pmd()` |
| `commstools.metrics` | `evm()`, `snr()`, `ber()` |
| `commstools.spectral` | `welch_psd()`, `shift_frequency()` |
| `commstools.plotting` | `constellation()`, `eye_diagram()`, `psd()`, `time_domain()`, `filter_response()`, `equalizer_result()` |
| `commstools.helpers` | `random_bits()`, `random_symbols()`, `normalize()`, `rms()`, `format_si()` |

---

## Running Tests

```bash
# CPU (default)
uv run pytest

# GPU (requires CuPy + CUDA)
uv run pytest --device=gpu

# Both backends
uv run pytest --device=all

# Single module
uv run pytest tests/test_equalization.py -v

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
uv sync              # install all dependencies
uv run pytest        # run the test suite
uv run ruff check .  # lint
```

---

## License

MIT
