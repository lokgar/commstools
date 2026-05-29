"""
`commstools` is a high-performance library for simulating and analyzing
digital communication systems. It provides a unified API for generating,
transforming, and assessing signals across diverse computational backends
(CPU, GPU, and JAX).

Main Features
-------------
- **Signal Abstractions**: Unified `Signal` and `SingleCarrierFrame` containers.
- **Modulation**: Support for PAM, PSK, and QAM (NRZ/RZ) with Gray coding.
- **Impairments**: Simulation of AWGN, Phase Noise, and Frequency Offset.
- **Synchronization**: Time and frequency synchronization algorithms.
- **Execution backends**: Transparent NumPy, CuPy, and JAX support.

Subpackages
-----------
core :
    Primary data structures and signal analysis methods.
impairments :
    Channel models and signal degradation effects.
metrics :
    Quality assessment tools (BER, SNR, EVM).
timing :
    Timing synchronization: preamble sequences and fractional delay estimation.
frequency :
    Frequency offset estimation and correction algorithms.
recovery :
    Carrier phase recovery and cycle-slip correction.
mapping :
    Bits-to-symbols and LLR demapping logic.
filtering :
    Digital filter design and pulse shaping.
multirate :
    Resampling and polyphase filtering.
equalization :
    Adaptive and block channel equalization (LMS, RLS, CMA, ZF/MMSE).
spectral :
    Frequency domain analysis and transformations.
"""

import warnings

__version__ = "3.4.1"

from . import equalization, frequency, impairments, metrics, recovery, timing
from .core import Preamble, Signal, SingleCarrierFrame
from .io import load_npz, save_npz
from .logger import set_log_level
from .plotting import apply_default_theme

# Filter the specific warning message using a regular expression match
warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")

__all__ = [
    "__version__",
    "Preamble",
    "Signal",
    "SingleCarrierFrame",
    "equalization",
    "frequency",
    "impairments",
    "load_npz",
    "metrics",
    "recovery",
    "save_npz",
    "set_log_level",
    "timing",
]

apply_default_theme()
