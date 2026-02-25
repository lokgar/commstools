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
sync :
    Synchronization and timing recovery algorithms.
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

__version__ = "0.1.0"

from . import equalization, impairments, metrics, sync
from .core import Preamble, Signal, SingleCarrierFrame
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
    "impairments",
    "metrics",
    "set_log_level",
    "sync",
]

apply_default_theme()
