"""
Commstools: A library for digital communication signal processing.

This package provides tools for:
- Generating standard baseband waveforms (PAM, PSK, QAM).
- Simulating channel impairments.
- Performing filtering and pulse shaping.
- Visualizing signals (Time, Frequency, Eye diagrams).
- Supporting execution on both CPU (NumPy) and GPU (CuPy).
"""

import warnings

from . import baseband, impairments
from .core import Frame, Signal
from .logger import set_log_level
from .plotting import apply_default_theme

# Filter the specific warning message using a regular expression match
warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")

__all__ = [
    "Signal",
    "Frame",
    "baseband",
    "impairments",
    "set_log_level",
]

apply_default_theme()
