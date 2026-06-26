"""
Signal analysis and characterization.

Post-processing and diagnostic routines that operate on recovered signals to
quantify their properties, as opposed to the DSP stages that *produce* those
signals (synchronization, equalization, recovery, ...).  Functions here are
grouped by the property they characterize; new analyses can be added as
independent groups without disturbing the others.
"""

from .allan import allan_deviation
from .characterize import characterize_carrier_phase
from .drift import frequency_drift_metrics, separate_drift_phase_noise
from .linewidth import (
    fm_noise_psd,
    linewidth_beta_separation,
    linewidth_increment,
)
from .trajectory import carrier_phase_trajectory

__all__ = [
    "allan_deviation",
    "carrier_phase_trajectory",
    "characterize_carrier_phase",
    "fm_noise_psd",
    "frequency_drift_metrics",
    "linewidth_beta_separation",
    "linewidth_increment",
    "separate_drift_phase_noise",
]
