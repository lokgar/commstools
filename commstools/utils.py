"""
Utility functions.

This module provides general helper functions used across the library:
- Array normalization (unity_gain, unit_energy, max_amplitude, average_power).
- Format SI prefixes (format_si).
"""

from typing import Optional

from .backend import ArrayType, dispatch
from .logger import logger


def normalize(x: ArrayType, mode: str = "unity_gain") -> ArrayType:
    """
    Normalize array based on the specified mode.

    Args:
        x: Input array.
        mode: Normalization mode.
            'unity_gain': Sum of elements is 1.
            'unit_energy': Sum of squared magnitudes is 1.
            'max_amplitude': Maximum absolute value is 1.
            'average_power': Mean of squared magnitudes is 1.

    Returns:
        Normalized array.

    Raises:
        ValueError: If the normalization factor is zero (Numpy only).
    """
    logger.debug(f"Normalizing array (mode: {mode}).")
    x, xp, _ = dispatch(x)

    if mode == "unity_gain":
        norm_factor = xp.sum(x)
    elif mode == "unit_energy":
        norm_factor = xp.sqrt(xp.sum(xp.abs(x) ** 2))
    elif mode == "max_amplitude":
        norm_factor = xp.max(xp.abs(x))
    elif mode == "average_power":
        norm_factor = xp.sqrt(xp.mean(xp.abs(x) ** 2))
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    # Handle division by zero safely for both NumPy and CuPy
    # Avoid control flow based on data values to prevent host-device synchronization.
    safe_norm = xp.where(norm_factor == 0, 1.0, norm_factor)
    result = x / safe_norm

    # Zero out result if norm was 0 (since 0/0 or x/0 should be handled)
    # If norm_factor is 0, it means the signal is all zeros (for energy/power/max/sum)
    # So the normalized result should also be all zeros.
    return xp.where(norm_factor == 0, xp.zeros(x.shape, dtype=x.dtype), result)


def format_si(value: Optional[float], unit: str = "Hz") -> str:
    """
    Format a value with SI prefixes.

    Args:
        value: The value to format.
        unit: The unit string (e.g., 'Hz', 'Baud', 's').

    Returns:
        Formatted string (e.g., '10.00 MHz').
    """
    import numpy as np

    if value is None:
        return "None"

    if abs(value) == 0:
        return f"0.00 {unit}"

    # Standard SI prefixes
    si_units = {
        -5: "f",
        -4: "p",
        -3: "n",
        -2: "µ",
        -1: "m",
        0: "",
        1: "k",
        2: "M",
        3: "G",
        4: "T",
        5: "P",
    }

    rank = int(np.floor(np.log10(abs(value)) / 3))
    # clamp to supported range
    rank = max(min(si_units.keys()), min(rank, max(si_units.keys())))

    scaled = value / (1000.0**rank)
    return f"{scaled:.2f} {si_units[rank]}{unit}"
