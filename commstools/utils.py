"""
Utility functions.

This module provides general helper functions used across the library:
- Random bit generation (`random_bits`).
- Random symbol generation (`random_symbols`).
- Array normalization (`normalize`).
- Format SI prefixes (`format_si`).
- Input array validation and coercion (`validate_array`).
- Linear interpolation (`interp1d`).
"""

from typing import Any, Optional

import numpy as np

from .backend import ArrayType, dispatch, get_array_module, is_cupy_available, to_device
from .logger import logger

try:
    import cupy as cp
except ImportError:
    cp = None


def random_bits(length: int, seed: Optional[int] = None) -> ArrayType:
    """
    Generates a sequence of random bits (0s and 1s).
    Uses numpy.random.default_rng() (PCG64 algorithm)
    for seed consistency, as cupy and numpy give different
    random sequences for the same seed.

    Args:
        length: Length of the sequence to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of bits (0s and 1s) as a NumPy or CuPy array.
    """
    logger.debug(f"Generating {length} random bits (seed={seed}).")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length)

    if is_cupy_available():
        bits = to_device(bits, "gpu")

    return bits


def random_symbols(
    num_symbols: int,
    modulation: str,
    order: int,
    seed: Optional[int] = None,
    dtype: Optional[Any] = np.complex64,
) -> ArrayType:
    """
    Generates a sequence of random modulation symbols.

    Args:
        num_symbols: Number of symbols to generate.
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order.
        seed: Random seed for reproducibility.
        dtype: Output dtype (e.g., np.complex64, np.complex128). Default: complex64.

    Returns:
        Array of complex symbols on the default backend (GPU if available).
    """
    from . import mapping

    k = int(np.log2(order))
    bits = random_bits(num_symbols * k, seed=seed)
    return mapping.map_bits(bits, modulation, order, dtype=dtype)


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
        ValueError: If the normalization mode is unknown.
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


def validate_array(
    v: Any, name: str = "array", complex_only: bool = False
) -> ArrayType:
    """
    Validates and coerces input into a backend-compatible array (NumPy or CuPy).

    Args:
        v: Input to validate (array-like, list, tuple).
        name: Name of the variable for error reporting.
        complex_only: If True, ensures the array is of complex type.

    Returns:
        The validated array.

    Raises:
        ValueError: If the input cannot be converted to a supported array type.
    """
    if v is None:
        return None

    # Coerce lists/tuples or other array-likes to numpy arrays initially
    if not isinstance(v, (np.ndarray, getattr(cp, "ndarray", type(None)))):
        try:
            v = np.asarray(v)
        except Exception:
            raise ValueError(f"Could not convert {name} of type {type(v)} to array.")

    if complex_only and not np.iscomplexobj(v):
        xp = get_array_module(v)
        v = v.astype(xp.complex128)

    return v


def interp1d(x: ArrayType, x_p: ArrayType, f_p: ArrayType, axis: int = -1) -> ArrayType:
    """
    Linear interpolation logic (future-safe replacement for scipy.interpolate.interp1d).
    interpolates f_p (values) at query points x, given sample points x_p.

    Args:
        x: Query points.
        x_p: Sample points.
        f_p: Values at sample points.
        axis: Axis along which to perform interpolation.

    Returns:
        Interpolated values.
    """
    logger.debug(f"Performing linear interpolation (axis={axis}).")
    x, xp, _ = dispatch(x)

    # Ensure other inputs are on the same backend
    x_p = xp.asarray(x_p)
    f_p = xp.asarray(f_p)

    # Move axis to end for easier handling
    f_p = xp.swapaxes(f_p, axis, -1)

    # Find indices such that xp[i-1] <= x < xp[i]
    idxs = xp.searchsorted(x_p, x)
    idxs = xp.clip(idxs, 1, len(x_p) - 1)

    # Get the bounding points
    x0 = x_p[idxs - 1]
    x1 = x_p[idxs]

    # Calculate weights
    denominator = x1 - x0
    # Avoid division by zero
    denominator[denominator == 0] = 1.0
    weights = (x - x0) / denominator

    # Get the bounding values
    # f_p is (..., T)
    # idxs is (M,)
    # We want result (..., M)

    y0 = f_p[..., idxs - 1]
    y1 = f_p[..., idxs]

    result = y0 * (1 - weights) + y1 * weights

    # Move axis back
    result = xp.swapaxes(result, axis, -1)

    return result
