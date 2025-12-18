"""
Utility functions.

This module provides general helper functions used across the library:
- Array normalization (unity_gain, unit_energy, max_amplitude, average_power).
- Linear interpolation (backend-agnostic).
"""

from .backend import ArrayType, dispatch


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
