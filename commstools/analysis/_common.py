"""Shared helpers and constants for the carrier-phase analysis package."""

import numpy as np

__all__ = [
    "_BETA_SLOPE",
    "_FWHM_FROM_AREA",
    "_as_2d",
    "_pairing_variance",
    "_scalar_or_array",
]

# β-separation-line slope (Di Domenico 2010): S_f(f) = (8 ln2 / π²) · f
_BETA_SLOPE = 8.0 * np.log(2.0) / (np.pi**2)
# FWHM linewidth from the integrated FM-noise area above the β-line:
# Δν = sqrt(8 ln2 · A)
_FWHM_FROM_AREA = 8.0 * np.log(2.0)


def _as_2d(arr):
    """Promote SISO ``(N,)`` to ``(1, N)``; return ``(arr2d, was_1d)``."""
    return (arr[None, :], True) if arr.ndim == 1 else (arr, False)


def _scalar_or_array(values):
    """Collapse a length-1 per-channel result to a Python float."""
    values = np.asarray(values, dtype=np.float64)
    return float(values[0]) if values.size == 1 else values


def _pairing_variance(y2, d2, xp):
    """Total wrapped phase-error increment variance for a channel pairing."""
    pe = xp.angle(y2 * xp.conj(d2)).astype(xp.float64)
    return float(xp.sum(xp.var(xp.diff(pe, axis=-1), axis=-1)))
