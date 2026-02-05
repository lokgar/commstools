"""
Spectral analysis tools.

This module provides functions for spectral estimation.
"""

from typing import Optional, Tuple, Union

from .backend import ArrayType, dispatch


def welch_psd(
    samples: ArrayType,
    sampling_rate: float,
    nperseg: int = 256,
    detrend: Optional[Union[str, bool]] = False,
    average: Optional[str] = "mean",
    return_onesided: Optional[bool] = None,
) -> Tuple[ArrayType, ArrayType]:
    """
    Compute the Power Spectral Density (PSD) using Welch's method.

    Args:
        samples: Input signal samples.
        sampling_rate: Sampling rate in Hz.
        nperseg: Length of each segment.
        detrend: Detrend method (default: False).
        average: Averaging method (default: 'mean').
        return_onesided: If True, return a one-sided spectrum for real data.
                         If False, return a two-sided spectrum.
                         If None (default), behavior depends on input type:
                         - Complex data: False (two-sided, centered)
                         - Real data: True (one-sided)

    Returns:
        Tuple of (frequency_axis, psd_values).
    """
    samples, xp, sp = dispatch(samples)
    is_complex = xp.iscomplexobj(samples)

    if return_onesided is None:
        return_onesided = not is_complex

    # scipy.signal.welch returns onesided by default for real, two-sided for complex
    # unless return_onesided is explicitly set.
    # Note: scipy's return_onesided argument serves to force one-sided for real data.
    # It cannot force one-sided for complex data (always raises error).
    # For complex data, it always returns two-sided (0 to fs).

    if is_complex and return_onesided:
        raise ValueError("Cannot compute one-sided PSD for complex data.")

    f, Pxx = sp.signal.welch(
        samples,
        fs=sampling_rate,
        nperseg=nperseg,
        detrend=detrend,
        average=average,
        return_onesided=return_onesided,
    )

    if not return_onesided:
        # Shift zero frequency to center
        f = xp.fft.fftshift(f)
        Pxx = xp.fft.fftshift(Pxx)

    return f, Pxx
