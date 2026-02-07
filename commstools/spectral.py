"""
Spectral tools.

This module provides functions for spectral-related operations:
- Frequency shift.
- PSD estimation (Welch's method).
"""

from typing import Optional, Tuple, Union

from .backend import ArrayType, dispatch
from .logger import logger


def shift_frequency(
    samples: ArrayType, offset: float, fs: float
) -> Tuple[ArrayType, float]:
    """
    Apply a frequency offset to the signal.

    Shifts the signal spectrum by `offset` Hz.
    To maintain phase continuity (circularity), the applied offset is quantized
    to the frequency resolution of the signal (fs / N).

    Args:
        samples: Input signal samples.
        offset: Desired frequency offset in Hz.
        fs: Sampling rate in Hz.

    Returns:
        Tuple of (shifted_samples, actual_offset).
    """
    samples, xp, _ = dispatch(samples)

    # Axis -1 is time
    n = samples.shape[-1]
    df = fs / n

    # Quantize offset to nearest bin to ensure phase continuity
    k = xp.round(offset / df)
    actual_offset = k * df

    if not xp.isclose(offset, actual_offset):
        logger.warning(
            f"Requested offset {offset:.3f} Hz quantized to {actual_offset:.3f} Hz "
            f"(step {df:.3f} Hz) to maintain phase continuity."
        )
    else:
        logger.debug(f"Applying frequency offset: {actual_offset:.3f} Hz.")

    # Time vector
    t = xp.arange(n) / fs

    # Apply mixing
    # exp(j * 2 * pi * f * t)
    phase = 2 * xp.pi * actual_offset * t
    mixer = xp.exp(1j * phase)

    # Broadcast mixer if samples is multidimensional (N_channels, N_samples)
    if samples.ndim > 1:
        # Reshape mixer to (1, N) from (N,)
        mixer = mixer.reshape(1, -1)

    return samples * mixer, float(actual_offset)


def welch_psd(
    samples: ArrayType,
    sampling_rate: float,
    nperseg: int = 256,
    detrend: Optional[Union[str, bool]] = False,
    average: Optional[str] = "mean",
    return_onesided: Optional[bool] = None,
    axis: int = -1,
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
        axis: Axis along which the PSD is computed. Defaults to -1 (Time-Last).

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
        axis=axis,
    )

    if not return_onesided:
        # Shift zero frequency to center
        # f is typically 1D array of frequencies
        f = xp.fft.fftshift(f)
        # Pxx needs shift along the frequency axis
        Pxx = xp.fft.fftshift(Pxx, axes=axis)

    return f, Pxx
