"""
Spectral analysis and frequency-domain processing.

This module provides high-performance routines for spectral estimation and
manipulation, optimized for both CPU and GPU backends. It supports Welch's
Power Spectral Density (PSD) method and phase-continuous frequency shifting.

Functions
---------
shift_frequency :
    Applies a complex frequency shift (mixing) to a signal.
welch_psd :
    Estimates the Power Spectral Density using Welch's method.
"""

from typing import Optional, Tuple, Union

from .backend import ArrayType, dispatch
from .logger import logger


def shift_frequency(
    samples: ArrayType, offset: float, fs: float
) -> Tuple[ArrayType, float]:
    """
    Applies a frequency offset (complex mixing) to a signal.

    This function shifts the signal spectrum by a specified offset in Hz
    by multiplying the samples with a complex phasor:
    $s_{shifted}(t) = s(t) \\cdot e^{j 2 \\pi f_{offset} t}$

    To maintain phase continuity and prevent spectral leakage when the
    signal is treated as periodic (e.g., in circular convolution or
    FFT-based operations), the applied offset is quantized to the
    fundamental frequency resolution of the signal ($\\Delta f = f_s / N$).

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    offset : float
        Target frequency shift in Hz. Positive values shift the spectrum
        towards higher frequencies.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    shifted_samples : array_like
        The frequency-shifted signal on the same backend as the input.
    actual_offset : float
        The actual quantized frequency shift applied to the signal.

    Notes
    -----
    The quantization ensures that the applied shift corresponds to an
    integer number of cycles over the signal duration, which is critical
    for preserving the circularity of the signal's phase.
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

    # Broadcast mixer to match samples shape: (1, ..., 1, N)
    if samples.ndim > 1:
        mixer = mixer.reshape((1,) * (samples.ndim - 1) + (-1,))

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
    Estimates the Power Spectral Density (PSD) using Welch's method.

    Welch's method provides a lower-variance estimate of the PSD
    compared to a raw periodogram by averaging spectra computed over
    overlapping segments of the signal.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float
        Sampling rate in Hz.
    nperseg : int, default 256
        Length of each segment. A longer segment increases frequency
        resolution but also increases the variance of the estimate.
    detrend : str or bool, default False
        Specifies how to detrend each segment (e.g., 'constant', 'linear').
    average : {"mean", "median"}, default "mean"
        Method to use for averaging segments. Median is more robust to
        transient outliers.
    return_onesided : bool, optional
        If True, returns a one-sided spectrum (frequencies 0 to $f_s/2$)
        for real-valued data. For complex data, only two-sided spectra
        (frequencies $-f_s/2$ to $f_s/2$) are supported.
    axis : int, default -1
        The axis along which to compute the PSD.

    Returns
    -------
    f : array_like
        Array of sample frequencies.
    Pxx : array_like
        Power spectral density (linear scale, units: $V^2/Hz$).

    Raises
    ------
    ValueError
        If `return_onesided` set to True for complex-valued inputs.
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
