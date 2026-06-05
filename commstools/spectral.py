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

from typing import Any, Optional, Tuple, Union

from .backend import ArrayType, dispatch
from .logger import logger


def shift_frequency(
    samples: ArrayType, offset: float, sampling_rate: float
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
    sampling_rate : float
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
    df = sampling_rate / n

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
    t = xp.arange(n) / sampling_rate

    # Apply mixing
    # exp(j * 2 * pi * f * t)
    # Phase is computed at float64 accuracy (xp.pi is float64), then the mixer
    # is cast to the signal's complex precision to prevent silent promotion of
    # complex64/float32 signals to complex128/float64.
    phase = 2 * xp.pi * actual_offset * t
    mixer = xp.exp(1j * phase)  # complex128
    if xp.iscomplexobj(samples):
        target_cdtype = samples.dtype
    else:
        target_cdtype = xp.complex64 if samples.dtype == xp.float32 else xp.complex128
    mixer = mixer.astype(target_cdtype)

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
    window: Union[str, Tuple[Any, ...], Any] = "hann",
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    scaling: str = "density",
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
    window : str or tuple or array_like, default "hann"
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        `noverlap = nperseg // 2`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`.
    scaling : {"density", "spectrum"}, default "density"
        Selects between computing the power spectral density ('density')
        where Pxx has units of V**2/Hz and computing the power spectrum
        ('spectrum') where Pxx has units of V**2.
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
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        average=average,
    )

    if not return_onesided:
        # Shift zero frequency to center
        # f is typically 1D array of frequencies
        f = xp.fft.fftshift(f)
        # Pxx needs shift along the frequency axis
        Pxx = xp.fft.fftshift(Pxx, axes=axis)

    return f, Pxx


def spectrogram(
    samples: ArrayType,
    sampling_rate: float,
    window: Union[str, Tuple[Any, ...], Any] = "hann",
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: Optional[Union[str, bool]] = False,
    return_onesided: Optional[bool] = None,
    scaling: str = "density",
    axis: int = -1,
    mode: str = "psd",
) -> Tuple[ArrayType, ArrayType, ArrayType]:
    """
    Computes a spectrogram with consecutive Fourier transforms.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float
        Sampling rate in Hz.
    window : str or tuple or array_like, default "hann"
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values.
    nperseg : int, default 256
        Length of each segment. A longer segment increases frequency
        resolution but also increases the variance of the estimate.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        `noverlap = nperseg // 2`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`.
    detrend : str or bool, default False
        Specifies how to detrend each segment (e.g., 'constant', 'linear').
    return_onesided : bool, optional
        If True, returns a one-sided spectrum (frequencies 0 to $f_s/2$)
        for real-valued data. For complex data, only two-sided spectra
        are supported.
    scaling : {"density", "spectrum"}, default "density"
        Selects between computing the power spectral density ('density')
        where Sxx has units of V**2/Hz and computing the power spectrum
        ('spectrum') where Sxx has units of V**2.
    axis : int, default -1
        The axis along which to compute the spectrogram.
    mode : {"psd", "complex", "magnitude", "angle", "phase"}, default "psd"
        Type of spectrogram to return. Options are 'psd', 'complex',
        'magnitude', 'angle', 'phase'.

    Returns
    -------
    f : array_like
        Array of sample frequencies.
    t : array_like
        Array of segment times.
    Sxx : array_like
        Spectrogram of the signal.

    Raises
    ------
    ValueError
        If `return_onesided` set to True for complex-valued inputs.
    """
    samples, xp, sp = dispatch(samples)
    is_complex = xp.iscomplexobj(samples)

    if return_onesided is None:
        return_onesided = not is_complex

    if is_complex and return_onesided:
        raise ValueError("Cannot compute one-sided spectrogram for complex data.")

    f, t, Sxx = sp.signal.spectrogram(
        samples,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        mode=mode,
    )

    if not return_onesided:
        # Shift zero frequency to center
        f = xp.fft.fftshift(f)
        # Sxx frequency axis is at position axis_pos in output
        ndim = samples.ndim
        axis_pos = axis % ndim
        Sxx = xp.fft.fftshift(Sxx, axes=axis_pos)

    return f, t, Sxx

