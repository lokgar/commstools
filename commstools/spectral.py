"""
Spectral analysis and frequency-domain processing.

This module provides high-performance routines for spectral estimation and
manipulation, optimized for both CPU and GPU backends. It supports Welch's
Power Spectral Density (PSD) method and phase-continuous frequency shifting.

Functions
---------
shift_frequency :
    Applies a complex frequency shift (mixing) to a signal.
add_pilot_tone :
    Superimposes a CW pilot tone for pilot-tone-aided carrier phase recovery.
welch_psd :
    Estimates the Power Spectral Density using Welch's method.
"""

import math
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


def add_pilot_tone(
    samples: ArrayType,
    sampling_rate: float,
    frequency_hz: float,
    power_ratio_db: float = -15.0,
    phase_init: float = 0.0,
    renormalize: bool = False,
) -> Tuple[ArrayType, float]:
    r"""
    Add a continuous-wave (CW) pilot tone to a baseband waveform.

    Superimposes a complex exponential
    :math:`a\,e^{\,j(2\pi f_p n / f_s + \phi_0)}` on the oversampled samples.
    The tone propagates through the same channel and local oscillator as the
    data, so it acquires the **same** carrier frequency offset and phase noise.
    At the receiver, isolating the tone and reading its phase recovers that
    common phase directly — see
    :func:`~commstools.recovery.recover_carrier_phase_pilot_tone` (recovery)
    and :func:`~commstools.frequency.find_bias_tone` (tone localisation).

    This is a **transmit-side** operation, intended to be applied to a
    pulse-shaped, oversampled waveform (``sps > 1``), *before* any channel
    impairments.  Place the tone in a guard band just outside the occupied
    signal bandwidth so it can be cleanly separated by a narrowband filter at
    the receiver:

    .. math::

        \tfrac{(1+\beta)}{2}\,R_s \;<\; |f_p| \;<\; \tfrac{f_s}{2}

    where :math:`\beta` is the roll-off and :math:`R_s` the symbol rate.  A DC
    tone (``frequency_hz=0``) is also possible (a "residual carrier") but
    overlaps the data spectrum and incurs an SNR penalty unless the data has a
    spectral null at DC.

    Parameters
    ----------
    samples : array_like
        Complex baseband samples. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
        The tone is added to every channel, each scaled to its own power.
    sampling_rate : float
        Sampling rate :math:`f_s` in Hz.
    frequency_hz : float
        Requested tone frequency :math:`f_p` in Hz, in ``(-f_s/2, f_s/2)``.
        Quantized to the nearest FFT bin ``f_s/N`` (see Notes) — the **actual**
        applied frequency is returned.
    power_ratio_db : float, default -15.0
        Pilot-to-signal power ratio (PSR) in dB:
        ``10·log10(P_tone / P_signal)``, where ``P_signal`` is the mean
        per-channel sample power of the input.  Typical range ``-20`` to
        ``-10`` dB — enough tone SNR for reliable phase tracking while keeping
        the data SNR penalty (``10·log10(1 + 10^(PSR/10))``) below a few tenths
        of a dB.
    phase_init : float, default 0.0
        Initial tone phase :math:`\phi_0` in radians.  Acts as a known phase
        reference; it appears as a constant offset in the recovered phase and
        is absorbed by the usual post-CPR ambiguity resolution.
    renormalize : bool, default False
        If ``True``, rescale each channel after adding the tone so its mean
        power matches the input, preserving the library power-normalisation
        invariant (``E[|x|²] = 1/sps``).  The data then gives up a sliver of
        power to the tone (the SNR penalty above).  If ``False`` (default), the
        tone is added on top and the total power rises by ``1 + 10^(PSR/10)``.

    Returns
    -------
    samples : array_like
        Samples with the pilot tone added, same shape, dtype, and backend as
        the input.
    actual_frequency : float
        The grid-quantized tone frequency in Hz actually applied (see Notes).
        Store this (e.g. in :attr:`Signal.pilot_tone_hz`) and pass it to the
        receiver, since it — not the requested value — is where the tone sits.

    Notes
    -----
    **Grid quantization (buffer-periodic playback).**  The requested frequency
    is snapped to the nearest FFT bin, :math:`f_s/N`, exactly as
    :func:`shift_frequency` snaps a mixing offset.  This makes the tone complete
    an integer number of cycles over the ``N``-sample buffer, so it is
    *seamless across the loop boundary* when an AWG/DAC plays the buffer
    repeatedly — an off-grid tone would jump in phase at each wrap and radiate
    spurs spaced at :math:`f_s/N`.  The quantization error is at most
    :math:`f_s/2N`.

    The per-channel tone amplitude is

    .. math::

        a = \sqrt{P_\text{signal} \cdot 10^{\,\text{PSR}/10}}, \qquad
        P_\text{signal} = \tfrac{1}{N}\sum_n |x[n]|^2 .

    The phase ramp is accumulated in ``float64`` and wrapped to
    :math:`[-\pi, \pi)` before the complex exponential, matching the precision
    pattern of :func:`~commstools.frequency.correct_static_frequency_offset` so
    that ``complex64`` waveforms do not suffer trig argument-reduction error for
    large ``N``.

    Examples
    --------
    >>> # Tone in the guard band of a 16-QAM RRC waveform at sps=4
    >>> f_p = 0.62 * sig.symbol_rate            # just outside (1+β)/2 · Rs
    >>> sig.samples, sig.pilot_tone_hz = add_pilot_tone(
    ...     sig.samples, sig.sampling_rate, f_p, power_ratio_db=-15.0
    ... )
    """
    if not (-sampling_rate / 2.0 < frequency_hz < sampling_rate / 2.0):
        raise ValueError(
            f"frequency_hz={frequency_hz} must lie in (-fs/2, fs/2) = "
            f"(±{sampling_rate / 2.0:.3g}) Hz."
        )

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)
    C, N = samples.shape

    # Snap to the FFT bin grid so the tone is buffer-periodic (loop-seamless on
    # an AWG/DAC), mirroring shift_frequency's quantization.
    df = sampling_rate / N
    actual_frequency = float(round(frequency_hz / df) * df)
    if abs(actual_frequency - frequency_hz) > 1e-12 * max(1.0, abs(frequency_hz)):
        logger.warning(
            f"add_pilot_tone: requested {frequency_hz:.3f} Hz quantized to "
            f"{actual_frequency:.3f} Hz (grid step fs/N={df:.3f} Hz) for "
            f"buffer-periodic (loop-seamless) playback."
        )

    # Per-channel signal power and the tone amplitude that realises the PSR.
    p_signal = xp.mean(xp.abs(samples) ** 2, axis=-1, keepdims=True)  # (C, 1) float
    psr_lin = 10.0 ** (power_ratio_db / 10.0)
    amp = xp.sqrt(p_signal * psr_lin)  # (C, 1)

    # Phase ramp in float64; wrap to [-π, π) before exp so complex64 targets
    # avoid argument-reduction error on long ramps (cf. correct_static_frequency_offset).
    two_pi = 2.0 * xp.pi
    n = xp.arange(N, dtype=xp.float64)  # (N,)
    phase = two_pi * actual_frequency * n / sampling_rate + phase_init  # (N,) float64
    phase = phase - xp.round(phase / two_pi) * two_pi

    dtype_real = xp.float32 if samples.dtype == xp.complex64 else xp.float64
    tone = xp.exp(1j * phase.astype(dtype_real)).astype(samples.dtype)  # (N,)
    out = samples + amp.astype(samples.dtype) * tone[None, :]  # (C, N)

    if renormalize:
        # Restore each channel to its original mean power.
        p_out = xp.mean(xp.abs(out) ** 2, axis=-1, keepdims=True)  # (C, 1)
        out = out * xp.sqrt(p_signal / p_out).astype(samples.dtype)

    logger.info(
        f"add_pilot_tone: f_p={actual_frequency:.3g} Hz, PSR={power_ratio_db:.1f} dB "
        f"(amp/√P_sig={math.sqrt(psr_lin):.3g}), phase_init={phase_init:.3g} rad, "
        f"renormalize={renormalize} [C={C}, N={N}]"
    )

    if was_1d:
        return out[0], actual_frequency
    return out, actual_frequency


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
