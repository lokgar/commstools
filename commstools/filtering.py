"""
Digital filtering and pulse shaping.

This module provides routines for design and application of digital filters
commonly used in communication systems. It supports both standard FIR filters
and specialized pulse-shaping filters, with high-performance execution on
both CPU and GPU backends.

Functions
---------
gaussian_taps :
    Gaussian pulse-shaping filter design.
smoothrect_taps :
    Gaussian-smoothed rectangular pulse design.
rrc_taps :
    Root Raised Cosine (RRC) filter design.
rc_taps :
    Raised Cosine (RC) filter design.
lowpass_taps, highpass_taps, bandpass_taps, bandstop_taps :
    Standard FIR filter design using the window method.
fir_filter :
    Generic FIR filtering operation via FFT convolution.
shape_pulse :
    Primary interface for applying pulse shaping to symbols.
matched_filter :
    Matched filtering operation maximizing SNR in AWGN.
"""

from typing import Any

import numpy as np
import scipy

from .backend import ArrayType, dispatch
from .logger import logger
from .multirate import expand
from .helpers import normalize

# ============================================================================
# FILTER DESIGN - TAP GENERATORS
# ============================================================================
# gaussian_taps: Gaussian filter taps
# rrc_taps: Root Raised Cosine filter taps
# rc_taps: Raised Cosine filter taps
# lowpass_taps: Low pass filter taps
# highpass_taps: High pass filter taps
# bandpass_taps: Band pass filter taps
# bandstop_taps: Band stop filter taps


def gaussian_taps(sps: float, span: int = 4, bt: float = 0.3) -> np.ndarray:
    """
    Generates Gaussian pulse-shaping filter taps.

    The Gaussian filter is typically used in GMSK/GFSK modulation to minimize
    occupied bandwidth while introducing controlled Inter-Symbol Interference (ISI).

    Parameters
    ----------
    sps : float
        Samples per symbol.
    span : int, default 4
        Total filter span in symbols. The number of taps will be `span * sps + 1`
        to ensure symmetry.
    bt : float, default 0.3
        Bandwidth-Time (BT) product. Lower values result in narrower bandwidths
        but more ISI.

    Returns
    -------
    ndarray
        Gaussian filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug(f"Generating Gaussian taps: sps={sps}, span={span}, bt={bt}")
    # Ensure odd number of taps to have a center peak
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Gaussian function
    # h(t) = (sqrt(pi)/alpha) * exp(-(pi*t/alpha)^2)
    # where alpha = sqrt(ln(2)/2)/B
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    return normalize(h, "unit_energy")


def smoothrect_taps(
    sps: int, span: int, bt: float = 1.0, pulse_width: float = 1.0
) -> ArrayType:
    """
    Generates a perfectly centered Gaussian-smoothed rectangular pulse.

    This method uses the analytical closed-form solution (Error Function)
    to avoid the 0.5 sample shift artifact typically caused by convolving
    odd/even discrete arrays.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    span : int
        Filter span in symbols. The number of taps will be approximately `span * sps`.
    bt : float, default 1.0
        Bandwidth-Time (BT) product of the Gaussian smoothing filter.
    pulse_width : float, default 1.0
        Width of the rectangular pulse in symbol periods. Use 1.0 for NRZ
        and 0.5 for RZ signaling.

    Returns
    -------
    ndarray
        Gaussian-smoothed rectangular pulse taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug(
        f"Generating SmoothRect taps: sps={sps}, span={span}, bt={bt}, width={pulse_width}"
    )
    # Ensure odd number of taps to have a center peak
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Calculate Sigma from BT
    # Relationship: B = sqrt(ln 2) / (2 * pi * sigma)
    # So sigma = sqrt(ln 2) / (2 * pi * B)
    # B = bt / pulse_width (If bt is B * pulse_width product)
    # sigma = pulse_width * sqrt(ln 2) / (2 * np.pi * bt)
    sigma = pulse_width * np.sqrt(np.log(2)) / (2 * np.pi * bt)

    # Analytical Formula (Convolved Rect and Gaussian)
    # The 'width' of the rect is pulse_width symbols
    # Rect is from -width/2 to width/2
    # The convolution of a rectangle with a Gaussian is given by the difference of erfs
    w_half = pulse_width / 2.0
    h = 0.5 * (
        scipy.special.erf((t + w_half) / (sigma * np.sqrt(2)))
        - scipy.special.erf((t - w_half) / (sigma * np.sqrt(2)))
    )

    return normalize(h, "unit_energy")


def rrc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
    """
    Generates Root Raised Cosine (RRC) filter taps.

    RRC filters are used at both the transmitter (pulse shaping) and
    receiver (matched filtering) to satisfy the Nyquist ISI criterion.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor ($\alpha$), range [0, 1].
    span : int, default 8
        Filter span in symbols.

    Returns
    -------
    ndarray
        RRC filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug(f"Generating RRC taps: sps={sps}, rolloff={rolloff}, span={span}")
    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Avoid division by zero
    # 1. t = 0
    # 2. t = +/- 1/(4*rolloff)

    # Initialize array
    h = np.zeros_like(t)

    # Case 1: t = 0
    idx_0 = np.isclose(t, 0)
    h = np.where(idx_0, 1.0 - rolloff + (4 * rolloff / np.pi), h)

    # Case 2: t = +/- 1/(4*rolloff)
    if rolloff > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (4 * rolloff))
        h = np.where(
            idx_singularity,
            (rolloff / np.sqrt(2))
            * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            ),
            h,
        )
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    # Case 3: General case
    idx_general = ~(idx_0 | idx_singularity)

    numer = np.sin(np.pi * t * (1 - rolloff)) + 4 * rolloff * t * np.cos(
        np.pi * t * (1 + rolloff)
    )
    denom = np.pi * t * (1 - (4 * rolloff * t) ** 2)

    # Avoid invalid value warning by making den safe
    denom_safe = np.where(idx_general, denom, 1.0)
    h = np.where(idx_general, numer / denom_safe, h)

    return normalize(h, "unit_energy")


def rc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> ArrayType:
    """
    Generates Raised Cosine (RC) filter taps.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor ($\alpha$), range [0, 1].
    span : int, default 8
        Filter span in symbols.

    Returns
    -------
    ndarray
        RC filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug(f"Generating RC taps: sps={sps}, rolloff={rolloff}, span={span}")
    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Avoid division by zero
    # Singularities at t = +/- 1 / (2 * rolloff)

    # Initialize array
    h = np.zeros_like(t)

    # General case mask
    # Denominator: 1 - (2 * rolloff * t)**2
    # Singularity when 2 * rolloff * |t| = 1 => |t| = 1 / (2 * rolloff)

    if rolloff > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (2 * rolloff))
        # Value at singularity: (pi / 4) * sinc(1 / (2 * rolloff))
        # sinc(x) = sin(pi * x) / (pi * x)
        # arg = 1 / (2 * rolloff)
        # val = (pi / 4) * sin(pi * arg) / (pi * arg)
        #     = (pi / 4) * sin(pi / (2 * rolloff)) * (2 * rolloff / pi)
        #     = (rolloff / 2) * sin(pi / (2 * rolloff))
        val_singularity = (rolloff / 2) * np.sin(np.pi / (2 * rolloff))
        h = np.where(idx_singularity, val_singularity, h)
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    idx_general = ~idx_singularity

    # h(t) = sinc(t) * cos(pi * alpha * t) / (1 - (2 * alpha * t)^2)
    # sinc(t) = sin(pi * t) / (pi * t) (normalized sinc)

    # To avoid t=0 in sinc division, use np.sinc which handles 0 safely
    sinc_t = np.sinc(t)
    cos_t = np.cos(np.pi * rolloff * t)
    denom = 1 - (2 * rolloff * t) ** 2

    # We masked out where denom is 0, so safe to divide where idx_general is true
    # However we compute everywhere then mask, so denom should not be 0 to avoid warning/NaN if backend evals strict
    # backend.where usually evals both branches
    # So we set denom to 1 where it is 0
    denom_safe = np.where(idx_singularity, 1.0, denom)

    res = sinc_t * cos_t / denom_safe
    h = np.where(idx_general, res, h)

    return normalize(h, "unit_energy")


def lowpass_taps(
    num_taps: int,
    cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Lowpass FIR filter using the window method.

    Parameters
    ----------
    num_taps : int
        Number of filter coefficients.
    cutoff : float
        Cutoff frequency in Hz.
    sampling_rate : float, default 1.0
        The sampling rate of the signal in Hz.
    window : str, default "hamming"
        Type of window function to apply (e.g., 'hamming', 'blackman').

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug(f"Designing Lowpass FIR: cutoff={cutoff} Hz, taps={num_taps}")
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=True
    )
    return h


def highpass_taps(
    num_taps: int,
    cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Highpass FIR filter using the window method.

    Parameters
    ----------
    num_taps : int
        Number of filter coefficients. For highpass filters, this should
        typically be an odd integer to avoid a zero at the Nyquist frequency.
    cutoff : float
        Cutoff frequency in Hz.
    sampling_rate : float, default 1.0
        The sampling rate of the signal in Hz.
    window : str, default "hamming"
        Type of window function to apply.

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug(f"Designing Highpass FIR: cutoff={cutoff} Hz, taps={num_taps}")
    # pass_zero=False for highpass
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=False
    )
    return h


def bandpass_taps(
    num_taps: int,
    low_cutoff: float,
    high_cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Bandpass FIR filter using the window method.

    Parameters
    ----------
    num_taps : int
        Number of filter coefficients.
    low_cutoff : float
        Lower cutoff frequency in Hz.
    high_cutoff : float
        Upper cutoff frequency in Hz.
    sampling_rate : float, default 1.0
        The sampling rate of the signal in Hz.
    window : str, default "hamming"
        Type of window function to apply.

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug(
        f"Designing Bandpass FIR: range=[{low_cutoff}, {high_cutoff}] Hz, taps={num_taps}"
    )
    # pass_zero=False for bandpass
    h = scipy.signal.firwin(
        num_taps,
        [low_cutoff, high_cutoff],
        window=window,
        fs=sampling_rate,
        pass_zero=False,
    )
    return h


def bandstop_taps(
    num_taps: int,
    low_cutoff: float,
    high_cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Bandstop FIR filter using the window method.

    Parameters
    ----------
    num_taps : int
        Number of filter coefficients. Should typically be odd.
    low_cutoff : float
        Lower cutoff frequency in Hz.
    high_cutoff : float
        Upper cutoff frequency in Hz.
    sampling_rate : float, default 1.0
        The sampling rate of the signal in Hz.
    window : str, default "hamming"
        Type of window function to apply.

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug(
        f"Designing Bandstop FIR: range=[{low_cutoff}, {high_cutoff}] Hz, taps={num_taps}"
    )
    # pass_zero=True for bandstop
    h = scipy.signal.firwin(
        num_taps,
        [low_cutoff, high_cutoff],
        window=window,
        fs=sampling_rate,
        pass_zero=True,
    )
    return h


# ============================================================================
# FILTERING OPERATIONS
# ============================================================================
# _ols_forward:  OLS block windowing + batch FFT (shared scaffold)
# _ols_backward: OLS batch IFFT + symmetric discard + reshape (shared scaffold)
# ols_fir_filter: Public OLS FIR convolution (long-tap / memory-bounded)
# fir_filter: Generic FIR filtering operation (short-to-medium taps)
# matched_filter: Apply matched filter (time-reversed conjugate of pulse shape)
# shape_pulse: Apply pulse shaping to symbols


def _ols_forward(samples: ArrayType, N_fft: int):
    """
    Overlap-and-save forward pass: block windowing and batch FFT.

    This is the shared OLS scaffolding used by both ``ols_fir_filter`` (SISO
    scalar convolution) and ``zf_equalizer`` (MIMO per-bin matrix multiply).
    It should be called on samples that have already been dispatched to the
    correct backend and shaped as ``(num_ch, N)``.

    Parameters
    ----------
    samples : array_like
        Input samples. Shape: ``(num_ch, N)``. Must be 2-D.
    N_fft : int
        FFT block size. Must be a power of 2 and satisfy
        ``N_fft // 4 >= filter_length`` so the causal/anti-causal guard
        regions fully contain the filter transients.

    Returns
    -------
    Y : array_like
        Batch FFT of all OLS windows. Shape: ``(num_ch, num_blocks, N_fft)``.
    meta : dict
        Scaffold parameters required by ``_ols_backward``:
        ``{'N': int, 'B': int, 'discard': int, 'num_blocks': int}``.
    """
    _, xp, _ = dispatch(samples)
    num_ch, N = samples.shape
    B = N_fft // 2        # 50 % hop — maximises block reuse
    discard = N_fft // 4  # symmetric guard: absorbs causal & anti-causal transients
    num_blocks = (N + B - 1) // B

    # Pre-pad by discard so the first valid output aligns with sample 0.
    # Post-pad to fill the last block window completely.
    pad_left = discard
    pad_right = num_blocks * B - N + discard
    samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))

    # Zero-copy window extraction via as_strided (view, not copy).
    stride = samples_padded.strides
    windows = xp.lib.stride_tricks.as_strided(
        samples_padded,
        shape=(num_ch, num_blocks, N_fft),
        strides=(stride[0], B * stride[1], stride[1]),
    )

    Y = xp.fft.fft(windows, n=N_fft, axis=-1)  # (num_ch, num_blocks, N_fft)
    meta = {"N": N, "B": B, "discard": discard, "num_blocks": num_blocks}
    return Y, meta


def _ols_backward(X_hat_f: ArrayType, meta: dict) -> ArrayType:
    """
    Overlap-and-save backward pass: batch IFFT, symmetric discard, reshape.

    Parameters
    ----------
    X_hat_f : array_like
        Frequency-domain blocks after per-bin processing.
        Shape: ``(num_ch, num_blocks, N_fft)``.
    meta : dict
        Scaffold parameters returned by ``_ols_forward``.

    Returns
    -------
    array_like
        Time-domain output trimmed to the original signal length ``N``.
        Shape: ``(num_ch, N)``.
    """
    _, xp, _ = dispatch(X_hat_f)
    N = meta["N"]
    B = meta["B"]
    discard = meta["discard"]
    N_fft = X_hat_f.shape[-1]
    num_ch = X_hat_f.shape[0]

    x_hat = xp.fft.ifft(X_hat_f, n=N_fft, axis=-1)
    # Keep the center B samples of each block (symmetric discard of guard regions).
    valid = x_hat[:, :, discard : discard + B]
    out = valid.reshape(num_ch, -1)[:, :N]
    return out


def ols_fir_filter(
    samples: ArrayType, taps: ArrayType, N_fft: int = None, center: bool = True
) -> ArrayType:
    """
    Overlap-and-save FIR filter for long-tap or large-signal convolution.

    Implements the overlap-and-save (OLS) block-processing algorithm, which
    processes the signal in fixed-size FFT blocks. This makes it suitable
    for filters with long impulse responses (e.g., chromatic dispersion
    compensation, group-delay equalizers) where a single full-signal FFT
    would be memory-prohibitive on GPU.

    For short filters on moderate-length signals, ``fir_filter`` (which
    uses scipy's FFT convolution) is equally efficient and simpler.

    Parameters
    ----------
    samples : array_like
        Input signal. Shape: ``(N,)`` for SISO or ``(C, N)`` for
        multi-channel.
    taps : array_like
        FIR filter coefficients. Shape: ``(L,)``.
    N_fft : int, optional
        FFT block size. Must be a power of 2. Defaults to
        ``max(1024, next_power_of_2(4 * L))`` so that the 25 % guard
        region is at least ``L`` samples long.
    center : bool, default True
        When ``True`` (default), the output alignment matches
        ``fir_filter`` (scipy ``mode='same'``, center-aligned at tap
        ``L // 2``).  The output at position ``n`` equals
        ``sum_k x[n + L//2 - k] * taps[k]``, which is correct for
        pulse-shaped signals where the filter group delay must be
        compensated before symbol sampling.

        When ``False``, the output is the causal linear convolution
        ``y[n] = sum_k x[n-k] * taps[k]`` (equivalent to
        ``numpy.convolve(x, taps, mode='full')[:N]``).  Use this when
        you need the raw causal impulse response (e.g. measuring filter
        step response) or when writing CD/dispersion compensation where
        the two-sided inverse filter alignment is handled externally.

    Returns
    -------
    array_like
        Filtered signal, same shape as ``samples``.

    Notes
    -----
    A symmetric guard of ``N_fft // 4`` samples is discarded from each
    block edge, so ``N_fft // 4 >= len(taps)`` must hold.

    The ``center=True`` path post-pads the input by ``L // 2`` zeros
    before OLS processing and trims the same number of leading output
    samples — a zero-copy shift that costs one extra OLS block at most.
    """
    samples, xp, _ = dispatch(samples)
    taps = xp.asarray(taps)
    is_real = not xp.iscomplexobj(samples) and not xp.iscomplexobj(taps)
    out_dtype = samples.dtype  # capture before any reshape

    # Signal drives precision: cast taps to match signal so float64 tap
    # generators do not silently upcast complex64 signals via FFT multiply.
    target_tap_dtype = samples.real.dtype if not xp.iscomplexobj(taps) else samples.dtype
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    L = len(taps)
    half = L // 2

    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]

    N = samples.shape[-1]

    if N_fft is None:
        N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())

    logger.debug(
        f"ols_fir_filter: L={L}, N={N}, N_fft={N_fft}, "
        f"num_ch={samples.shape[0]}, center={center}"
    )

    H = xp.fft.fft(taps, n=N_fft)  # frequency response of the filter

    if center:
        # Post-pad by half so the OLS can compute full_conv[half : half+N].
        # This matches scipy's mode='same' (center-aligned, group-delay compensated),
        # which is required for correct eye-opening after pulse-shaped filtering.
        samples_ext = xp.pad(samples, ((0, 0), (0, half)))
        Y, meta = _ols_forward(samples_ext, N_fft)
        X_hat_f = Y * H
        out_ext = _ols_backward(X_hat_f, meta)  # shape: (num_ch, N + half)
        out = out_ext[:, half:]                 # trim leading half → shape: (num_ch, N)
    else:
        Y, meta = _ols_forward(samples, N_fft)
        X_hat_f = Y * H
        out = _ols_backward(X_hat_f, meta)

    if is_real:
        out = out.real  # strip IFFT imaginary noise for real inputs
    elif out.dtype != out_dtype:
        out = out.astype(out_dtype)  # guard complex inputs (e.g. complex64 → complex128)
    return out[0] if was_1d else out


def fir_filter(samples: ArrayType, taps: ArrayType, axis: int = -1) -> ArrayType:
    """
    Apply a Finite Impulse Response (FIR) filter to signal samples.

    The filter is applied via FFT-based convolution for high throughput,
    efficiently handling both CPU and GPU backends.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N_samples).
    taps : array_like
        FIR filter coefficients (impulse response). Shape: (N_taps,).
    axis : int, default -1
        The axis along which the filter is applied (typically the Time axis).

    Returns
    -------
    array_like
        Filtered samples with the same shape as `samples` (mode='same').
    """
    logger.debug(
        f"Applying FIR filter via convolution ({len(taps)} taps, axis={axis})."
    )
    samples, xp, sp = dispatch(samples)

    # Ensure taps are on the correct backend
    taps = xp.asarray(taps)

    # Signal drives precision: cast taps to match signal dtype so numpy/scipy
    # type-promotion rules do not silently upcast float32/complex64 signals.
    target_tap_dtype = samples.real.dtype if not xp.iscomplexobj(taps) else samples.dtype
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    if samples.ndim > 1:
        # Ensure axis is positive
        axis = axis % samples.ndim

        new_shape = [1] * samples.ndim
        new_shape[axis] = len(taps)
        taps_nd = taps.reshape(new_shape)

        result = sp.signal.convolve(samples, taps_nd, mode="same", method="fft")
    else:
        # 1D case
        result = sp.signal.convolve(samples, taps, mode="same", method="fft")

    # Belt-and-suspenders: scipy may still promote internally (version-dependent)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)
    return result


def shape_pulse(
    symbols: ArrayType,
    sps: float,
    pulse_shape: str = "none",
    **kwargs: Any,
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence.

    Parameters
    ----------
    symbols : array_like
        Input symbol sequence. Shape: (..., N_symbols).
    sps : float
        Samples per symbol (upsampling factor).
    pulse_shape : {"none", "rect", "smoothrect", "gaussian", "rrc", "rc", "sinc"}, default "none"
        Identifier for the pulse shaping filter type.
    **kwargs : Any
        Filter-specific parameters:
        filter_span (int): Filter span in symbols (default 10).
        rrc_rolloff / rc_rolloff (float): Rolloff factor (default 0.35).
        smoothrect_bt / gaussian_bt (float): BT product (default 1.0 / 0.3).

    Returns
    -------
    array_like
        The pulse-shaped waveform at rate `sps * symbol_rate`. The peak
        absolute amplitude is normalized to 1.0.

    Notes
    -----
    This method implements pulse shaping via polyphase resampling, which is
    computationally more efficient than zero-stuffing followed by convolution.
    """
    logger.debug(f"Applying pulse shaping: {pulse_shape}")

    # Extract parameters with defaults
    filter_span = kwargs.get("filter_span", 10)
    rrc_rolloff = kwargs.get("rrc_rolloff", 0.35)
    rc_rolloff = kwargs.get("rc_rolloff", 0.35)
    smoothrect_bt = kwargs.get("smoothrect_bt", 1.0)
    gaussian_bt = kwargs.get("gaussian_bt", 0.3)
    pulse_width = kwargs.get("pulse_width", 1.0)

    # Support explicit rz flag
    if kwargs.get("rz", False):
        pulse_width = 0.5

    symbols, xp, sp = dispatch(symbols)

    if pulse_shape == "none":
        if kwargs.get("rz", False):
            logger.debug("RZ signaling requested, using rect pulse shape")
            pulse_shape = "rect"
        else:
            logger.debug("Pulse shaping disabled, expanding symbols by sps")
            return normalize(expand(symbols, int(sps), axis=-1), "peak")

    if pulse_shape == "rect":
        h = xp.ones(int(sps * pulse_width))
    elif pulse_shape == "smoothrect":
        # Note: Tap generators return NumPy arrays
        h = smoothrect_taps(
            sps, span=filter_span, bt=smoothrect_bt, pulse_width=pulse_width
        )
    elif pulse_shape == "gaussian":
        h = gaussian_taps(sps, span=filter_span, bt=gaussian_bt)
    elif pulse_shape == "rrc":
        h = rrc_taps(sps, span=filter_span, rolloff=rrc_rolloff)
    elif pulse_shape == "rc":
        h = rc_taps(sps, span=filter_span, rolloff=rc_rolloff)
    elif pulse_shape == "sinc":
        # Sinc pulse shaping is equivalent to RRC with rolloff=0
        h = rrc_taps(sps, span=filter_span, rolloff=0.0)
    else:
        raise ValueError(f"Not implemented pulse shape: {pulse_shape}")

    # Ensure h is on the correct backend and matches symbol precision.
    # Tap generators return float64; casting here prevents scipy's resample_poly
    # from promoting complex64 symbols to complex128.
    h = xp.asarray(h).astype(symbols.real.dtype)

    # Apply Pulse Shaping via Polyphase Resampling
    res = sp.signal.resample_poly(symbols, int(sps), 1, window=h, axis=-1)
    if res.dtype != symbols.dtype:
        res = res.astype(symbols.dtype)

    return normalize(res, "peak")


def matched_filter(
    samples: ArrayType,
    pulse_taps: ArrayType,
    taps_normalization: str = "unit_energy",
    normalize_output: bool = False,
    axis: int = -1,
) -> ArrayType:
    """
    Applies a matched filter to the received signal.

    The matched filter is the time-reversed complex conjugate of the pulse
    shaping filter. It maximizes the Signal-to-Noise Ratio (SNR) in the
    presence of AWGN.

    Parameters
    ----------
    samples : array_like
        Input received samples. Shape: (..., N_samples).
    pulse_taps : array_like
        Taps of the pulse-shaping filter used at the transmitter.
        Shape: (N_taps,).
    taps_normalization : {"unit_energy", "unity_gain"}, default "unit_energy"
        Designates how the matched filter taps are normalized.
    normalize_output : bool, default False
        Whether to normalize the output amplitude to 1.0.
    axis : int, default -1
        The axis along which to apply the filter.

    Returns
    -------
    array_like
        Matched filtered samples. Shape: (..., N_samples).
    """
    logger.debug(f"Applying Matched Filter (taps length={len(pulse_taps)}).")
    samples, xp, _ = dispatch(samples)

    # Matched filter is conjugate and time-reversed version of pulse
    # Ensure pulse_taps is on correct backend
    pulse_taps = xp.asarray(pulse_taps)
    matched_taps = xp.conj(pulse_taps[::-1])

    if taps_normalization == "unity_gain":
        matched_taps = normalize(matched_taps, mode="unity_gain")
    elif taps_normalization == "unit_energy":
        matched_taps = normalize(matched_taps, mode="unit_energy")
    else:
        raise ValueError(
            f"Unknown taps_normalization: {taps_normalization!r}. "
            "Use 'unity_gain' or 'unit_energy'."
        )

    # Apply filter
    output = fir_filter(samples, matched_taps, axis=axis)

    if normalize_output:
        output = normalize(output, mode="peak")

    return output
