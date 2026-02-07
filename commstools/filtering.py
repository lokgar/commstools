"""
Digital filtering and pulse shaping.

This module implements digital filter design and application routines:
- Pulse shaping filter design (RRC, RC, Gaussian, SmoothRect).
- Standard FIR filter design (Lowpass, Highpass, Bandpass, Bandstop).
- FIR filtering and matched filtering operations.

Taps are generated on CPU as they are short.
"""

from typing import Any

import numpy as np
import scipy

from .backend import ArrayType, dispatch
from .logger import logger
from .multirate import expand
from .utils import normalize

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


def gaussian_taps(sps: float, span: int = 4, bt: float = 0.3) -> ArrayType:
    """
    Generates Gaussian filter taps.

    Args:
        sps: Samples per symbol.
        span: Total filter span in symbols.
        bt: Bandwidth-Time product.

    Returns:
        Gaussian filter taps with unity gain normalization.
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

    return normalize(h, "unity_gain")


def smoothrect_taps(
    sps: int, span: int, bt: float = 1.0, pulse_width: float = 1.0
) -> ArrayType:
    """
    Generates a perfectly centered Gaussian-smoothed rectangular pulse
    using the analytical closed-form solution (Error Function).

    This avoids the 0.5 sample shift artifact caused by convolving
    odd/even discrete arrays.

    Args:
        sps: Samples per symbol.
        span: Filter span in symbols.
        bt: Bandwidth-Time product of the Gaussian filter.
        pulse_width: Width of the rectangular pulse in symbol periods.
                     Default is 1.0 (NRZ), for RZ use 0.5.

    Returns:
        Centered pulse shaping taps with unity gain normalization.
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

    return normalize(h, "unity_gain")


def rrc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> ArrayType:
    """
    Generates Root Raised Cosine (RRC) filter taps.

    Args:
        sps: Samples per symbol.
        rolloff: Roll-off factor (0 to 1).
        span: Total filter span in symbols.

    Returns:
        RRC filter taps with unity gain normalization.
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

    return normalize(h, "unity_gain")


def rc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> ArrayType:
    """
    Generates Raised Cosine (RC) filter taps.

    Args:
        sps: Samples per symbol.
        rolloff: Roll-off factor (0 to 1).
        span: Total filter span in symbols.

    Returns:
        RC filter taps with unity gain normalization.
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

    return normalize(h, "unity_gain")


def lowpass_taps(
    num_taps: int,
    cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Lowpass FIR filter.

    Args:
        num_taps: Number of filter coefficients.
        cutoff: Cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        window: Window function type.

    Returns:
        Filter taps.
    """
    logger.debug(f"Designing Lowpass FIR: cutoff={cutoff} Hz, taps={num_taps}")
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=True
    )
    return normalize(h, "unity_gain")


def highpass_taps(
    num_taps: int,
    cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Highpass FIR filter.

    Args:
        num_taps: Number of filter coefficients (should be odd).
        cutoff: Cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        window: Window function type.

    Returns:
        Filter taps.
    """
    logger.debug(f"Designing Highpass FIR: cutoff={cutoff} Hz, taps={num_taps}")
    # pass_zero=False for highpass
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=False
    )
    return normalize(h, "unity_gain")


def bandpass_taps(
    num_taps: int,
    low_cutoff: float,
    high_cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Bandpass FIR filter.

    Args:
        num_taps: Number of filter coefficients.
        low_cutoff: Lower cutoff frequency in Hz.
        high_cutoff: Upper cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        window: Window function type.

    Returns:
        Filter taps.
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
    return normalize(h, "unity_gain")


def bandstop_taps(
    num_taps: int,
    low_cutoff: float,
    high_cutoff: float,
    sampling_rate: float = 1.0,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Bandstop FIR filter.

    Args:
        num_taps: Number of filter coefficients (should be odd).
        low_cutoff: Lower cutoff frequency in Hz.
        high_cutoff: Upper cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        window: Window function type.

    Returns:
        Filter taps.
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
    return normalize(h, "unity_gain")


# ============================================================================
# FILTERING OPERATIONS
# ============================================================================
# fir_filter: Generic FIR filtering operation
# matched_filter: Apply matched filter (time-reversed conjugate of pulse shape)
# shape_pulse: Apply pulse shaping to symbols


def fir_filter(samples: ArrayType, taps: ArrayType, axis: int = -1) -> ArrayType:
    """
    Apply FIR filter via convolution.

    Args:
        samples: Input sample array.
        taps: FIR filter taps (impulse response, should be odd length).
        axis: Axis along which to apply the filter.

    Returns:
        Filtered samples.
    """
    logger.debug(
        f"Applying FIR filter via convolution ({len(taps)} taps, axis={axis})."
    )
    samples, xp, sp = dispatch(samples)
    # Ensure taps are on the correct backend
    taps = xp.asarray(taps)

    # Handle N-D convolution
    # If samples is N-D, we need to reshape taps to broadcast correctly
    # or iterate.
    # sc.signal.convolve computes N-dimensional convolution.
    # To filter along one axis only, we can use 1D taps expanded to matching dimensions.

    if samples.ndim > 1:
        # Construct slices to expand taps
        # E.g. if samples is (Time, Channel) and axis=0,
        # taps should be (Taps, 1) -> broadcasting will apply convolution along axis 0
        # Wait, sp.signal.convolve broadcasts correctly?
        # Typically we rely on convolve1d or explicit reshaping.
        # But 'convolve' does N-D convolution.
        # If we reshape taps to be (Taps, 1), and convolve with (Time, Channel),
        # it will convolve axis 0 with taps, and axis 1 with 1 (identity).

        # Ensure axis is positive
        axis = axis % samples.ndim

        new_shape = [1] * samples.ndim
        new_shape[axis] = len(taps)
        taps_nd = taps.reshape(new_shape)

        return sp.signal.convolve(samples, taps_nd, mode="same", method="fft")
    else:
        # 1D case
        return sp.signal.convolve(samples, taps, mode="same", method="fft")


def shape_pulse(
    symbols: ArrayType,
    sps: float,
    pulse_shape: str = "none",
    **kwargs: Any,
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence.

    Args:
        symbols: Input symbol array.
        sps: Samples per symbol (upsampling factor).
        pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
        **kwargs: Pulse shaping parameters:
            filter_span (int): Filter span in symbols (default: 10).
            rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
            rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
            smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
            gaussian_bt (float): BT product for Gaussian filter (default: 0.3).

    Returns:
        The shaped sample array at rate (sps * symbol_rate).
    """
    logger.debug(f"Applying pulse shaping: {pulse_shape}")

    # Extract parameters with defaults
    filter_span = kwargs.get("filter_span", 10)
    rrc_rolloff = kwargs.get("rrc_rolloff", 0.35)
    rc_rolloff = kwargs.get("rc_rolloff", 0.35)
    smoothrect_bt = kwargs.get("smoothrect_bt", 1.0)
    gaussian_bt = kwargs.get("gaussian_bt", 0.3)
    pulse_width = 1.0  # Hardcoded to 1.0 (NRZ)

    symbols, xp, sp = dispatch(symbols)

    # Determine processing axis
    axis = -1

    if pulse_shape == "none":
        logger.info("Pulse shaping disabled, expanding symbols by sps")
        return normalize(expand(symbols, int(sps), axis=axis), "max_amplitude")

    elif pulse_shape == "rect":
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

    # Ensure h is on the correct backend
    h = xp.asarray(h)

    # Apply Pulse Shaping via Polyphase Resampling
    # efficient_polyphase_resample handles CuPy stability workaround for multidimensional arrays
    from .multirate import polyphase_resample

    res = polyphase_resample(symbols, int(sps), 1, window=h, axis=axis)

    return normalize(res, "max_amplitude")


def matched_filter(
    samples: ArrayType,
    pulse_taps: ArrayType,
    taps_normalization: str = "unity_gain",
    normalize_output: bool = False,
    axis: int = -1,
) -> ArrayType:
    """
    Apply matched filter (time-reversed conjugate of pulse shape).

    Args:
        samples: Received sample array.
        pulse_taps: Pulse shape filter taps.
        taps_normalization: Normalization to apply to the matched filter taps.
                            Options: 'unity_gain', 'unit_energy'.
        normalize_output: If True, normalizes the output samples to have a maximum
                          absolute value of 1.0.
        axis: Axis along which to apply the filter.

    Returns:
        Matched filtered samples.
    """
    logger.debug(f"Applying Matched Filter (taps length={len(pulse_taps)}).")
    samples, xp, _ = dispatch(samples)

    # Matched filter is conjugate and time-reversed version of pulse
    # Ensure pulse_taps is on correct backend
    pulse_taps = xp.asarray(pulse_taps)
    matched_taps = xp.conj(pulse_taps[::-1])

    if taps_normalization == "unity_gain" or taps_normalization == "gain":
        matched_taps = normalize(matched_taps, mode="unity_gain")
    elif taps_normalization == "unit_energy" or taps_normalization == "energy":
        matched_taps = normalize(matched_taps, mode="unit_energy")
    else:
        raise ValueError(f"Not implemented taps normalization: {taps_normalization}")

    # Apply filter
    output = fir_filter(samples, matched_taps, axis=axis)

    if normalize_output:
        output = normalize(output, mode="max_amplitude")

    return output
