import numpy as np
from typing import Any
from ..core.backend import get_backend, ArrayType


# ============================================================================
# FILTER DESIGN - TAP GENERATORS
# ============================================================================


def boxcar_taps(sps: float) -> ArrayType:
    """
    Generates boxcar (rectangular) filter taps.

    Args:
        sps: Samples per symbol.

    Returns:
        Array of ones with length equal to sps.
    """
    backend = get_backend()
    return backend.ones(int(sps))


def gaussian_taps(sps: float, bt: float = 0.3, span: int = 2) -> ArrayType:
    """
    Generates Gaussian filter taps.

    Args:
        sps: Samples per symbol.
        bt: Bandwidth-Time product.
        span: Filter span in symbols.

    Returns:
        Gaussian filter taps.
    """
    backend = get_backend()
    # Using numpy for coefficient calculation, then converting to backend

    t = np.arange(-span * sps, span * sps + 1) / sps
    # Gaussian function: h(t) = (sqrt(pi)/a) * exp(-(pi*t/a)^2) where a = sqrt(ln(2)/2)/B
    # Simplified for comms usually:
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    # Normalize: sum(h) = sps (preserves amplitude after convolution with expanded sequence)
    h = h / np.sum(h) * sps

    return backend.array(h)


def rrc_taps(sps: float, rolloff: float = 0.35, span: int = 4) -> ArrayType:
    """
    Generates Root Raised Cosine (RRC) filter taps.

    Args:
        sps: Samples per symbol.
        rolloff: Roll-off factor (0 to 1).
        span: Filter span in symbols.

    Returns:
        RRC filter taps.
    """
    backend = get_backend()

    # Use numpy for calculation
    t = np.arange(-span * sps, span * sps + 1) / sps

    # Avoid division by zero
    # 1. t = 0
    # 2. t = +/- 1/(4*rolloff)

    # Initialize array
    h = np.zeros_like(t)

    # Case 1: t = 0
    idx_0 = np.isclose(t, 0)
    h[idx_0] = 1.0 - rolloff + (4 * rolloff / np.pi)

    # Case 2: t = +/- 1/(4*rolloff)
    if rolloff > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (4 * rolloff))
        h[idx_singularity] = (rolloff / np.sqrt(2)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
        )
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    # Case 3: General case
    idx_general = ~(idx_0 | idx_singularity)
    t_gen = t[idx_general]

    num = np.sin(np.pi * t_gen * (1 - rolloff)) + 4 * rolloff * t_gen * np.cos(
        np.pi * t_gen * (1 + rolloff)
    )
    den = np.pi * t_gen * (1 - (4 * rolloff * t_gen) ** 2)
    h[idx_general] = num / den

    # Normalize to unit energy (standard for matched filtering)
    h = h / np.sqrt(np.sum(h**2))

    return backend.array(h)


def sinc_taps(sps: float, bandwidth: float = 0.5, span: int = 8) -> ArrayType:
    """
    Generates ideal sinc interpolation filter taps.

    Args:
        sps: Samples per symbol.
        bandwidth: Normalized bandwidth (0-1), where 1 is Nyquist.
        span: Filter span in symbols.

    Returns:
        Windowed sinc filter taps.
    """
    backend = get_backend()

    # Create time vector
    t = np.arange(-span * sps, span * sps + 1) / sps

    # Sinc function
    h = np.sinc(t * bandwidth)

    # Apply Hamming window
    n = np.arange(len(h))
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (len(h) - 1))
    h = h * window

    # Normalize to preserve DC gain
    h = h / np.sum(h) * sps

    return backend.array(h)


def get_taps(filter_type: str, sps: float, **kwargs: Any) -> ArrayType:
    """
    Factory function to generate filter taps.

    Args:
        filter_type: Filter type ('none', 'boxcar', 'gaussian', 'rrc', 'sinc').
        sps: Samples per symbol.
        **kwargs: Additional arguments for specific filters (e.g., rolloff, bt, span).

    Returns:
        Filter taps.
    """
    filter_type = filter_type.lower()
    if filter_type == "none":
        backend = get_backend()
        return backend.ones(1)
    elif filter_type == "boxcar":
        return boxcar_taps(sps)
    elif filter_type == "gaussian":
        return gaussian_taps(sps, **kwargs)
    elif filter_type == "rrc":
        return rrc_taps(sps, **kwargs)
    elif filter_type == "sinc":
        return sinc_taps(sps, **kwargs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


# ============================================================================
# CORE BUILDING BLOCKS - SIGNAL PROCESSING PRIMITIVES
# ============================================================================
# fir_filter: Generic FIR filtering operation
# expand: Zero-insertion upsampling
# upsample: Complete upsampling (expand + fir_filter)
# decimate: Anti-aliasing filter + downsampling
# resample: Rational rate conversion (up/down)


def fir_filter(
    samples: ArrayType,
    taps: ArrayType,
    mode: str = "same",
) -> ArrayType:
    """
    Apply FIR filter to signal via convolution.

    Generic filtering function used throughout the DSP pipeline.
    This is the fundamental operation for pulse shaping, matched filtering,
    and interpolation.

    Args:
        samples: Input sample array.
        taps: FIR filter taps (impulse response).
        mode: Convolution mode ('same', 'full', 'valid').
            'same': Output same length as input (centered).
            'full': Full convolution (length = len(samples) + len(taps) - 1).
            'valid': Only where sequences fully overlap.

    Returns:
        Filtered samples.
    """
    backend = get_backend()
    return backend.convolve(samples, taps, mode=mode, method="fft")


def expand(symbols: ArrayType, factor: int) -> ArrayType:
    """
    Zero-insertion: Insert (factor-1) zeros between each symbol.

    This is the first step in upsampling. The signal is expanded by inserting
    zeros, which creates spectral images that should be filtered out.

    Args:
        symbols: Input symbol array.
        factor: Expansion factor (samples per symbol).

    Returns:
        Expanded array with zeros inserted (length = len(symbols) * factor).
    """
    backend = get_backend()
    return backend.expand(symbols, int(factor))


def upsample(
    samples: ArrayType,
    factor: int,
    filter_type: str = "sinc",
    **kwargs: Any,
) -> ArrayType:
    """
    Upsample signal: Expansion (zero-insertion) + anti-imaging filtering.

    Increases sample rate by inserting zeros and applying lowpass filter
    to suppress spectral images.

    Args:
        samples: Input sample array.
        factor: Upsampling factor.
        filter_type: Filter type for interpolation ('sinc', 'rrc', 'gaussian').
        **kwargs: Additional arguments for filter design.

    Returns:
        Upsampled samples at (factor * original_rate).
    """
    # Expand signal (zero-insertion)
    expanded = expand(samples, factor)

    # Design and apply interpolation filter
    taps = get_taps(filter_type, float(factor), **kwargs)
    return fir_filter(expanded, taps)


def decimate(
    samples: ArrayType,
    factor: int,
    filter_type: str = "fir",
    **kwargs: Any,
) -> ArrayType:
    """
    Decimate signal: Anti-aliasing filter followed by downsampling.

    Reduces the sample rate by filtering to remove high-frequency content
    (which would alias) and then keeping every Nth sample.

    Args:
        samples: Input sample array.
        factor: Decimation factor.
        filter_type: Filter type ('fir', 'iir'). Only 'fir' currently supported.
        **kwargs: Additional filter parameters.

    Returns:
        Decimated samples at (original_rate / factor).
    """
    backend = get_backend()
    zero_phase = kwargs.get("zero_phase", True)
    return backend.decimate(
        samples, int(factor), ftype=filter_type, zero_phase=zero_phase
    )


def resample(
    samples: ArrayType,
    up: int,
    down: int,
    **kwargs: Any,
) -> ArrayType:
    """
    Rational resampling: Upsample by 'up', downsample by 'down'.

    Implements efficient polyphase filtering for arbitrary rational rate conversion.
    New rate = original_rate * (up / down).

    Args:
        samples: Input sample array.
        up: Upsampling factor.
        down: Downsampling factor.
        **kwargs: Additional parameters (reserved for future use).

    Returns:
        Resampled samples at rate (original_rate * up / down).
    """
    backend = get_backend()
    return backend.resample_poly(samples, int(up), int(down))


# ============================================================================
# FILTERING OPERATIONS
# ============================================================================


def matched_filter(
    samples: ArrayType,
    pulse_taps: ArrayType,
    mode: str = "same",
) -> ArrayType:
    """
    Apply matched filter (time-reversed conjugate of pulse shape).

    The matched filter is optimal for detecting signals in AWGN. It maximizes
    the signal-to-noise ratio at the sampling instant.

    Args:
        samples: Received sample array.
        pulse_taps: Pulse shape filter taps.
        mode: Convolution mode ('same', 'full', 'valid').

    Returns:
        Matched filtered samples.
    """
    backend = get_backend()

    # Matched filter is conjugate and time-reversed version of pulse
    matched_taps = backend.conj(pulse_taps[::-1])

    # Apply filter
    return fir_filter(samples, matched_taps, mode=mode)


# ============================================================================
# HIGH-LEVEL OPERATIONS
# ============================================================================


def shape_pulse(
    symbols: ArrayType,
    sps: float,
    pulse_shape: str = "none",
    **kwargs: Any,
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence.

    High-level function that combines expansion and pulse shaping filtering.
    This is the standard transmit-side operation for digital communications.

    Args:
        symbols: Input symbol array.
        sps: Samples per symbol (upsampling factor).
        pulse_shape: Type of pulse shaping ('none', 'boxcar', 'gaussian', 'rrc', 'sinc').
        **kwargs: Additional arguments for filter generation.

    Returns:
        Shaped sample array at (sps * symbol_rate).
    """
    # 1. Resolve taps
    taps = get_taps(pulse_shape, sps, **kwargs)

    # 2. Determine convolution mode
    # Default to 'same' (centered) which is correct for RRC, Gaussian, etc.
    mode = "same"

    # Special handling for causal filters
    # Boxcar is causal (starts at 0), so 'same' would shift it left.
    # if pulse_shape.lower() == "boxcar":
    #     mode = "full"

    # 3. Expand (zero-insertion)
    expanded = expand(symbols, int(sps))

    # Optimization: If taps is just [1], return expanded signal
    if taps.shape == (1,) and taps[0] == 1:
        return expanded

    # 4. Filter
    shaped = fir_filter(expanded, taps, mode=mode)

    # 5. Post-process (Truncate if full)
    # if mode == "full":
    #     # Truncate to expected length (num_symbols * sps)
    #     expected_len = int(symbols.shape[0] * sps)
    #     shaped = shaped[:expected_len]

    return shaped
