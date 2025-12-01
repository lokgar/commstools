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
        Array of normalized taps with unity gain normaliazation.
    """
    backend = get_backend()
    taps = backend.ones(int(sps))
    return taps / backend.sum(taps)


def gaussian_taps(sps: float, bt: float = 0.3, span: int = 2) -> ArrayType:
    """
    Generates Gaussian filter taps.

    Args:
        sps: Samples per symbol.
        bt: Bandwidth-Time product.
        span: Filter span in symbols.

    Returns:
        Gaussian filter taps with unity gain normalization.
    """
    backend = get_backend()
    # Using numpy for coefficient calculation, then converting to backend

    t = np.arange(-span * sps, span * sps + 1) / sps
    # Gaussian function: h(t) = (sqrt(pi)/a) * exp(-(pi*t/a)^2) where a = sqrt(ln(2)/2)/B
    # Simplified for comms usually:
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    # Normalize to Unity Gain
    h = h / np.sum(h)

    return backend.array(h)


def rrc_taps(sps: float, rolloff: float = 0.35, span: int = 4) -> ArrayType:
    """
    Generates Root Raised Cosine (RRC) filter taps.

    Args:
        sps: Samples per symbol.
        rolloff: Roll-off factor (0 to 1).
        span: Filter span in symbols.

    Returns:
        RRC filter taps with unity gain normalization.
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

    # Normalize to Unity Gain
    h = h / np.sum(h)

    return backend.array(h)


def sinc_taps(num_taps: int, cutoff_norm: float, window: str = "blackman") -> ArrayType:
    """
    Generates Windowed Sinc Low Pass filter taps.

    Args:
        num_taps (int): Total number of coefficients (should be odd).
        cutoff_norm (float): Normalized cutoff frequency (0.0 to 0.5).
                             0.5 = Nyquist (fs/2).
        window (str): Type of window to apply ('blackman', 'hamming', or 'none').

    Returns:
        Sinc filter taps with unity gain normalization.
    """
    backend = get_backend()

    # Create centered grid
    center = (num_taps - 1) / 2
    n = np.arange(num_taps) - center

    # Generic Sinc Formula: 2 * fc * sinc(2 * fc * n)
    # This works for ANY cutoff, not just the interpolation case.
    h = 2 * cutoff_norm * np.sinc(2 * cutoff_norm * n)

    # Apply Window
    if window == "blackman":
        win = np.blackman(num_taps)
    elif window == "hamming":
        win = np.hamming(num_taps)
    else:
        win = np.ones(num_taps)

    h = h * win

    # Unity Gain Normalization
    h = h / np.sum(h)

    return backend.array(h)


def sinc_interpolation_taps(
    factor: float, span: int = 10, bandwidth_fraction: float = 1.0
) -> ArrayType:
    """
    Generates Sinc interpolation filter taps for upsampling.

    Args:
        factor: Upsampling factor.
        span: Filter span in symbols.
        bandwidth_fraction: Defines the cutoff relative to the output Nyquist.
                            1.0 = Cutoff at exactly Nyquist (Ideal).
                            < 1.0 = Cutoff lower than Nyquist (Allows rolloff).

    Returns:
        Sinc interpolation filter taps with unity gain normalization.
    """
    backend = get_backend()

    num_taps = int(2 * span * factor) + 1

    cutoff = (1.0 / (2.0 * factor)) * bandwidth_fraction

    h = sinc_taps(num_taps, cutoff)

    # Normalize to Unity Gain
    h = h / np.sum(h)
    h *= factor

    return backend.array(h)


# ============================================================================
# FILTERING OPERATIONS
# ============================================================================
# fir_filter: Generic FIR filtering operation
# matched_filter: Apply matched filter (time-reversed conjugate of pulse shape)
# shape_pulse: Apply pulse shaping to symbols


def fir_filter(samples: ArrayType, taps: ArrayType, mode: str = "same") -> ArrayType:
    """
    Apply FIR filter via convolution.

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


def matched_filter(
    samples: ArrayType, pulse_taps: ArrayType, mode: str = "same"
) -> ArrayType:
    """
    Apply matched filter (time-reversed conjugate of pulse shape).

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

    # Enforce Unit Energy
    # Calculate energy
    energy_sq = backend.sum(backend.abs(matched_taps) ** 2)

    # Avoid division by zero or redundant calculation if already 1.0
    # (Using a small epsilon for float comparison)
    if energy_sq > 0 and backend.abs(energy_sq - 1.0) > 1e-6:
        matched_taps = matched_taps / backend.sqrt(energy_sq)

    # Apply filter
    return fir_filter(samples, matched_taps, mode=mode)


def shape_pulse(
    symbols: ArrayType, sps: float, span: int, pulse_shape: str = "none", **kwargs: Any
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence.

    Args:
        symbols: Input symbol array.
        sps: Samples per symbol (upsampling factor).
        span: Pulse shaping filter span in symbols.
        pulse_shape: Type of pulse shaping ('none', 'boxcar', 'gaussian', 'rrc').
        **kwargs: Additional arguments for filter generation.

    Returns:
        Shaped sample array at rate (sps * symbol_rate).
    """
    backend = get_backend()

    expanded = backend.expand(symbols, int(sps))

    if pulse_shape == "none":
        h = backend.ones(1)
    elif pulse_shape == "boxcar":
        h = boxcar_taps(sps)
    elif pulse_shape == "gaussian":
        h = gaussian_taps(sps, span=span, **kwargs)
    elif pulse_shape == "rrc":
        h = rrc_taps(sps, span=span, **kwargs)
    elif pulse_shape == "sinc":
        # Sinc pulse shaping is equivalent to RRC with rolloff=0
        h = rrc_taps(sps, span=span, rolloff=0.0, **kwargs)
    else:
        raise ValueError(f"Not implemented pulse shape: {pulse_shape}")

    # Optimization: If h is just [1], return expanded scaled by sqrt(sps)
    if h.shape == (1,) and backend.abs(h[0] - 1.0) < 1e-10:
        return expanded * backend.sqrt(float(sps))

    # Enforce Unit Energy
    # Calculate energy
    energy_sq = backend.sum(backend.abs(h) ** 2)

    # Avoid division by zero or redundant calculation if already 1.0
    # (Using a small epsilon for float comparison)
    if energy_sq > 0 and backend.abs(energy_sq - 1.0) > 1e-6:
        h = h / backend.sqrt(energy_sq)

    scaled_h = h * backend.sqrt(float(sps))

    return fir_filter(expanded, scaled_h, mode="same")
