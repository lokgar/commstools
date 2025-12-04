import numpy as np

from ..core.backend import ArrayType, get_backend, ensure_on_backend
from ..dsp.utils import normalize

# ============================================================================
# FILTER DESIGN - TAP GENERATORS
# ============================================================================
# boxcar_taps: Boxcar filter taps
# gaussian_taps: Gaussian filter taps
# rrc_taps: Root Raised Cosine filter taps
# sinc_taps: Windowed Sinc Low Pass filter taps
# sinc_interpolation_taps: Sinc interpolation filter taps


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
    return normalize(taps, mode="unity_gain")


def gaussian_taps(sps: float, bt: float = 0.3, span: int = 4) -> ArrayType:
    """
    Generates Gaussian filter taps.

    Args:
        sps: Samples per symbol.
        bt: Bandwidth-Time product.
        span: Total filter span in symbols.

    Returns:
        Gaussian filter taps with unity gain normalization.
    """
    backend = get_backend()

    # Ensure odd number of taps to have a center peak
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    # Gaussian function: h(t) = (sqrt(pi)/a) * exp(-(pi*t/a)^2) where a = sqrt(ln(2)/2)/B
    # Simplified for comms usually:
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    # Normalize to Unity Gain
    h = normalize(h, mode="unity_gain")

    return backend.array(h)


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
    backend = get_backend()

    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    # Use numpy for calculation
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps

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
    h = normalize(h, mode="unity_gain")

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
    h = normalize(h, mode="unity_gain")

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

    num_taps = int(span * factor)
    if num_taps % 2 == 0:
        num_taps += 1

    cutoff = (1.0 / (2.0 * factor)) * bandwidth_fraction

    h = sinc_taps(num_taps, cutoff)

    # Normalize to Unity Gain
    h = normalize(h, mode="unity_gain")

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
    samples = ensure_on_backend(samples)
    backend = get_backend()
    return backend.convolve(samples, taps, mode=mode, method="fft")


def shape_pulse(
    symbols: ArrayType,
    sps: float,
    pulse_shape: str = "none",
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    gaussian_bt: float = 0.3,
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence.

    Args:
        symbols: Input symbol array.
        sps: Samples per symbol (upsampling factor).
        pulse_shape: Type of pulse shaping ('none', 'boxcar', 'gaussian', 'rrc').
        filter_span: Pulse shaping filter span in symbols.
        rrc_rolloff: Roll-off factor for RRC filter.
        gaussian_bt: Bandwidth-Time product for Gaussian filter.

    Returns:
        Shaped sample array at rate (sps * symbol_rate), normalized
    """
    symbols = ensure_on_backend(symbols)
    backend = get_backend()

    expanded = backend.expand(symbols, int(sps))

    if pulse_shape == "none":
        h = backend.ones(1)
    elif pulse_shape == "boxcar":
        h = boxcar_taps(sps)
    elif pulse_shape == "gaussian":
        h = gaussian_taps(sps, span=filter_span, bt=gaussian_bt)
    elif pulse_shape == "rrc":
        h = rrc_taps(sps, span=filter_span, rolloff=rrc_rolloff)
    elif pulse_shape == "sinc":
        # Sinc pulse shaping is equivalent to RRC with rolloff=0
        h = rrc_taps(sps, span=filter_span, rolloff=0.0)
    else:
        raise ValueError(f"Not implemented pulse shape: {pulse_shape}")

    return normalize(fir_filter(expanded, h, mode="same"), mode="max_amplitude")


def matched_filter(
    samples: ArrayType,
    pulse_taps: ArrayType,
    taps_normalization: str = "unity_gain",
    mode: str = "same",
    normalize_output: bool = False,
) -> ArrayType:
    """
    Apply matched filter (time-reversed conjugate of pulse shape).

    Args:
        samples: Received sample array.
        pulse_taps: Pulse shape filter taps.
        taps_normalization: Normalization to apply to the matched filter taps.
                            Options: 'unity_gain', 'unit_energy'.
        mode: Convolution mode ('same', 'full', 'valid').
        normalize_output: If True, normalizes the output samples to have a maximum
                          absolute value of 1.0.

    Returns:
        Matched filtered samples.
    """
    samples = ensure_on_backend(samples)
    backend = get_backend()

    # Matched filter is conjugate and time-reversed version of pulse
    matched_taps = backend.conj(pulse_taps[::-1])

    if taps_normalization == "unity_gain" or taps_normalization == "gain":
        matched_taps = normalize(matched_taps, mode="unity_gain")
    elif taps_normalization == "unit_energy" or taps_normalization == "energy":
        matched_taps = normalize(matched_taps, mode="unit_energy")
    else:
        raise ValueError(f"Not implemented taps normalization: {taps_normalization}")

    # Apply filter
    output = fir_filter(samples, matched_taps, mode=mode)

    if normalize_output:
        output = normalize(output, mode="max_amplitude")

    return output
