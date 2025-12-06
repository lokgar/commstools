from .backend import ArrayType, get_backend, ensure_on_backend
from .utils import normalize

# ============================================================================
# FILTER DESIGN - TAP GENERATORS
# ============================================================================
# boxcar_taps: Boxcar filter taps
# gaussian_taps: Gaussian filter taps
# rrc_taps: Root Raised Cosine filter taps
# rc_taps: Raised Cosine filter taps
# lowpass_taps: Low pass filter taps
# highpass_taps: High pass filter taps
# bandpass_taps: Band pass filter taps
# bandstop_taps: Band stop filter taps


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

    t = (backend.arange(num_taps) - (num_taps - 1) / 2) / sps
    # Gaussian function: h(t) = (sqrt(pi)/a) * exp(-(pi*t/a)^2) where a = sqrt(ln(2)/2)/B
    # Simplified for comms usually:
    alpha = backend.sqrt(backend.log(2) / 2) / bt
    h = (backend.sqrt(backend.pi) / alpha) * backend.exp(
        -((backend.pi * t / alpha) ** 2)
    )

    # Normalize to Unity Gain
    h = normalize(h, mode="unity_gain")

    return h


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

    # Use backend for calculation
    t = (backend.arange(num_taps) - (num_taps - 1) / 2) / sps

    # Avoid division by zero
    # 1. t = 0
    # 2. t = +/- 1/(4*rolloff)

    # Initialize array
    h = backend.zeros_like(t)

    # Case 1: t = 0
    idx_0 = backend.isclose(t, 0)
    h = backend.where(idx_0, 1.0 - rolloff + (4 * rolloff / backend.pi), h)

    # Case 2: t = +/- 1/(4*rolloff)
    if rolloff > 0:
        idx_singularity = backend.isclose(backend.abs(t), 1 / (4 * rolloff))
        h = backend.where(
            idx_singularity,
            (rolloff / backend.sqrt(2))
            * (
                (1 + 2 / backend.pi) * backend.sin(backend.pi / (4 * rolloff))
                + (1 - 2 / backend.pi) * backend.cos(backend.pi / (4 * rolloff))
            ),
            h,
        )
    else:
        idx_singularity = backend.zeros_like(t, dtype=bool)

    # Case 3: General case
    idx_general = ~(idx_0 | idx_singularity)

    num = backend.sin(backend.pi * t * (1 - rolloff)) + 4 * rolloff * t * backend.cos(
        backend.pi * t * (1 + rolloff)
    )
    den = backend.pi * t * (1 - (4 * rolloff * t) ** 2)
    h = backend.where(idx_general, num / den, h)

    # Normalize to Unity Gain
    h = normalize(h, mode="unity_gain")

    return h


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
    backend = get_backend()

    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = (backend.arange(num_taps) - (num_taps - 1) / 2) / sps

    # Avoid division by zero
    # Singularities at t = +/- 1 / (2 * rolloff)

    # Initialize array
    h = backend.zeros_like(t)

    # General case mask
    # Denominator: 1 - (2 * rolloff * t)**2
    # Singularity when 2 * rolloff * |t| = 1 => |t| = 1 / (2 * rolloff)

    if rolloff > 0:
        idx_singularity = backend.isclose(backend.abs(t), 1 / (2 * rolloff))
        # Value at singularity: (pi / 4) * sinc(1 / (2 * rolloff))
        # sinc(x) = sin(pi * x) / (pi * x)
        # arg = 1 / (2 * rolloff)
        # val = (pi / 4) * sin(pi * arg) / (pi * arg)
        #     = (pi / 4) * sin(pi / (2 * rolloff)) * (2 * rolloff / pi)
        #     = (rolloff / 2) * sin(pi / (2 * rolloff))
        val_singularity = (rolloff / 2) * backend.sin(backend.pi / (2 * rolloff))
        h = backend.where(idx_singularity, val_singularity, h)
    else:
        idx_singularity = backend.zeros_like(t, dtype=bool)

    idx_general = ~idx_singularity

    # h(t) = sinc(t) * cos(pi * alpha * t) / (1 - (2 * alpha * t)^2)
    # sinc(t) = sin(pi * t) / (pi * t) (normalized sinc)

    # To avoid t=0 in sinc division, use backend.sinc which handles 0 safely
    sinc_t = backend.sinc(t)
    cos_t = backend.cos(backend.pi * rolloff * t)
    denom = 1 - (2 * rolloff * t) ** 2

    # We masked out where denom is 0, so safe to divide where idx_general is true
    # However we compute everywhere then mask, so denom should not be 0 to avoid warning/NaN if backend evals strict
    # backend.where usually evals both branches
    # So we set denom to 1 where it is 0
    denom_safe = backend.where(idx_singularity, 1.0, denom)

    res = sinc_t * cos_t / denom_safe
    h = backend.where(idx_general, res, h)

    return normalize(h, mode="unity_gain")


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
    backend = get_backend()
    return backend.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=True
    )


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
    backend = get_backend()

    # pass_zero=False for highpass
    return backend.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=False
    )


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
    backend = get_backend()

    # pass_zero=False for bandpass
    return backend.firwin(
        num_taps,
        [low_cutoff, high_cutoff],
        window=window,
        fs=sampling_rate,
        pass_zero=False,
    )


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
    backend = get_backend()
    # pass_zero=True for bandstop
    return backend.firwin(
        num_taps,
        [low_cutoff, high_cutoff],
        window=window,
        fs=sampling_rate,
        pass_zero=True,
    )


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
        pulse_shape: Type of pulse shaping ('none', 'boxcar', 'gaussian', 'rrc', 'rc', 'sinc').
        filter_span: Pulse shaping filter span in symbols.
        rrc_rolloff: Roll-off factor for RRC filter.
        gaussian_bt: Bandwidth-Time product for Gaussian filter.

    Returns:
        Shaped sample array at rate (sps * symbol_rate), normalized
    """
    symbols = ensure_on_backend(symbols)
    backend = get_backend()

    if pulse_shape == "none":
        return normalize(backend.expand(symbols, int(sps)), "max_amplitude")

    elif pulse_shape == "boxcar":
        h = boxcar_taps(sps)
    elif pulse_shape == "gaussian":
        h = gaussian_taps(sps, span=filter_span, bt=gaussian_bt)
    elif pulse_shape == "rrc":
        h = rrc_taps(sps, span=filter_span, rolloff=rrc_rolloff)
    elif pulse_shape == "rc":
        h = rc_taps(sps, span=filter_span, rolloff=rrc_rolloff)
    elif pulse_shape == "sinc":
        # Sinc pulse shaping is equivalent to RRC with rolloff=0
        h = rrc_taps(sps, span=filter_span, rolloff=0.0)
    else:
        raise ValueError(f"Not implemented pulse shape: {pulse_shape}")

    # Use polyphase implementation for efficiency:
    return normalize(
        backend.resample_poly(symbols, int(sps), 1, window=h), "max_amplitude"
    )


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
