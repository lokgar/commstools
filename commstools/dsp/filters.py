import numpy as np
from typing import Any
from ..core.backend import get_backend, ArrayType


def boxcar_taps(samples_per_symbol: int) -> ArrayType:
    """
    Generates a boxcar filter taps.

    Args:
        samples_per_symbol: Number of samples per symbol.

    Returns:
        Array of ones.
    """
    backend = get_backend()
    return backend.ones(samples_per_symbol)


def gaussian_taps(samples_per_symbol: int, bt: float = 0.3, span: int = 2) -> ArrayType:
    """
    Generates a Gaussian filter taps.

    Args:
        samples_per_symbol: Number of samples per symbol.
        bt: Bandwidth-Time product.
        span: Filter span in symbols.

    Returns:
        Gaussian filter taps.
    """
    backend = get_backend()
    # Using numpy for coefficient calculation, then converting to backend

    t = (
        np.arange(-span * samples_per_symbol, span * samples_per_symbol + 1)
        / samples_per_symbol
    )
    # Gaussian function: h(t) = (sqrt(pi)/a) * exp(-(pi*t/a)^2) where a = sqrt(ln(2)/2)/B
    # Simplified for comms usually:
    alpha = np.sqrt(np.log(2) / 2) / bt
    pulse = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    # Normalize energy to 1 (or peak to 1 depending on convention, usually energy for filters)
    # But for pulse shaping, often we want unit energy per symbol or unit peak.
    # Let's normalize so sum is sps (unit DC gain for rect) or unit energy.
    # Standard convention: sum(h) = sps (preserves amplitude after convolution with upsampled sequence)

    pulse = pulse / np.sum(pulse) * samples_per_symbol

    return backend.array(pulse)


def rrc_taps(
    samples_per_symbol: int, rolloff: float = 0.35, span: int = 4
) -> ArrayType:
    """
    Generates a Root Raised Cosine (RRC) filter taps.

    Args:
        samples_per_symbol: Number of samples per symbol.
        rolloff: Roll-off factor (0 to 1).
        span: Filter span in symbols.

    Returns:
        RRC filter taps.
    """
    backend = get_backend()

    # Use numpy for calculation
    t = (
        np.arange(-span * samples_per_symbol, span * samples_per_symbol + 1)
        / samples_per_symbol
    )

    # Avoid division by zero
    # 1. t = 0
    # 2. t = +/- 1/(4*alpha)

    # Initialize array
    h = np.zeros_like(t)

    # Case 1: t = 0
    idx_0 = np.isclose(t, 0)
    h[idx_0] = 1.0 - rolloff + (4 * rolloff / np.pi)

    # Case 2: t = +/- 1/(4*alpha)
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

    # Normalize
    # Usually we want the filter to have unit energy
    # h = h / np.sqrt(np.sum(h**2))

    # Or unit peak for simple testing?
    # Let's stick to unit energy which is standard for matched filtering
    h = h / np.sqrt(np.sum(h**2))

    return backend.array(h)


def get_taps(filter_type: str, samples_per_symbol: int, **kwargs: Any) -> ArrayType:
    """
    Factory function to generate filter taps.

    Args:
        filter_type: Filter type ('none', 'boxcar', 'gaussian', 'rrc').
        samples_per_symbol: Number of samples per symbol.
        **kwargs: Additional arguments for specific filters (e.g., alpha, bt, span).

    Returns:
        Filter taps.
    """
    filter_type = filter_type.lower()
    if filter_type == "none":
        backend = get_backend()
        return backend.ones(1)
    elif filter_type == "boxcar":
        return boxcar_taps(samples_per_symbol)
    elif filter_type == "gaussian":
        return gaussian_taps(samples_per_symbol, **kwargs)
    elif filter_type == "rrc":
        return rrc_taps(samples_per_symbol, **kwargs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def upsample(symbols: ArrayType, samples_per_symbol: int) -> ArrayType:
    """
    Upsamples a symbol sequence by inserting zeros.

    Args:
        symbols: Input symbol array.
        samples_per_symbol: Upsampling factor.

    Returns:
        Upsampled array (length = len(symbols) * sps).
    """
    backend = get_backend()
    n_symbols = symbols.shape[0]
    n_samples = n_symbols * samples_per_symbol

    out = backend.zeros(n_samples, dtype=symbols.dtype)

    if backend.name == "jax":
        out = out.at[::samples_per_symbol].set(symbols)
    else:
        out[::samples_per_symbol] = symbols

    return out


def shape_pulse(
    symbols: ArrayType,
    taps: ArrayType,
    samples_per_symbol: int,
    mode: str = "same",
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence by upsampling and convolving with filter taps.

    Args:
        symbols: Input symbol array.
        taps: Filter taps (pulse shape).
        samples_per_symbol: Upsampling factor.
        mode: Convolution mode ('full', 'valid', 'same'). Defaults to 'same'.

    Returns:
        Shaped sample array.
    """
    backend = get_backend()

    # 1. Upsample
    upsampled = upsample(symbols, samples_per_symbol)

    # Optimization: If taps is just [1] (filter is identity), return upsampled directly
    if taps.shape == (1,) and taps[0] == 1:
        return upsampled

    # 2. Convolve
    return backend.convolve(upsampled, taps, mode=mode, method="fft")
