import numpy as np
from typing import Optional, Union, Literal
from ..core.backend import get_backend, ArrayType


def rect(samples_per_symbol: int) -> ArrayType:
    """
    Generates a rectangular pulse (all ones).

    Args:
        samples_per_symbol: Number of samples per symbol.

    Returns:
        Array of ones.
    """
    backend = get_backend()
    return backend.ones(samples_per_symbol)


def gaussian(samples_per_symbol: int, bt: float = 0.3, span: int = 2) -> ArrayType:
    """
    Generates a Gaussian pulse.

    Args:
        samples_per_symbol: Number of samples per symbol.
        bt: Bandwidth-Time product.
        span: Filter span in symbols.

    Returns:
        Gaussian pulse taps.
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


def rrc(samples_per_symbol: int, alpha: float = 0.35, span: int = 4) -> ArrayType:
    """
    Generates a Root Raised Cosine (RRC) pulse.

    Args:
        samples_per_symbol: Number of samples per symbol.
        alpha: Roll-off factor (0 to 1).
        span: Filter span in symbols.

    Returns:
        RRC pulse taps.
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
    h[idx_0] = 1.0 - alpha + (4 * alpha / np.pi)

    # Case 2: t = +/- 1/(4*alpha)
    if alpha > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (4 * alpha))
        h[idx_singularity] = (alpha / np.sqrt(2)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
        )
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    # Case 3: General case
    idx_general = ~(idx_0 | idx_singularity)
    t_gen = t[idx_general]

    num = np.sin(np.pi * t_gen * (1 - alpha)) + 4 * alpha * t_gen * np.cos(
        np.pi * t_gen * (1 + alpha)
    )
    den = np.pi * t_gen * (1 - (4 * alpha * t_gen) ** 2)
    h[idx_general] = num / den

    # Normalize
    # Usually we want the filter to have unit energy
    # h = h / np.sqrt(np.sum(h**2))

    # Or unit peak for simple testing?
    # Let's stick to unit energy which is standard for matched filtering
    h = h / np.sqrt(np.sum(h**2))

    return backend.array(h)


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


def apply_pulse_shape(
    symbols: ArrayType,
    samples_per_symbol: int,
    filter_taps: Optional[ArrayType] = None,
    kind: Literal["rect", "custom"] = "rect",
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence.

    Args:
        symbols: Input symbol array.
        samples_per_symbol: Upsampling factor.
        filter_taps: Custom filter taps. Required if kind='custom'.
        kind: Type of pulse shaping.
            - 'rect': Rectangular pulse (Sample-and-Hold).
            - 'custom': Convolves upsampled sequence with filter_taps.

    Returns:
        Shaped sample array.
    """
    backend = get_backend()

    if kind == "rect":
        # Optimized path for rect (sample-and-hold)
        # Reshape and tile approach
        symbols_col = symbols.reshape((-1, 1))
        ones_row = backend.ones((1, samples_per_symbol), dtype=symbols.dtype)
        matrix = symbols_col * ones_row
        return matrix.reshape((-1,))

    elif kind == "custom":
        if filter_taps is None:
            raise ValueError("filter_taps must be provided for kind='custom'")

        # 1. Upsample
        upsampled = upsample(symbols, samples_per_symbol)

        # 2. Convolve
        return backend.convolve(upsampled, filter_taps, mode="full")

    else:
        raise ValueError(f"Unknown pulse shape kind: {kind}")
