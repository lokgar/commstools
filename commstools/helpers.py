"""
General library utility functions.

This module provides essential helper routines used throughout the library,
including random sequence generation, SI prefix formatting, and
multi-backend array validation.

Functions
---------
random_bits :
    Generates reproducible random binary sequences.
random_symbols :
    Generates random modulation symbols.
rms :
    Computes the Root-Mean-Square (RMS) value of an array.
normalize :
    Applies various normalization strategies (unit energy, max amplitude, etc.).
format_si :
    Converts numeric values into human-readable SI-formatted strings.
validate_array :
    Ensures input data is coerced into a supported backend array type.
interp1d :
    High-performance linear interpolation for NumPy and CuPy backends.
cross_correlate_fft :
    Vectorized FFT-based cross-correlation for 1D and multichannel signals.
expand_preamble_mimo :
    Expands a single preamble waveform to a MIMO preamble for N_tx antennas.
"""

from typing import Any, Optional

import numpy as np

from .backend import ArrayType, dispatch, get_array_module, is_cupy_available, to_device
from .logger import logger

try:
    import cupy as cp
except ImportError:
    cp = None


def random_bits(length: int, seed: Optional[int] = None) -> ArrayType:
    """
    Generates a sequence of random binary bits (0s and 1s).

    Uses `numpy.random.default_rng()` for consistent seed behavior across
    different platforms and backends.

    Parameters
    ----------
    length : int
        Total number of bits to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array_like
        Array of bits (0 or 1). Shape: (length,).
        Data type is `int8`.
    """
    logger.debug(f"Generating {length} random bits (seed={seed}).")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length, dtype="int8")

    if is_cupy_available():
        bits = to_device(bits, "gpu")

    return bits


def random_symbols(
    num_symbols: int,
    modulation: str,
    order: int,
    seed: Optional[int] = None,
    unipolar: Optional[bool] = None,
) -> ArrayType:
    """
    Generates a sequence of random modulation symbols.

    This is a high-level utility that combines bit generation and mapping
    to produce synthetic symbol sequences.

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate.
    modulation : {"psk", "qam", "ask"}
        The modulation scheme identifier.
    order : int
        Modulation order (e.g., 4, 16, 64).
    seed : int, optional
        Random seed for reproducible results.
    unipolar : bool, default False
        If True, use unipolar constellation (ASK/PAM).

    Returns
    -------
    array_like
        Array of symbols on the active device (CPU or GPU).
        Dtype is ``complex64`` for PSK/QAM, ``float32`` for ASK/PAM.
    """
    from . import mapping

    k = int(np.log2(order))
    bits = random_bits(num_symbols * k, seed=seed)
    return mapping.map_bits(bits, modulation, order, unipolar=unipolar)


def rms(x: ArrayType, axis: Optional[int] = None, keepdims: bool = False) -> ArrayType:
    """
    Computes the Root-Mean-Square (RMS) value of an array.

    RMS is defined as: $\\sqrt{E[|x|^2]}$.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to compute the RMS. If None, computes global RMS.
    keepdims : bool, default False
        If True, the reduced axes are left in the result as dimensions with size one.

    Returns
    -------
    array_like or float
        The RMS value of the input.
    """
    x, xp, _ = dispatch(x)
    return xp.sqrt(xp.mean(xp.abs(x) ** 2, axis=axis, keepdims=keepdims))


def normalize(
    x: ArrayType, mode: str = "unity_gain", axis: Optional[int] = None
) -> ArrayType:
    """
    Normalizes an array according to the specified strategy.

    Parameters
    ----------
    x : array_like
        Input signal or filter taps.
    mode : {"unity_gain", "unit_energy", "peak", "average_power", "rms"}, default "unity_gain"
        Normalization strategy:
        - "unity_gain": Sum of elements is 1.0 (DC gain normalization).
          Preserves signal levels (e.g., 5V -> 5V). Used for general filters.
        - "unit_energy": L2-norm is 1.0 ($\\sum |x|^2 = 1$).
          Preserves total energy/noise power. Used for pulse shaping and matched filters.
        - "peak": Peak absolute value is 1.0.
          **Crucially**: For complex signals, normalizes Real and Imaginary components
          independently to fit within DAC limits ($|I| \\le 1, |Q| \\le 1$).
          This maximizes dynamic range without clipping either independent channel.
        - "average_power" or "rms": Mean power (RMS) is 1.0 ($E[|x|^2] = 1$).
          Normalizes the composite complex signal power. Used for symbol constellations.
    axis : int, optional
        The axis along which to compute the normalization factor.
        If `None`, normalizes the entire array globally.

    Returns
    -------
    array_like
        The normalized array.
    """
    logger.debug(f"Normalizing array (mode: {mode}, axis={axis}).")
    x, xp, _ = dispatch(x)

    # keepdims for proper broadcasting when axis is specified
    keepdims = axis is not None

    if mode == "unity_gain":
        # DC gain = 1: H(0) = sum(h) = 1
        # Use case: filter taps where you want unity passband gain
        norm_factor = xp.sum(x, axis=axis, keepdims=keepdims)

    elif mode == "unit_energy":
        # L2 norm = 1: ||x||₂ = sqrt(sum(|x|²)) = 1
        # Use case: matched filter taps (preserves SNR after correlation)
        norm_factor = xp.sqrt(xp.sum(xp.abs(x) ** 2, axis=axis, keepdims=keepdims))

    elif mode == "peak":
        # Peak normalization: max of any channel = 1
        # For complex: max(max(|I|), max(|Q|)) to prevent DAC/ADC clipping
        # For real: max(|x|) = 1
        if xp.iscomplexobj(x):
            max_real = xp.max(xp.abs(x.real), axis=axis, keepdims=keepdims)
            max_imag = xp.max(xp.abs(x.imag), axis=axis, keepdims=keepdims)
            norm_factor = xp.maximum(max_real, max_imag)
        else:
            norm_factor = xp.max(xp.abs(x), axis=axis, keepdims=keepdims)

    elif mode in ("average_power", "rms"):
        # RMS = 1: sqrt(mean(|x|²)) = 1, so mean(|x|²) = 1
        # Use case: signals where average power should be normalized
        norm_factor = rms(x, axis=axis, keepdims=keepdims)

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    # Handle division by zero safely for both NumPy and CuPy
    # Avoid control flow based on data values to prevent host-device synchronization.
    safe_norm = xp.where(norm_factor == 0, 1.0, norm_factor)
    result = x / safe_norm

    # If norm_factor is 0, the input was all zeros → output should also be zeros
    return xp.where(norm_factor == 0, xp.zeros(x.shape, dtype=x.dtype), result)


def format_si(value: Optional[float], unit: str = "Hz") -> str:
    """
    Formats a numeric value into a human-readable string with SI prefixes.

    Automatically selects the appropriate SI prefix (e.g., k, M, G, m, u, n)
    based on the magnitude of the value. Supports a wide range from
    femto ($10^{-15}$) to Peta ($10^{15}$).

    Parameters
    ----------
    value : float or None
        The numeric value to format. If `None`, returns "None".
    unit : str, default "Hz"
        The unit suffix to append (e.g., 'Hz', 'Baud', 's', 'W').

    Returns
    -------
    str
        The formatted string (e.g., '10.00 MHz', '50.00 ns').
    """
    if value is None:
        return "None"

    if abs(value) == 0:
        return f"0.00 {unit}"

    # Standard SI prefixes
    si_units = {
        -5: "f",
        -4: "p",
        -3: "n",
        -2: "µ",
        -1: "m",
        0: "",
        1: "k",
        2: "M",
        3: "G",
        4: "T",
        5: "P",
    }

    rank = int(np.floor(np.log10(abs(value)) / 3))
    # clamp to supported range
    rank = max(min(si_units.keys()), min(rank, max(si_units.keys())))

    scaled = value / (1000.0**rank)
    return f"{scaled:.2f} {si_units[rank]}{unit}"


def validate_array(
    v: Any, name: str = "array", complex_only: bool = False
) -> ArrayType:
    """
    Validates and coerces input data into a backend-compatible array.

    Handles conversion from Python scalars, lists, and tuples into NumPy
    or CuPy arrays. Can optionally enforce complex-valued data types.

    Parameters
    ----------
    v : array_like or any
        Input data to validate.
    name : str, default "array"
        Variable name used in error messages.
    complex_only : bool, default False
        If True, ensures the resulting array is complex-valued.

    Returns
    -------
    array_like
        A backend-native array (NumPy/CuPy).

    Raises
    ------
    ValueError
        If the input cannot be converted to a supported array type.
    """
    if v is None:
        return None

    # Coerce lists/tuples or other array-likes to numpy arrays initially
    if not isinstance(v, (np.ndarray, getattr(cp, "ndarray", type(None)))):
        try:
            v = np.asarray(v)
        except Exception:
            raise ValueError(f"Could not convert {name} of type {type(v)} to array.")

    # Ensure it's a numeric array (not object, string, etc.)
    if v.dtype.kind not in "biufc":
        raise ValueError(
            f"Expected numeric array for {name}, got dtype {v.dtype} (kind {v.dtype.kind})"
        )

    if complex_only and not np.iscomplexobj(v):
        xp = get_array_module(v)
        v = v.astype(xp.complex128)

    return v


def interp1d(x: ArrayType, x_p: ArrayType, f_p: ArrayType, axis: int = -1) -> ArrayType:
    """
    Performs 1D linear interpolation across a specified axis.

    This function provides a backend-agnostic (NumPy/CuPy) implementation of
    linear interpolation, serving as a high-performance replacement for
    generic interpolation routines.

    Parameters
    ----------
    x : array_like
        Target coordinates (query points).
    x_p : array_like
        Original sample coordinates (must be monotonically increasing).
    f_p : array_like
        Original sample values at coordinates `x_p`.
    axis : int, default -1
        The axis along which to perform interpolation.

    Returns
    -------
    array_like
        Interpolated values at the target coordinates `x`.
    """
    logger.debug(f"Performing linear interpolation (axis={axis}).")
    x, xp, _ = dispatch(x)

    # Ensure other inputs are on the same backend
    x_p = xp.asarray(x_p)
    f_p = xp.asarray(f_p)

    # Move axis to end for easier handling
    f_p = xp.swapaxes(f_p, axis, -1)

    # Find indices such that xp[i-1] <= x < xp[i]
    idxs = xp.searchsorted(x_p, x)
    idxs = xp.clip(idxs, 1, len(x_p) - 1)

    # Get the bounding points
    x0 = x_p[idxs - 1]
    x1 = x_p[idxs]

    # Calculate weights
    denominator = x1 - x0
    # Avoid division by zero
    denominator[denominator == 0] = 1.0
    weights = (x - x0) / denominator

    # Get the bounding values
    # f_p is (..., T)
    # idxs is (M,)
    # We want result (..., M)

    y0 = f_p[..., idxs - 1]
    y1 = f_p[..., idxs]

    result = y0 * (1 - weights) + y1 * weights

    # Move axis back
    result = xp.swapaxes(result, axis, -1)

    return result


def cross_correlate_fft(
    signal: ArrayType,
    template: ArrayType,
    mode: str = "full",
) -> ArrayType:
    """
    Vectorized FFT-based cross-correlation.

    Computes the cross-correlation of ``signal`` with ``template`` using
    the frequency-domain multiplication approach. Handles 1D and 2D
    (multichannel) inputs natively via ``axis=-1`` broadcasting — no
    Python loops over channels.

    Parameters
    ----------
    signal : array_like
        Input signal. Shape: ``(N,)`` or ``(C, N)``.
    template : array_like
        Reference sequence. Shape: ``(L,)`` or ``(C, L)``.
        If ``(1, L)`` and signal is ``(C, N)``, the template is
        broadcast across all channels.
    mode : {"full", "same", "valid"}, default "full"
        Output size:
        - ``"full"``: length ``N + L - 1``.
        - ``"same"``: length ``N`` (centered).
        - ``"valid"``: length ``max(N, L) - min(N, L) + 1``.

    Returns
    -------
    array_like
        Complex cross-correlation with shape matching the input
        dimensionality and the selected ``mode``.
    """
    signal, xp, _ = dispatch(signal)
    template = xp.asarray(template)

    was_1d = signal.ndim == 1
    if was_1d:
        signal = signal[None, :]
    if template.ndim == 1:
        template = template[None, :]

    N = signal.shape[-1]
    L = template.shape[-1]
    full_len = N + L - 1

    # Power-of-2 FFT length for efficiency
    n_fft = (
        1 << full_len.bit_length()
        if isinstance(full_len, int)
        else 1 << int(full_len).bit_length()
    )

    # FFT-based correlation: R[k] = IFFT(FFT(signal) * conj(FFT(template)))
    # Circular correlation places positive lags at 0..N-1 and negative lags
    # wrap to n_fft-(L-1)..n_fft-1.  Rearrange to match scipy layout:
    # lags [-(L-1), ..., -1, 0, 1, ..., N-1]  (total = N + L - 1).
    SIG = xp.fft.fft(signal, n_fft, axis=-1)
    TPL = xp.fft.fft(template, n_fft, axis=-1)
    corr_circ = xp.fft.ifft(SIG * xp.conj(TPL), axis=-1)

    # Gather negative lags (indices n_fft-(L-1) .. n_fft-1) then positive (0 .. N-1)
    neg_lags = corr_circ[..., -(L - 1) :]  # length L-1
    pos_lags = corr_circ[..., :N]  # length N
    corr = xp.concatenate([neg_lags, pos_lags], axis=-1)  # length N+L-1

    # Apply mode trimming
    if mode == "same":
        start = (L - 1) // 2
        corr = corr[..., start : start + N]
    elif mode == "valid":
        valid_len = max(N, L) - min(N, L) + 1
        start = min(N, L) - 1
        corr = corr[..., start : start + valid_len]
    # mode == "full": no trimming needed

    if was_1d:
        return corr[0]
    return corr


def expand_preamble_mimo(
    base_waveform: ArrayType, num_streams: int, mode: str = "same"
) -> ArrayType:
    """
    Expands a base preamble waveform into a MIMO preamble structure.

    Parameters
    ----------
    base_waveform : ArrayType
        Base preamble samples, shape (L,) or (1, L).
    num_streams : int
        Number of transmit streams.
    mode : str
        MIMO mode: "same" (broadcast) or "time_orthogonal".

    Returns
    -------
    ArrayType
        Expanded preamble, shape (C, L_mimo).
    """
    if num_streams <= 1:
        return base_waveform

    xp = get_array_module(base_waveform)

    # Ensure (1, L)
    if base_waveform.ndim == 1:
        base_waveform = base_waveform[None, :]

    if mode == "same":
        # Broadcast: (C, L)
        return xp.tile(base_waveform, (num_streams, 1))

    elif mode == "time_orthogonal":
        # Time-Orthogonal: (C, L * C)
        # [ P 0 0 ]
        # [ 0 P 0 ]
        # [ 0 0 P ]
        L = base_waveform.shape[-1]
        dtype = base_waveform.dtype

        # Flattened view strategy to avoid loop
        # We want to place `base_waveform` at offsets 0, L+1*row_stride, 2L+2*row_stride...
        # But for time-orthogonal:
        # p[0, 0:L]
        # p[1, L:2L]
        # ...

        # Create empty array
        total_len = L * num_streams
        p_mimo = xp.zeros((num_streams, total_len), dtype=dtype)

        # We can simulate this by reshaping to (num_streams, num_streams, L)
        # and filling the diagonal blocks (i, i, :)

        p_view = p_mimo.reshape(num_streams, num_streams, L)

        # Optimized approach using strided write:
        # p_view[i, i, :] = base_waveform
        # Advanced indexing:
        indices = xp.arange(num_streams)
        p_view[indices, indices, :] = base_waveform[0]

        return p_mimo

    else:
        # Fallback to broadcast (or raise error? "same" is safe default)
        return xp.tile(base_waveform, (num_streams, 1))
