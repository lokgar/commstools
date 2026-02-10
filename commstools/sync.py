"""
Synchronization and frame detection utilities.

This module provides routines for time and frequency synchronization,
including the generation of optimal synchronization sequences (Barker,
Zadoff-Chu) and robust frame detection algorithms using cross-correlation.

Functions
---------
barker_sequence :
    Generates Barker codes with optimal auto-correlation.
zadoff_chu_sequence :
    Generates Constant Amplitude Zero Auto-Correlation (CAZAC) sequences.
correlate :
    Multi-backend cross-correlation for sequence detection.
detect_frame :
    Identifies frame start position via preamble correlation.
generate_preamble_bits :
    Factory for standard synchronization bit sequences.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch, is_cupy_available, to_device
from .logger import logger

if TYPE_CHECKING:
    from .core import Preamble, Signal


# Standard Barker codes
_BARKER_SEQUENCES = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def barker_sequence(length: int) -> ArrayType:
    """
    Generates a Barker sequence of the specified length.

    Barker sequences are binary sequences (+1, -1) with optimal cyclic
    auto-correlation properties, where the sidelobes are at most 1. They are
    widely used for frame synchronization and pulse compression.

    Parameters
    ----------
    length : {2, 3, 4, 5, 7, 11, 13}
        Total length of the Barker sequence.

    Returns
    -------
    array_like
        BPSK symbols (+1.0, -1.0). Shape: (length,).
        Backend depends on system availability (CuPy if available, else NumPy).

    Raises
    ------
    ValueError
        If the requested length is not a valid Barker length.

    Examples
    --------
    >>> barker_sequence(7)
    array([ 1.,  1.,  1., -1., -1.,  1., -1.], dtype=float32)
    """
    if length not in _BARKER_SEQUENCES:
        valid = sorted(_BARKER_SEQUENCES.keys())
        raise ValueError(f"No Barker sequence of length {length}. Valid: {valid}")

    seq = np.array(_BARKER_SEQUENCES[length], dtype=np.float32)

    if is_cupy_available():
        seq = to_device(seq, "gpu")

    logger.debug(f"Generated Barker-{length} sequence.")
    return seq


def zadoff_chu_sequence(length: int, root: int = 1) -> ArrayType:
    """
    Generates a Zadoff-Chu (ZC) synchronization sequence.

    ZC sequences are Complex-valued, Constant Amplitude Zero
    Auto-Correlation (CAZAC) sequences. They possess the unique property
    that their periodic auto-correlation is zero at all non-zero lags,
    and their DFT is also a ZC sequence. This makes them ideal for
    timing and frequency synchronization in systems like LTE and 5G NR.

    Parameters
    ----------
    length : int
        The sequence length ($N_{ZC}$). For optimal cross-correlation
        properties, this should be a prime number.
    root : int, default 1
        The root index ($u$). Must be relatively prime to `length`.

    Returns
    -------
    array_like
        Complex Zadoff-Chu symbols of unit magnitude.
        Shape: (length,). Data type: `complex64`.

    Notes
    -----
    - For odd lengths: $x[n] = \exp(-j \frac{\pi u n (n+1)}{N_{ZC}})$
    - For even lengths: $x[n] = \exp(-j \frac{\pi u n^2}{N_{ZC}})$
    - ZC sequences have exceptionally low Peak-to-Average Power Ratio (PAPR).
    """
    if length < 1:
        raise ValueError("Length must be positive.")
    if root < 1 or root >= length:
        raise ValueError(f"Root must be in [1, {length - 1}].")

    # Determine backend
    if is_cupy_available():
        import cupy as cp

        xp = cp
    else:
        xp = np

    # ZC formula: x[n] = exp(-j * pi * u * n * (n+1) / N)
    n = xp.arange(length)
    if length % 2 == 0:
        # Even length: x[n] = exp(-j * pi * u * n^2 / N)
        seq = xp.exp(-1j * xp.pi * root * n * n / length)
    else:
        # Odd length: x[n] = exp(-j * pi * u * n * (n+1) / N)
        seq = xp.exp(-1j * xp.pi * root * n * (n + 1) / length)

    seq = seq.astype(xp.complex64)

    logger.debug(f"Generated ZC sequence: length={length}, root={root}.")
    return seq


def correlate(
    signal: ArrayType,
    template: ArrayType,
    mode: str = "full",
    normalize: bool = False,
) -> ArrayType:
    """
    Cross-correlates a signal with a template sequence.

    Uses FFT-based correlation for optimal performance on both CPU and GPU.
    This is essential for detecting synchronization sequences (preambles)
    in received data.

    Parameters
    ----------
    signal : array_like
        Input signal samples. Shape: (N_samples,) or (N_channels, N_samples).
    template : array_like
        The reference sequence to correlate against. Shape: (N_template,).
    mode : {"full", "same", "valid"}, default "full"
        The output size mode of the correlation.
    normalize : bool, default False
        If True, the output is normalized by the template energy.

    Returns
    -------
    array_like
        Correlation metric. Shape depends on `mode` and `signal` dimensions.
    """
    signal, xp, sp = dispatch(signal)
    template = xp.asarray(template)

    # Handle MIMO: correlate each channel independently
    if signal.ndim == 2:
        # Apply along time axis (axis=-1)
        results = []
        for ch in range(signal.shape[0]):
            corr = _correlate_1d(signal[ch], template, mode, normalize, xp, sp)
            results.append(corr)
        return xp.stack(results, axis=0)

    return _correlate_1d(signal, template, mode, normalize, xp, sp)


def _correlate_1d(
    signal: ArrayType,
    template: ArrayType,
    mode: str,
    normalize: bool,
    xp,
    sp,
) -> ArrayType:
    """
    Internal helper for 1D cross-correlation.

    Parameters
    ----------
    signal : array_like
        The input sequence.
    template : array_like
        The reference sequence.
    mode : {"full", "same", "valid"}
        Correlation output mode.
    normalize : bool
        Whether to apply energy-based normalization.
    xp : module
        Active array backend (NumPy/CuPy).
    sp : module
        Active signal processing backend (SciPy/CuPyX).

    Returns
    -------
    array_like
        The correlation metric.
    """
    # Use scipy's correlate which handles mode correctly
    # For matched filtering, we want correlation (not convolution)
    # scipy.correlate computes: sum_k(signal[n+k] * conj(template[k]))

    corr = sp.signal.correlate(signal, template, mode=mode)

    if normalize:
        # Normalize by template energy
        template_energy = xp.sum(xp.abs(template) ** 2)
        if template_energy > 0:
            corr = corr / xp.sqrt(template_energy)

            # Also normalize by signal energy in each window (for true normalized correlation)
            # This is expensive, so we use a simpler approximation
            signal_energy = xp.sum(xp.abs(signal) ** 2)
            if signal_energy > 0:
                corr = corr / xp.sqrt(signal_energy / len(signal) * len(template))

    return corr


def detect_frame(
    signal: Union[ArrayType, "Signal"],
    preamble: Union[ArrayType, "Preamble"],
    threshold: float = 0.5,
    return_metric: bool = False,
    search_range: Optional[Tuple[int, int]] = None,
) -> Union[int, Tuple[int, float]]:
    """
    Detects the start of a frame via preamble correlation.

    This function performs a sliding cross-correlation between the received
    signal and a known reference preamble. The frame start is identified
    as the index of the maximum correlation peak that exceeds the specified
    relative threshold.

    Parameters
    ----------
    signal : array_like or Signal
        Received signal samples. For multidimensional (MIMO) signals,
        the first channel is used for detection.
    preamble : array_like or Preamble
        The known preamble symbols (reference sequence).
    threshold : float, default 0.5
        Detection threshold normalized between 0 and 1. A peak is only
        considered valid if the normalized correlation coefficient exceeds
        this value.
    return_metric : bool, default False
        If True, returns both the detected index and the peak metric.
    search_range : tuple of int, optional
        A `(start, end)` sample range to restrict the detection search.
        Useful for gated receivers or known frame intervals.

    Returns
    -------
    frame_start : int
        The sample index where the preamble begins.
    peak_metric : float, optional
        The normalized correlation coefficient at the detected peak.
        Only returned if `return_metric=True`.

    Raises
    ------
    ValueError
        If no correlation peak is found that satisfies the threshold criteria.

    Notes
    -----
    The returned index is compensated for the preamble length and corresponds
    to the very first sample of the detected preamble.
    """
    from .core import Preamble, Signal

    # Extract arrays
    if isinstance(signal, Signal):
        sig_array = signal.samples
    else:
        sig_array = signal

    if isinstance(preamble, Preamble):
        preamble_array = preamble.symbols
    else:
        preamble_array = preamble

    sig_array, xp, _ = dispatch(sig_array)
    preamble_array = xp.asarray(preamble_array)

    # Handle MIMO: use first channel for detection (preamble usually same on all)
    if sig_array.ndim == 2:
        sig_1d = sig_array[0]
    else:
        sig_1d = sig_array

    # Apply search range
    offset = 0
    if search_range is not None:
        start, end = search_range
        sig_1d = sig_1d[start:end]
        offset = start

    # Correlate with preamble (use complex conjugate for matched filter)
    corr = correlate(sig_1d, preamble_array, mode="same", normalize=False)

    # Take absolute value for magnitude
    corr_mag = xp.abs(corr)

    # Find peak
    peak_idx = int(xp.argmax(corr_mag))
    peak_val = float(corr_mag[peak_idx])

    # Compute normalized metric: peak / (preamble_energy * signal_energy)
    # This gives a value in [0, 1] for matched signals
    preamble_energy = float(xp.sum(xp.abs(preamble_array) ** 2))
    signal_energy = float(xp.mean(xp.abs(sig_1d) ** 2)) * len(preamble_array)

    if preamble_energy > 0 and signal_energy > 0:
        # Normalized correlation coefficient
        normalized_peak = peak_val / xp.sqrt(preamble_energy * signal_energy)
        normalized_peak = float(
            min(normalized_peak, 1.0)
        )  # Clamp due to numerical issues
    else:
        normalized_peak = 0.0

    if normalized_peak < threshold:
        raise ValueError(
            f"No correlation peak above threshold {threshold} "
            f"(max: {normalized_peak:.3f})"
        )

    # Adjust for preamble length and offset
    # In "same" mode, peak is at center when preamble aligned
    preamble_len = len(preamble_array)
    frame_start = peak_idx - preamble_len // 2 + offset

    # Clamp to valid range
    frame_start = max(0, frame_start)

    logger.debug(
        f"Frame detected at sample {frame_start} (metric: {normalized_peak:.3f})"
    )

    if return_metric:
        return frame_start, normalized_peak
    return frame_start


def generate_preamble_bits(sequence_type: str, length: int, **kwargs) -> ArrayType:
    """
    Generates standard bit sequences for synchronization preambles.

    This utility provides a consistent interface for creating binary
    sequences suitable for use with the `Preamble` and `Signal` classes.

    Parameters
    ----------
    sequence_type : {"barker", "zc", "zadoff_chu", "random"}
        The type of sequence to generate.
    length : int
        The desired sequence length in bits or symbols.
    **kwargs : Any
        Additional parameters for sequence generation (e.g., `root` for
        ZC sequences, `seed` for random sequences).

    Returns
    -------
    array_like
        An array of bits (0s and 1s).
    """
    if sequence_type.lower() == "barker":
        # Barker sequence as bits: +1 -> 1, -1 -> 0
        seq = barker_sequence(length)
        xp = np if not is_cupy_available() else dispatch(seq)[1]
        bits = ((seq + 1) / 2).astype(xp.int32)
        return bits

    elif sequence_type.lower() in ("zc", "zadoff_chu"):
        root = kwargs.get("root", 1)
        # Generate ZC and quantize to BPSK bits
        zc = zadoff_chu_sequence(length, root=root)
        xp = np if not is_cupy_available() else dispatch(zc)[1]
        # Map: real > 0 -> 1, else 0
        bits = (zc.real > 0).astype(xp.int32)
        return bits

    elif sequence_type.lower() == "random":
        from .utils import random_bits

        seed = kwargs.get("seed", None)
        return random_bits(length, seed=seed)

    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")
