"""
Synchronization utilities for frame detection and sequence generation.

This module provides primitives for:
- **Cross-correlation**: For preamble detection and timing recovery.
- **Frame detection**: Finding frame start via preamble correlation.
- **Barker sequences**: Low auto-correlation sync sequences.
- **Zadoff-Chu sequences**: CAZAC sequences for timing/frequency sync.
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
    Generate a Barker sequence of specified length.

    Barker sequences have optimal auto-correlation properties with sidelobe
    levels at most 1/N of the peak. Commonly used for frame synchronization.

    Args:
        length: Sequence length. Must be one of: 2, 3, 4, 5, 7, 11, 13.

    Returns:
        Array of BPSK symbols (+1/-1) on default backend.

    Raises:
        ValueError: If length is not a valid Barker length.

    Note:
        Only lengths with known Barker sequences are supported.
        No Barker sequences exist for lengths > 13.
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
    Generate a Zadoff-Chu (ZC) sequence.

    ZC sequences are CAZAC (Constant Amplitude Zero Auto-Correlation) sequences
    used extensively in LTE/5G for synchronization signals (PSS, SRS).

    Args:
        length: Sequence length N (should be prime for optimal properties).
        root: Root index u (must be coprime with length). Default: 1.

    Returns:
        Complex array of unit-magnitude samples on default backend.

    Note:
        - ZC sequences have perfect periodic auto-correlation.
        - Cross-correlation between different roots is low.
        - For LTE PSS, use lengths 63 or 839 with roots 25, 29, 34.

    Example:
        >>> zc = zadoff_chu_sequence(63, root=25)  # LTE PSS
        >>> assert np.allclose(np.abs(zc), 1.0)  # Unit magnitude
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
    Cross-correlation between signal and template.

    Uses FFT-based correlation for efficiency with long sequences.

    Args:
        signal: Input signal array. Shape: (N,) or (Channels, N).
        template: Template/reference to correlate against. Shape: (M,).
        mode: Output size mode:
            - "full": Full correlation, length N+M-1 (default).
            - "same": Same size as signal, length N.
            - "valid": Only complete overlaps, length max(N,M) - min(N,M) + 1.
        normalize: If True, normalize by template energy for detection threshold.

    Returns:
        Correlation output array.

    Note:
        For frame detection, the peak location indicates timing offset.
        Normalize=True gives output in [0, 1] for threshold-based detection.
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
    """1D correlation helper."""
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
    Detect frame start via preamble correlation.

    Finds the sample index where the preamble begins by correlating
    the received signal with the known preamble sequence.

    Args:
        signal: Received signal (array or Signal object).
                For Signal objects, uses samples directly.
        preamble: Known preamble (array or Preamble object).
                  For Preamble objects, uses symbols.
        threshold: Detection threshold (0-1). Peak must exceed this.
        return_metric: If True, also return the normalized correlation peak.
        search_range: Optional (start, end) sample range to search.
                      Default: search entire signal.

    Returns:
        frame_start: Sample index where preamble begins.
        If return_metric=True: (frame_start, peak_metric).

    Raises:
        ValueError: If no peak exceeds threshold.

    Note:
        - For oversampled signals, ensure preamble is at same sample rate.
        - The returned index is relative to the start of the signal (or search_range).
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
    Generate standard preamble bit sequences.

    Args:
        sequence_type: Type of sequence:
            - "barker": Barker sequence (length must be valid Barker length).
            - "zc" or "zadoff_chu": Zadoff-Chu (returns BPSK-quantized version).
            - "random": Random bits.
        length: Sequence length (in bits for Barker/random, symbols for ZC).
        **kwargs: Additional parameters (e.g., root for ZC).

    Returns:
        Bit array (0s and 1s) suitable for Preamble class.
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
