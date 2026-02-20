"""
Multirate signal processing and resampling.

This module provides high-performance implementations of multirate
operations, including interpolation, decimation, and rational rate
conversion using polyphase filter banks.

Functions
---------
decimate_to_symbol_rate :
    Optimized symbol extraction after matched filtering.
expand :
    Inserts zeros between samples (zero-stuffing).
upsample :
    Increases sampling rate by an integer factor with anti-imaging.
decimate :
    Reduces sampling rate with anti-aliasing filtering.
resample :
    High-level interface for arbitrary rate changes.
"""

from fractions import Fraction
from typing import Any, Optional

from .backend import ArrayType, dispatch
from .logger import logger


def decimate_to_symbol_rate(
    samples: ArrayType,
    sps: int,
    offset: int = 0,
    axis: int = -1,
) -> ArrayType:
    """
    Decimates an oversampled signal to symbol-rate by direct slicing.

    This function should be used **after** matched filtering to extract
    pulse-shaped symbols at $1 \text{ sps}$. It does not apply additional
    filtering, which is correct since the matched filter has already
    performed optimal noise suppression.

    Parameters
    ----------
    samples : array_like
        Input matched-filtered signal. Shape: (..., N_samples).
    sps : int
        Input samples per symbol (decimation factor).
    offset : int, default 0
        Sampling phase offset in samples [0, sps-1]. Adjust this to
        sample at the peak of the impulse response (center of the eye).
    axis : int, default -1
        The axis along which to downsample.

    Returns
    -------
    array_like
        Symbols at 1 sps. Shape: (..., N_samples / sps).
    """
    logger.debug(f"Downsampling to symbols: sps={sps}, offset={offset}")
    samples, xp, _ = dispatch(samples)

    # Build slicing for arbitrary axis
    slices = [slice(None)] * samples.ndim
    slices[axis] = slice(offset, None, sps)

    return samples[tuple(slices)]


def expand(samples: ArrayType, factor: int, axis: int = -1) -> ArrayType:
    """
    Inserts zeros between samples (up-sampling by zero-stuffing).

    This operation increases the sampling rate by an integer factor by
    inserting `factor - 1` zeros between each original sample. This is the
    first step in traditional interpolation but requires subsequent
    filtering to remove spectral images.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N_samples).
    factor : int
        The expansion factor (number of output samples per input sample).
    axis : int, default -1
        The axis along which to perform expansion.

    Returns
    -------
    array_like
        The expanded sample array with zeros inserted.
        Shape: (..., N_samples * factor).
    """
    logger.debug(f"Inserting zeros (expansion factor={factor}).")
    samples, xp, _ = dispatch(samples)

    n_in = samples.shape[axis]
    n_out = n_in * factor

    # Construct output shape
    out_shape = list(samples.shape)
    out_shape[axis] = n_out

    out = xp.zeros(out_shape, dtype=samples.dtype)

    # Slice logic to insert
    # We want out[..., ::factor, ...] = samples
    # Construct slices dynamically
    slices = [slice(None)] * samples.ndim
    slices[axis] = slice(None, None, factor)
    out[tuple(slices)] = samples

    return out


def upsample(samples: ArrayType, factor: int, axis: int = -1) -> ArrayType:
    """
    Increases the sampling rate by an integer factor with filtering.

    This is a convenience wrapper around `resample_poly` that performs
    both zero-insertion (expansion) and anti-imaging filtering to suppress
    spectral replicas.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N_samples).
    factor : int
        The interpolation factor.
    axis : int, default -1
        The axis along which to perform upsampling.

    Returns
    -------
    array_like
        The upsampled signal. Shape: (..., N_samples * factor).
    """
    logger.debug(f"Upsampling by factor {factor} (polyphase, axis={axis}).")
    samples, xp, sp = dispatch(samples)
    return sp.signal.resample_poly(samples, factor, 1, axis=axis)


def decimate(
    samples: ArrayType,
    factor: int,
    method: str = "decimate",
    axis: int = -1,
    **kwargs: Any,
) -> ArrayType:
    """
    Reduces the sampling rate with anti-aliasing filtering.

    Decimation combines lowpass filtering (to prevent aliasing) with
    downsampling (keeping every Nth sample).

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N_samples).
    factor : int
        The decimation factor.
    method : {"decimate", "polyphase"}, default "decimate"
        The implementation strategy:
        - "decimate": Uses `scipy.signal.decimate` (Chebyshev I or FIR).
        - "polyphase": Uses `resample_poly` for filter-and-sample.
    axis : int, default -1
        The axis along which to perform decimation.
    **kwargs : Any
        Additional parameters passed to the underlying filter design, such
        as `zero_phase` or `ftype`.

    Returns
    -------
    array_like
        The decimated signal. Shape: (..., N_samples / factor).

    Notes
    -----
    Do NOT use this function for symbol extraction after a matched filter.
    Matched filters already perform optimal noise suppression and
    anti-aliasing; adding an extra decimation filter will degrade the
    signal. Use `decimate_to_symbol_rate` instead.
    """
    logger.debug(f"Decimating by factor {factor} (method: {method}).")
    samples, _, sp = dispatch(samples)

    if method == "decimate":
        # scipy.signal.decimate (includes antialiasing)
        zero_phase = kwargs.get("zero_phase", True)
        ftype = kwargs.get("ftype", "fir")
        return sp.signal.decimate(
            samples, int(factor), ftype=ftype, axis=axis, zero_phase=zero_phase
        )

    elif method == "polyphase":
        # resample_poly with up=1
        return sp.signal.resample_poly(samples, 1, int(factor), axis=axis)

    else:
        raise ValueError(f"Unknown decimation method: {method}")


def resample(
    samples: ArrayType,
    up: Optional[int] = None,
    down: Optional[int] = None,
    sps_in: Optional[float] = None,
    sps_out: Optional[float] = None,
    axis: int = -1,
) -> ArrayType:
    """
    Performs rational resampling of a signal.

    Changes the sampling rate of the input by a rational factor. The rate
    can be specified either as direct integer factors (`up`, `down`) or
    relative to symbols (`sps_in`, `sps_out`).

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N_samples).
    up : int, optional
        Integer upsampling factor.
    down : int, optional
        Integer downsampling factor.
    sps_in : float, optional
        Input samples per symbol.
    sps_out : float, optional
        Target samples per symbol.
    axis : int, default -1
        The axis along which to perform resampling.

    Returns
    -------
    array_like
        The resampled signal. Shape: (..., N_samples * Ratio).

    Raises
    ------
    ValueError
        If parameters are insufficient or contradictory.
    """
    if (up is not None or down is not None) and (
        sps_in is not None or sps_out is not None
    ):
        raise ValueError("Cannot specify both (up, down) and (sps_in, sps_out).")

    if sps_in is not None and sps_out is not None:
        ratio = Fraction(sps_out / sps_in).limit_denominator()
        up = ratio.numerator
        down = ratio.denominator
    elif up is None or down is None:
        raise ValueError("Must specify either (up, down) or (sps_in, sps_out).")

    logger.debug(f"Resampling by rational factor {up}/{down} (polyphase, axis={axis}).")
    samples, xp, sp = dispatch(samples)
    return sp.signal.resample_poly(samples, int(up), int(down), axis=axis)
