"""
Multirate signal processing and resampling.

This module provides high-performance implementations of multirate
operations, including interpolation, decimation, and rational rate
conversion using polyphase filter banks.

Functions
---------
polyphase_resample :
    Performs rational rate conversion via polyphase filtering.
resample :
    High-level interface for arbitrary rate changes.
upsample :
    Increases sampling rate by an integer factor with anti-imaging.
decimate :
    Reduces sampling rate with anti-aliasing filtering.
expand :
    Inserts zeros between samples (zero-stuffing).
downsample_to_symbols :
    Optimized symbol extraction after matched filtering.
"""

from fractions import Fraction
from typing import Any, Optional

from .backend import ArrayType, dispatch
from .logger import logger


def polyphase_resample(
    samples: ArrayType,
    up: int,
    down: int,
    axis: int = -1,
    window: Optional[ArrayType] = None,
) -> ArrayType:
    """
    Rational resampling using polyphase filtering.

    This method changes the sampling rate of a signal by a rational factor
    `up/down`. It applies an anti-aliasing/anti-imaging filter during the
    process to prevent spectral overlap.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N_samples).
    up : int
        Upsampling factor (interpolation).
    down : int
        Downsampling factor (decimation).
    axis : int, default -1
        The axis along which to perform resampling.
    window : array_like, optional
        Custom FIR filter taps (window) to use for the polyphase filter.
        If None, a Kaiser window is used by default. Shape: (N_taps,).

    Returns
    -------
    array_like
        Resampled signal. Shape: (..., N_samples * up / down).
        Backend (NumPy/CuPy) matches the input `samples`.

    Notes
    -----
    The CuPy implementation includes a stability workaround for
    multidimensional arrays by iterating over channels when necessary,
    avoiding certain CUDA-level kernel errors in `resample_poly`.
    """
    samples, xp, sp = dispatch(samples)

    # Check if CuPy and multidimensional
    # cupyx.scipy.signal.resample_poly crashes with CUDA_ERROR_INVALID_VALUE on some 2D/3D inputs
    # even with correct axis alignment. We iterate over channels to avoid this.
    is_cupy = xp.__name__ == "cupy"
    if is_cupy and samples.ndim > 1:
        # Move processing axis to -1 for canonical iteration
        # Note: axis might be negative
        samples_moved = xp.moveaxis(samples, axis, -1)
        original_shape = samples_moved.shape
        n_samples = original_shape[-1]

        # Flatten non-processing dimensions: (C1, C2, ..., N) -> (FlatC, N)
        samples_flat = samples_moved.reshape(-1, n_samples)

        # Prepare kwargs
        kwargs = {}
        if window is not None:
            kwargs["window"] = window

        # Pre-allocate output: compute output length from first row
        first_out = sp.signal.resample_poly(
            samples_flat[0], up, down, axis=-1, **kwargs
        )
        n_out = first_out.shape[-1]
        n_channels = samples_flat.shape[0]

        # Pre-allocate result array (avoid list.append() in DSP loops)
        result_flat = xp.empty((n_channels, n_out), dtype=samples_flat.dtype)
        result_flat[0] = first_out

        for i in range(1, n_channels):
            result_flat[i] = sp.signal.resample_poly(
                samples_flat[i], up, down, axis=-1, **kwargs
            )

        res = result_flat

        # Reshape back to (C1, C2, ..., NewN)
        new_shape = list(original_shape)
        new_shape[-1] = res.shape[-1]
        res = res.reshape(new_shape)

        # Move axis back
        res = xp.moveaxis(res, -1, axis)
        return res
    else:
        if window is not None:
            return sp.signal.resample_poly(samples, up, down, axis=axis, window=window)
        return sp.signal.resample_poly(samples, up, down, axis=axis)


def downsample_to_symbols(
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
        Samples per symbol (decimation factor).
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

    This is a convenience wrapper around `polyphase_resample` that performs
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
    return polyphase_resample(samples, factor, 1, axis=axis)


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
        - "polyphase": Uses `polyphase_resample` for filter-and-sample.
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
    signal. Use `downsample_to_symbols` instead.
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
        return polyphase_resample(samples, 1, int(factor), axis=axis)

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
    return polyphase_resample(samples, int(up), int(down), axis=axis)
