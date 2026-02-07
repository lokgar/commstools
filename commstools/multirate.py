"""
Multirate signal processing.

This module provides efficient implementations of multirate operations:
- Upsampling (interpolation).
- Downsampling (decimation).
- Rational rate resampling (polyphase filtering).
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
    Safe wrapper for polyphase resampling that handles CuPy multidimensional stability issues.

    Args:
        samples: Input samples.
        up: Upsampling factor.
        down: Downsampling factor.
        axis: Processing axis.
        window: FIR filter window (taps).
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


def expand(samples: ArrayType, factor: int, axis: int = -1) -> ArrayType:
    """
    Zero-insertion: Insert (factor-1) zeros between each sample.

    Args:
        samples: Input sample array.
        factor: Expansion factor (samples per symbol).

    Returns:
        Expanded array with zeros inserted (length = len(samples) * factor) on the same backend.
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
    Upsampling: Expansion (zero-insertion) + anti-imaging filter.

    Increases sample rate by inserting zeros and applying lowpass filter
    to suppress spectral images.

    Args:
        samples: Input sample array.
        factor: Upsampling factor.
        axis: Axis along which to upsample.

    Returns:
        Upsampled samples at rate (factor * original_rate) on the same backend.
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
    Decimate: Anti-aliasing filter followed by downsampling.

    Reduces the sample rate by filtering to remove high-frequency content
    (which would alias) and then keeping every Nth sample.

    Args:
        samples: Input sample array.
        factor: Decimation factor.
        method: Decimation method ('decimate', 'polyphase').
        axis: Axis along which to decimate.
        **kwargs: Additional filter parameters for 'decimate' method.

    Returns:
        Decimated samples at rate (original_rate / factor) on the same backend.
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
    Rational resampling: Upsample by 'up', downsample by 'down'.

    Can also specify 'sps_in' and 'sps_out' to calculate 'up' and 'down' automatically.
    New rate = original_rate * (up / down) = original_rate * (sps_out / sps_in).

    Args:
        samples: Input sample array.
        up: Upsampling factor.
        down: Downsampling factor.
        sps_in: Input samples per symbol.
        sps_out: Target samples per symbol.
        axis: Axis along which to resample.

    Returns:
        Resampled samples at rate (original_rate * up / down) on the same backend.

    Raises:
        ValueError: If arguments are invalid or insufficient.
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
