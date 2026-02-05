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


def expand(samples: ArrayType, factor: int) -> ArrayType:
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

    n_in = samples.shape[0]
    n_out = n_in * factor
    out = xp.zeros(n_out, dtype=samples.dtype)
    out[::factor] = samples
    return out


def upsample(samples: ArrayType, factor: int) -> ArrayType:
    """
    Upsampling: Expansion (zero-insertion) + anti-imaging filter.

    Increases sample rate by inserting zeros and applying lowpass filter
    to suppress spectral images.

    Args:
        samples: Input sample array.
        factor: Upsampling factor.

    Returns:
        Upsampled samples at rate (factor * original_rate) on the same backend.
    """
    logger.debug(f"Upsampling by factor {factor} (polyphase).")
    samples, _, sp = dispatch(samples)
    return sp.signal.resample_poly(samples, factor, 1)


def decimate(
    samples: ArrayType, factor: int, method: str = "decimate", **kwargs: Any
) -> ArrayType:
    """
    Decimate: Anti-aliasing filter followed by downsampling.

    Reduces the sample rate by filtering to remove high-frequency content
    (which would alias) and then keeping every Nth sample.

    Args:
        samples: Input sample array.
        factor: Decimation factor.
        method: Decimation method ('decimate', 'polyphase').
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
            samples, int(factor), ftype=ftype, zero_phase=zero_phase
        )

    elif method == "polyphase":
        # resample_poly with up=1
        return sp.signal.resample_poly(samples, 1, int(factor))

    else:
        raise ValueError(f"Unknown decimation method: {method}")


def resample(
    samples: ArrayType,
    up: Optional[int] = None,
    down: Optional[int] = None,
    sps_in: Optional[float] = None,
    sps_out: Optional[float] = None,
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

    logger.debug(f"Resampling by rational factor {up}/{down} (polyphase).")
    samples, _, sp = dispatch(samples)
    return sp.signal.resample_poly(samples, int(up), int(down))
