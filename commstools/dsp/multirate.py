from typing import Any
from ..core.backend import ensure_on_backend, get_backend, ArrayType


def expand(samples: ArrayType, factor: int) -> ArrayType:
    """
    Zero-insertion: Insert (factor-1) zeros between each sample.

    Args:
        samples: Input sample array.
        factor: Expansion factor (samples per symbol).

    Returns:
        Expanded array with zeros inserted (length = len(samples) * factor).
    """
    samples = ensure_on_backend(samples)
    backend = get_backend()
    return backend.expand(samples, int(factor))


def upsample(samples: ArrayType, factor: int) -> ArrayType:
    """
    Upsampling: Expansion (zero-insertion) + anti-imaging filtering.

    Increases sample rate by inserting zeros and applying lowpass filter
    to suppress spectral images.

    Args:
        samples: Input sample array.
        factor: Upsampling factor.

    Returns:
        Upsampled samples at rate (factor * original_rate).
    """
    # Use polyphase upsampling for efficiency
    samples = ensure_on_backend(samples)
    backend = get_backend()

    # resample_poly(x, up, down)
    # upsample by factor means up=factor, down=1
    return backend.resample_poly(samples, int(factor), 1)


def decimate(
    samples: ArrayType, factor: int, filter_type: str = "fir", **kwargs: Any
) -> ArrayType:
    """
    Decimate: Anti-aliasing filter followed by downsampling.

    Reduces the sample rate by filtering to remove high-frequency content
    (which would alias) and then keeping every Nth sample.

    Args:
        samples: Input sample array.
        factor: Decimation factor.
        filter_type: Filter type ('fir', 'iir'). Only 'fir' currently supported.
        **kwargs: Additional filter parameters.

    Returns:
        Decimated samples at rate (original_rate / factor).
    """
    samples = ensure_on_backend(samples)
    backend = get_backend()
    zero_phase = kwargs.get("zero_phase", True)
    return backend.decimate(
        samples, int(factor), ftype=filter_type, zero_phase=zero_phase
    )


def resample(samples: ArrayType, up: int, down: int) -> ArrayType:
    """
    Rational resampling: Upsample by 'up', downsample by 'down'.

    Implements efficient polyphase filtering for arbitrary rational rate conversion.
    New rate = original_rate * (up / down).

    Args:
        samples: Input sample array.
        up: Upsampling factor.
        down: Downsampling factor.

    Returns:
        Resampled samples at rate (original_rate * up / down).
    """
    samples = ensure_on_backend(samples)
    backend = get_backend()
    return backend.resample_poly(samples, int(up), int(down))
