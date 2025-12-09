from typing import Any

from .backend import ArrayType, ensure_on_backend, get_sp, get_xp


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
    xp = get_xp()

    n_in = samples.shape[0]
    n_out = n_in * factor
    out = xp.zeros(n_out, dtype=samples.dtype)
    out[::factor] = samples
    return out


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
    samples = ensure_on_backend(samples)
    sp = get_sp()
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
        Decimated samples at rate (original_rate / factor).
    """
    samples = ensure_on_backend(samples)
    sp = get_sp()

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
    sp = get_sp()
    return sp.signal.resample_poly(samples, int(up), int(down))
