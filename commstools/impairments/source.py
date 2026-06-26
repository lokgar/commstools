"""Optical/electronic source impairments (laser linewidth phase noise)."""

import math

from ..backend import ArrayType, dispatch
from ..logger import logger

__all__ = ["apply_phase_noise"]


def apply_phase_noise(
    samples: ArrayType,
    sampling_rate: float,
    linewidth: float,
    seed: int | None = None,
    shared_lo: bool = False,
) -> ArrayType:
    """
    Adds laser / oscillator phase noise modelled as a Wiener (random-walk) process.

    Each sample is rotated by an accumulated phase drawn from a discrete Wiener
    process whose per-sample variance is set by the laser linewidth:

        phi[n] = sum_{k=0}^{n} delta_k,  delta_k ~ N(0, 2*pi*delta_nu / f_s)

        r[n] = s[n] * exp(j * phi[n])

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    sampling_rate : float
        Sampling rate in Hz.
    linewidth : float
        Combined transmitter + receiver laser linewidth delta_nu in Hz.
        Typical values: 100 kHz (narrow-linewidth laser) to 10 MHz (DFB).
    seed : int, optional
        Random seed for reproducible noise.
    shared_lo : bool, default False
        When ``False`` (default), each channel receives independent phase noise
        (separate oscillators / lasers per TX-RX path).
        When ``True``, a single phase noise trajectory is shared across all
        channels (common local oscillator in a coherent system).

    Returns
    -------
    array_like
        Phase-noise-impaired signal, same shape, dtype, and backend as input.

    Examples
    --------
    >>> noisy = apply_phase_noise(sig.samples, linewidth=100e3,
    ...                           sampling_rate=sig.sampling_rate)
    """
    logger.info(
        f"Applying phase noise (linewidth={linewidth:.3g} Hz, shared_lo={shared_lo})."
    )

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)
    C, N = samples.shape

    variance_per_sample = 2.0 * math.pi * linewidth / sampling_rate
    std_per_sample = math.sqrt(variance_per_sample)

    rng = xp.random.RandomState(seed) if seed is not None else xp.random

    if shared_lo:
        # One trajectory shared across all channels
        increments = rng.normal(0.0, std_per_sample, N).astype(xp.float64)
        phase = xp.cumsum(increments)  # (N,)
        result = samples * xp.exp(1j * phase[None, :])
    else:
        # Independent trajectory per channel
        increments = rng.normal(0.0, std_per_sample, (C, N)).astype(xp.float64)
        phase = xp.cumsum(increments, axis=-1)  # (C, N)
        result = samples * xp.exp(1j * phase)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    if was_1d:
        return result[0]
    return result
