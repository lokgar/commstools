"""
Impairment models for communication channels.

This module provides functions to simulate various physical channel impairments,
allowing for the evaluation of receiver performance under realistic conditions.
Currently supported:
- **Additive White Gaussian Noise (AWGN)**: Adds random noise to the signal based on a target SNR.
"""

from typing import TYPE_CHECKING, Union

from .backend import ArrayType, dispatch
from .logger import logger

if TYPE_CHECKING:
    from .core import Signal


def add_gaussian_noise(
    signal: Union[ArrayType, "Signal"], snr_db: float
) -> Union[ArrayType, "Signal"]:
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal to achieve a target SNR.

    Args:
        signal: The input signal (NumPy/CuPy array or `Signal` object).
        snr_db: The desired Signal-to-Noise Ratio (SNR) in decibels.

    Returns:
        The noisy signal. If the input was a `Signal` object, a new `Signal`
        instance with updated samples is returned.
    """
    logger.info(f"Adding Gaussian noise (SNR target: {snr_db:.2f} dB).")
    # Check if signal is a Signal object
    from .core import Signal

    if isinstance(signal, Signal):
        samples = signal.samples
    else:
        samples = signal

    samples, xp, _ = dispatch(samples)

    signal_power = xp.mean(xp.abs(samples) ** 2)
    snr_linear = 10 ** (snr_db / 10)

    # Handle very low SNR or infinite noise case
    if snr_linear <= 1e-20:
        # Fallback to huge noise
        noise_power = signal_power / 1e-20
    else:
        noise_power = signal_power / snr_linear

    noise_std = xp.sqrt(noise_power)

    # Handle complex signals
    is_complex = xp.iscomplexobj(samples)

    if is_complex:
        # For complex noise, power is split between real and imag
        noise_std_component = xp.sqrt(noise_power / 2)

        # Generate noise on backend
        noise = xp.random.normal(
            0, noise_std_component, samples.shape
        ) + 1j * xp.random.normal(0, noise_std_component, samples.shape)
    else:
        # Real noise
        noise = xp.random.normal(0, noise_std, samples.shape)

    # noise is already on backend
    noisy_samples = samples + noise

    if isinstance(signal, Signal):
        sig = signal.copy()
        sig.samples = noisy_samples
        return sig
    else:
        return noisy_samples
