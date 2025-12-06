import numpy as np
from .backend import ArrayType


def calculate_snr(signal_power: float, noise_power: float) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        signal_power: The power of the signal.
        noise_power: The power of the noise.

    Returns:
        The SNR in decibels.
    """
    if noise_power <= 0:
        return float("inf")  # Or handle as an error, depending on requirements
    return 10 * np.log10(signal_power / noise_power)


def add_gaussian_noise(signal: ArrayType, snr_db: float) -> ArrayType:
    """
    Adds Gaussian noise to a signal to achieve a specified SNR.

    Args:
        signal: The input signal (array).
        snr_db: The desired Signal-to-Noise Ratio in decibels.

    Returns:
        The signal with added Gaussian noise.
    """
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    if snr_linear <= 0:
        # If SNR is very low or negative, noise power will be very high
        noise_power = signal_power / 1e-10  # Effectively infinite noise
    else:
        noise_power = signal_power / snr_linear

    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, signal.shape)
    return signal + noise
