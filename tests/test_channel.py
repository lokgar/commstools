import numpy as np
import pytest
from commstools.channel import calculate_snr, add_gaussian_noise


def test_calculate_snr():
    signal_pow = 10.0
    noise_pow = 1.0
    snr = calculate_snr(signal_pow, noise_pow)
    assert snr == 10.0  # 10 log10(10) = 10


def test_add_gaussian_noise():
    # Create a constant signal
    sig = np.ones(1000)
    target_snr = 10.0
    noisy = add_gaussian_noise(sig, snr_db=target_snr)

    assert len(noisy) == len(sig)

    # Check if SNR is roughly correct
    noise = noisy - sig
    signal_power = np.mean(sig**2)
    noise_power = np.mean(noise**2)
    measured_snr = 10 * np.log10(signal_power / noise_power)

    # Allow some statistical deviation
    assert np.isclose(measured_snr, target_snr, atol=1.0)
