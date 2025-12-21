import pytest
import numpy as np
from commstools import impairments, backend


def test_awgn(backend_device, xp):
    data = np.ones(1000, dtype=complex)
    data = backend.to_device(data, backend_device)

    snr_db = 10.0

    noisy = impairments.add_gaussian_noise(data, snr_db)

    assert isinstance(noisy, xp.ndarray)
    assert noisy.shape == data.shape
    assert not xp.allclose(noisy, data)

    # Check approximate noise power
    # Signal power is 1.0. SNR=10dB => ratio=10. Noise power = 0.1
    noise = noisy - data
    measured_noise_power = xp.mean(xp.abs(noise) ** 2)

    if backend_device == "gpu":
        measured_noise_power = float(measured_noise_power)

    assert 0.08 < measured_noise_power < 0.12
