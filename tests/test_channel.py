import pytest
import numpy as np
from commstools import channel


def test_awgn(backend_device, xp):
    data = xp.ones(1000, dtype=complex)
    snr_db = 10.0

    # Run AWGN
    # We need to make sure 'data' is compatible with the backend used inside channel.awgn
    # channel.awgn usually ensures backend usage or uses get_backend().

    # If we want to test on GPU, we should ensure the input is on GPU or backend is set.
    if backend_device == "gpu":
        from commstools.backend import set_backend

        set_backend("gpu")
        data = xp.asarray(data)

    noisy = channel.add_gaussian_noise(data, snr_db)

    assert isinstance(noisy, xp.ndarray)
    assert noisy.shape == data.shape
    assert not xp.allclose(noisy, data)  # Should have noise

    # Rough check of noise power?
    # Signal power is 0 (zeros). Noise power should be related to SNR...
    # Wait, if signal power is 0, SNR definition P_sig/P_noise -> P_noise = P_sig / 10^(SNR/10).
    # If P_sig is 0, P_noise is 0.
    # So actually output should be 0??
    # Let's check implementation.
    # Usually implementation measures signal power. If 0, it might default to something or noise is 0.

    # Let's use non-zero signal
    data = xp.ones(1000, dtype=complex)
    noisy = channel.add_gaussian_noise(data, snr_db)
    # Signal power = 1.
    # Noise power = 1 / 10^(10/10) = 0.1
    noise = noisy - data
    measured_noise_power = xp.mean(xp.abs(noise) ** 2)

    # On GPU checking scalar
    if backend_device == "gpu":
        measured_noise_power = float(measured_noise_power)

    assert 0.08 < measured_noise_power < 0.12  # Allow statistical variance

    if backend_device == "gpu":
        from commstools.backend import set_backend

        set_backend("cpu")
