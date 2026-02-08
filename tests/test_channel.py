from commstools import impairments


def test_awgn(backend_device, xp):
    data = xp.ones(1000, dtype=complex)

    snr_db = 10.0

    noisy = impairments.add_awgn(data, snr_db)

    assert isinstance(noisy, xp.ndarray)
    assert noisy.shape == data.shape
    assert not xp.allclose(noisy, data)

    # Check approximate noise power
    # Signal power is 1.0. SNR=10dB => ratio=10. Noise power = 0.1
    noise = noisy - data
    measured_noise_power = xp.mean(xp.abs(noise) ** 2)

    # xp.mean returns a 0-d array on numpy/cupy, casting to float handles both
    measured_noise_power = float(measured_noise_power)

    assert 0.08 < measured_noise_power < 0.12
