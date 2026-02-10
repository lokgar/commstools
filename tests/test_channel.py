"""Tests for channel impairments and signal degradation models."""

from commstools import impairments


def test_awgn(backend_device, xp):
    """Verify that AWGN addition produces correct noise power levels."""
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

    # Expected noise power is 10^(-SNR/10) = 0.1
    assert 0.08 < measured_noise_power < 0.12


def test_awgn_signal_object(backend_device, xp):
    """Verify add_awgn with Signal objects."""
    from commstools.core import Signal

    sig = Signal.pam(order=2, num_symbols=100, sps=4, symbol_rate=1e6)

    noisy_sig = impairments.add_awgn(sig, esn0_db=10)

    assert isinstance(noisy_sig, Signal)
    assert noisy_sig.samples.shape == sig.samples.shape
    assert noisy_sig.sps == 4


def test_awgn_real_data(backend_device, xp):
    """Verify add_awgn with real-valued data."""
    data = xp.ones(1000, dtype=xp.float32)
    noisy = impairments.add_awgn(data, esn0_db=10)

    assert xp.isrealobj(noisy)
    assert not xp.allclose(noisy, data)


def test_awgn_low_snr(backend_device, xp):
    """Verify add_awgn with extremely low SNR values."""
    data = xp.ones(100, dtype=complex)
    # This should trigger the esn0_linear <= 1e-20 branch
    noisy = impairments.add_awgn(data, esn0_db=-300)
    measured_power = xp.mean(xp.abs(noisy) ** 2)
    assert float(measured_power) > 1e15
