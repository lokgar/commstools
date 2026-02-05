import pytest
from commstools import spectral
from commstools.backend import to_device


def test_welch_psd_real(backend_device, xp):
    """Test PSD for real-valued signals."""
    if backend_device == "gpu":
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not installed")

    # Generate a simple sine wave
    fs = 100.0
    t = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.sin(2 * xp.pi * freq * t)

    samples = to_device(samples, backend_device)

    # 1. Default (one-sided for real)
    f, p = spectral.welch_psd(samples, sampling_rate=fs, nperseg=256)

    assert isinstance(f, xp.ndarray)
    assert isinstance(p, xp.ndarray)

    # Check frequency range: 0 to fs/2
    assert xp.min(f) >= 0
    assert xp.max(f) <= fs / 2 + 1e-6  # small tolerance

    # Check peak frequency
    peak_idx = xp.argmax(p)
    peak_freq = f[peak_idx]
    assert xp.abs(peak_freq - freq) < (fs / 256)  # Resolution check

    # 2. Force two-sided
    f2, p2 = spectral.welch_psd(
        samples, sampling_rate=fs, nperseg=256, return_onesided=False
    )

    # Check frequency range: centered around 0 (fftshifted)
    assert xp.min(f2) < 0

    # Peak should appear at +freq and -freq
    # Tolerant search near positive peak
    peak_idx_pos = xp.argmax(p2 * (f2 > 0))  # mask negative
    peak_freq_pos = f2[peak_idx_pos]
    assert xp.abs(peak_freq_pos - freq) < (fs / 256)


def test_welch_psd_complex(backend_device, xp):
    """Test PSD for complex-valued signals."""
    if backend_device == "gpu":
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not installed")

    # Complex exponential
    fs = 100.0
    t = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.exp(1j * 2 * xp.pi * freq * t)

    samples = to_device(samples, backend_device)

    # 1. Default (two-sided for complex)
    f, p = spectral.welch_psd(samples, sampling_rate=fs, nperseg=256)

    assert isinstance(f, xp.ndarray)
    assert xp.min(f) < 0  # centered

    peak_idx = xp.argmax(p)
    peak_freq = f[peak_idx]

    assert xp.abs(peak_freq - freq) < (fs / 256)

    # 2. Try force one-sided (should fail)
    with pytest.raises(ValueError, match="Cannot compute one-sided PSD"):
        spectral.welch_psd(samples, sampling_rate=fs, return_onesided=True)
