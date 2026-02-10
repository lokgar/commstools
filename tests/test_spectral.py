"""Tests for spectral analysis routines (Welch PSD, Frequency shifting)."""

import pytest

from commstools import spectral


def test_welch_psd_real(backend_device, xp):
    """Verify Welch PSD estimation for real-valued signals, including one-sided/two-sided modes."""
    # Generate a simple sine wave
    fs = 100.0
    t = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.sin(2 * xp.pi * freq * t)

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
    """Verify Welch PSD estimation for complex-valued signals."""
    # Complex exponential
    fs = 100.0
    t = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.exp(1j * 2 * xp.pi * freq * t)

    # 1. Default (two-sided for complex)
    f, p = spectral.welch_psd(samples, sampling_rate=fs, nperseg=256)

    assert isinstance(f, xp.ndarray)
    assert xp.min(f) < 0  # centered

    peak_idx = xp.argmax(p)
    peak_freq = f[peak_idx]

    assert xp.abs(peak_freq - freq) < (fs / 256)

    # 2. Try force one-sided (should fail for complex)
    with pytest.raises(ValueError, match="Cannot compute one-sided PSD"):
        spectral.welch_psd(samples, sampling_rate=fs, return_onesided=True)


def test_shift_frequency(backend_device, xp):
    """Verify complex frequency shifting and energy preservation."""
    # 1. Exact integer shift
    # fs=100, N=100 -> df=1Hz. Shift by 10Hz.
    fs = 100.0
    N = 100
    t = xp.arange(N) / fs
    # Signal at 20 Hz
    s = xp.exp(1j * 2 * xp.pi * 20 * t)

    shifted, actual = spectral.shift_frequency(s, offset=10.0, fs=fs)
    assert actual == 10.0

    # New frequency should be 30 Hz
    f_axis = xp.fft.fftfreq(N, 1 / fs)
    peak_idx = xp.argmax(xp.abs(xp.fft.fft(shifted)))
    peak_freq = f_axis[peak_idx]
    assert xp.isclose(peak_freq, 30.0)

    # 2. Quantized shift
    # Shift by 10.5 Hz. Should be quantized to integer multiple of df.
    shifted_q, actual_q = spectral.shift_frequency(s, offset=10.5, fs=fs)
    # Check it is integer multiple of df=1
    assert actual_q % 1.0 == 0.0
    assert abs(actual_q - 10.5) <= 0.5

    # 3. Energy preservation (unitary)
    energy_in = xp.sum(xp.abs(s) ** 2)
    energy_out = xp.sum(xp.abs(shifted_q) ** 2)
    assert xp.isclose(energy_in, energy_out)
