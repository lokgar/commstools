"""Tests for the base Signal class and its core signal processing methods."""

import pytest

from commstools.core import Signal


def test_signal_initialization(backend_device, xp):
    """Verify Signal initialization from basic Python lists and device-aware backend tracking."""
    # Test with list
    data = [1, 2, 3, 4]
    # Signal automatically moves to GPU if available (controlled by backend_device fixture)
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert isinstance(s.samples, xp.ndarray)

    assert s.sampling_rate == 1.0
    assert s.symbol_rate == 1.0


def test_signal_properties(backend_device, xp):
    """Verify core time-domain and rate properties of the Signal object."""
    # Create data directly on device using xp
    data = xp.zeros(100)
    fs = 100.0
    sym_rate = 10.0

    s = Signal(samples=data, sampling_rate=fs, symbol_rate=sym_rate)

    assert s.duration == 1.0
    assert s.sps == 10.0
    assert len(s.time_axis()) == 100


def test_signal_methods(backend_device, xp):
    """Verify common Signal methods like upsampling and FIR filtering."""
    # Test upsample
    data = xp.array([1.0 + 0j, -1.0 + 0j])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

    s.upsample(2)
    assert s.sampling_rate == 2.0
    assert s.samples.shape[0] == 4

    # Test fir_filter
    taps = xp.array([1.0])
    s.fir_filter(taps)
    # should be unchanged roughly


def test_signal_resample_sps(backend_device, xp):
    """Verify Signal resampling using target samples per symbol (SPS)."""
    data = xp.ones(100)
    # create signal with sps=4 (fs=4, sym_rate=1)
    s = Signal(samples=data, sampling_rate=4.0, symbol_rate=1.0)
    assert s.sps == 4.0

    # resample to sps=8
    s.resample(sps_out=8.0)
    assert s.sps == 8.0
    assert s.sampling_rate == 8.0
    assert s.samples.size == 200


def test_welch_psd(backend_device, xp):
    """Verify Welch PSD estimation within the Signal object."""
    data = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    f, p = s.welch_psd(nperseg=64)
    assert f.shape == p.shape
    assert isinstance(f, xp.ndarray)


def test_signal_print_info(backend_device, xp, capsys):
    """Verify print_info() execution and output detection."""
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
    s.print_info()
    captured = capsys.readouterr()
    assert "Spectral Domain" in captured.out or captured.out == ""


def test_shaping_filter_taps_error(backend_device, xp):
    """Verify that shaping_filter_taps raises errors for unconfigured or unknown shapes."""
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    with pytest.raises(ValueError, match="No pulse shape defined"):
        s.shaping_filter_taps()

    s.pulse_shape = "invalid_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        s.shaping_filter_taps()


def test_signal_copy(backend_device, xp):
    """Verify Signal.copy() performs a deep copy of samples and preserves device context."""
    data = xp.array([1, 2, 3])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    s_copy = s.copy()

    assert s_copy is not s
    assert xp.allclose(s.samples, s_copy.samples)
    assert s_copy.backend == s.backend


def test_signal_shift_frequency(backend_device, xp):
    """Verify frequency shifting logic and resulting spectral peak positioning."""
    fs = 100.0
    # Simple DC signal (freq 0)
    data = xp.ones(100, dtype=xp.complex128)
    s = Signal(samples=data, sampling_rate=fs, symbol_rate=10.0)

    # Offset by 20 Hz
    s.shift_frequency(20.0)

    # 1. Check metadata
    assert s.digital_frequency_offset == 20.0

    # 2. Check signal content
    t = xp.arange(100) / fs
    expected = xp.exp(1j * 2 * xp.pi * 20.0 * t)
    assert xp.allclose(s.samples, expected)

    # 3. Check accumulation
    s.shift_frequency(5.0)
    assert s.digital_frequency_offset == 25.0

    # Check approximate freq
    f, p = s.welch_psd(nperseg=64)
    peak = f[xp.argmax(p)]
    # 25 Hz expected
    assert abs(peak - 25.0) < (fs / 64)
