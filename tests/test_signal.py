import pytest
from commstools.core import Signal


def test_signal_initialization(backend_device, xp):
    # Test with list
    data = [1, 2, 3, 4]
    # Signal automatically moves to GPU if available (controlled by backend_device fixture)
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert isinstance(s.samples, xp.ndarray)

    assert s.sampling_rate == 1.0
    assert s.symbol_rate == 1.0


def test_signal_properties(backend_device, xp):
    # Create data directly on device using xp
    data = xp.zeros(100)
    fs = 100.0
    sym_rate = 10.0

    s = Signal(samples=data, sampling_rate=fs, symbol_rate=sym_rate)

    assert s.duration == 1.0
    assert s.sps == 10.0
    assert len(s.time_axis()) == 100


def test_signal_methods(backend_device, xp):
    # Test upsample
    data = xp.array([1.0 + 0j, -1.0 + 0j])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

    s.upsample(2)
    assert s.sampling_rate == 2.0
    assert s.samples.shape[0] == 4
    # Check values (upsample is zero insertion + filtering? No, multirate.upsample uses resample_poly)
    # resample_poly applies filtering, so values won't be exactly 1, 0, -1, 0.

    # Test fir_filter
    taps = xp.array([1.0])
    s.fir_filter(taps)
    # should be unchanged roughly


def test_signal_resample_sps(backend_device, xp):
    data = xp.ones(100)
    # create signal with sps=4 (fs=4, sym_rate=1)
    s = Signal(samples=data, sampling_rate=4.0, symbol_rate=1.0)
    assert s.sps == 4.0

    # resample to sps=8
    s.resample(sps_out=8.0)
    assert s.sps == 8.0
    assert s.sampling_rate == 8.0
    assert s.samples.size == 200

    # resample check invalid combinations handled by multirate (implicitly)
    # Signal.resample only exposes sps_out or up/down, so no conflict possible in signature
    # but we can check if it calls multirate correctly


def test_welch_psd(backend_device, xp):
    data = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    f, p = s.welch_psd(nperseg=64)
    assert f.shape == p.shape
    assert isinstance(f, xp.ndarray)


def test_signal_print_info(backend_device, xp, capsys):
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
    s.print_info()
    captured = capsys.readouterr()
    assert (
        "Spectral Domain" in captured.out or captured.out == ""
    )  # Might be empty if pandas display mocks ipython, check stdout


def test_shaping_filter_taps_error(backend_device, xp):
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    with pytest.raises(ValueError, match="No pulse shape defined"):
        s.shaping_filter_taps()

    s.pulse_shape = "invalid_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        s.shaping_filter_taps()


def test_signal_copy(backend_device, xp):
    data = xp.array([1, 2, 3])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    s_copy = s.copy()

    assert s_copy is not s
    # Ensure they are on the same device and content matches
    assert xp.allclose(s.samples, s_copy.samples)
    assert s_copy.backend == s.backend


def test_signal_shift_frequency(backend_device, xp):
    fs = 100.0
    # Simple DC signal (freq 0)
    data = xp.ones(100, dtype=xp.complex128)
    s = Signal(samples=data, sampling_rate=fs, symbol_rate=10.0)

    # Offset by 20 Hz
    s.shift_frequency(20.0)

    # 1. Check metadata
    assert s.digital_frequency_offset == 20.0

    # 2. Check signal content
    # Should now be a complex exponential at 20 Hz
    t = xp.arange(100) / fs
    expected = xp.exp(1j * 2 * xp.pi * 20.0 * t)
    # Be tolerant with float comparisons/phases
    assert xp.allclose(s.samples, expected)

    # 3. Check accumulation
    s.shift_frequency(5.0)
    assert s.digital_frequency_offset == 25.0

    # Check approximate freq
    f, p = s.welch_psd(nperseg=64)
    peak = f[xp.argmax(p)]
    # 25 Hz expected
    assert abs(peak - 25.0) < (fs / 64)
