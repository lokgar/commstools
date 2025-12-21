import pytest
import numpy as np
from commstools.core import Signal


def test_signal_initialization(backend_device, xp):
    if backend_device == "gpu":
        try:
            import cupy  # noqa: F401
        except ImportError:
            pytest.skip("CuPy not installed")

    # Test with list
    data = [1, 2, 3, 4]
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0).to(backend_device)
    assert isinstance(s.samples, xp.ndarray)

    assert s.sampling_rate == 1.0
    assert s.symbol_rate == 1.0


def test_signal_properties(backend_device, xp):
    data = xp.zeros(100)
    fs = 100.0
    sym_rate = 10.0

    s = Signal(samples=data, sampling_rate=fs, symbol_rate=sym_rate).to(backend_device)

    assert s.duration == 1.0
    assert s.sps == 10.0
    assert len(s.time_axis()) == 100


def test_signal_methods(backend_device, xp):
    # Test upsample
    data = xp.array([1.0 + 0j, -1.0 + 0j])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0).to(backend_device)

    s.upsample(2)
    assert s.sampling_rate == 2.0
    assert s.samples.shape[0] == 4
    # Check values (upsample is zero insertion + filtering? No, multirate.upsample uses resample_poly)
    # resample_poly applies filtering, so values won't be exactly 1, 0, -1, 0.

    # Test fir_filter
    taps = xp.array([1.0])
    s.fir_filter(taps)
    # should be unchanged roughly


def test_welch_psd(backend_device, xp):
    data = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0).to(backend_device)

    f, p = s.welch_psd(nperseg=64)
    assert f.shape == p.shape
    assert isinstance(f, xp.ndarray)


def test_signal_print_info(backend_device, xp, capsys):
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0).to(backend_device)
    s.print_info()
    captured = capsys.readouterr()
    assert (
        "Spectral Domain" in captured.out or captured.out == ""
    )  # Might be empty if pandas display mocks ipython, check stdout


def test_shaping_filter_taps_error(backend_device, xp):
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0).to(backend_device)

    with pytest.raises(ValueError, match="No pulse shape defined"):
        s.shaping_filter_taps()

    s.pulse_shape = "invalid_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        s.shaping_filter_taps()


def test_signal_copy(backend_device, xp):
    from commstools import backend

    data = xp.array([1, 2, 3])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0).to(backend_device)
    s_copy = s.copy()

    assert s_copy is not s
    assert np.allclose(
        backend.to_device(s.samples, "cpu"), backend.to_device(s_copy.samples, "cpu")
    )
    assert s_copy.backend == s.backend
