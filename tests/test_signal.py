import pytest
import numpy as np
import pytest
from commstools.signal import Signal


def test_signal_initialization(backend_device, xp):
    if backend_device == "gpu":
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not installed")

    # Test with list
    data = [1, 2, 3, 4]
    if backend_device == "gpu":
        # Signal will auto-convert to backend in __post_init__ via to_device or manually
        # But Signal init logic checks if valid array.
        # Let's create Signal and move it.
        s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0).to("gpu")
        assert isinstance(s.samples, xp.ndarray)
    else:
        s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
        assert isinstance(s.samples, np.ndarray)

    assert s.sampling_rate == 1.0
    assert s.symbol_rate == 1.0


def test_signal_properties(backend_device, xp):
    data = xp.zeros(100)
    fs = 100.0
    sym_rate = 10.0

    s = Signal(samples=data, sampling_rate=fs, symbol_rate=sym_rate)
    if backend_device == "gpu":
        s.to("gpu")

    assert s.duration == 1.0
    assert s.sps == 10.0
    assert len(s.time_axis()) == 100


def test_signal_methods(backend_device, xp):
    # Test upsample
    data = xp.array([1.0 + 0j, -1.0 + 0j])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    if backend_device == "gpu":
        s.to("gpu")

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
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
    if backend_device == "gpu":
        s.to("gpu")

    f, p = s.welch_psd(nperseg=64)
    assert f.shape == p.shape
    assert isinstance(f, xp.ndarray)
