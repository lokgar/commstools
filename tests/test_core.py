import numpy as np
import pytest
from commstools import Signal
from commstools.core.backend import get_backend, set_backend


def test_signal_creation_cpu():
    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)
    assert isinstance(sig.samples, np.ndarray)
    assert sig.backend.name == "numpy"


def test_signal_to_gpu():
    try:
        import cupy
    except ImportError:
        pytest.skip("CuPy not installed")

    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    sig_gpu = sig.to("gpu")
    assert isinstance(sig_gpu.samples, cupy.ndarray)
    assert sig_gpu.backend.name == "cupy"

    # Original signal is modified in-place
    assert isinstance(sig.samples, cupy.ndarray)


def test_signal_to_jax():
    try:
        import jax
    except ImportError:
        pytest.skip("JAX not installed")

    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    jax_arr = sig.to_jax()
    assert isinstance(jax_arr, (jax.Array, np.ndarray))


def test_mutability():
    samples = np.zeros(10)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    # Test direct attribute assignment
    sig.samples = np.ones(10)
    assert np.all(sig.samples == 1)

    # Test method in-place modification
    original_id = id(sig)
    sig.fir_filter(taps=np.ones(1))
    assert id(sig) == original_id
    # Convolving ones with [1] should be ones (mode='same')
    assert np.all(sig.samples == 1)


def test_signal_to_gpu_inplace():
    try:
        import cupy
    except ImportError:
        pytest.skip("CuPy not installed")

    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)
    original_id = id(sig)

    sig.to("gpu")
    assert id(sig) == original_id
    assert isinstance(sig.samples, cupy.ndarray)
    assert sig.backend.name == "cupy"
