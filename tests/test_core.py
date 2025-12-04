import numpy as np
import pytest
from commstools import Signal
from commstools.core.backend import get_backend, set_backend


def test_signal_creation_cpu():
    print("Testing Signal creation on CPU...")
    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)
    assert isinstance(sig.samples, np.ndarray)
    assert sig.backend.name == "numpy"
    print("PASS")


def test_signal_to_gpu():
    print("Testing Signal.to('gpu')...")
    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    try:
        import cupy

        sig_gpu = sig.to("gpu")
        assert isinstance(sig_gpu.samples, cupy.ndarray)
        assert sig_gpu.backend.name == "cupy"
        print("PASS (CuPy available)")
    except ImportError:
        print("SKIP (CuPy not available)")
        try:
            sig.to("gpu")
        except ImportError:
            print("PASS (Correctly raised ImportError)")


def test_signal_to_jax():
    print("Testing Signal.to_jax()...")
    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    try:
        import jax

        jax_arr = sig.samples_to_jax()
        assert isinstance(
            jax_arr, (jax.Array, np.ndarray)
        )  # jax.Array or numpy array if on cpu
        print("PASS (JAX available)")
    except ImportError:
        print("SKIP (JAX not available)")
        try:
            sig.samples_to_jax()
        except ImportError:
            print("PASS (Correctly raised ImportError)")


def test_mutability():
    print("Testing Signal mutability...")
    samples = np.zeros(10)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    # Test direct attribute assignment
    sig.samples = np.ones(10)
    assert np.all(sig.samples == 1)

    # Test method in-place modification
    original_id = id(sig)
    sig.fir_filter(taps=np.ones(1))
    assert id(sig) == original_id
    assert np.all(
        sig.samples == 1
    )  # Convolving ones with [1] should be ones (mode='same')

    print("PASS")


def test_signal_to_gpu_inplace():
    print("Testing Signal.to('gpu') in-place...")
    samples = np.random.randn(100) + 1j * np.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)
    original_id = id(sig)

    try:
        import cupy

        sig.to("gpu")
        assert id(sig) == original_id
        assert isinstance(sig.samples, cupy.ndarray)
        assert sig.backend.name == "cupy"
        print("PASS (CuPy available)")
    except ImportError:
        print("SKIP (CuPy not available)")
        try:
            sig.to("gpu")
        except ImportError:
            print("PASS (Correctly raised ImportError)")


if __name__ == "__main__":
    test_signal_creation_cpu()
    test_mutability()
    test_signal_to_gpu()
    test_signal_to_gpu_inplace()
    test_signal_to_jax()
