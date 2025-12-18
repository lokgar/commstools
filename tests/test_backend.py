import pytest
import numpy as np
from commstools import backend


def test_get_array_module():
    # Test CPU
    arr_cpu = np.array([1, 2, 3])
    assert backend.get_array_module(arr_cpu) == np

    # Test List
    assert backend.get_array_module([1, 2, 3]) == np


def test_to_device(backend_device, xp):
    data = np.array([1, 2, 3])

    # Move to target device
    device_data = backend.to_device(data, backend_device)

    assert isinstance(device_data, xp.ndarray)
    assert np.allclose(backend.to_device(device_data, "cpu"), data)

    if backend_device == "cpu":
        assert backend.get_array_module(device_data) == np
    elif backend_device == "gpu":
        # If gpu test runs, we assume cupy is present (checked by fixture)
        import cupy as cp

        assert backend.get_array_module(device_data) == cp


def test_dispatch(backend_device, xp):
    data = np.array([1, 2, 3])

    # Pre-move data to device manually to simulate input from that device
    data_in = backend.to_device(data, backend_device)

    out_data, out_xp, out_sp = backend.dispatch(data_in)

    assert out_xp == xp
    assert isinstance(out_data, xp.ndarray)
    assert hasattr(out_sp, "signal")  # Check if it looks like scipy/cupyx.scipy


def test_cpu_only_toggle():
    from commstools import backend

    # Ensure clean state
    backend.use_cpu_only(False)
    initial_status = backend.is_cupy_available()

    # Force CPU
    backend.use_cpu_only(True)
    assert backend.is_cupy_available() is False
    assert backend._FORCE_CPU is True

    # Restore
    backend.use_cpu_only(False)
    assert backend.is_cupy_available() == initial_status


def test_jax_interop(backend_device, xp):
    try:
        import jax.numpy as jnp
        import jax
    except ImportError:
        pytest.skip("JAX not installed")

    # Create data on backend device
    # Create data on backend device
    data = xp.array([1.0, 2.0, 3.0])

    # To JAX
    if backend_device == "cpu":
        # Ensure CPU only is enforced within this test context
        backend.use_cpu_only(True)

    jax_arr = backend.to_jax(data)
    assert isinstance(jax_arr, jnp.ndarray)

    # From JAX
    # Note: from_jax might return numpy or cupy depending on context/availability
    # If on GPU, we expect JAX to likely stay on GPU if jax[cuda] is working,
    # but strictly from_jax should return backend-compatible array

    back_arr = backend.from_jax(jax_arr)

    if backend_device == "cpu":
        assert isinstance(back_arr, np.ndarray)
    elif backend_device == "gpu":
        # Ideally it should be cupy, but fallback to numpy is valid if DLPack fails or JAX is CPU-only
        assert isinstance(back_arr, (np.ndarray, xp.ndarray))

    # Test round trip values
    assert np.allclose(backend.to_device(back_arr, "cpu"), [1.0, 2.0, 3.0])

    # Reset configuration to avoid polluting other tests if fixture doesn't handle it
    backend.use_cpu_only(False)
