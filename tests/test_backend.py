"""Tests for the backend management and device-agnostic dispatch system."""

import warnings

import numpy as np
import pytest

from commstools import backend

warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")


def test_get_array_module():
    """Verify that get_array_module correctly identifies NumPy for host data."""
    # Test CPU
    arr_cpu = np.array([1, 2, 3])
    assert backend.get_array_module(arr_cpu) == np

    # Test List (should default to NumPy)
    assert backend.get_array_module([1, 2, 3]) == np


def test_to_device(backend_device, xp):
    """Verify data transfer between CPU and the target device."""
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
    """Verify the dispatch system returns correct array and signal modules."""
    data = np.array([1, 2, 3])

    # Pre-move data to device manually to simulate input from that device
    data_in = backend.to_device(data, backend_device)

    out_data, out_xp, out_sp = backend.dispatch(data_in)

    assert out_xp == xp
    assert isinstance(out_data, xp.ndarray)
    assert hasattr(out_sp, "signal")  # Check if it looks like scipy/cupyx.scipy


def test_cpu_only_toggle():
    """Verify that forcing CPU mode correctly disables GPU detection."""
    # Ensure clean state
    # Save original state to restore
    original_force = backend._FORCE_CPU

    backend.use_cpu_only(False)
    # Check if cupy is physically available on system
    # We can't easily check this without potentially importing cupy if we haven't already,
    # but backend.is_cupy_available() relies on module level _CUPY_AVAILABLE.

    # Force CPU
    backend.use_cpu_only(True)
    assert backend.is_cupy_available() is False

    # Restore
    backend.use_cpu_only(False)
    # If it was available before force, it should be available now.

    # Restore original state
    backend.use_cpu_only(original_force)


def test_jax_interop(backend_device, xp):
    """Verify interoperability between core backends and JAX using DLPack."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")

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


def test_use_cpu_only(xp):
    """Test use_cpu_only forces CPU backend."""
    from commstools import backend

    # Save original state
    original_force = backend._FORCE_CPU

    try:
        backend.use_cpu_only(True)
        assert backend.is_cupy_available() is False
        assert backend.get_array_module(np.array([1])) == np

        # Should raise ImportError if we try to force GPU
        with pytest.raises(ImportError):
            backend.to_device(np.array([1]), "gpu")

    finally:
        # Restore state
        backend.use_cpu_only(original_force)


def test_to_device_errors(xp):
    """Test error paths in to_device."""
    from commstools import backend

    with pytest.raises(ValueError, match="Unknown device"):
        backend.to_device(np.array([1]), "tpu")


def test_dispatch_list(xp):
    """Test dispatch with list input."""
    # Should convert list to array
    from commstools import backend, multirate

    data, x, s = backend.dispatch([1, 2, 3])
    assert isinstance(data, x.ndarray)
    assert x in (np, getattr(multirate, "cp", None))  # generic check


def test_jax_conversions(backend_device, xp):
    """Test JAX conversion utilities with real JAX if available."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")

    from commstools import backend, Signal

    # 1. Numpy -> JAX
    arr_np = np.array([1, 2, 3])
    arr_jax = backend.to_jax(arr_np)
    assert isinstance(arr_jax, jnp.ndarray)

    # 2. JAX -> Numpy
    arr_back = backend.from_jax(arr_jax)
    assert isinstance(arr_back, np.ndarray)
    assert np.allclose(arr_back, arr_np)

    # 3. Signal methods
    sig = Signal(samples=arr_np, sampling_rate=1.0, symbol_rate=1.0)
    jax_sig = sig.export_samples_to_jax()
    assert isinstance(jax_sig, jnp.ndarray)

    sig.update_samples_from_jax(jax_sig)
    assert isinstance(sig.samples, xp.ndarray)
    if backend_device == "cpu":
        assert np.allclose(sig.samples, arr_np)
    else:
        assert xp.allclose(sig.samples, xp.asarray(arr_np))
