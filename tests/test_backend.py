import pytest
import numpy as np
from commstools import backend


def test_get_backend(backend_device):
    # This test might be tricky if we want to force a backend,
    # but backend_device fixture just tells us what to expect.
    # We can try to force it.

    if backend_device == "gpu":
        try:
            import cupy

            # We can't easily force global state in tests without side effects,
            # but we can check if CuPyBackend is returned when we look for it.
            # Assuming the environment has cupy if backend_device is gpu.
            pass
        except ImportError:
            pytest.skip("CuPy not installed")

    # Just check we get a backend object
    b = backend.get_backend()
    # Check if it has expected methods
    assert hasattr(b, "xp")
    assert hasattr(b, "sp")


def test_ensure_on_backend(backend_device, xp):
    data = np.array([1, 2, 3])
    # If we are in a test parameterized for GPU, we want to ensure
    # the function converts implicit inputs to the *current* backend?
    # Actually ensure_on_backend usually means "ensure it is on the active backend".
    # But pytest doesn't handle global active backend switching automatically unless we do it.

    # We should probably set the backend in the test if we want to test that specific behavior.
    if backend_device == "gpu":
        backend.set_backend("gpu")
        out = backend.ensure_on_backend(data)
        assert isinstance(out, xp.ndarray)
        # Restore defaults
        backend.set_backend("cpu")
    else:
        backend.set_backend("cpu")
        out = backend.ensure_on_backend(data)
        assert isinstance(out, np.ndarray)


def test_to_host(backend_device, xp):
    data = xp.array([1, 2, 3])
    host_data = backend.to_host(data)
    assert isinstance(host_data, np.ndarray)
    assert np.allclose(host_data, [1, 2, 3])
