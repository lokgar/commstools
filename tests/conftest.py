import pytest
import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

from commstools.backend import NumpyBackend, CupyBackend, set_backend


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run tests on: cpu, gpu, or all",
    )


def pytest_generate_tests(metafunc):
    if "backend_device" in metafunc.fixturenames:
        device_opt = metafunc.config.getoption("--device")
        if device_opt == "all":
            params = ["cpu", "gpu"]
        elif device_opt == "cpu":
            params = ["cpu"]
        elif device_opt == "gpu":
            params = ["gpu"]
        else:
            params = ["cpu"]  # Default fallback

        metafunc.parametrize("backend_device", params)


@pytest.fixture
def backend_device(request):
    """
    Fixture that returns the backend device name.
    Skips GPU tests if CuPy is not available or functional.
    """
    device = request.param
    if device == "gpu":
        if not _CUPY_AVAILABLE:
            pytest.skip("CuPy not installed, skipping GPU tests")
        try:
            # Aggressive check for functionality
            cp.zeros(1)
            cp.random.default_rng(0)
            # cp.random.randn(1) # This seems to crash differently on some envs, let's try-catch it
            try:
                cp.random.randn(1)
            except ImportError:
                # catch specific libcurand error
                raise
        except Exception as e:
            pytest.skip(f"CuPy installed but not functional (missing libs?): {e}")

    return device


@pytest.fixture
def xp(backend_device):
    """Returns the array module (numpy or cupy) for the current backend."""
    if backend_device == "cpu":
        return np
    elif backend_device == "gpu":
        return cp
    return np
