import pytest
import numpy as np
from commstools import backend

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


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

        metafunc.parametrize("backend_device", params, indirect=True)


@pytest.fixture
def backend_device(request):
    """
    Fixture that returns the backend device name.
    Skips GPU tests if CuPy is not available or functional.
    Forces CPU mode when device is 'cpu' to ensure isolation.
    """

    device = request.param
    if device == "gpu":
        backend.use_cpu_only(False)
        if not _CUPY_AVAILABLE:
            pytest.skip("CuPy not installed, skipping GPU tests")
        try:
            # Aggressive check for functionality
            cp.zeros(1)
            try:
                cp.random.randn(1)
            except ImportError:
                # catch specific libcurand error
                raise
        except Exception as e:
            pytest.skip(f"CuPy installed but not functional (missing libs?): {e}")

    elif device == "cpu":
        # Force CPU to prevent accidental GPU usage in "cpu" tests
        backend.use_cpu_only(True)

    yield device

    # Restore default state (allow GPU) after test
    backend.use_cpu_only(False)


@pytest.fixture
def xp(backend_device):
    """Returns the array module (numpy or cupy) for the current backend."""
    if backend_device == "cpu":
        return np
    elif backend_device == "gpu":
        return cp
    return np
