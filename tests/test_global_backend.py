import numpy as np
from commstools.backend import set_backend, get_backend, ensure_on_backend, to_host

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def test_global_backend_enforcement():
    if not _CUPY_AVAILABLE:
        print("Skipping GPU tests (CuPy not installed)")
        return

    # Store original state (implicitly numpy/cpu usually, but let's just reset to cpu)
    try:
        # 1. Test ensure_on_backend with GPU
        set_backend("gpu")
        data_cpu = np.array([1, 2, 3])
        data_gpu = ensure_on_backend(data_cpu)
        assert isinstance(data_gpu, cp.ndarray), "Data should be moved to GPU"
        assert get_backend().name == "gpu"

        # 2. Test ensure_on_backend with CPU
        set_backend("cpu")
        data_gpu = cp.array([1, 2, 3])
        data_cpu = ensure_on_backend(data_gpu)
        assert isinstance(data_cpu, np.ndarray), "Data should be moved to CPU"
        assert get_backend().name == "cpu"

        # 3. Test to_host
        set_backend("gpu")
        data_gpu = cp.array([1, 2, 3])
        data_host = to_host(data_gpu)
        assert isinstance(data_host, np.ndarray), "to_host should return NumPy array"

        data_cpu = np.array([1, 2, 3])
        data_host = to_host(data_cpu)
        assert isinstance(data_host, np.ndarray), "to_host should return NumPy array"

        print("Global backend enforcement tests passed!")
    finally:
        set_backend("cpu")


if __name__ == "__main__":
    test_global_backend_enforcement()
