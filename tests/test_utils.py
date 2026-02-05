import pytest
import numpy as np
from commstools import utils
from commstools import backend


def test_normalize(backend_device, xp):
    data = np.array([1.0, 2.0, 0.5])

    # Move to target backend
    data = backend.to_device(data, backend_device)

    norm = utils.normalize(data, mode="max_amplitude")
    assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

    norm_power = utils.normalize(data, mode="average_power")
    # Mean power should be 1
    mean_pwr = xp.mean(xp.abs(norm_power) ** 2)

    if backend_device == "gpu":
        mean_pwr = float(mean_pwr)
    assert np.isclose(mean_pwr, 1.0)


def test_interp1d(backend_device, xp):
    """Test linear interpolation."""
    if backend_device == "gpu":
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not installed")

    # Linear function y = 2x
    x_p = xp.array([0.0, 1.0, 2.0, 3.0])
    f_p = 2.0 * x_p

    x_p = backend.to_device(x_p, backend_device)
    f_p = backend.to_device(f_p, backend_device)

    # Query points
    x = xp.array([0.5, 1.5, 2.5])
    x = backend.to_device(x, backend_device)

    result = utils.interp1d(x, x_p, f_p)

    expected = 2.0 * x
    assert xp.allclose(result, expected)

    # Test boundary clipping (extrapolation clamps to boundaries)
    x_out = xp.array([-0.5, 3.5])
    x_out = backend.to_device(x_out, backend_device)

    result_out = utils.interp1d(x_out, x_p, f_p)

    # Check extrapolation behavior based on searchsorted + clip
    # For -0.5: clamped to first segment indices
    # Correct calculation: x0=0.0, x1=1.0, y0=0.0, y1=2.0. weight = -0.5. res = 0 - 1 = -1.0.
    assert xp.isclose(result_out[0], -1.0)

    # For 3.5: clamped to last segment
    # Correct calculation: x0=2.0, x1=3.0, y0=4.0, y1=6.0. weight = 1.5. res = 4*(-0.5) + 6*1.5 = 7.0
    assert xp.isclose(result_out[1], 7.0)
