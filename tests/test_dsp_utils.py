import pytest
import numpy as np
from commstools import set_backend
from commstools.dsp import utils

# Test both backends
backends = ["numpy"]
try:
    import jax

    backends.append("jax")
except ImportError:
    pass


class TestNormalization:
    """Test normalization functions."""

    @pytest.mark.parametrize("backend_name", backends)
    def test_normalize_unity_gain(self, backend_name):
        set_backend(backend_name)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = utils.normalize(x, mode="unity_gain")

        assert abs(np.sum(normalized) - 1.0) < 1e-6
        # Check relative values are preserved
        assert abs(normalized[1] / normalized[0] - 2.0) < 1e-6

    @pytest.mark.parametrize("backend_name", backends)
    def test_normalize_unit_energy(self, backend_name):
        set_backend(backend_name)
        x = np.array([3.0, 4.0])  # Energy = 9 + 16 = 25. Sqrt(25) = 5.
        normalized = utils.normalize(x, mode="unit_energy")

        energy = np.sum(np.abs(normalized) ** 2)
        assert abs(energy - 1.0) < 1e-6
        # Expected: [3/5, 4/5] = [0.6, 0.8]
        assert abs(normalized[0] - 0.6) < 1e-6

    @pytest.mark.parametrize("backend_name", backends)
    def test_normalize_max_amplitude(self, backend_name):
        set_backend(backend_name)
        x = np.array([1.0, -5.0, 2.0])
        normalized = utils.normalize(x, mode="max_amplitude")

        assert abs(np.max(np.abs(normalized)) - 1.0) < 1e-6
        # Expected: [0.2, -1.0, 0.4]
        assert abs(normalized[1] + 1.0) < 1e-6

    @pytest.mark.parametrize("backend_name", backends)
    def test_normalize_average_power(self, backend_name):
        set_backend(backend_name)
        x = np.array([2.0, 2.0])  # Power = (4+4)/2 = 4. Sqrt(4) = 2.
        normalized = utils.normalize(x, mode="average_power")

        avg_power = np.mean(np.abs(normalized) ** 2)
        assert abs(avg_power - 1.0) < 1e-6
        # Expected: [1.0, 1.0]
        assert abs(normalized[0] - 1.0) < 1e-6

    @pytest.mark.parametrize("backend_name", backends)
    def test_normalization_zero_input(self, backend_name):
        set_backend(backend_name)
        x = np.zeros(5)

        # Should return zeros (handled gracefully)
        normalized = utils.normalize(x, mode="unity_gain")
        np.testing.assert_array_equal(normalized, np.zeros_like(x))
