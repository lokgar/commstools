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
