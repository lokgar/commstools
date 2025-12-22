import pytest
import numpy as np
from commstools import utils, backend


def test_random_bits(backend_device, xp):
    bits = utils.random_bits(100, seed=42)
    assert len(bits) == 100
    assert np.all((bits == 0) | (bits == 1))

    # Optional: Verify we can move it to device
    bits_dev = backend.to_device(bits, backend_device)
    assert isinstance(bits_dev, xp.ndarray)
