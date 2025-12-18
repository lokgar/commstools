import pytest
import numpy as np
from commstools import filtering
from commstools import backend


def test_rrc_taps(backend_device, xp):
    # Tap generation is now always on CPU/NumPy
    taps = filtering.rrc_taps(sps=4, span=10, rolloff=0.35)

    assert isinstance(taps, np.ndarray)
    assert len(taps) > 0
    assert hasattr(taps, "shape")


def test_fir_filter(backend_device, xp):
    data = np.ones(100)
    taps = np.ones(5) / 5.0  # Moving average

    # Move to target backend
    data = backend.to_device(data, backend_device)
    # Taps can be passed as numpy, fir_filter handles conversion

    filtered = filtering.fir_filter(data, taps)

    assert isinstance(filtered, xp.ndarray)
    assert len(filtered) == len(data)
