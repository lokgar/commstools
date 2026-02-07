import numpy as np
from commstools import filtering


def test_rrc_taps():
    # Tap generation is now always on CPU/NumPy
    taps = filtering.rrc_taps(sps=4, span=10, rolloff=0.35)

    assert isinstance(taps, np.ndarray)
    assert len(taps) > 0
    assert hasattr(taps, "shape")


def test_fir_filter(backend_device, xp):
    # Create data directly on device using xp
    data = xp.ones(100)
    taps = xp.ones(5) / 5.0  # Moving average

    # taps is also xp array now, filtering.fir_filter handles it

    filtered = filtering.fir_filter(data, taps)

    assert isinstance(filtered, xp.ndarray)
    assert len(filtered) == len(data)
