import pytest
import numpy as np
from commstools import multirate, backend


def test_upsample(backend_device, xp):
    data = np.array([1.0, 2.0, 3.0])
    data = backend.to_device(data, backend_device)

    factor = 2
    out = multirate.upsample(data, factor)

    assert isinstance(out, xp.ndarray)
    assert out.size >= data.size * factor - factor


def test_decimate(backend_device, xp):
    data = np.zeros(100)
    data[::2] = 1.0
    data = backend.to_device(data, backend_device)

    factor = 2
    out = multirate.decimate(data, factor)

    assert isinstance(out, xp.ndarray)
    assert out.size <= data.size // factor + 1


def test_resample(backend_device, xp):
    data = np.ones(100)
    data = backend.to_device(data, backend_device)

    up = 3
    down = 2
    out = multirate.resample(data, up, down)

    assert isinstance(out, xp.ndarray)
    expected_size = int(data.size * up / down)
    assert abs(out.size - expected_size) < 5
