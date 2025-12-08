import pytest
import numpy as np
from commstools import multirate, backend


def test_upsample(backend_device, xp):
    backend.set_backend(backend_device)
    data = xp.array([1.0, 2.0, 3.0])
    factor = 2
    # multirate.upsample uses polyphase filter, so output length approx len*factor
    out = multirate.upsample(data, factor)

    # Valid output type
    assert isinstance(out, xp.ndarray)
    # Check shape roughly
    # resample_poly implementation details might vary slightly on edges or filtering
    # but factor is 2.
    assert out.size >= data.size * factor - factor


def test_decimate(backend_device, xp):
    backend.set_backend(backend_device)
    data = xp.zeros(100)
    data[::2] = 1.0  # signal
    factor = 2
    out = multirate.decimate(data, factor)

    assert isinstance(out, xp.ndarray)
    assert out.size <= data.size // factor + 1


def test_resample(backend_device, xp):
    backend.set_backend(backend_device)
    data = xp.ones(100)
    up = 3
    down = 2
    out = multirate.resample(data, up, down)

    assert isinstance(out, xp.ndarray)
    expected_size = int(data.size * up / down)
    # Allow small off-by-one due to filter delays/padding
    assert abs(out.size - expected_size) < 5
