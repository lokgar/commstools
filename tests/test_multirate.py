import pytest
from commstools import multirate


def test_upsample(backend_device, xp):
    data = xp.array([1.0, 2.0, 3.0])

    factor = 2
    out = multirate.upsample(data, factor)

    assert isinstance(out, xp.ndarray)
    assert out.size >= data.size * factor - factor


def test_decimate(backend_device, xp):
    data = xp.zeros(100)
    data[::2] = 1.0

    factor = 2
    out = multirate.decimate(data, factor)

    assert isinstance(out, xp.ndarray)
    assert out.size <= data.size // factor + 1


def test_resample(backend_device, xp):
    data = xp.ones(100)

    up = 3
    down = 2
    out = multirate.resample(data, up, down)

    assert isinstance(out, xp.ndarray)
    expected_size = int(data.size * up / down)
    assert abs(out.size - expected_size) < 5


def test_resample_sps(backend_device, xp):
    data = xp.ones(100)

    # integer ratio
    sps_in = 4
    sps_out = 8
    out = multirate.resample(data, sps_in=sps_in, sps_out=sps_out)
    assert out.size == 200

    # fractional ratio
    sps_in = 10
    sps_out = 25
    out = multirate.resample(data, sps_in=sps_in, sps_out=sps_out)
    assert out.size == 250


def test_resample_errors(backend_device, xp):
    data = xp.zeros(10)

    with pytest.raises(ValueError, match="Cannot specify both"):
        multirate.resample(data, up=2, sps_in=4)

    with pytest.raises(ValueError, match="Must specify either"):
        multirate.resample(data)
