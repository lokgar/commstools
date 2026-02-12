"""Tests for multirate signal processing routines (upsampling, decimation, resampling)."""

import pytest

from commstools import multirate


def test_upsample(backend_device, xp):
    """Verify integer upsampling increases signal length correctly."""
    data = xp.array([1.0, 2.0, 3.0])

    factor = 2
    out = multirate.upsample(data, factor)

    assert isinstance(out, xp.ndarray)
    assert out.size >= data.size * factor - factor


def test_decimate(backend_device, xp):
    """Verify integer decimation reduces signal length correctly."""
    data = xp.zeros(100)
    data[::2] = 1.0

    factor = 2
    out = multirate.decimate(data, factor)

    assert isinstance(out, xp.ndarray)
    assert out.size <= data.size // factor + 1


def test_resample(backend_device, xp):
    """Verify rational resampling produce expected output size."""
    data = xp.ones(100)

    up = 3
    down = 2
    out = multirate.resample(data, up, down)

    assert isinstance(out, xp.ndarray)
    expected_size = int(data.size * up / down)
    assert abs(out.size - expected_size) < 5


def test_resample_sps(backend_device, xp):
    """Verify resampling based on input/output samples per symbol ratios."""
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
    """Verify that inconsistent resampling parameters raise appropriate errors."""
    data = xp.zeros(10)

    with pytest.raises(ValueError, match="Cannot specify both"):
        multirate.resample(data, up=2, sps_in=4)

    with pytest.raises(ValueError, match="Must specify either"):
        multirate.resample(data)


def test_downsample_to_symbols(backend_device, xp):
    """Verify downsampling (picking) symbols from an upsampled stream."""
    sps = 4
    # 4 symbols, each repeated sps times
    data = xp.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=xp.float32)

    # Sample at center of first symbol (offset 2)
    syms = multirate.downsample_to_symbols(data, sps=sps, offset=0)
    assert xp.array_equal(syms, xp.array([1, 2, 3, 4]))

    # MIMO case
    data_mimo = xp.stack([data, data * 10])
    syms_mimo = multirate.downsample_to_symbols(data_mimo, sps=sps, axis=-1)
    assert syms_mimo.shape == (2, 4)
    assert xp.array_equal(syms_mimo[1], xp.array([10, 20, 30, 40]))


def test_expand(backend_device, xp):
    """Verify up-sampling by zero-stuffing correctly inserts zeros."""
    data = xp.array([1, 2, 3], dtype=xp.float32)
    factor = 3
    expanded = multirate.expand(data, factor)

    # Expected: 1, 0, 0, 2, 0, 0, 3, 0, 0
    expected = xp.array([1, 0, 0, 2, 0, 0, 3, 0, 0])
    assert xp.array_equal(expanded, expected)


def test_decimate_polyphase(backend_device, xp):
    """Verify decimation using the polyphase method."""
    data = xp.ones(100)
    factor = 2
    out = multirate.decimate(data, factor, method="polyphase")

    assert isinstance(out, xp.ndarray)
    assert out.size == 50

    with pytest.raises(ValueError, match="Unknown decimation method"):
        multirate.decimate(data, factor, method="unknown")


def test_polyphase_resample_cupy_multidim(backend_device, xp):
    """
    Cover the manual loop workaround in polyphase_resample for CuPy
    when input has >1 dimensions.
    """
    if backend_device != "gpu":
        pytest.skip("Test specifically targets CuPy workaround path")

    # 2 channels, 10 samples
    samples = xp.ones((2, 10))
    # Resample by 2/1
    res = multirate.polyphase_resample(samples, up=2, down=1, axis=-1)

    assert res.shape == (2, 20)
    assert isinstance(res, xp.ndarray)
    assert res.size == 40
