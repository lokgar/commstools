"""Tests for general utility routines (normalization, interpolation)."""

import numpy as np

from commstools import utils


def test_normalize(backend_device, xp):
    """Verify max-amplitude and average-power normalization modes."""
    data = xp.array([1.0, 2.0, 0.5])

    norm = utils.normalize(data, mode="max_amplitude")
    assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

    norm_power = utils.normalize(data, mode="average_power")
    # Mean power should be 1
    mean_pwr = xp.mean(xp.abs(norm_power) ** 2)

    # Use float() to ensure compatibility for scalar comparisons
    assert np.isclose(float(mean_pwr), 1.0)


def test_interp1d(backend_device, xp):
    """Verify linear interpolation and extrapolation behavior."""
    # Linear function y = 2x
    x_p = xp.array([0.0, 1.0, 2.0, 3.0])
    f_p = 2.0 * x_p

    # Query points
    x = xp.array([0.5, 1.5, 2.5])

    result = utils.interp1d(x, x_p, f_p)

    expected = 2.0 * x
    assert xp.allclose(result, expected)

    # Test extrapolation (linear based on nearest segment)
    x_out = xp.array([-0.5, 3.5])

    result_out = utils.interp1d(x_out, x_p, f_p)

    # For -0.5: clamped to first segment indices
    # Correct calculation: x0=0.0, x1=1.0, y0=0.0, y1=2.0 -> y = 2x -> -1.0
    assert xp.isclose(result_out[0], -1.0)

    # For 3.5: clamped to last segment
    # Correct calculation: x0=2.0, x1=3.0, y0=4.0, y1=6.0 -> y = 2x -> 7.0
    assert xp.isclose(result_out[1], 7.0)


def test_normalize_unity_gain(backend_device, xp):
    """Verify unity-gain normalization (sum of elements = 1)."""
    data = xp.array([1.0, 2.0, 3.0, 4.0])
    norm = utils.normalize(data, mode="unity_gain")
    assert xp.isclose(xp.sum(norm), 1.0)


def test_normalize_errors():
    """Verify that invalid normalization modes raise ValueError."""
    import pytest

    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        utils.normalize(data, mode="invalid")


def test_format_si():
    """Verify SI-prefix formatting for various magnitudes."""
    assert utils.format_si(None) == "None"
    assert utils.format_si(0) == "0.00 Hz"
    assert "1.00 MHz" in utils.format_si(1e6, "Hz")
    assert "500.00 mV" in utils.format_si(0.5, "V")


def test_validate_array(backend_device, xp):
    """Verify array validation and coercion logic."""
    # Test None
    assert utils.validate_array(None) is None

    # Test conversion
    arr = utils.validate_array([1, 2, 3])
    assert isinstance(arr, np.ndarray)

    # Test complex_only
    arr_c = utils.validate_array(np.array([1, 2], dtype=float), complex_only=True)
    assert np.iscomplexobj(arr_c)

    # Test error
    import pytest

    with pytest.raises(ValueError, match="Expected numeric array"):
        utils.validate_array("not an array")
