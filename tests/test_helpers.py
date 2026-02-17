"""Tests for helpers routines (normalization, interpolation)."""

import numpy as np
import pytest

from commstools import helpers


class Unconvertible:
    """Object that raises an exception during np.asarray conversion."""

    def __array__(self):
        raise TypeError("Cannot convert to array")


def test_normalize(backend_device, xp):
    """Verify max-amplitude and average-power normalization modes."""
    data = xp.array([1.0, 2.0, 0.5])

    norm = helpers.normalize(data, mode="peak")
    assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

    norm_power = helpers.normalize(data, mode="average_power")
    # Mean power should be 1
    mean_pwr = xp.mean(xp.abs(norm_power) ** 2)

    # Use float() to ensure compatibility for scalar comparisons
    assert xp.isclose(float(mean_pwr), 1.0)


def test_interp1d(backend_device, xp):
    """Verify linear interpolation and extrapolation behavior."""
    # Linear function y = 2x
    x_p = xp.array([0.0, 1.0, 2.0, 3.0])
    f_p = 2.0 * x_p

    # Query points
    x = xp.array([0.5, 1.5, 2.5])

    result = helpers.interp1d(x, x_p, f_p)

    expected = 2.0 * x
    assert xp.allclose(result, expected)

    # Test extrapolation (linear based on nearest segment)
    x_out = xp.array([-0.5, 3.5])

    result_out = helpers.interp1d(x_out, x_p, f_p)

    # For -0.5: clamped to first segment indices
    # Correct calculation: x0=0.0, x1=1.0, y0=0.0, y1=2.0 -> y = 2x -> -1.0
    assert xp.isclose(result_out[0], -1.0)

    # For 3.5: clamped to last segment
    # Correct calculation: x0=2.0, x1=3.0, y0=4.0, y1=6.0 -> y = 2x -> 7.0
    assert xp.isclose(result_out[1], 7.0)


def test_normalize_unity_gain(backend_device, xp):
    """Verify unity-gain normalization (sum of elements = 1)."""
    data = xp.array([1.0, 2.0, 3.0, 4.0])
    norm = helpers.normalize(data, mode="unity_gain")
    assert xp.isclose(xp.sum(norm), 1.0)


def test_normalize_errors(backend_device, xp):
    """Verify that invalid normalization modes raise ValueError."""
    import pytest

    data = xp.array([1, 2, 3])
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        helpers.normalize(data, mode="invalid")


def test_format_si(backend_device, xp):
    """Verify SI-prefix formatting for various magnitudes."""
    assert helpers.format_si(None) == "None"
    assert helpers.format_si(0) == "0.00 Hz"
    assert "1.00 MHz" in helpers.format_si(1e6, "Hz")
    assert "500.00 mV" in helpers.format_si(0.5, "V")


def test_validate_array(backend_device, xp):
    """Verify array validation and coercion logic."""
    # Test None
    assert helpers.validate_array(None) is None

    # Test conversion
    arr = helpers.validate_array([1, 2, 3])
    assert isinstance(arr, (xp.ndarray, np.ndarray))

    # Test complex_only
    arr_c = helpers.validate_array(xp.array([1, 2], dtype=float), complex_only=True)
    assert xp.iscomplexobj(arr_c)

    # Test error
    import pytest

    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array("not an array")

    # Test non-numeric array (object/string)
    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array(np.array(["a", "b"]))

    # Test completely non-convertible type (though np.asarray is permissive)
    # Using something that raises in np.asarray (if possible)
    # Actually most things get converted to object arrays if possible, which hits 290


def test_validate_array_errors(backend_device, xp):
    """Verify validation errors for unsupported types."""
    # Non-convertible type
    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array(np.array(["a", "b"]), name="array")

    # Non-numeric dtype
    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array(np.array(["a", "b"]), name="array")


def test_format_si_edge_cases(backend_device, xp):
    """Verify SI formatting for None and 0."""
    assert helpers.format_si(None) == "None"
    assert helpers.format_si(0) == "0.00 Hz"
    assert "Hz" in helpers.format_si(100)


def test_normalize_zeros(backend_device, xp):
    """Verify normalization of all-zero arrays."""
    x = xp.zeros(10)
    # Norm factor is 0, should returned zeros instead of NaN
    out = helpers.normalize(x, mode="unit_energy")
    assert xp.all(out == 0)
    assert out.shape == (10,)


def test_normalize_invalid_mode(backend_device, xp):
    """Verify error for unknown normalization mode."""
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        helpers.normalize(xp.ones(5), mode="invalid")


def test_rms_axis(backend_device, xp):
    """Verify RMS calculation across specific axis."""
    x = xp.array([[1.0, 1.0], [2.0, 2.0]])
    # Overall RMS: sqrt((1+1+4+4)/4) = sqrt(2.5)
    assert xp.allclose(helpers.rms(x), xp.sqrt(2.5))
    # Per-row RMS: [1, 2]
    assert xp.allclose(helpers.rms(x, axis=1), [1.0, 2.0])


def test_random_symbols_unipolar(backend_device, xp):
    """Test unipolar flag in random_symbols."""
    syms = helpers.random_symbols(10, "ask", 4, unipolar=True)
    assert xp.all(syms >= 0)


def test_validate_array_complex_only(backend_device, xp):
    """Verify complex_only flag in validate_array."""
    arr = xp.array([1, 2, 3], dtype=float)
    out = helpers.validate_array(arr, complex_only=True)
    assert xp.iscomplexobj(out)
    assert xp.all(out.real == arr)
    assert xp.all(out.imag == 0)


def test_random_bits_gpu(backend_device, xp):
    """Cover the to_device branch in random_bits when on GPU."""
    if backend_device != "gpu":
        pytest.skip("Test targets GPU branch")

    bits = helpers.random_bits(100)
    assert isinstance(bits, xp.ndarray)
    assert len(bits) == 100


def test_validate_array_exception(backend_device, xp):
    """Cover the 'except Exception' block in validate_array."""
    # This object raises TypeError when np.asarray tries to convert it
    obj = Unconvertible()

    with pytest.raises(ValueError, match="Could not convert"):
        helpers.validate_array(obj)


def test_expand_preamble_mimo_time_orthogonal(backend_device, xp):
    """Verify time_orthogonal MIMO preamble expansion (lines 489-523)."""
    from commstools.helpers import expand_preamble_mimo

    base = xp.array([1.0 + 0j, -1.0 + 0j, 1.0 + 0j])
    result = expand_preamble_mimo(base, num_streams=3, mode="time_orthogonal")

    # Shape: (3, 9) = (num_streams, L * num_streams)
    assert result.shape == (3, 9)

    # Verify block-diagonal structure
    # Channel 0: [base, 0, 0]
    assert xp.allclose(result[0, :3], base)
    assert xp.allclose(result[0, 3:], 0)

    # Channel 1: [0, base, 0]
    assert xp.allclose(result[1, :3], 0)
    assert xp.allclose(result[1, 3:6], base)
    assert xp.allclose(result[1, 6:], 0)

    # Channel 2: [0, 0, base]
    assert xp.allclose(result[2, :6], 0)
    assert xp.allclose(result[2, 6:], base)


def test_expand_preamble_mimo_single_stream(backend_device, xp):
    """Verify expand_preamble_mimo returns unchanged for 1 stream."""
    from commstools.helpers import expand_preamble_mimo

    base = xp.array([1.0, -1.0])
    result = expand_preamble_mimo(base, num_streams=1, mode="same")
    assert xp.array_equal(result, base)


def test_expand_preamble_mimo_unknown_mode(backend_device, xp):
    """Verify expand_preamble_mimo falls back to broadcast for unknown mode (line 521-523)."""
    from commstools.helpers import expand_preamble_mimo

    base = xp.array([1.0, -1.0])
    result = expand_preamble_mimo(base, num_streams=2, mode="unknown_mode")

    # Falls back to tile (same as "same")
    assert result.shape == (2, 2)
    assert xp.allclose(result[0], base)
    assert xp.allclose(result[1], base)
