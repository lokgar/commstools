"""Tests for helpers routines (normalization, interpolation, random generation)."""

import numpy as np
import pytest

from commstools import helpers


class Unconvertible:
    """Object that raises an exception during np.asarray conversion."""

    def __array__(self):
        raise TypeError("Cannot convert to array")


# ============================================================================
# RANDOM GENERATION
# ============================================================================


def test_random_bits(backend_device, xp):
    """Verify random bit generation produces correct length, binary values, and device."""
    bits = helpers.random_bits(100, seed=42)
    assert len(bits) == 100
    assert xp.all((bits == 0) | (bits == 1))
    assert isinstance(bits, xp.ndarray)


def test_random_bits_no_seed(backend_device, xp):
    """Verify random_bits without a seed produces a result on the active device."""
    bits = helpers.random_bits(100)
    assert isinstance(bits, xp.ndarray)
    assert len(bits) == 100


def test_random_symbols_unipolar(backend_device, xp):
    """Verify unipolar flag in random_symbols produces non-negative values."""
    syms = helpers.random_symbols(10, "ask", 4, unipolar=True)
    assert xp.all(syms >= 0)


# ============================================================================
# NORMALIZATION
# ============================================================================


def test_normalize(backend_device, xp):
    """Verify peak and average-power normalization modes."""
    data = xp.array([1.0, 2.0, 0.5])

    norm = helpers.normalize(data, mode="peak")
    assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

    norm_power = helpers.normalize(data, mode="average_power")
    assert xp.isclose(float(xp.mean(xp.abs(norm_power) ** 2)), 1.0)


def test_normalize_unity_gain(backend_device, xp):
    """Verify unity-gain normalization (sum of elements = 1)."""
    data = xp.array([1.0, 2.0, 3.0, 4.0])
    norm = helpers.normalize(data, mode="unity_gain")
    assert xp.isclose(xp.sum(norm), 1.0)


def test_normalize_zeros(backend_device, xp):
    """Verify all-zero array normalization returns zeros without NaN."""
    x = xp.zeros(10)
    out = helpers.normalize(x, mode="unit_energy")
    assert xp.all(out == 0)
    assert out.shape == (10,)


def test_normalize_invalid_mode(backend_device, xp):
    """Verify unknown normalization mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        helpers.normalize(xp.ones(5), mode="invalid")


# ============================================================================
# SI FORMATTING
# ============================================================================


def test_format_si(backend_device, xp):
    """Verify SI-prefix formatting for common magnitudes."""
    assert helpers.format_si(None) == "None"
    assert helpers.format_si(0) == "0.00 Hz"
    assert "1.00 MHz" in helpers.format_si(1e6, "Hz")
    assert "500.00 mV" in helpers.format_si(0.5, "V")
    assert "Hz" in helpers.format_si(100)


# ============================================================================
# ARRAY VALIDATION
# ============================================================================


def test_validate_array(backend_device, xp):
    """Verify array validation: None passthrough, list conversion, complex_only, error paths."""
    assert helpers.validate_array(None) is None

    arr = helpers.validate_array([1, 2, 3])
    assert isinstance(arr, (xp.ndarray, np.ndarray))

    arr_c = helpers.validate_array(xp.array([1, 2], dtype=float), complex_only=True)
    assert xp.iscomplexobj(arr_c)

    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array("not an array")

    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array(np.array(["a", "b"]))


def test_validate_array_complex_only(backend_device, xp):
    """Verify complex_only flag zero-extends the imaginary part."""
    arr = xp.array([1, 2, 3], dtype=float)
    out = helpers.validate_array(arr, complex_only=True)
    assert xp.iscomplexobj(out)
    assert xp.all(out.real == arr)
    assert xp.all(out.imag == 0)


def test_validate_array_exception(backend_device, xp):
    """Verify the except-block in validate_array raises ValueError for unconvertible input."""
    obj = Unconvertible()
    with pytest.raises(ValueError, match="Could not convert"):
        helpers.validate_array(obj)


# ============================================================================
# RMS
# ============================================================================


def test_rms_axis(backend_device, xp, xpt):
    """Verify RMS over all elements and per-row."""
    x = xp.array([[1.0, 1.0], [2.0, 2.0]])
    xpt.assert_allclose(helpers.rms(x), xp.sqrt(2.5))
    xpt.assert_allclose(helpers.rms(x, axis=1), [1.0, 2.0])


# ============================================================================
# MIMO PREAMBLE EXPANSION
# ============================================================================


def test_expand_preamble_mimo_time_orthogonal(backend_device, xp, xpt):
    """Verify time-orthogonal MIMO preamble expansion produces block-diagonal structure."""
    from commstools.helpers import expand_preamble_mimo

    base = xp.array([1.0 + 0j, -1.0 + 0j, 1.0 + 0j])
    result = expand_preamble_mimo(base, num_streams=3, mode="time_orthogonal")

    assert result.shape == (3, 9)  # (num_streams, L * num_streams)
    xpt.assert_allclose(result[0, :3], base)
    xpt.assert_allclose(result[0, 3:], 0)
    xpt.assert_allclose(result[1, :3], 0)
    xpt.assert_allclose(result[1, 3:6], base)
    xpt.assert_allclose(result[1, 6:], 0)
    xpt.assert_allclose(result[2, :6], 0)
    xpt.assert_allclose(result[2, 6:], base)


def test_expand_preamble_mimo_single_stream(backend_device, xp, xpt):
    """Verify expand_preamble_mimo returns the base waveform unchanged for 1 stream."""
    from commstools.helpers import expand_preamble_mimo

    base = xp.array([1.0, -1.0])
    result = expand_preamble_mimo(base, num_streams=1, mode="same")
    xpt.assert_array_equal(result, base)


def test_expand_preamble_mimo_unknown_mode(backend_device, xp, xpt):
    """Verify expand_preamble_mimo broadcasts the base waveform for an unknown mode."""
    from commstools.helpers import expand_preamble_mimo

    base = xp.array([1.0, -1.0])
    result = expand_preamble_mimo(base, num_streams=2, mode="unknown_mode")

    assert result.shape == (2, 2)
    xpt.assert_allclose(result[0], base)
    xpt.assert_allclose(result[1], base)


# ============================================================================
# DTYPE PRESERVATION TESTS
# ============================================================================


def test_normalize_preserves_float32_dtype(backend_device, xp):
    """normalize: float32 input → float32 output across all modes."""
    x = xp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    for mode in ("unity_gain", "unit_energy", "peak", "average_power"):
        out = helpers.normalize(x, mode=mode)
        assert out.dtype == xp.float32, f"mode={mode!r}: expected float32, got {out.dtype}"


def test_rms_preserves_float32_dtype(backend_device, xp):
    """rms: float32 input → float32 output."""
    x = xp.asarray(np.ones(64, dtype=np.float32))
    out = helpers.rms(x)
    assert out.dtype == xp.float32, f"Expected float32, got {out.dtype}"


def test_normalize_preserves_complex64_dtype(backend_device, xp):
    """normalize: complex64 input → complex64 output."""
    x = xp.asarray(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
    for mode in ("unit_energy", "peak", "average_power"):
        out = helpers.normalize(x, mode=mode)
        assert out.dtype == xp.complex64, f"mode={mode!r}: expected complex64, got {out.dtype}"
