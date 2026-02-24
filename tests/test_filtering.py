"""Tests for digital filtering and pulse shaping tap generation."""

import pytest
import numpy as np

from commstools import filtering


def test_rrc_taps(backend_device, xp):
    """Verify Root Raised Cosine (RRC) filter tap generation."""
    # Tap generation is now always on CPU/NumPy
    taps = filtering.rrc_taps(sps=4, span=10, rolloff=0.35)

    assert isinstance(taps, np.ndarray)
    assert len(taps) > 0
    assert hasattr(taps, "shape")


def test_fir_filter(backend_device, xp):
    """Verify FIR filtering across different computational backends."""
    # Create data directly on device using xp
    data = xp.ones(100)
    taps = xp.ones(5) / 5.0  # Moving average

    filtered = filtering.fir_filter(data, taps)

    assert isinstance(filtered, xp.ndarray)
    assert len(filtered) == len(data)


def test_rc_taps(backend_device, xp):
    """Verify Raised Cosine (RC) filter tap generation and normalization."""
    # RC taps are generated on CPU
    sps = 4
    rolloff = 0.5
    taps = filtering.rc_taps(sps=sps, rolloff=rolloff, span=8)

    assert isinstance(taps, np.ndarray)
    assert len(taps) == 8 * 4 + 1
    # Unit energy check
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_gaussian_taps(backend_device, xp):
    """Verify Gaussian filter tap generation and normalization."""
    sps = 4
    bt = 0.3
    taps = filtering.gaussian_taps(sps=sps, bt=bt, span=4)

    assert isinstance(taps, np.ndarray)
    assert len(taps) == 4 * 4 + 1
    # Unit energy check
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_smoothrect_taps(backend_device, xp):
    """Verify Gaussian-smoothed rectangular pulse generation."""
    sps = 8
    span = 4
    taps = filtering.smoothrect_taps(sps=sps, span=span, bt=1.0)

    assert isinstance(taps, np.ndarray)
    assert len(taps) == span * sps + 1
    # Unit energy check
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_rc_taps_rolloff_zero(backend_device, xp):
    """Verify RC filter tap generation for zero rolloff (Brick-wall)."""
    taps = filtering.rc_taps(sps=4, rolloff=0.0, span=8)
    assert isinstance(taps, np.ndarray)
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_highpass_taps(backend_device, xp):
    """Verify highpass FIR filter design."""
    taps = filtering.highpass_taps(num_taps=31, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 31


def test_bandpass_taps(backend_device, xp):
    """Verify bandpass FIR filter design."""
    taps = filtering.bandpass_taps(
        num_taps=31, low_cutoff=0.1, high_cutoff=0.3, sampling_rate=1.0
    )
    assert len(taps) == 31


def test_bandstop_taps(backend_device, xp):
    """Verify bandstop FIR filter design."""
    taps = filtering.bandstop_taps(
        num_taps=31, low_cutoff=0.1, high_cutoff=0.3, sampling_rate=1.0
    )
    assert len(taps) == 31


def test_lowpass_taps(backend_device, xp):
    """Verify lowpass FIR filter design."""
    taps = filtering.lowpass_taps(num_taps=31, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 31


def test_rrc_taps_rolloff_zero(backend_device, xp):
    """Verify RRC taps when rolloff is 0 (sinc filter)."""
    taps = filtering.rrc_taps(sps=4, rolloff=0, span=8)
    assert len(taps) > 0
    # Should match sinc
    t = xp.linspace(-4, 4, len(taps))
    expected = xp.sinc(t)
    expected = expected / xp.sqrt(xp.sum(expected**2))
    assert xp.allclose(taps, expected, atol=1e-3)


def test_shape_pulse_variants(backend_device, xp):
    """Verify shape_pulse for less common shapes."""
    symbols = xp.array([1, -1, 1, -1])

    # RC
    res_rc = filtering.shape_pulse(symbols, sps=4, pulse_shape="rc")
    assert len(res_rc) == 16

    # Sinc
    res_sinc = filtering.shape_pulse(symbols, sps=4, pulse_shape="sinc")
    assert len(res_sinc) == 16

    # Invalid
    with pytest.raises(ValueError, match="Not implemented pulse shape"):
        filtering.shape_pulse(symbols, sps=4, pulse_shape="magic")


def test_matched_filter_normalization(backend_device, xp):
    """Verify matched filter normalization options."""
    samples = np.ones(100)
    pulse = np.ones(10)

    # Unity gain
    out_gain = filtering.matched_filter(samples, pulse, taps_normalization="unity_gain")
    assert out_gain.shape == (100,)

    # Max amplitude output
    out_max = filtering.matched_filter(samples, pulse, normalize_output=True)
    assert np.max(np.abs(out_max)) == pytest.approx(1.0)

    # Invalid normalization
    with pytest.raises(ValueError, match="Not implemented taps normalization"):
        filtering.matched_filter(samples, pulse, taps_normalization="magic")


def test_smoothrect_pulse(backend_device, xp):
    """Verify smoothrect pulse shaping."""
    symbols = xp.array([1, 1])
    res = filtering.shape_pulse(symbols, sps=8, pulse_shape="smoothrect", filter_span=4)
    assert len(res) == 16


def test_shape_pulse_none_with_rz(backend_device, xp):
    """shape_pulse with pulse_shape='none' and rz=True should use rect (lines 550-551)."""
    symbols = xp.array([1, -1, 1], dtype=xp.complex64)
    # With rz=True and 'none' pulse, it should expand with rect pulse
    result = filtering.shape_pulse(symbols, sps=4, pulse_shape="none", rz=True)
    assert result is not None
    assert len(result) > 0
