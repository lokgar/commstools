"""Tests for digital filtering and pulse shaping tap generation."""

import numpy as np

from commstools import filtering


def test_rrc_taps():
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


def test_rc_taps():
    """Verify Raised Cosine (RC) filter tap generation and normalization."""
    # RC taps are generated on CPU
    sps = 4
    rolloff = 0.5
    taps = filtering.rc_taps(sps=sps, rolloff=rolloff, span=8)

    assert isinstance(taps, np.ndarray)
    assert len(taps) == 8 * 4 + 1
    # Unit energy check
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_gaussian_taps():
    """Verify Gaussian filter tap generation and normalization."""
    sps = 4
    bt = 0.3
    taps = filtering.gaussian_taps(sps=sps, bt=bt, span=4)

    assert isinstance(taps, np.ndarray)
    assert len(taps) == 4 * 4 + 1
    # Unit energy check
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_smoothrect_taps():
    """Verify Gaussian-smoothed rectangular pulse generation."""
    sps = 8
    span = 4
    taps = filtering.smoothrect_taps(sps=sps, span=span, bt=1.0)

    assert isinstance(taps, np.ndarray)
    assert len(taps) == span * sps + 1
    # Unit energy check
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_rc_taps_rolloff_zero():
    """Verify RC filter tap generation for zero rolloff (Brick-wall)."""
    taps = filtering.rc_taps(sps=4, rolloff=0.0, span=8)
    assert isinstance(taps, np.ndarray)
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_highpass_taps():
    """Verify highpass FIR filter design."""
    taps = filtering.highpass_taps(num_taps=31, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 31
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_bandpass_taps():
    """Verify bandpass FIR filter design."""
    taps = filtering.bandpass_taps(
        num_taps=31, low_cutoff=0.1, high_cutoff=0.3, sampling_rate=1.0
    )
    assert len(taps) == 31
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_bandstop_taps():
    """Verify bandstop FIR filter design."""
    taps = filtering.bandstop_taps(
        num_taps=31, low_cutoff=0.1, high_cutoff=0.3, sampling_rate=1.0
    )
    assert len(taps) == 31
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_lowpass_taps():
    """Verify lowpass FIR filter design."""
    taps = filtering.lowpass_taps(num_taps=31, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 31
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)
