"""Tests for pulse shaping, matched filtering, and filter tap generation in Signal objects."""

import pytest

from commstools.core import Signal


def test_signal_pulse_params(backend_device, xp):
    """Verify that pulse shaping parameters (e.g., rolloff) are correctly stored and utilized."""
    sig = Signal.pam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rrc",
        rrc_rolloff=0.5,
    )
    assert sig.pulse_shape == "rrc"
    assert getattr(sig, "pulse_params", None) is None
    assert sig.rrc_rolloff == 0.5

    # Check if taps generation works
    taps = sig.shaping_filter_taps()
    assert taps is not None
    assert len(taps) > 0


def test_matched_filter_auto_taps(backend_device, xp):
    """Verify that matched_filter correctly auto-generates and applies the correct taps."""
    sig = Signal.pam(
        order=2,
        bipolar=True,
        num_symbols=100,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rrc",
    )

    # Copy signal to compare
    sig_before = sig.copy()

    # Apply matched filter (should auto-generate taps)
    sig.matched_filter()

    # Check that samples changed (filtering happened)
    assert not xp.allclose(sig.samples, sig_before.samples)


def test_rzpam_pulse_params(backend_device, xp):
    """Verify pulse parameters for Return-to-Zero (RZ) PAM signals."""
    sig = Signal.pam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        mode="rz",
        pulse_shape="smoothrect",
        smoothrect_bt=0.5,
    )
    assert sig.pulse_shape == "smoothrect"
    assert sig.smoothrect_bt == 0.5

    taps = sig.shaping_filter_taps()
    assert len(taps) > 0


def test_rz_rect_taps_length(backend_device, xp):
    """Verify that RZ rectangular pulse taps have the correct half-symbol length."""
    # RZ rect with sps=4 should have length 2 (sps * 0.5)
    sig = Signal.pam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        mode="rz",
        pulse_shape="rect",
    )
    taps = sig.shaping_filter_taps()
    assert len(taps) == 2
    assert xp.allclose(taps, xp.ones(2))


def test_unknown_pulse_shape(backend_device, xp):
    """Verify that attempting to generate taps for unsupported shapes raises errors."""
    sig = Signal(samples=[1, 2], sampling_rate=10, symbol_rate=5)
    # No pulse shape
    with pytest.raises(ValueError, match="No pulse shape defined"):
        sig.shaping_filter_taps()

    sig.pulse_shape = "unknown_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        sig.shaping_filter_taps()


def test_rect_pulse_taps(backend_device, xp):
    """Verify that standard rectangular pulse shaping produces all-ones taps."""
    # Rect pulse should return ones
    sig = Signal.pam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rect",
    )
    taps = sig.shaping_filter_taps()
    assert xp.allclose(taps, xp.ones(4))
