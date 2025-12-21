import pytest
import numpy as np

from commstools import baseband
from commstools.core import Signal


def test_signal_pulse_params(backend_device):
    sig = baseband.pam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rrc",
        rrc_rolloff=0.5,
    ).to(backend_device)
    assert sig.pulse_shape == "rrc"
    assert sig.pulse_params is not None
    assert sig.pulse_params["rrc_rolloff"] == 0.5

    # Check if taps generation works
    taps = sig.shaping_filter_taps()
    assert taps is not None
    assert len(taps) > 0


def test_matched_filter_auto_taps(backend_device):
    sig = baseband.pam(
        order=2,
        bipolar=True,
        num_symbols=100,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rrc",
    ).to(backend_device)

    # Copy signal to compare
    sig_before = sig.copy()

    # Apply matched filter (should auto-generate taps)
    sig.matched_filter()

    # Check that samples changed (filtering happened)
    # It's unlikely that matched filter leaves signal identical
    assert not np.allclose(sig.samples, sig_before.samples)

    # Check backend compatibility (mocking or ensuring it runs)
    # The tests run with numpy by default unless configured otherwise


def test_rzpam_pulse_params(backend_device):
    sig = baseband.rzpam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="smoothrect",
        smoothrect_bt=0.5,
    ).to(backend_device)
    assert sig.pulse_shape == "smoothrect"
    assert sig.pulse_params["smoothrect_bt"] == 0.5

    taps = sig.shaping_filter_taps()
    assert len(taps) > 0
    # Check that pulse width is respected (internally smoothrect uses it)


def test_rz_rect_taps_length(backend_device):
    # RZ rect with sps=4 should have length 2 (sps * 0.5)
    sig = baseband.rzpam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rect",
    ).to(backend_device)
    taps = sig.shaping_filter_taps()
    assert len(taps) == 2
    assert np.allclose(taps, np.ones(2))


def test_unknown_pulse_shape(backend_device):
    sig = Signal(samples=[1, 2], sampling_rate=10, symbol_rate=5).to(backend_device)
    # No pulse shape
    with pytest.raises(ValueError, match="No pulse shape defined"):
        sig.shaping_filter_taps()

    sig.pulse_shape = "unknown_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        sig.shaping_filter_taps()


def test_rect_pulse_taps(backend_device):
    # Rect pulse should return ones
    sig = baseband.pam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rect",
    ).to(backend_device)
    taps = sig.shaping_filter_taps()
    assert np.allclose(taps, np.ones(4))
