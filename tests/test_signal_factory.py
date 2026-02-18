"""Tests for Signal factory methods (PAM, QAM, Waveform generation)."""

import pytest

from commstools.core import Signal


def test_pam_waveform(backend_device, xp):
    """Verify basic PAM signal generation."""
    sig = Signal.pam(order=2, unipolar=False, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)


def test_rzpam_waveform(backend_device, xp):
    """Verify Return-to-Zero (RZ) PAM signal generation."""
    sig = Signal.pam(
        order=2,
        unipolar=False,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        rz=True,
        pulse_shape="rect",
    )
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)

    # Check invalid pulse shape for RZ mode
    with pytest.raises(ValueError, match="not allowed for RZ PAM"):
        Signal.pam(
            order=2,
            unipolar=False,
            num_symbols=10,
            sps=4,
            symbol_rate=1e3,
            rz=True,
            pulse_shape="rrc",
        )


def test_qam_waveform(backend_device, xp):
    """Verify standard QAM signal generation."""
    sig = Signal.qam(order=16, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
