import pytest

from commstools import baseband


def test_pam_waveform(backend_device, xp):
    # Only testing generation runs and returns Signal
    # waveforms usually runs on CPU unless specified?
    # baseband.generate_baseband internal logic might use numpy or backend.

    # We should ensure backend is clear.
    # Current implementation of waveforms seems to support backend via filtering calls,
    # but generate_waveform creates 'bits' using sequences.random_bits which is numpy based?
    # Let's check if we can force it.
    # Actually waveforms.py uses 'get_backend()' inside some functions.

    # If we run tests with --device=gpu, we should check if waveforms works.
    # Note: mixing numpy inputs with GPU backend might cause issues if not handled.

    sig = baseband.pam(order=2, bipolar=True, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    # By default it might be on CPU if inputs were CPU.
    # We can check sig.to(backend_device) works.
    sig.to(backend_device)
    assert isinstance(sig.samples, xp.ndarray)


def test_rzpam_waveform(backend_device, xp):
    sig = baseband.rzpam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rect",
    )
    assert sig.samples.size > 0
    sig.to(backend_device)
    assert isinstance(sig.samples, xp.ndarray)

    # Check invalid pulse shape
    with pytest.raises(ValueError):
        baseband.rzpam(
            order=2,
            bipolar=True,
            num_symbols=10,
            sps=4,
            symbol_rate=1e3,
            pulse_shape="rrc",
        )


def test_qam_waveform(backend_device, xp):
    sig = baseband.qam(order=16, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    sig.to(backend_device)
    assert isinstance(sig.samples, xp.ndarray)
