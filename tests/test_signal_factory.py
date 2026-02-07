import pytest

from commstools.core import Signal


def test_pam_waveform(backend_device, xp):
    sig = Signal.pam(order=2, bipolar=True, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)


def test_rzpam_waveform(backend_device, xp):
    sig = Signal.pam(
        order=2,
        bipolar=True,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        mode="rz",
        pulse_shape="rect",
    )
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)

    # Check invalid pulse shape
    with pytest.raises(ValueError):
        Signal.pam(
            order=2,
            bipolar=True,
            num_symbols=10,
            sps=4,
            symbol_rate=1e3,
            mode="rz",
            pulse_shape="rrc",
        )


def test_qam_waveform(backend_device, xp):
    sig = Signal.qam(order=16, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
