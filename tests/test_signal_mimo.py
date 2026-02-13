"""Tests for multi-stream (MIMO) Signal and Frame generation support."""

import pytest
from pydantic import ValidationError

from commstools.core import Preamble, Signal, SingleCarrierFrame


def test_signal_generate_mimo(backend_device, xp):
    """Verify MIMO signal generation via high-level factories."""
    # Test MIMO generation via factories
    sig = Signal.qam(order=4, num_symbols=100, sps=4, symbol_rate=1e6, num_streams=2)

    # Check shape: (num_streams, num_symbols * sps)
    expected_samples = 100 * 4
    assert sig.samples.shape == (2, expected_samples)
    assert sig.num_streams == 2
    assert sig.sps == 4.0

    # Check if streams are not identical (random seed should diverge or be handled)
    assert not xp.allclose(sig.samples[0], sig.samples[1])


def test_frame_mimo_generation(backend_device, xp):
    """Verify basic MIMO frame generation with guard intervals."""
    frame = SingleCarrierFrame(
        payload_len=100,
        symbol_rate=1e6,
        num_streams=2,
        pilot_pattern="none",
        guard_type="zero",
        guard_len=10,
    )

    sig = frame.to_signal(sps=1, pulse_shape="none")
    # Length: 100 payload + 10 guard = 110 symbols
    assert sig.samples.shape == (2, 110)
    assert sig.num_streams == 2

    # Check guard interval (zeros)
    # Last 10 samples (axis=-1)
    assert xp.all(sig.samples[:, -10:] == 0)


def test_frame_mimo_pilots(backend_device, xp):
    """Verify that pilot patterns are correctly applied across all MIMO streams."""
    frame = SingleCarrierFrame(
        payload_len=10,
        symbol_rate=1e6,
        num_streams=2,
        pilot_pattern="comb",
        pilot_period=2,
    )
    # len=10 payload. period=2 -> 1 pilot, 1 data.
    # total len = 10 data -> 10 periods -> 20 symbols.

    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert sig.samples.shape == (2, 20)

    # Check mask and body
    mask, _ = frame._generate_pilot_mask()
    # Mask is 1D (time), applicable to all streams
    assert len(mask) == 20
    assert xp.sum(mask) == 10


def test_frame_mimo_preamble_broadcasting(backend_device, xp):
    """Verify that a 1D preamble is correctly broadcasted across all MIMO streams."""
    # Create Barker-13 preamble
    preamble = Preamble(sequence_type="barker", length=13)
    frame = SingleCarrierFrame(
        payload_len=20, symbol_rate=1e6, num_streams=2, preamble=preamble
    )

    sig = frame.to_signal(sps=1, pulse_shape="none")
    # Total: 13 preamble + 20 payload = 33
    assert sig.samples.shape == (2, 33)

    # Check preamble on both streams (should be broadcast)
    # (Channels, Time)
    assert xp.allclose(sig.samples[0, :13], preamble.symbols)
    assert xp.allclose(sig.samples[1, :13], preamble.symbols)


def test_frame_mimo_waveform(backend_device, xp):
    """Verify MIMO waveform generation with pulse shaping."""
    frame = SingleCarrierFrame(payload_len=10, symbol_rate=1e6, num_streams=2)

    sig = frame.to_signal(sps=4, pulse_shape="rect")
    # 10 symbols * 4 sps = 40 samples
    assert sig.samples.shape == (2, 40)
    assert sig.sps == 4.0


def test_signal_mimo_transpose(backend_device, xp):
    """
    Test the transposition heuristic in Signal.validate_samples.
    If we pass (100, 2), it should detect Time > Channels and transpose to (2, 100).
    """
    data = xp.zeros((100, 2))
    # Note: logger.warning is used, not warnings.warn, so pytest.warns won't catch it.
    # We just verify the transposition logic happened.
    sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert sig.samples.shape == (2, 100)


def test_signal_invalid_ndim(backend_device, xp):
    """Test >2D array raises ValueError (wrapped in ValidationError)."""
    data = xp.zeros((2, 2, 2))
    with pytest.raises(ValidationError) as excparams:
        Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert "3 dimensions" in str(excparams.value)
