import pytest
import numpy as np
from commstools.core import SingleCarrierFrame


def test_sc_frame_none():
    frame = SingleCarrierFrame(payload_len=100, symbol_rate=1e6, pilot_pattern="none")
    sig = frame.generate_sequence()
    assert len(sig.samples) == 100
    assert sig.symbol_rate == 1e6


def test_sc_frame_comb():
    # payload=10, period=4 -> data_per_period=3
    # 3 periods: 3*3=9 data. remainder=1.
    # total_length = 3*4 + 1 (data) + 1 (leading pilot) = 14
    # Mask [T, F, F, F, T, F, F, F, T, F, F, F, T, F] -> 4 pilots, 10 data
    frame = SingleCarrierFrame(
        payload_len=10, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
    )
    mask, length = frame._generate_pilot_mask()
    assert length == 14
    assert np.sum(mask) == 4

    sig = frame.generate_sequence()
    assert len(sig.samples) == 14


def test_sc_frame_block():
    # payload=10, period=4, len=2 -> data_per_block=2
    # num_blocks = ceil(10/2) = 5
    # Mask [T, T, F, F] * 5 -> [T, T, F, F, T, T, F, F, T, T, F, F, T, T, F, F, T, T, F, F]
    # False indices: 2, 3, 6, 7, 10, 11, 14, 15, 18, 19
    # 10th False is at index 19.
    # Truncated length: 19 + 1 = 20.
    frame = SingleCarrierFrame(
        payload_len=10,
        symbol_rate=1e6,
        pilot_pattern="block",
        pilot_period=4,
        pilot_block_len=2,
    )
    mask, length = frame._generate_pilot_mask()
    assert length == 20
    assert np.sum(mask) == 10

    sig = frame.generate_sequence()
    assert len(sig.samples) == 20


def test_sc_frame_guard_zero():
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, guard_type="zero", guard_len=20
    )
    sig = frame.generate_sequence()
    assert len(sig.samples) == 120
    assert np.all(sig.samples[-20:] == 0)


def test_sc_frame_guard_cp():
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, guard_type="cp", guard_len=20
    )
    sig = frame.generate_sequence()
    assert len(sig.samples) == 120
    # CP should match the last 20 samples of the *original* body
    # Original body length is 100.
    # sig.samples = [body[-20:], body]
    assert np.allclose(sig.samples[:20], sig.samples[-20:])


def test_sc_frame_preamble():
    preamble = np.ones(50) + 1j * np.ones(50)
    frame = SingleCarrierFrame(payload_len=100, symbol_rate=1e6, preamble=preamble)
    sig = frame.generate_sequence()
    assert len(sig.samples) == 150
    assert np.allclose(sig.samples[:50], preamble)
